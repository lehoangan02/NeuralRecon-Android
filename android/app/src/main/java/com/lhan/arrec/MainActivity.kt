package com.lhan.arrec

import android.Manifest
import android.app.Activity
import android.content.pm.PackageManager
import android.graphics.Color
import android.opengl.GLES20
import android.opengl.GLSurfaceView
import android.os.Bundle
import android.util.Log
import android.view.Gravity
import android.widget.Button
import android.widget.FrameLayout
import android.widget.TextView
import com.google.ar.core.ArCoreApk
import com.google.ar.core.CameraConfig
import com.google.ar.core.CameraConfigFilter
import com.google.ar.core.Config
import com.google.ar.core.Session
import com.google.ar.core.TrackingState
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import javax.microedition.khronos.egl.EGLConfig
import javax.microedition.khronos.opengles.GL10

class MainActivity : Activity(), GLSurfaceView.Renderer {
    private lateinit var surface: GLSurfaceView
    private lateinit var status: TextView
    private lateinit var button: Button

    private var session: Session? = null
    private var installRequested = false
    private var textureSet = false
    private var viewportChanged = false
    private var vpWidth = 0
    private var vpHeight = 0
    private val background = BackgroundRenderer()

    @Volatile private var recorder: Recorder? = null
    private var streamer: Streamer? = null
    private var lastStatusMs = 0L
    private var configSummary = ""

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        surface = GLSurfaceView(this).apply {
            preserveEGLContextOnPause = true
            setEGLContextClientVersion(2)
            setEGLConfigChooser(8, 8, 8, 8, 16, 0)
            setRenderer(this@MainActivity)
            renderMode = GLSurfaceView.RENDERMODE_CONTINUOUSLY
            setWillNotDraw(false)
        }
        status = TextView(this).apply {
            setTextColor(Color.WHITE)
            setBackgroundColor(0x99000000.toInt())
            setPadding(24, 24, 24, 24)
            textSize = 13f
            text = "starting..."
        }
        button = Button(this).apply {
            text = "START RECORDING"
            setOnClickListener { toggleRecording() }
        }
        val root = FrameLayout(this)
        root.addView(surface)
        root.addView(status, FrameLayout.LayoutParams(-1, -2, Gravity.TOP))
        root.addView(button, FrameLayout.LayoutParams(-1, -2, Gravity.BOTTOM).apply { setMargins(48, 0, 48, 96) })
        setContentView(root)
    }

    override fun onResume() {
        super.onResume()
        if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(arrayOf(Manifest.permission.CAMERA), 1)
            return
        }
        if (session == null) {
            try {
                when (ArCoreApk.getInstance().requestInstall(this, !installRequested)) {
                    ArCoreApk.InstallStatus.INSTALL_REQUESTED -> {
                        installRequested = true
                        return
                    }
                    ArCoreApk.InstallStatus.INSTALLED -> {}
                }
                session = createSession()
                textureSet = false
            } catch (e: Exception) {
                setStatus("ARCore init failed: $e")
                Log.e(TAG, "init", e)
                return
            }
        }
        try {
            session?.resume()
        } catch (e: Exception) {
            setStatus("resume failed: $e")
            session = null
            return
        }
        surface.onResume()
    }

    override fun onPause() {
        super.onPause()
        if (recorder != null) toggleRecording()
        surface.onPause()
        session?.pause()
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        if (grantResults.firstOrNull() == PackageManager.PERMISSION_GRANTED) recreate()
        else setStatus("camera permission is required")
    }

    private fun createSession(): Session {
        val s = Session(this)
        val filter = CameraConfigFilter(s).setFacingDirection(CameraConfig.FacingDirection.BACK)
        val configs = s.getSupportedCameraConfigs(filter)

        val sb = StringBuilder()
        for (c in configs) {
            sb.append("cpu=${c.imageSize} gpu=${c.textureSize} fps=${c.fpsRange} depth=${c.depthSensorUsage}\n")
        }
        Log.i(TAG, "Supported camera configs:\n$sb")

        // Prefer the largest 4:3 CPU image up to 1920x1440 (matches the 4:3 input NeuralRecon resizes to);
        // otherwise fall back to the largest available.
        val fourThree = configs.filter {
            it.imageSize.width * 3 == it.imageSize.height * 4 && it.imageSize.width <= 1920
        }
        val chosen = (if (fourThree.isNotEmpty()) fourThree else configs)
            .maxByOrNull { it.imageSize.width * it.imageSize.height }
        if (chosen != null) s.cameraConfig = chosen
        val cpu = s.cameraConfig.imageSize
        configSummary = "CPU image ${cpu.width}x${cpu.height} (${configs.size} configs, see cameraconfigs.txt)"
        Log.i(TAG, "Chosen: $configSummary")

        // Fixed focus keeps the focal length stable, which matters for reconstruction.
        s.configure(Config(s).apply { focusMode = Config.FocusMode.FIXED })

        File(getExternalFilesDir(null), "cameraconfigs.txt").writeText("chosen=${cpu.width}x${cpu.height}\n$sb")
        return s
    }

    private fun toggleRecording() {
        val r = recorder
        if (r == null) {
            val name = "rec_" + SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
            val dir = File(getExternalFilesDir(null), name).also { it.mkdirs() }
            streamer = Streamer()
            recorder = Recorder(dir, streamer)
            button.text = "STOP"
            setStatus("recording to $name")
        } else {
            recorder = null
            r.finish()
            streamer = null
            button.text = "START RECORDING"
            setStatus("saved ${r.count} frames")
        }
    }

    private fun setStatus(msg: String) = runOnUiThread { status.text = msg }

    override fun onSurfaceCreated(gl: GL10?, config: EGLConfig?) {
        GLES20.glClearColor(0.1f, 0.1f, 0.1f, 1f)
        background.createOnGlThread()
        textureSet = false
    }

    override fun onSurfaceChanged(gl: GL10?, width: Int, height: Int) {
        GLES20.glViewport(0, 0, width, height)
        vpWidth = width
        vpHeight = height
        viewportChanged = true
    }

    @Suppress("DEPRECATION")
    override fun onDrawFrame(gl: GL10?) {
        GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT or GLES20.GL_DEPTH_BUFFER_BIT)
        val s = session ?: return
        if (!textureSet) {
            s.setCameraTextureName(background.textureId)
            textureSet = true
        }
        if (viewportChanged) {
            val rot = windowManager.defaultDisplay.rotation
            s.setDisplayGeometry(rot, vpWidth, vpHeight)
            viewportChanged = false
        }
        val frame = try {
            s.update()
        } catch (e: Exception) {
            return
        }
        background.draw(frame)

        val cam = frame.camera
        val tracking = cam.trackingState == TrackingState.TRACKING
        val r = recorder
        if (r != null && tracking) r.capture(frame, cam)

        val now = System.currentTimeMillis()
        if (now - lastStatusMs > 300) {
            lastStatusMs = now
            val rec = if (r != null) "REC ${r.count} frames" else "idle"
            val pc = if (r == null) "" else if (streamer?.connected == true) "\nPC: streaming" else "\nPC: not connected (saving on phone only)"
            setStatus("$configSummary\ntracking=${cam.trackingState} ($rec)$pc")
        }
    }

    companion object {
        private const val TAG = "ArRecorder"
    }
}
