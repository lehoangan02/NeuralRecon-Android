package com.lhan.arrec

import android.graphics.ImageFormat
import android.graphics.Rect
import android.graphics.YuvImage
import android.media.Image
import com.google.ar.core.Camera
import com.google.ar.core.Frame
import com.google.ar.core.exceptions.NotYetAvailableException
import java.io.ByteArrayOutputStream
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.Locale
import java.util.concurrent.Executors

/**
 * Saves, for each captured ARCore frame, the CPU camera image as a JPEG and one CSV row holding the
 * camera pose and intrinsics from the same Frame. Both are in the unrotated sensor-image frame
 * (Camera.getPose + Camera.getImageIntrinsics), which is what tools/process_arcore_data.py expects.
 *
 * poses.csv columns:
 *   idx,timestamp_ns,tx,ty,tz,qx,qy,qz,qw,fx,fy,cx,cy,img_w,img_h,intr_w,intr_h
 * pose = camera-to-world, ARCore world (Y up), OpenGL camera (+X right, +Y up, -Z forward).
 * If a Streamer is given, every saved frame is also sent to the PC for the live preview.
 */
class Recorder(
    private val dir: File,
    private val streamer: Streamer? = null,
    private val minIntervalNs: Long = 80_000_000L
) {
    private val imagesDir = File(dir, "images").also { it.mkdirs() }
    private val csv = File(dir, "poses.csv").bufferedWriter().also {
        it.write("idx,timestamp_ns,tx,ty,tz,qx,qy,qz,qw,fx,fy,cx,cy,img_w,img_h,intr_w,intr_h\n")
    }
    private val io = Executors.newSingleThreadExecutor()
    private var lastTs = 0L
    var count = 0
        private set

    fun capture(frame: Frame, cam: Camera) {
        val ts = frame.timestamp
        if (ts - lastTs < minIntervalNs) return

        val image = try {
            frame.acquireCameraImage()
        } catch (e: NotYetAvailableException) {
            return
        }
        val nv21: ByteArray
        val w: Int
        val h: Int
        try {
            w = image.width
            h = image.height
            nv21 = toNv21(image)
        } finally {
            image.close()
        }

        val pose = cam.pose
        val t = pose.translation
        val q = pose.rotationQuaternion
        val intr = cam.imageIntrinsics
        val f = intr.focalLength
        val c = intr.principalPoint
        val dims = intr.imageDimensions

        val idx = count++
        lastTs = ts
        val row = String.format(
            Locale.US, "%d,%d,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.6f,%.6f,%.6f,%.6f,%d,%d,%d,%d\n",
            idx, ts, t[0], t[1], t[2], q[0], q[1], q[2], q[3], f[0], f[1], c[0], c[1], w, h, dims[0], dims[1]
        )
        io.execute {
            val jpg = ByteArrayOutputStream(64 * 1024).also {
                YuvImage(nv21, ImageFormat.NV21, w, h, null).compressToJpeg(Rect(0, 0, w, h), 95, it)
            }.toByteArray()
            File(imagesDir, String.format(Locale.US, "%06d.jpg", idx)).writeBytes(jpg)
            csv.write(row)
            streamer?.offer(packet(idx, ts, t, q, f, c, w, h, jpg))
        }
    }

    /** Wire format read by live_demo.py: struct '<4sIq11fHHI' followed by the JPEG bytes. */
    private fun packet(
        idx: Int, ts: Long, t: FloatArray, q: FloatArray, f: FloatArray, c: FloatArray, w: Int, h: Int, jpg: ByteArray
    ): ByteArray {
        val b = ByteBuffer.allocate(HEADER_BYTES + jpg.size).order(ByteOrder.LITTLE_ENDIAN)
        b.put(MAGIC).putInt(idx).putLong(ts)
        for (v in t) b.putFloat(v)
        for (v in q) b.putFloat(v)
        b.putFloat(f[0]).putFloat(f[1]).putFloat(c[0]).putFloat(c[1])
        b.putShort(w.toShort()).putShort(h.toShort()).putInt(jpg.size)
        b.put(jpg)
        return b.array()
    }

    fun finish() {
        io.execute {
            csv.close()
            streamer?.close()  // after the last frame was queued, so the PC receives all of them
        }
        io.shutdown()
    }

    private fun toNv21(image: Image): ByteArray {
        val w = image.width
        val h = image.height
        val out = ByteArray(w * h * 3 / 2)
        val yp = image.planes[0]
        val up = image.planes[1]
        val vp = image.planes[2]

        val yBuf = yp.buffer
        for (r in 0 until h) {
            yBuf.position(r * yp.rowStride)
            yBuf.get(out, r * w, w)
        }

        val uBuf = up.buffer
        val vBuf = vp.buffer
        var o = w * h
        for (r in 0 until h / 2) {
            for (col in 0 until w / 2) {
                out[o++] = vBuf.get(r * vp.rowStride + col * vp.pixelStride)
                out[o++] = uBuf.get(r * up.rowStride + col * up.pixelStride)
            }
        }
        return out
    }

    companion object {
        private val MAGIC = "NRP1".toByteArray(Charsets.US_ASCII)
        private const val HEADER_BYTES = 68
    }
}
