package com.lhan.arrec

import java.io.IOException
import java.net.Socket
import java.util.concurrent.ArrayBlockingQueue
import java.util.concurrent.TimeUnit

/**
 * Sends packets to live_demo.py on the PC. The PC is reached over USB with `adb reverse tcp:PORT tcp:PORT`
 * (live_demo.py runs that itself), so 127.0.0.1 on the phone is the PC.
 *
 * offer() never blocks the camera loop: if the link is slow the oldest packets are dropped. Frames are saved
 * on the phone regardless, so a dropped packet only costs live preview detail. close() lets the sender drain
 * what is queued and then end the connection, which tells the PC the recording is over.
 */
class Streamer(private val host: String = "127.0.0.1", private val port: Int = 8777) {
    private val queue = ArrayBlockingQueue<ByteArray>(120)

    @Volatile private var running = true
    @Volatile var connected = false
        private set

    init {
        Thread { loop() }.apply {
            isDaemon = true
            start()
        }
    }

    fun offer(packet: ByteArray) {
        while (!queue.offer(packet)) queue.poll()
    }

    fun close() {
        running = false
    }

    private fun loop() {
        while (running || queue.isNotEmpty()) {
            try {
                Socket(host, port).use { socket ->
                    socket.tcpNoDelay = true
                    connected = true
                    val out = socket.getOutputStream()
                    while (running || queue.isNotEmpty()) {
                        val packet = queue.poll(200, TimeUnit.MILLISECONDS) ?: continue
                        out.write(packet)
                    }
                    out.flush()
                }
                return
            } catch (e: IOException) {
                // PC not listening yet or link dropped; keep the queue and retry
            } finally {
                connected = false
            }
            if (!running && queue.isEmpty()) return
            try {
                Thread.sleep(500)
            } catch (e: InterruptedException) {
                return
            }
        }
    }
}
