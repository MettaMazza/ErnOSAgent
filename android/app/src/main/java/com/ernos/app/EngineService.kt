package com.ernos.app

import android.app.*
import android.content.Intent
import android.os.Build
import android.os.IBinder
import android.util.Log
import androidx.core.app.NotificationCompat
import java.io.File

/**
 * Foreground service that runs the Ern-OS Rust engine + llama-server.
 * Loads libernos.so via JNI and calls startEngine() which boots Axum on :3000.
 * Also manages the llama-server binary for local/hybrid compute modes.
 */
class EngineService : Service() {

    companion object {
        private const val TAG = "ErnOS.Engine"
        private const val CHANNEL_ID = "ernos_engine"
        private const val NOTIFICATION_ID = 1
    }

    // JNI bridge to Rust engine
    external fun startEngine(dataDir: String, providerUrl: String, computeMode: String)

    init {
        System.loadLibrary("ernos")
    }

    private var engineThread: Thread? = null
    private var llamaProcess: Process? = null

    override fun onCreate() {
        super.onCreate()
        createNotificationChannel()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        val notification = buildNotification("Starting...")
        startForeground(NOTIFICATION_ID, notification)

        val prefs = getSharedPreferences("ernos", MODE_PRIVATE)
        val mode = prefs.getString("compute_mode", "local") ?: "local"
        val hostIp = prefs.getString("host_ip", "") ?: ""

        // Determine provider URL based on compute mode
        val providerUrl = when (mode) {
            "host" -> "http://$hostIp:8080/v1/chat/completions"
            "hybrid" -> "http://127.0.0.1:8080/v1/chat/completions"
            else -> "http://127.0.0.1:8080/v1/chat/completions" // local
        }

        // Start llama-server for local/hybrid modes
        if (mode != "host") {
            startLlamaServer()
        }

        // Start Rust engine in background thread
        val dataDir = filesDir.absolutePath
        engineThread = Thread {
            Log.i(TAG, "Starting Ern-OS engine: mode=$mode, dataDir=$dataDir")
            try {
                startEngine(dataDir, providerUrl, mode)
            } catch (e: Exception) {
                Log.e(TAG, "Engine crashed: ${e.message}", e)
            }
        }.apply {
            name = "ernos-engine"
            isDaemon = true
            start()
        }

        updateNotification("Running — $mode mode")
        return START_STICKY
    }

    private fun startLlamaServer() {
        try {
            val llamaBinary = extractLlamaServer()
            if (llamaBinary == null) {
                Log.w(TAG, "llama-server binary not found in assets")
                return
            }

            val modelDir = File(getExternalFilesDir(null), "models")
            val modelFile = modelDir.listFiles()?.firstOrNull { it.name.endsWith(".gguf") }

            if (modelFile == null) {
                Log.w(TAG, "No model file (.gguf) found in ${modelDir.absolutePath}")
                updateNotification("Waiting for model download...")
                return
            }

            Log.i(TAG, "Starting llama-server with model: ${modelFile.name}")
            llamaProcess = ProcessBuilder(
                llamaBinary.absolutePath,
                "-m", modelFile.absolutePath,
                "--host", "127.0.0.1",
                "--port", "8080",
                "-ngl", "99",
            ).redirectErrorStream(true).start()

            Log.i(TAG, "llama-server started (PID: ${llamaProcess?.pid()})")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to start llama-server: ${e.message}", e)
        }
    }

    private fun extractLlamaServer(): File? {
        val targetFile = File(filesDir, "bin/llama-server")
        if (targetFile.exists() && targetFile.canExecute()) {
            return targetFile
        }

        return try {
            targetFile.parentFile?.mkdirs()
            assets.open("bin/llama-server").use { input ->
                targetFile.outputStream().use { output ->
                    input.copyTo(output)
                }
            }
            targetFile.setExecutable(true)
            Log.i(TAG, "Extracted llama-server to ${targetFile.absolutePath}")
            targetFile
        } catch (e: Exception) {
            Log.w(TAG, "llama-server not bundled: ${e.message}")
            null
        }
    }

    private fun createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                CHANNEL_ID, "Ern-OS Engine",
                NotificationManager.IMPORTANCE_LOW
            ).apply {
                description = "Ern-OS AI engine running in background"
            }
            val nm = getSystemService(NotificationManager::class.java)
            nm.createNotificationChannel(channel)
        }
    }

    private fun buildNotification(text: String): Notification {
        val pendingIntent = PendingIntent.getActivity(
            this, 0,
            Intent(this, MainActivity::class.java),
            PendingIntent.FLAG_IMMUTABLE
        )
        return NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle("Ern-OS")
            .setContentText(text)
            .setSmallIcon(android.R.drawable.ic_dialog_info)
            .setContentIntent(pendingIntent)
            .setOngoing(true)
            .build()
    }

    private fun updateNotification(text: String) {
        val nm = getSystemService(NotificationManager::class.java)
        nm.notify(NOTIFICATION_ID, buildNotification(text))
    }

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onDestroy() {
        llamaProcess?.destroyForcibly()
        engineThread?.interrupt()
        super.onDestroy()
    }
}
