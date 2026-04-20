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
            // Check if model exists, download if missing
            val modelDir = File(getExternalFilesDir(null), "models")
            val modelFile = modelDir.listFiles()?.firstOrNull { it.name.endsWith(".gguf") }
            if (modelFile == null) {
                updateNotification("Downloading Gemma 4B model...")
                Thread {
                    val mgr = ModelManager(this)
                    mgr.downloadIfMissing { progress ->
                        updateNotification("Downloading model: ${progress}%")
                    }
                    updateNotification("Model ready — starting inference")
                    startLlamaServer()
                }.start()
            } else {
                startLlamaServer()
            }
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
            val binDir = extractLlamaBinaries()
            if (binDir == null) {
                Log.w(TAG, "llama-server binaries not found in assets")
                return
            }

            val llamaBinary = File(binDir, "llama-server")
            if (!llamaBinary.exists() || !llamaBinary.canExecute()) {
                Log.w(TAG, "llama-server binary missing or not executable")
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
            val env = arrayOf("LD_LIBRARY_PATH=${binDir.absolutePath}")
            llamaProcess = Runtime.getRuntime().exec(
                arrayOf(
                    llamaBinary.absolutePath,
                    "-m", modelFile.absolutePath,
                    "--host", "127.0.0.1",
                    "--port", "8080",
                    "-ngl", "99",
                ),
                env
            )

            // Log llama-server output in background
            Thread {
                llamaProcess?.inputStream?.bufferedReader()?.use { reader ->
                    reader.forEachLine { line ->
                        Log.i("$TAG.llama", line)
                        if (line.contains("listening on")) {
                            updateNotification("Inference engine ready")
                        }
                    }
                }
            }.apply { isDaemon = true; start() }

            Log.i(TAG, "llama-server started (pid=${llamaProcess?.toString()})")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to start llama-server: ${e.message}", e)
        }
    }

    /**
     * Extract all files from assets/bin/ (llama-server + shared libs).
     * Returns the directory containing the extracted binaries, or null if not bundled.
     */
    private fun extractLlamaBinaries(): File? {
        val targetDir = File(filesDir, "bin")
        val marker = File(targetDir, ".extracted")

        // Skip if already extracted (same app version)
        if (marker.exists() && File(targetDir, "llama-server").canExecute()) {
            Log.i(TAG, "llama-server binaries already extracted")
            return targetDir
        }

        return try {
            targetDir.mkdirs()
            val assetFiles = assets.list("bin") ?: emptyArray()
            if (assetFiles.isEmpty()) {
                Log.w(TAG, "No files in assets/bin/")
                return null
            }

            for (filename in assetFiles) {
                val targetFile = File(targetDir, filename)
                assets.open("bin/$filename").use { input ->
                    targetFile.outputStream().use { output ->
                        input.copyTo(output)
                    }
                }
                targetFile.setExecutable(true)
                Log.i(TAG, "Extracted: $filename (${targetFile.length()} bytes)")
            }

            // Write marker so we don't re-extract next time
            marker.writeText(System.currentTimeMillis().toString())
            Log.i(TAG, "All llama binaries extracted to ${targetDir.absolutePath}")
            targetDir
        } catch (e: Exception) {
            Log.w(TAG, "Failed to extract llama binaries: ${e.message}")
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
