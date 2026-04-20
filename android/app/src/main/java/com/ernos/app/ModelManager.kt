package com.ernos.app

import android.content.Context
import android.util.Log
import java.io.File
import java.net.URL
import java.security.MessageDigest

/**
 * Manages Gemma 4B model download on first launch.
 * Downloads from HuggingFace, verifies SHA256, caches locally.
 */
class ModelManager(private val context: Context) {

    companion object {
        private const val TAG = "ErnOS.Model"
        private const val MODEL_FILENAME = "gemma-3-4b-it-Q4_K_M.gguf"
        private const val MODEL_URL = "https://huggingface.co/ggml-org/gemma-3-4b-it-GGUF/resolve/main/gemma-3-4b-it-Q4_K_M.gguf"
        private const val EXPECTED_SHA256 = "" // Set when URL is finalized
    }

    interface DownloadListener {
        fun onProgress(bytesDownloaded: Long, totalBytes: Long)
        fun onComplete(modelFile: File)
        fun onError(error: String)
    }

    fun getModelDir(): File {
        val dir = File(context.getExternalFilesDir(null), "models")
        dir.mkdirs()
        return dir
    }

    fun isModelDownloaded(): Boolean {
        val modelFile = File(getModelDir(), MODEL_FILENAME)
        return modelFile.exists() && modelFile.length() > 1_000_000_000 // >1GB sanity check
    }

    fun getModelFile(): File? {
        val modelFile = File(getModelDir(), MODEL_FILENAME)
        return if (modelFile.exists()) modelFile else null
    }

    /**
     * Download the model in background. Must be called from a background thread.
     */
    fun downloadModel(listener: DownloadListener) {
        val modelFile = File(getModelDir(), MODEL_FILENAME)
        val tempFile = File(getModelDir(), "$MODEL_FILENAME.download")

        try {
            Log.i(TAG, "Starting model download: $MODEL_URL")

            val connection = URL(MODEL_URL).openConnection()
            connection.connectTimeout = 30_000
            connection.readTimeout = 60_000
            val totalBytes = connection.contentLengthLong

            connection.getInputStream().use { input ->
                tempFile.outputStream().use { output ->
                    val buffer = ByteArray(8192)
                    var bytesRead: Long = 0
                    var count: Int
                    var lastProgressWrite: Long = 0

                    while (input.read(buffer).also { count = it } != -1) {
                        output.write(buffer, 0, count)
                        bytesRead += count
                        listener.onProgress(bytesRead, totalBytes)

                        // Write progress file every 500KB for WebUI polling
                        if (bytesRead - lastProgressWrite > 512_000) {
                            lastProgressWrite = bytesRead
                            writeProgressFile(bytesRead, totalBytes)
                        }
                    }
                }
            }
            // Clear progress file on completion
            progressFile().delete()

            // Verify SHA256 if configured
            if (EXPECTED_SHA256.isNotEmpty()) {
                val actualHash = sha256(tempFile)
                if (actualHash != EXPECTED_SHA256) {
                    tempFile.delete()
                    listener.onError("SHA256 mismatch: expected $EXPECTED_SHA256, got $actualHash")
                    return
                }
                Log.i(TAG, "SHA256 verified: $actualHash")
            }

            // Move to final location
            tempFile.renameTo(modelFile)
            Log.i(TAG, "Model downloaded: ${modelFile.absolutePath} (${modelFile.length()} bytes)")
            listener.onComplete(modelFile)

        } catch (e: Exception) {
            tempFile.delete()
            Log.e(TAG, "Download failed: ${e.message}", e)
            listener.onError("Download failed: ${e.message}")
        }
    }

    private fun progressFile(): File {
        return File(context.filesDir, "model_download_progress.json")
    }

    private fun writeProgressFile(bytesDownloaded: Long, totalBytes: Long) {
        try {
            val pct = if (totalBytes > 0) ((bytesDownloaded * 100) / totalBytes).toInt() else 0
            val dlMb = bytesDownloaded / (1024 * 1024)
            val totalMb = if (totalBytes > 0) totalBytes / (1024 * 1024) else 0
            progressFile().writeText("""{"downloading":true,"progress":$pct,"downloaded_mb":$dlMb,"total_mb":$totalMb}""")
        } catch (_: Exception) {}
    }

    private fun sha256(file: File): String {
        val digest = MessageDigest.getInstance("SHA-256")
        file.inputStream().use { input ->
            val buffer = ByteArray(8192)
            var count: Int
            while (input.read(buffer).also { count = it } != -1) {
                digest.update(buffer, 0, count)
            }
        }
        return digest.digest().joinToString("") { "%02x".format(it) }
    }

    /**
     * Download the model if not present. Blocks until complete.
     * Reports progress as an integer percentage (0-100) via the callback.
     */
    fun downloadIfMissing(onProgress: (Int) -> Unit) {
        if (isModelDownloaded()) {
            Log.i(TAG, "Model already downloaded")
            return
        }

        Log.i(TAG, "Model not found — starting download")
        val latch = java.util.concurrent.CountDownLatch(1)
        var error: String? = null

        downloadModel(object : DownloadListener {
            override fun onProgress(bytesDownloaded: Long, totalBytes: Long) {
                if (totalBytes > 0) {
                    onProgress(((bytesDownloaded * 100) / totalBytes).toInt())
                }
            }
            override fun onComplete(modelFile: File) {
                Log.i(TAG, "Model download complete: ${modelFile.name}")
                latch.countDown()
            }
            override fun onError(err: String) {
                error = err
                Log.e(TAG, "Model download error: $err")
                latch.countDown()
            }
        })

        latch.await()
        if (error != null) {
            Log.e(TAG, "Model download failed: $error")
        }
    }
}
