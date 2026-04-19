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
        private const val MODEL_FILENAME = "gemma-4b.gguf"
        private const val MODEL_URL = "https://huggingface.co/google/gemma-3-4b-it-GGUF/resolve/main/gemma-3-4b-it.gguf"
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

                    while (input.read(buffer).also { count = it } != -1) {
                        output.write(buffer, 0, count)
                        bytesRead += count
                        listener.onProgress(bytesRead, totalBytes)
                    }
                }
            }

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
}
