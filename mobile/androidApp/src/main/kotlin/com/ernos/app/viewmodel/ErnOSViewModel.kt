package com.ernos.app.viewmodel

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

/**
 * ErnOS ViewModel — bridges Compose UI ↔ Rust core.
 *
 * All state management happens in Rust. This ViewModel:
 * 1. Holds a reference to the UniFFI-generated ErnOSEngine
 * 2. Exposes engine state as Compose-observable StateFlows
 * 3. Delegates all actions to the Rust engine
 *
 * When UniFFI is integrated, replace the stub calls with:
 *   `uniffi.ernosagent.ErnOSEngine`
 */
class ErnOSViewModel(application: Application) : AndroidViewModel(application) {

    // ── Engine State ──

    private val _isReady = MutableStateFlow(false)
    val isReady: StateFlow<Boolean> = _isReady.asStateFlow()

    private val _inferenceMode = MutableStateFlow("Hybrid")
    val inferenceMode: StateFlow<String> = _inferenceMode.asStateFlow()

    private val _loadedModel = MutableStateFlow<String?>(null)
    val loadedModel: StateFlow<String?> = _loadedModel.asStateFlow()

    private val _isDesktopConnected = MutableStateFlow(false)
    val isDesktopConnected: StateFlow<Boolean> = _isDesktopConnected.asStateFlow()

    private val _statusSummary = MutableStateFlow("No model loaded")
    val statusSummary: StateFlow<String> = _statusSummary.asStateFlow()

    // ── Chat State ──

    private val _messages = MutableStateFlow<List<ChatMessage>>(emptyList())
    val messages: StateFlow<List<ChatMessage>> = _messages.asStateFlow()

    private val _isGenerating = MutableStateFlow(false)
    val isGenerating: StateFlow<Boolean> = _isGenerating.asStateFlow()

    private val _streamingContent = MutableStateFlow("")
    val streamingContent: StateFlow<String> = _streamingContent.asStateFlow()

    // ── Download State ──

    private val _downloadProgress = MutableStateFlow(0f)
    val downloadProgress: StateFlow<Float> = _downloadProgress.asStateFlow()

    private val _isDownloading = MutableStateFlow(false)
    val isDownloading: StateFlow<Boolean> = _isDownloading.asStateFlow()

    // ── Actions ──

    /** Initialize the engine with the app's data directory. */
    fun initialize() {
        viewModelScope.launch {
            val dataDir = getApplication<Application>().filesDir.absolutePath
            // TODO: UniFFI call — ErnOSEngine.new(dataDir)
            _isReady.value = false // Will become true when model is loaded
        }
    }

    /** Set the inference mode. */
    fun setInferenceMode(mode: String) {
        _inferenceMode.value = mode
        // TODO: UniFFI call — engine.setInferenceMode(mode)
    }

    /** Download and load a model. */
    fun downloadModel(modelName: String) {
        viewModelScope.launch {
            _isDownloading.value = true
            _downloadProgress.value = 0f

            // TODO: UniFFI call — engine.downloadModel(modelName, progressCallback)
            // Simulate progress for now
            for (i in 0..100) {
                kotlinx.coroutines.delay(50)
                _downloadProgress.value = i / 100f
            }

            _isDownloading.value = false
            _loadedModel.value = modelName
            _isReady.value = true
            _statusSummary.value = "$modelName │ ${_inferenceMode.value}"
        }
    }

    /** Send a chat message. */
    fun sendMessage(content: String) {
        if (_isGenerating.value) return

        viewModelScope.launch {
            _isGenerating.value = true
            _messages.value = _messages.value + ChatMessage("user", content)
            _streamingContent.value = ""

            // TODO: UniFFI call — engine.chat(content, streamCallback)
            // The Rust engine will:
            //   1. Build context (system prompt + memory + history)
            //   2. Run ReAct loop with Observer audit
            //   3. Stream tokens back via callback
            //   4. Return full response

            // Stub: simulate streaming
            val stubResponse = "This is a placeholder response from the ErnOS mobile engine stub. " +
                "When UniFFI is integrated, this will run the full ReAct loop through either " +
                "the on-device Gemma 4 E2B model or relay to your desktop's 26B model."

            for (char in stubResponse) {
                _streamingContent.value += char
                kotlinx.coroutines.delay(20)
            }

            _messages.value = _messages.value + ChatMessage("assistant", stubResponse)
            _streamingContent.value = ""
            _isGenerating.value = false
        }
    }

    /** Connect to desktop by QR code. */
    fun connectDesktopQR(qrPayload: String) {
        viewModelScope.launch {
            // TODO: UniFFI call — engine.connectDesktopQR(qrPayload)
            _isDesktopConnected.value = true
        }
    }

    /** Connect to desktop by manual IP. */
    fun connectDesktopManual(ip: String, port: Int = 3000) {
        viewModelScope.launch {
            // TODO: UniFFI call — engine.connectDesktopManual(ip, port)
            _isDesktopConnected.value = true
        }
    }

    /** Disconnect from desktop. */
    fun disconnectDesktop() {
        // TODO: UniFFI call — engine.disconnectDesktop()
        _isDesktopConnected.value = false
    }
}

/** Simple chat message DTO. */
data class ChatMessage(
    val role: String,
    val content: String,
    val timestamp: Long = System.currentTimeMillis(),
)
