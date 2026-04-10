import SwiftUI
import Combine

/// ErnOS ViewModel — bridges SwiftUI ↔ Rust core.
///
/// All state management happens in Rust. This ViewModel:
/// 1. Holds a reference to the UniFFI-generated ErnOSEngine
/// 2. Exposes engine state as @Published properties for SwiftUI
/// 3. Delegates all actions to the Rust engine
///
/// When UniFFI is integrated, replace the stub calls with:
///   `import ErnOSAgentFFI`
///   `let engine = ErnOSEngine(dataDir: ...)`
@MainActor
class ErnOSViewModel: ObservableObject {

    // MARK: - Engine State

    @Published var isReady = false
    @Published var inferenceMode = "Hybrid"
    @Published var loadedModel: String? = nil
    @Published var isDesktopConnected = false
    @Published var desktopName = ""
    @Published var statusSummary = "No model loaded"

    // MARK: - Chat State

    @Published var messages: [ChatMessage] = []
    @Published var isGenerating = false
    @Published var streamingContent = ""

    // MARK: - Download State

    @Published var downloadProgress: Float = 0
    @Published var isDownloading = false
    @Published var downloadingModel: String? = nil

    // MARK: - UI State

    @Published var showQRScanner = false
    @Published var showManualPairing = false
    @Published var selectedMode = "Hybrid"

    // MARK: - Model Info

    let availableModels: [ModelInfo] = [
        ModelInfo(
            name: "Gemma 4 E2B",
            description: "2.3B params, fast, ideal for 6-8 GB RAM devices",
            size: "3.1 GB"
        ),
        ModelInfo(
            name: "Gemma 4 E4B",
            description: "4.5B params, higher quality, needs 12+ GB RAM",
            size: "5.0 GB"
        ),
    ]

    var recommendedModel: String {
        // Auto-detect based on device RAM
        let physicalMemory = ProcessInfo.processInfo.physicalMemory
        let ramGB = physicalMemory / (1024 * 1024 * 1024)
        return ramGB >= 12 ? "Gemma 4 E4B" : "Gemma 4 E2B"
    }

    // MARK: - Actions

    /// Initialize the engine with the app's data directory.
    func initialize() {
        let dataDir = FileManager.default
            .urls(for: .documentDirectory, in: .userDomainMask)
            .first!
            .appendingPathComponent("ernos")
            .path

        // TODO: UniFFI call — ErnOSEngine(dataDir: dataDir)
        isReady = false
    }

    /// Download and load a model.
    func downloadModel(_ name: String) {
        guard !isDownloading else { return }

        isDownloading = true
        downloadingModel = name
        downloadProgress = 0

        // TODO: UniFFI call — engine.downloadModel(name, progressCallback)
        Task {
            // Simulate download progress
            for i in 0...100 {
                try? await Task.sleep(nanoseconds: 50_000_000) // 50ms
                downloadProgress = Float(i) / 100.0
            }

            isDownloading = false
            downloadingModel = nil
            loadedModel = name
            isReady = true
            statusSummary = "\(name) │ \(inferenceMode)"
        }
    }

    /// Send a chat message.
    func sendMessage(_ content: String) {
        guard !isGenerating else { return }

        isGenerating = true
        messages.append(ChatMessage(role: "user", content: content))
        streamingContent = ""

        // TODO: UniFFI call — engine.chat(content, streamCallback)
        Task {
            let stubResponse = "This is a placeholder response from the ErnOS mobile engine stub. " +
                "When UniFFI is integrated, this will run the full ReAct loop through either " +
                "the on-device Gemma 4 model or relay to your desktop's 26B model."

            for char in stubResponse {
                streamingContent += String(char)
                try? await Task.sleep(nanoseconds: 20_000_000) // 20ms
            }

            messages.append(ChatMessage(role: "assistant", content: stubResponse))
            streamingContent = ""
            isGenerating = false
        }
    }

    /// Connect to desktop via QR code.
    func connectDesktopQR(_ payload: String) {
        // TODO: UniFFI call — engine.connectDesktopQR(payload)
        isDesktopConnected = true
        desktopName = "Desktop"
    }

    /// Connect to desktop via manual IP.
    func connectDesktopManual(ip: String, port: Int = 3000) {
        // TODO: UniFFI call — engine.connectDesktopManual(ip, port)
        isDesktopConnected = true
        desktopName = ip
    }

    /// Disconnect from desktop.
    func disconnectDesktop() {
        // TODO: UniFFI call — engine.disconnectDesktop()
        isDesktopConnected = false
        desktopName = ""
    }
}
