import SwiftUI

/// ErnOS — iOS App Entry Point.
///
/// This is a thin rendering shell. ALL intelligence (inference, memory,
/// learning, ReAct loop, Observer audit) lives in the Rust core, accessed
/// via UniFFI-generated Swift bindings.
///
/// The SwiftUI app is organized into views:
///   - SetupView: first-run model download + desktop pairing
///   - ChatView: main chat interface with streaming tokens
///   - DashboardView: neural activity, memory, steering controls
///   - GlassesView: Meta Ray-Ban camera/mic overlay
///   - SettingsView: inference mode, model management, desktop config
@main
struct ErnOSApp: App {
    @StateObject private var viewModel = ErnOSViewModel()

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(viewModel)
        }
    }
}

/// Root content view with navigation.
struct ContentView: View {
    @EnvironmentObject var viewModel: ErnOSViewModel

    var body: some View {
        NavigationStack {
            if viewModel.isReady {
                ChatView()
            } else {
                SetupView()
            }
        }
    }
}

// MARK: - Setup View

/// First-run setup: download model + pair with desktop.
struct SetupView: View {
    @EnvironmentObject var viewModel: ErnOSViewModel

    var body: some View {
        VStack(spacing: 32) {
            // Logo
            Image(systemName: "brain.head.profile")
                .font(.system(size: 80))
                .foregroundStyle(
                    LinearGradient(
                        colors: [.purple, .blue, .cyan],
                        startPoint: .topLeading,
                        endPoint: .bottomTrailing
                    )
                )

            Text("ErnOS")
                .font(.system(size: 48, weight: .bold, design: .rounded))

            Text("Self-evolving AI that learns from every interaction")
                .font(.subheadline)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
                .padding(.horizontal, 40)

            Spacer()

            // Model selection
            VStack(spacing: 16) {
                Text("Choose Your Model")
                    .font(.headline)

                ForEach(viewModel.availableModels, id: \.name) { model in
                    ModelCard(
                        model: model,
                        isRecommended: model.name == viewModel.recommendedModel,
                        isDownloading: viewModel.isDownloading && viewModel.downloadingModel == model.name,
                        progress: viewModel.downloadProgress,
                        onDownload: { viewModel.downloadModel(model.name) }
                    )
                }
            }

            // Desktop pairing
            VStack(spacing: 12) {
                Text("Link to Desktop (Optional)")
                    .font(.headline)

                HStack(spacing: 16) {
                    Button("Scan QR") {
                        viewModel.showQRScanner = true
                    }
                    .buttonStyle(.borderedProminent)

                    Button("Enter IP") {
                        viewModel.showManualPairing = true
                    }
                    .buttonStyle(.bordered)
                }

                if viewModel.isDesktopConnected {
                    Label("Connected to \(viewModel.desktopName)", systemImage: "checkmark.circle.fill")
                        .foregroundStyle(.green)
                }
            }

            Spacer()
        }
        .padding()
        .sheet(isPresented: $viewModel.showQRScanner) {
            Text("QR Scanner — Coming in Phase 5")
        }
        .sheet(isPresented: $viewModel.showManualPairing) {
            ManualPairingSheet()
        }
    }
}

// MARK: - Chat View

/// Main chat interface with streaming tokens.
struct ChatView: View {
    @EnvironmentObject var viewModel: ErnOSViewModel
    @State private var inputText = ""
    @FocusState private var isInputFocused: Bool

    var body: some View {
        VStack(spacing: 0) {
            // Status bar
            HStack {
                Circle()
                    .fill(viewModel.isDesktopConnected ? .green : .gray)
                    .frame(width: 8, height: 8)

                Text(viewModel.statusSummary)
                    .font(.caption)
                    .foregroundStyle(.secondary)

                Spacer()

                Menu {
                    NavigationLink("Dashboard") {
                        DashboardView()
                    }
                    NavigationLink("Settings") {
                        SettingsView()
                    }
                    NavigationLink("Glasses") {
                        GlassesView()
                    }
                } label: {
                    Image(systemName: "ellipsis.circle")
                }
            }
            .padding(.horizontal)
            .padding(.vertical, 8)

            Divider()

            // Messages
            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(spacing: 12) {
                        ForEach(viewModel.messages) { message in
                            MessageBubble(message: message)
                        }

                        // Streaming content
                        if !viewModel.streamingContent.isEmpty {
                            MessageBubble(
                                message: ChatMessage(
                                    role: "assistant",
                                    content: viewModel.streamingContent
                                )
                            )
                            .id("streaming")
                        }
                    }
                    .padding()
                }
                .onChange(of: viewModel.streamingContent) { _ in
                    withAnimation {
                        proxy.scrollTo("streaming", anchor: .bottom)
                    }
                }
            }

            Divider()

            // Input
            HStack(spacing: 12) {
                TextField("Ask ErnOS...", text: $inputText, axis: .vertical)
                    .textFieldStyle(.plain)
                    .focused($isInputFocused)
                    .lineLimit(1...5)

                Button {
                    guard !inputText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return }
                    viewModel.sendMessage(inputText)
                    inputText = ""
                } label: {
                    Image(systemName: viewModel.isGenerating ? "stop.fill" : "arrow.up.circle.fill")
                        .font(.title2)
                }
                .disabled(inputText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty && !viewModel.isGenerating)
            }
            .padding()
        }
        .navigationBarHidden(true)
    }
}

// MARK: - Dashboard View

/// Neural activity, memory stats, cognitive steering.
struct DashboardView: View {
    @EnvironmentObject var viewModel: ErnOSViewModel

    var body: some View {
        List {
            Section("Neural Activity") {
                Text("Token throughput: -- tok/s")
                Text("Context usage: --%")
                Text("Active features: --")
            }

            Section("Memory") {
                Text("Total memories: --")
                Text("Active lessons: --")
                Text("Training samples: --")
            }

            Section("Cognitive Steering") {
                Text("Configured via desktop Dashboard")
                Text("Steering vectors sync automatically")
            }
        }
        .navigationTitle("Dashboard")
    }
}

// MARK: - Settings View

/// Inference mode, model, desktop connection settings.
struct SettingsView: View {
    @EnvironmentObject var viewModel: ErnOSViewModel

    var body: some View {
        List {
            Section("Inference Mode") {
                Picker("Mode", selection: $viewModel.selectedMode) {
                    Text("Local (on-device)").tag("Local")
                    Text("Remote (desktop)").tag("Remote")
                    Text("Hybrid (smart routing)").tag("Hybrid")
                    Text("Chain-of-Agents").tag("ChainOfAgents")
                }
                .pickerStyle(.inline)
            }

            Section("Model") {
                if let model = viewModel.loadedModel {
                    Label(model, systemImage: "brain")
                } else {
                    Text("No model loaded")
                        .foregroundStyle(.secondary)
                }
            }

            Section("Desktop Connection") {
                if viewModel.isDesktopConnected {
                    Label("Connected", systemImage: "checkmark.circle.fill")
                        .foregroundStyle(.green)
                    Button("Disconnect", role: .destructive) {
                        viewModel.disconnectDesktop()
                    }
                } else {
                    Button("Scan QR Code") {
                        viewModel.showQRScanner = true
                    }
                    Button("Enter IP Manually") {
                        viewModel.showManualPairing = true
                    }
                }
            }
        }
        .navigationTitle("Settings")
    }
}

// MARK: - Glasses View

/// Meta Ray-Ban camera/mic integration overlay.
struct GlassesView: View {
    @EnvironmentObject var viewModel: ErnOSViewModel

    var body: some View {
        VStack(spacing: 24) {
            Image(systemName: "eyeglasses")
                .font(.system(size: 60))
                .foregroundStyle(.secondary)

            Text("Meta Ray-Ban Integration")
                .font(.title2)

            Text("Connect your Meta Ray-Ban Smart Glasses to use the camera for visual queries and the microphone for hands-free interaction.")
                .font(.body)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
                .padding(.horizontal, 32)

            Button("Connect Glasses") {
                // TODO: Meta Wearables Device Access Toolkit
            }
            .buttonStyle(.borderedProminent)

            Spacer()
        }
        .padding(.top, 60)
        .navigationTitle("Glasses")
    }
}

// MARK: - Shared Components

struct MessageBubble: View {
    let message: ChatMessage

    var body: some View {
        HStack {
            if message.role == "user" { Spacer() }

            Text(message.content)
                .padding(12)
                .background(
                    message.role == "user"
                        ? Color.blue.opacity(0.2)
                        : Color(.systemGray6)
                )
                .clipShape(RoundedRectangle(cornerRadius: 16))
                .frame(maxWidth: 300, alignment: message.role == "user" ? .trailing : .leading)

            if message.role == "assistant" { Spacer() }
        }
    }
}

struct ModelCard: View {
    let model: ModelInfo
    let isRecommended: Bool
    let isDownloading: Bool
    let progress: Float
    let onDownload: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text(model.name)
                    .font(.headline)
                if isRecommended {
                    Text("Recommended")
                        .font(.caption)
                        .foregroundStyle(.white)
                        .padding(.horizontal, 8)
                        .padding(.vertical, 2)
                        .background(.blue)
                        .clipShape(Capsule())
                }
            }
            Text(model.description)
                .font(.caption)
                .foregroundStyle(.secondary)

            if isDownloading {
                ProgressView(value: progress)
            } else {
                Button("Download (\(model.size))", action: onDownload)
                    .buttonStyle(.bordered)
            }
        }
        .padding()
        .background(Color(.systemGray6))
        .clipShape(RoundedRectangle(cornerRadius: 12))
    }
}

struct ManualPairingSheet: View {
    @EnvironmentObject var viewModel: ErnOSViewModel
    @State private var ipAddress = ""
    @Environment(\.dismiss) var dismiss

    var body: some View {
        NavigationStack {
            Form {
                TextField("Desktop IP Address", text: $ipAddress)
                    .keyboardType(.decimalPad)
                Button("Connect") {
                    viewModel.connectDesktopManual(ip: ipAddress)
                    dismiss()
                }
                .disabled(ipAddress.isEmpty)
            }
            .navigationTitle("Manual Pairing")
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel") { dismiss() }
                }
            }
        }
    }
}

// MARK: - Data Models

struct ModelInfo: Identifiable {
    let id = UUID()
    let name: String
    let description: String
    let size: String
}

struct ChatMessage: Identifiable {
    let id = UUID()
    let role: String
    let content: String
}
