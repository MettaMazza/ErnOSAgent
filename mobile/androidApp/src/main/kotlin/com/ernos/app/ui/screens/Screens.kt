package com.ernos.app.ui.screens

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.ernos.app.viewmodel.ErnOSViewModel

/**
 * Setup screen — first-run wizard for model download + desktop pairing.
 */
@Composable
fun SetupScreen(
    viewModel: ErnOSViewModel,
    onSetupComplete: () -> Unit,
) {
    val isReady by viewModel.isReady.collectAsState()
    val isDownloading by viewModel.isDownloading.collectAsState()
    val progress by viewModel.downloadProgress.collectAsState()

    LaunchedEffect(isReady) {
        if (isReady) onSetupComplete()
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(24.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        Spacer(Modifier.height(48.dp))

        Icon(
            imageVector = Icons.Filled.Psychology,
            contentDescription = "ErnOS",
            modifier = Modifier.size(80.dp),
            tint = MaterialTheme.colorScheme.primary,
        )
        Spacer(Modifier.height(16.dp))
        Text("ErnOS", style = MaterialTheme.typography.displayMedium)
        Text(
            "Self-evolving AI that learns from every interaction",
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )

        Spacer(Modifier.height(48.dp))

        Text("Choose Your Model", style = MaterialTheme.typography.titleMedium)
        Spacer(Modifier.height(16.dp))

        // E2B Card
        ModelDownloadCard(
            name = "Gemma 4 E2B",
            description = "2.3B params • Fast • 6-8 GB RAM",
            size = "3.1 GB",
            isRecommended = true,
            isDownloading = isDownloading,
            progress = progress,
            onDownload = { viewModel.downloadModel("Gemma 4 E2B") },
        )
        Spacer(Modifier.height(12.dp))
        // E4B Card
        ModelDownloadCard(
            name = "Gemma 4 E4B",
            description = "4.5B params • Higher quality • 12+ GB RAM",
            size = "5.0 GB",
            isRecommended = false,
            isDownloading = isDownloading,
            progress = progress,
            onDownload = { viewModel.downloadModel("Gemma 4 E4B") },
        )

        Spacer(Modifier.weight(1f))

        // Desktop pairing
        Text("Link to Desktop (Optional)", style = MaterialTheme.typography.titleSmall)
        Spacer(Modifier.height(8.dp))
        Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
            OutlinedButton(onClick = { /* QR scanner */ }) {
                Icon(Icons.Filled.QrCodeScanner, "QR")
                Spacer(Modifier.width(8.dp))
                Text("Scan QR")
            }
            OutlinedButton(onClick = { /* Manual IP */ }) {
                Icon(Icons.Filled.Keyboard, "IP")
                Spacer(Modifier.width(8.dp))
                Text("Enter IP")
            }
        }
    }
}

@Composable
private fun ModelDownloadCard(
    name: String,
    description: String,
    size: String,
    isRecommended: Boolean,
    isDownloading: Boolean,
    progress: Float,
    onDownload: () -> Unit,
) {
    Card(modifier = Modifier.fillMaxWidth()) {
        Column(modifier = Modifier.padding(16.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text(name, style = MaterialTheme.typography.titleSmall)
                if (isRecommended) {
                    Spacer(Modifier.width(8.dp))
                    AssistChip(
                        onClick = {},
                        label = { Text("Recommended", style = MaterialTheme.typography.labelSmall) },
                    )
                }
            }
            Text(description, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
            Spacer(Modifier.height(8.dp))
            if (isDownloading) {
                LinearProgressIndicator(progress = { progress }, modifier = Modifier.fillMaxWidth())
            } else {
                Button(onClick = onDownload) { Text("Download ($size)") }
            }
        }
    }
}

/**
 * Chat screen — main chat interface with streaming tokens.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ChatScreen(
    viewModel: ErnOSViewModel,
    onNavigateToDashboard: () -> Unit,
    onNavigateToSettings: () -> Unit,
    onNavigateToGlasses: () -> Unit,
) {
    val messages by viewModel.messages.collectAsState()
    val isGenerating by viewModel.isGenerating.collectAsState()
    val streaming by viewModel.streamingContent.collectAsState()
    val statusSummary by viewModel.statusSummary.collectAsState()
    var inputText by remember { mutableStateOf("") }

    Scaffold(
        topBar = {
            TopAppBar(
                title = {
                    Text(statusSummary, style = MaterialTheme.typography.labelMedium)
                },
                actions = {
                    IconButton(onClick = onNavigateToDashboard) {
                        Icon(Icons.Filled.Dashboard, "Dashboard")
                    }
                    IconButton(onClick = onNavigateToGlasses) {
                        Icon(Icons.Filled.Visibility, "Glasses")
                    }
                    IconButton(onClick = onNavigateToSettings) {
                        Icon(Icons.Filled.Settings, "Settings")
                    }
                },
            )
        },
    ) { padding ->
        Column(modifier = Modifier.padding(padding).fillMaxSize()) {
            // Messages
            val listState = rememberLazyListState()
            LazyColumn(
                state = listState,
                modifier = Modifier.weight(1f).padding(horizontal = 16.dp),
            ) {
                items(messages) { msg ->
                    MessageBubble(role = msg.role, content = msg.content)
                    Spacer(Modifier.height(8.dp))
                }
                if (streaming.isNotEmpty()) {
                    item {
                        MessageBubble(role = "assistant", content = streaming)
                    }
                }
            }

            // Input bar
            Row(
                modifier = Modifier.padding(16.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                OutlinedTextField(
                    value = inputText,
                    onValueChange = { inputText = it },
                    modifier = Modifier.weight(1f),
                    placeholder = { Text("Ask ErnOS...") },
                    maxLines = 4,
                )
                Spacer(Modifier.width(8.dp))
                IconButton(
                    onClick = {
                        if (inputText.isNotBlank()) {
                            viewModel.sendMessage(inputText.trim())
                            inputText = ""
                        }
                    },
                    enabled = inputText.isNotBlank() || isGenerating,
                ) {
                    Icon(
                        if (isGenerating) Icons.Filled.Stop else Icons.Filled.Send,
                        "Send",
                    )
                }
            }
        }
    }
}

@Composable
private fun MessageBubble(role: String, content: String) {
    val isUser = role == "user"
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = if (isUser) Arrangement.End else Arrangement.Start,
    ) {
        Card(
            colors = CardDefaults.cardColors(
                containerColor = if (isUser)
                    MaterialTheme.colorScheme.primaryContainer
                else
                    MaterialTheme.colorScheme.surfaceVariant,
            ),
        ) {
            Text(
                text = content,
                modifier = Modifier.padding(12.dp).widthIn(max = 280.dp),
                style = MaterialTheme.typography.bodyMedium,
            )
        }
    }
}

/**
 * Dashboard screen — neural activity, memory, steering.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun DashboardScreen(viewModel: ErnOSViewModel, onBack: () -> Unit) {
    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Dashboard") },
                navigationIcon = {
                    IconButton(onClick = onBack) {
                        Icon(Icons.Filled.ArrowBack, "Back")
                    }
                },
            )
        },
    ) { padding ->
        LazyColumn(modifier = Modifier.padding(padding).padding(16.dp)) {
            item {
                Text("Neural Activity", style = MaterialTheme.typography.titleMedium)
                Spacer(Modifier.height(8.dp))
                Card(modifier = Modifier.fillMaxWidth()) {
                    Column(modifier = Modifier.padding(16.dp)) {
                        Text("Token throughput: -- tok/s")
                        Text("Context usage: --%")
                    }
                }
            }
            item { Spacer(Modifier.height(16.dp)) }
            item {
                Text("Memory", style = MaterialTheme.typography.titleMedium)
                Spacer(Modifier.height(8.dp))
                Card(modifier = Modifier.fillMaxWidth()) {
                    Column(modifier = Modifier.padding(16.dp)) {
                        Text("Total memories: --")
                        Text("Active lessons: --")
                    }
                }
            }
        }
    }
}

/**
 * Settings screen — inference mode, model management, desktop config.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun SettingsScreen(viewModel: ErnOSViewModel, onBack: () -> Unit) {
    val mode by viewModel.inferenceMode.collectAsState()
    val isConnected by viewModel.isDesktopConnected.collectAsState()

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Settings") },
                navigationIcon = {
                    IconButton(onClick = onBack) {
                        Icon(Icons.Filled.ArrowBack, "Back")
                    }
                },
            )
        },
    ) { padding ->
        LazyColumn(modifier = Modifier.padding(padding).padding(16.dp)) {
            item {
                Text("Inference Mode", style = MaterialTheme.typography.titleMedium)
                Spacer(Modifier.height(8.dp))
                listOf("Local", "Remote", "Hybrid", "ChainOfAgents").forEach { m ->
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        RadioButton(
                            selected = mode == m,
                            onClick = { viewModel.setInferenceMode(m) },
                        )
                        Text(m)
                    }
                }
            }
            item { Spacer(Modifier.height(16.dp)) }
            item {
                Text("Desktop Connection", style = MaterialTheme.typography.titleMedium)
                Spacer(Modifier.height(8.dp))
                if (isConnected) {
                    Text("✅ Connected")
                    TextButton(onClick = { viewModel.disconnectDesktop() }) {
                        Text("Disconnect")
                    }
                } else {
                    Text("Not connected", color = MaterialTheme.colorScheme.onSurfaceVariant)
                }
            }
        }
    }
}

/**
 * Glasses screen — Meta Ray-Ban integration.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun GlassesScreen(viewModel: ErnOSViewModel, onBack: () -> Unit) {
    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Glasses") },
                navigationIcon = {
                    IconButton(onClick = onBack) {
                        Icon(Icons.Filled.ArrowBack, "Back")
                    }
                },
            )
        },
    ) { padding ->
        Column(
            modifier = Modifier.padding(padding).fillMaxSize(),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.Center,
        ) {
            Icon(Icons.Filled.Visibility, "Glasses", modifier = Modifier.size(64.dp))
            Spacer(Modifier.height(16.dp))
            Text("Meta Ray-Ban Integration", style = MaterialTheme.typography.titleMedium)
            Spacer(Modifier.height(8.dp))
            Text(
                "Connect your Meta Ray-Ban Smart Glasses for visual queries and hands-free interaction.",
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                modifier = Modifier.padding(horizontal = 32.dp),
            )
            Spacer(Modifier.height(24.dp))
            Button(onClick = { /* Meta Wearables SDK */ }) {
                Text("Connect Glasses")
            }
        }
    }
}
