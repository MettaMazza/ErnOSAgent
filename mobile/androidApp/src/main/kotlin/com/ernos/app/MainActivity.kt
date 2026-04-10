package com.ernos.app

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import com.ernos.app.ui.screens.*
import com.ernos.app.ui.theme.ErnOSTheme
import com.ernos.app.viewmodel.ErnOSViewModel

/**
 * ErnOS — Main Android Activity.
 *
 * This is a thin rendering shell. ALL intelligence (inference, memory,
 * learning, ReAct loop, Observer audit) lives in the Rust core, accessed
 * via UniFFI-generated Kotlin bindings.
 *
 * The Compose UI is organized into screens:
 *   - SetupScreen: first-run model download + desktop pairing
 *   - ChatScreen: main chat interface with streaming tokens
 *   - DashboardScreen: neural activity, memory, steering controls
 *   - GlassesScreen: Meta Ray-Ban camera/mic overlay
 *   - SettingsScreen: inference mode, model management, desktop config
 */
class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()

        setContent {
            ErnOSTheme {
                ErnOSApp()
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ErnOSApp() {
    val navController = rememberNavController()
    val viewModel: ErnOSViewModel = viewModel()
    val isReady by viewModel.isReady.collectAsState()

    Scaffold(
        modifier = Modifier.fillMaxSize(),
    ) { padding ->
        NavHost(
            navController = navController,
            startDestination = if (isReady) "chat" else "setup",
            modifier = Modifier.padding(padding),
        ) {
            composable("setup") {
                SetupScreen(
                    viewModel = viewModel,
                    onSetupComplete = {
                        navController.navigate("chat") {
                            popUpTo("setup") { inclusive = true }
                        }
                    }
                )
            }
            composable("chat") {
                ChatScreen(
                    viewModel = viewModel,
                    onNavigateToDashboard = { navController.navigate("dashboard") },
                    onNavigateToSettings = { navController.navigate("settings") },
                    onNavigateToGlasses = { navController.navigate("glasses") },
                )
            }
            composable("dashboard") {
                DashboardScreen(
                    viewModel = viewModel,
                    onBack = { navController.popBackStack() },
                )
            }
            composable("settings") {
                SettingsScreen(
                    viewModel = viewModel,
                    onBack = { navController.popBackStack() },
                )
            }
            composable("glasses") {
                GlassesScreen(
                    viewModel = viewModel,
                    onBack = { navController.popBackStack() },
                )
            }
        }
    }
}
