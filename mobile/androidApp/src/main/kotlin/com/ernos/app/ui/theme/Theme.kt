package com.ernos.app.ui.theme

import android.app.Activity
import android.os.Build
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.runtime.SideEffect
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.toArgb
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalView
import androidx.core.view.WindowCompat

// ErnOS brand colors — purple/cyan gradient
private val ErnOSPurple = Color(0xFF7C4DFF)
private val ErnOSCyan = Color(0xFF00BCD4)
private val ErnOSDeepPurple = Color(0xFF311B92)

private val DarkColorScheme = darkColorScheme(
    primary = ErnOSPurple,
    secondary = ErnOSCyan,
    tertiary = Color(0xFFB388FF),
    background = Color(0xFF0D0D1A),
    surface = Color(0xFF1A1A2E),
    surfaceVariant = Color(0xFF252540),
    onPrimary = Color.White,
    onSecondary = Color.White,
    onBackground = Color(0xFFE1E1E6),
    onSurface = Color(0xFFE1E1E6),
)

private val LightColorScheme = lightColorScheme(
    primary = ErnOSPurple,
    secondary = ErnOSCyan,
    tertiary = ErnOSDeepPurple,
)

@Composable
fun ErnOSTheme(
    darkTheme: Boolean = isSystemInDarkTheme(),
    dynamicColor: Boolean = true,
    content: @Composable () -> Unit,
) {
    val colorScheme = when {
        dynamicColor && Build.VERSION.SDK_INT >= Build.VERSION_CODES.S -> {
            val context = LocalContext.current
            if (darkTheme) dynamicDarkColorScheme(context) else dynamicLightColorScheme(context)
        }
        darkTheme -> DarkColorScheme
        else -> LightColorScheme
    }

    val view = LocalView.current
    if (!view.isInEditMode) {
        SideEffect {
            val window = (view.context as Activity).window
            window.statusBarColor = colorScheme.background.toArgb()
            WindowCompat.getInsetsController(window, view).isAppearanceLightStatusBars = !darkTheme
        }
    }

    MaterialTheme(
        colorScheme = colorScheme,
        typography = Typography(),
        content = content,
    )
}
