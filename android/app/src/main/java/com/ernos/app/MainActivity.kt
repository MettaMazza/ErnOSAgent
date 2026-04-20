package com.ernos.app

import android.annotation.SuppressLint
import android.content.Intent
import android.os.Build
import android.os.Bundle
import android.view.View
import android.view.WindowInsetsController
import android.webkit.*
import androidx.appcompat.app.AppCompatActivity

/**
 * Main activity — fullscreen WebView pointing to the local Ern-OS engine.
 * The engine runs as a foreground service (EngineService) so it persists
 * when the user switches apps.
 *
 * Displays a loading screen while the engine boots, polling until ready.
 */
class MainActivity : AppCompatActivity() {

    private lateinit var webView: WebView

    @SuppressLint("SetJavaScriptEnabled")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // Fullscreen immersive
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            window.insetsController?.apply {
                hide(android.view.WindowInsets.Type.statusBars() or android.view.WindowInsets.Type.navigationBars())
                systemBarsBehavior = WindowInsetsController.BEHAVIOR_SHOW_TRANSIENT_BARS_BY_SWIPE
            }
        } else {
            @Suppress("DEPRECATION")
            window.decorView.systemUiVisibility = (
                View.SYSTEM_UI_FLAG_FULLSCREEN
                    or View.SYSTEM_UI_FLAG_HIDE_NAVIGATION
                    or View.SYSTEM_UI_FLAG_IMMERSIVE_STICKY
            )
        }

        // Start the engine service
        val serviceIntent = Intent(this, EngineService::class.java)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            startForegroundService(serviceIntent)
        } else {
            startService(serviceIntent)
        }

        // Configure WebView
        webView = findViewById(R.id.webView)
        webView.settings.apply {
            javaScriptEnabled = true
            domStorageEnabled = true
            databaseEnabled = true
            allowFileAccess = true
            mediaPlaybackRequiresUserGesture = false
            mixedContentMode = WebSettings.MIXED_CONTENT_ALWAYS_ALLOW
            javaScriptCanOpenWindowsAutomatically = true
            useWideViewPort = true
            loadWithOverviewMode = true
        }

        webView.webViewClient = object : WebViewClient() {
            override fun onReceivedError(
                view: WebView?, request: WebResourceRequest?,
                error: WebResourceError?
            ) {
                // Only retry for the main frame, not sub-resources
                if (request?.isForMainFrame == true) {
                    view?.postDelayed({
                        view.loadUrl("http://127.0.0.1:3000")
                    }, 2000)
                }
            }
        }

        webView.webChromeClient = WebChromeClient()

        // Show loading screen immediately, then poll for engine readiness
        showLoadingScreen()
    }

    /** Load an inline HTML loading page that polls the engine health endpoint. */
    private fun showLoadingScreen() {
        val loadingHtml = """
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
                <style>
                    * { margin: 0; padding: 0; box-sizing: border-box; }
                    body {
                        background: #06060e;
                        color: #e0e0e0;
                        font-family: -apple-system, sans-serif;
                        display: flex;
                        flex-direction: column;
                        align-items: center;
                        justify-content: center;
                        min-height: 100vh;
                        padding: 24px;
                    }
                    .logo {
                        font-size: 3rem;
                        font-weight: 700;
                        background: linear-gradient(135deg, #00FF88, #3b82f6);
                        -webkit-background-clip: text;
                        -webkit-text-fill-color: transparent;
                        margin-bottom: 24px;
                    }
                    .spinner {
                        width: 40px; height: 40px;
                        border: 3px solid rgba(0,255,136,0.15);
                        border-top-color: #00FF88;
                        border-radius: 50%;
                        animation: spin 1s linear infinite;
                        margin-bottom: 20px;
                    }
                    @keyframes spin { to { transform: rotate(360deg); } }
                    .status {
                        font-size: 0.9rem;
                        color: rgba(255,255,255,0.5);
                        text-align: center;
                    }
                    .dots::after {
                        content: '';
                        animation: dots 1.5s steps(4, end) infinite;
                    }
                    @keyframes dots {
                        0% { content: ''; }
                        25% { content: '.'; }
                        50% { content: '..'; }
                        75% { content: '...'; }
                    }
                    .progress-bar {
                        width: 240px; height: 4px;
                        background: rgba(255,255,255,0.1);
                        border-radius: 2px;
                        margin-top: 16px;
                        overflow: hidden;
                    }
                    .progress-fill {
                        height: 100%; width: 0%;
                        background: linear-gradient(90deg, #00FF88, #3b82f6);
                        border-radius: 2px;
                        transition: width 0.5s ease;
                    }
                    .sub {
                        font-size: 0.75rem;
                        color: rgba(255,255,255,0.35);
                        margin-top: 12px;
                        text-align: center;
                    }
                </style>
            </head>
            <body>
                <div class="logo">Ern-OS</div>
                <div class="spinner"></div>
                <div class="status" id="status">Starting engine<span class="dots"></span></div>
                <div class="progress-bar" id="progress" style="display:none">
                    <div class="progress-fill" id="progress-fill"></div>
                </div>
                <div class="sub" id="sub"></div>
                <script>
                    let phase = 'engine'; // 'engine' → 'provider'
                    let attempts = 0;
                    const status = document.getElementById('status');
                    const progress = document.getElementById('progress');
                    const progressFill = document.getElementById('progress-fill');
                    const sub = document.getElementById('sub');

                    function poll() {
                        attempts++;
                        if (phase === 'engine') {
                            // Phase 1: wait for Axum web server
                            fetch('http://127.0.0.1:3000/api/health')
                                .then(r => {
                                    if (r.ok) {
                                        phase = 'provider';
                                        attempts = 0;
                                        status.innerHTML = 'Checking inference provider<span class="dots"></span>';
                                        sub.textContent = '';
                                        setTimeout(poll, 500);
                                    } else { retry(); }
                                })
                                .catch(() => {
                                    if (attempts < 10) status.innerHTML = 'Starting engine<span class="dots"></span>';
                                    else status.innerHTML = 'Initializing<span class="dots"></span>';
                                    retry();
                                });
                        } else {
                            // Phase 2: wait for provider + show real download progress
                            Promise.all([
                                fetch('http://127.0.0.1:3000/api/status').then(r => r.json()).catch(() => null),
                                fetch('http://127.0.0.1:3000/api/model/download-progress').then(r => r.json()).catch(() => null)
                            ]).then(([statusData, dlData]) => {
                                if (statusData && statusData.provider_healthy) {
                                    status.innerHTML = '✓ Ready!';
                                    sub.textContent = statusData.model?.name || 'Connected';
                                    progress.style.display = 'none';
                                    setTimeout(() => {
                                        window.location.href = 'http://127.0.0.1:3000';
                                    }, 400);
                                } else if (dlData && dlData.downloading) {
                                    // Real download progress from ModelManager
                                    progress.style.display = 'block';
                                    progressFill.style.width = dlData.progress + '%';
                                    status.innerHTML = 'Downloading model — ' + dlData.progress + '%';
                                    sub.textContent = dlData.downloaded_mb + ' MB / ' + dlData.total_mb + ' MB';
                                    retry();
                                } else {
                                    // llama-server starting up, provider not ready yet
                                    progress.style.display = 'block';
                                    progressFill.style.width = (5 + Math.min(attempts * 2, 90)) + '%';
                                    if (attempts < 5) {
                                        status.innerHTML = 'Extracting inference engine<span class="dots"></span>';
                                        sub.textContent = 'Preparing llama-server for local AI';
                                    } else if (attempts < 30) {
                                        status.innerHTML = 'Starting inference engine<span class="dots"></span>';
                                        sub.textContent = 'Loading model into memory — this may take a moment';
                                    } else {
                                        status.innerHTML = 'Still loading model<span class="dots"></span>';
                                        sub.textContent = 'Large models take time on mobile — please wait';
                                    }
                                    retry();
                                }
                            });
                        }
                    }
                    function retry() { setTimeout(poll, 1500); }
                    setTimeout(poll, 1000);
                </script>
            </body>
            </html>
        """.trimIndent()

        webView.loadDataWithBaseURL(null, loadingHtml, "text/html", "UTF-8", null)
    }

    override fun onBackPressed() {
        if (webView.canGoBack()) {
            webView.goBack()
        } else {
            @Suppress("DEPRECATION")
            super.onBackPressed()
        }
    }

    override fun onDestroy() {
        webView.destroy()
        super.onDestroy()
    }
}
