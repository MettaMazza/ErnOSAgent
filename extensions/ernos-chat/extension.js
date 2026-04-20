// Ern-OS Code Extension v0.3.0
// Replaces GitHub Copilot Chat with a local-first AI panel.
// - Auto-disables Copilot extensions on activation
// - Registers Chat panel in bottom panel area (next to Terminal)
// - Cmd+L opens Ern-OS Chat, Cmd+I inline explains selection
// - Full markdown rendering, code highlighting, tool chips

const vscode = require('vscode');

let sidebarPanel;
let panelView;
let ws;
let currentSessionId = '';

function activate(context) {
    // ─── Phase 1: Auto-disable Copilot ───
    disableCopilot(context);

    // ─── Sidebar Webview (activity bar) ───
    const sidebarProvider = new ErnosChatProvider(context, 'sidebar');
    context.subscriptions.push(
        vscode.window.registerWebviewViewProvider('ernos-chat', sidebarProvider)
    );

    // ─── Panel Chat (bottom panel — replaces Copilot Chat) ───
    const panelProvider = new ErnosChatProvider(context, 'panel');
    context.subscriptions.push(
        vscode.window.registerWebviewViewProvider('ernos-chat-panel', panelProvider)
    );

    // ─── Commands ───
    context.subscriptions.push(
        vscode.commands.registerCommand('ernos.openChat', () => {
            vscode.commands.executeCommand('ernos-chat-panel.focus');
        }),
        vscode.commands.registerCommand('ernos.sendSelection', () => {
            sendEditorSelection('');
        }),
        vscode.commands.registerCommand('ernos.explainSelection', () => {
            sendEditorSelection('Explain this code:\n');
        }),
        vscode.commands.registerCommand('ernos.fixSelection', () => {
            sendEditorSelection('Fix any issues in this code:\n');
        }),
        vscode.commands.registerCommand('ernos.inlineChat', () => {
            vscode.commands.executeCommand('ernos-chat-panel.focus');
            setTimeout(() => sendEditorSelection('Explain this code:\n'), 300);
        })
    );

    connectWs();
}

// ─── Phase 1: Disable Copilot Extensions ───
async function disableCopilot(context) {
    const copilotIds = ['github.copilot', 'github.copilot-chat'];
    let anyDisabled = false;

    for (const id of copilotIds) {
        const ext = vscode.extensions.getExtension(id);
        if (ext) {
            try {
                await vscode.commands.executeCommand('workbench.extensions.disableExtension', id);
                anyDisabled = true;
            } catch (_) { /* may fail in some environments */ }
        }
    }

    const alreadyNotified = context.globalState.get('copilot-disabled-notified', false);
    if (anyDisabled && !alreadyNotified) {
        const action = await vscode.window.showInformationMessage(
            'Ern-OS: GitHub Copilot has been disabled — chat now routes to your local model. ' +
            'Use Cmd+L (Ctrl+L) to open Ern-OS Chat.',
            'Reload Now',
            'Later'
        );
        context.globalState.update('copilot-disabled-notified', true);
        if (action === 'Reload Now') {
            vscode.commands.executeCommand('workbench.action.reloadWindow');
        }
    }
}

// ─── WebSocket Connection ───
function getWsUrl() {
    return 'ws://127.0.0.1:3000/ws';
}

function connectWs() {
    if (ws && ws.readyState <= 1) return;
    try {
        const WebSocket = require('ws');
        ws = new WebSocket(getWsUrl());
    } catch (_e) {
        try { ws = new WebSocket(getWsUrl()); } catch (_e2) {
            broadcast({ type: 'status', connected: false });
            setTimeout(connectWs, 5000);
            return;
        }
    }
    ws.on ? wireNode(ws) : wireBrowser(ws);
}

function wireNode(sock) {
    sock.on('open', () => broadcast({ type: 'status', connected: true }));
    sock.on('message', (data) => {
        try { handleMsg(JSON.parse(data.toString())); } catch (_) {}
    });
    sock.on('close', () => {
        ws = null;
        broadcast({ type: 'status', connected: false });
        setTimeout(connectWs, 5000);
    });
    sock.on('error', () => { ws = null; });
}

function wireBrowser(sock) {
    sock.onopen = () => broadcast({ type: 'status', connected: true });
    sock.onmessage = (e) => { try { handleMsg(JSON.parse(e.data)); } catch (_) {} };
    sock.onclose = () => { ws = null; broadcast({ type: 'status', connected: false }); setTimeout(connectWs, 5000); };
    sock.onerror = () => { ws = null; };
}

function handleMsg(msg) {
    switch (msg.type) {
        case 'connected':
            broadcast({ type: 'model_info', model: msg.model || '' });
            break;
        case 'ack':
            if (msg.session_id) currentSessionId = msg.session_id;
            break;
        case 'text_delta': case 'thinking_delta':
        case 'tool_executing': case 'tool_completed':
        case 'done': case 'error':
        case 'audit_running': case 'audit_completed':
        case 'status':
            broadcast(msg);
            break;
    }
}

// ─── Broadcast to all active webviews ───
const activeViews = new Set();

function broadcast(msg) {
    for (const view of activeViews) {
        try { view.webview.postMessage(msg); } catch (_) {}
    }
}

function sendMessage(content) {
    if (!ws || (ws.readyState !== undefined && ws.readyState !== 1)) {
        vscode.window.showWarningMessage('Ern-OS: Not connected to engine.');
        connectWs();
        return;
    }
    ws.send(JSON.stringify({ type: 'chat', content, session_id: currentSessionId || '' }));
    broadcast({ type: 'user_message', content });
}

function sendEditorSelection(prefix) {
    const editor = vscode.window.activeTextEditor;
    if (!editor) return;
    const sel = editor.document.getText(editor.selection);
    if (!sel) { vscode.window.showWarningMessage('Ern-OS: No text selected.'); return; }
    const lang = editor.document.languageId;
    const file = editor.document.fileName.split('/').pop();
    sendMessage(`${prefix}\`${file}\`:\n\`\`\`${lang}\n${sel}\n\`\`\``);
}

// ─── Webview Provider (shared for sidebar + panel) ───
class ErnosChatProvider {
    constructor(context, mode) {
        this._ctx = context;
        this._mode = mode; // 'sidebar' or 'panel'
    }

    resolveWebviewView(view) {
        if (this._mode === 'sidebar') sidebarPanel = view;
        else panelView = view;

        activeViews.add(view);
        view.onDidDispose(() => activeViews.delete(view));

        view.webview.options = { enableScripts: true };
        view.webview.html = getChatHtml(this._mode);
        view.webview.onDidReceiveMessage((msg) => {
            if (msg.type === 'send') sendMessage(msg.content);
            if (msg.type === 'reconnect') connectWs();
        });

        // Send initial status
        const connected = ws && ws.readyState === 1;
        try { view.webview.postMessage({ type: 'status', connected }); } catch (_) {}
    }
}

// ─── Enhanced Chat HTML ───
function getChatHtml(mode) {
    const isPanel = mode === 'panel';
    return `<!DOCTYPE html>
<html><head>
<meta charset="UTF-8">
<style>
:root {
    --bg: var(--vscode-panel-background, var(--vscode-sideBar-background, #1e1e1e));
    --fg: var(--vscode-foreground, #ccc);
    --input-bg: var(--vscode-input-background, #2d2d2d);
    --input-fg: var(--vscode-input-foreground, #ccc);
    --input-border: var(--vscode-input-border, #444);
    --accent: var(--vscode-button-background, #00ff88);
    --accent-fg: var(--vscode-button-foreground, #000);
    --muted: var(--vscode-descriptionForeground, #888);
    --border: var(--vscode-panel-border, #333);
    --code-bg: var(--vscode-textCodeBlock-background, rgba(0,0,0,0.3));
    --badge-bg: rgba(0,255,136,0.1);
    --err: #ff4444;
}
* { margin:0; padding:0; box-sizing:border-box; }
body {
    font-family: var(--vscode-font-family, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif);
    font-size: 13px; color: var(--fg); background: var(--bg);
    display: flex; flex-direction: column; height: 100vh;
}

/* ── Header ── */
#header {
    display: flex; align-items: center; gap: 8px;
    padding: 8px 12px; border-bottom: 1px solid var(--border);
    flex-shrink: 0;
}
.dot { width: 7px; height: 7px; border-radius: 50%; background: #666; flex-shrink: 0; }
.dot.on { background: #00ff88; box-shadow: 0 0 6px rgba(0,255,136,0.5); }
#stxt { font-size: 11px; font-weight: 600; }
#model-badge {
    margin-left: auto; font-size: 10px; font-weight: 700;
    padding: 2px 8px; border-radius: 10px;
    background: var(--badge-bg); color: var(--accent);
    border: 1px solid rgba(0,255,136,0.15);
}

/* ── Messages ── */
#messages {
    flex: 1; overflow-y: auto; padding: 12px;
    scroll-behavior: smooth;
}
.msg { margin-bottom: 10px; animation: fadeIn 0.15s ease; }
@keyframes fadeIn { from { opacity: 0; transform: translateY(4px); } to { opacity: 1; transform: translateY(0); } }

.msg-user {
    background: var(--accent); color: var(--accent-fg);
    padding: 8px 14px; border-radius: 14px 14px 4px 14px;
    margin-left: ${isPanel ? '30%' : '15%'}; font-size: 12.5px; line-height: 1.5;
    word-break: break-word;
}
.msg-assist {
    background: var(--vscode-editor-background, #1a1a1a);
    border: 1px solid var(--border);
    padding: 10px 14px; border-radius: 14px 14px 14px 4px;
    margin-right: ${isPanel ? '10%' : '5%'}; font-size: 12.5px; line-height: 1.65;
    word-break: break-word;
}
.msg-assist p { margin: 4px 0; }
.msg-assist code {
    background: var(--code-bg); padding: 1px 5px; border-radius: 4px;
    font-family: var(--vscode-editor-font-family, 'SF Mono', 'Fira Code', monospace);
    font-size: 11.5px;
}
.msg-assist pre {
    background: var(--code-bg); padding: 10px 12px; border-radius: 8px;
    margin: 6px 0; overflow-x: auto; position: relative;
    border: 1px solid rgba(255,255,255,0.06);
}
.msg-assist pre code { background: none; padding: 0; font-size: 11.5px; line-height: 1.5; }
.msg-assist ul, .msg-assist ol { padding-left: 18px; margin: 4px 0; }
.msg-assist strong { color: var(--accent); font-weight: 700; }
.msg-assist h1,.msg-assist h2,.msg-assist h3 { margin: 8px 0 4px; color: var(--accent); }

/* ── Tool Chips ── */
.tool-chip {
    display: inline-flex; align-items: center; gap: 5px;
    padding: 4px 10px; border-radius: 8px; margin: 3px 0;
    font-size: 11px; font-weight: 600;
    background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08);
}
.tool-chip.running { color: var(--accent); }
.tool-chip.done { color: #8f8; opacity: 0.7; }
.tool-chip.fail { color: var(--err); opacity: 0.7; }
.tool-spinner { width: 10px; height: 10px; border: 2px solid var(--accent); border-top-color: transparent; border-radius: 50%; animation: spin 0.6s linear infinite; }
@keyframes spin { to { transform: rotate(360deg); } }

/* ── Thinking ── */
.thinking-block {
    margin: 4px 0 6px; padding: 6px 10px; border-radius: 8px;
    background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.06);
    cursor: pointer; font-size: 11px; color: var(--muted);
}
.thinking-label { font-weight: 600; }
.thinking-content { display: none; margin-top: 6px; white-space: pre-wrap; font-size: 11px; line-height: 1.5; max-height: 200px; overflow-y: auto; }
.thinking-block.expanded .thinking-content { display: block; }

/* ── Status ── */
.status-msg { text-align: center; padding: 4px; font-size: 11px; color: var(--muted); font-style: italic; }
.error-msg { background: rgba(255,68,68,0.08); color: var(--err); border: 1px solid rgba(255,68,68,0.15); padding: 8px 14px; border-radius: 10px; margin: 6px 0; font-size: 12px; }

/* ── Audit Badge ── */
.audit-chip { display: inline-flex; align-items: center; gap: 4px; font-size: 10px; color: var(--muted); padding: 2px 0; }

/* ── Input Bar ── */
#bar {
    display: flex; gap: 6px; padding: 8px 12px;
    border-top: 1px solid var(--border); flex-shrink: 0;
    align-items: flex-end;
}
#inp {
    flex: 1; background: var(--input-bg); color: var(--input-fg);
    border: 1px solid var(--input-border); border-radius: 10px;
    padding: 8px 12px; font-size: 12.5px; font-family: inherit;
    resize: none; min-height: 32px; max-height: 150px;
    line-height: 1.5; transition: border-color 0.15s;
}
#inp:focus { outline: none; border-color: var(--accent); box-shadow: 0 0 0 2px rgba(0,255,136,0.1); }
#btn {
    background: var(--accent); color: var(--accent-fg); border: none;
    border-radius: 10px; padding: 7px 14px; cursor: pointer;
    font-size: 12px; font-weight: 700; transition: transform 0.1s, opacity 0.1s;
}
#btn:hover { transform: translateY(-1px); }
#btn:active { transform: translateY(0); opacity: 0.8; }
</style></head><body>

<div id="header">
    <span class="dot" id="dot"></span>
    <span id="stxt">Connecting...</span>
    <span id="model-badge">—</span>
</div>

<div id="messages"></div>

<div id="bar">
    <textarea id="inp" placeholder="Ask Ern-OS... (Shift+Enter for newline)" rows="1"></textarea>
    <button id="btn">▶</button>
</div>

<script>
const vscode = acquireVsCodeApi();
const msgs = document.getElementById('messages');
const inp = document.getElementById('inp');
const dot = document.getElementById('dot');
const stxt = document.getElementById('stxt');
const badge = document.getElementById('model-badge');

let curEl = null; // current assistant message div
let curText = '';  // accumulated text for current response
let thinkEl = null;

// ── Auto-resize textarea ──
inp.addEventListener('input', () => {
    inp.style.height = 'auto';
    inp.style.height = Math.min(inp.scrollHeight, 150) + 'px';
});

inp.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); go(); }
});

document.getElementById('btn').addEventListener('click', go);

function go() {
    const t = inp.value.trim();
    if (!t) return;
    vscode.postMessage({ type: 'send', content: t });
    inp.value = '';
    inp.style.height = 'auto';
}

function scroll() { msgs.scrollTop = msgs.scrollHeight; }

function esc(s) { const d = document.createElement('div'); d.textContent = s; return d.innerHTML; }

// ── Simple markdown renderer ──
function md(text) {
    let h = esc(text);
    // Code blocks
    h = h.replace(/\`\`\`(\\w*?)\\n([\\s\\S]*?)\`\`\`/g, (_, lang, code) =>
        '<pre><code class="lang-' + lang + '">' + code + '</code></pre>');
    // Inline code
    h = h.replace(/\`([^\`]+)\`/g, '<code>$1</code>');
    // Bold
    h = h.replace(/\\*\\*(.+?)\\*\\*/g, '<strong>$1</strong>');
    // Italic
    h = h.replace(/\\*(.+?)\\*/g, '<em>$1</em>');
    // Headers
    h = h.replace(/^### (.+)$/gm, '<h3>$1</h3>');
    h = h.replace(/^## (.+)$/gm, '<h2>$1</h2>');
    h = h.replace(/^# (.+)$/gm, '<h1>$1</h1>');
    // Lists
    h = h.replace(/^- (.+)$/gm, '<li>$1</li>');
    h = h.replace(/(<li>.*<\\/li>)/s, '<ul>$1</ul>');
    // Line breaks (but not inside pre)
    h = h.replace(/\\n/g, '<br>');
    // Clean double <br> before/after block elements
    h = h.replace(/<br><(pre|h[1-3]|ul|ol|li)/g, '<$1');
    h = h.replace(/<\\/(pre|h[1-3]|ul|ol|li)><br>/g, '</$1>');
    return h;
}

function renderAssistant() {
    if (!curEl) return;
    const content = curEl.querySelector('.msg-content');
    if (content) content.innerHTML = md(curText);
    scroll();
}

// ── Message handling ──
window.addEventListener('message', e => {
    const d = e.data;
    switch (d.type) {
        case 'status':
            dot.className = 'dot' + (d.connected ? ' on' : '');
            stxt.textContent = d.connected ? 'Connected' : 'Disconnected';
            break;
        case 'model_info':
            badge.textContent = d.model || '—';
            break;
        case 'user_message': {
            const m = document.createElement('div');
            m.className = 'msg msg-user';
            m.textContent = d.content;
            msgs.appendChild(m);
            // Prepare assistant bubble
            curEl = document.createElement('div');
            curEl.className = 'msg msg-assist';
            curEl.innerHTML = '<div class="msg-content"></div>';
            msgs.appendChild(curEl);
            curText = '';
            thinkEl = null;
            scroll();
            break;
        }
        case 'thinking_delta':
            if (!curEl) break;
            if (!thinkEl) {
                thinkEl = document.createElement('div');
                thinkEl.className = 'thinking-block';
                thinkEl.innerHTML = '<div class="thinking-label">💭 Thinking <span style="font-size:10px;opacity:0.5">(click)</span></div><div class="thinking-content"></div>';
                thinkEl.onclick = () => thinkEl.classList.toggle('expanded');
                curEl.insertBefore(thinkEl, curEl.firstChild);
            }
            thinkEl.querySelector('.thinking-content').textContent += d.content;
            break;
        case 'text_delta':
            if (!curEl) {
                // Safety: create bubble if missing
                curEl = document.createElement('div');
                curEl.className = 'msg msg-assist';
                curEl.innerHTML = '<div class="msg-content"></div>';
                msgs.appendChild(curEl);
                curText = '';
            }
            curText += d.content;
            renderAssistant();
            break;
        case 'tool_executing': {
            if (!curEl) break;
            const chip = document.createElement('div');
            chip.className = 'tool-chip running';
            chip.id = 'tc-' + d.id;
            chip.innerHTML = '<div class="tool-spinner"></div> ' + esc(d.name);
            const content = curEl.querySelector('.msg-content');
            curEl.insertBefore(chip, content);
            scroll();
            break;
        }
        case 'tool_completed': {
            const chip = document.getElementById('tc-' + d.id);
            if (chip) {
                chip.className = 'tool-chip ' + (d.success ? 'done' : 'fail');
                chip.innerHTML = (d.success ? '✅' : '❌') + ' ' + esc(d.name);
            }
            break;
        }
        case 'audit_running': {
            if (!curEl) break;
            const a = document.createElement('div');
            a.className = 'audit-chip';
            a.id = 'audit-badge';
            a.textContent = '🔍 Observer audit...';
            curEl.appendChild(a);
            break;
        }
        case 'audit_completed': {
            const a = document.getElementById('audit-badge');
            if (a) a.textContent = d.approved ? '✅ Approved' : '⚠️ Retry';
            break;
        }
        case 'status':
            if (d.message) {
                const s = document.createElement('div');
                s.className = 'status-msg';
                s.textContent = d.message;
                msgs.appendChild(s);
                scroll();
            }
            break;
        case 'error': {
            const e = document.createElement('div');
            e.className = 'error-msg';
            e.textContent = '❌ ' + (d.message || 'Unknown error');
            msgs.appendChild(e);
            curEl = null; curText = '';
            scroll();
            break;
        }
        case 'done':
            renderAssistant(); // Final render
            curEl = null;
            curText = '';
            thinkEl = null;
            break;
    }
});
</script>
</body></html>`;
}

module.exports = { activate, deactivate: () => { if (ws) { ws.close(); ws = null; } } };
