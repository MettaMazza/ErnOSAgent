/* Ern-OS — WebUI Application */
/* Dual-layer inference, 7-tier memory, observer audit, autonomous learning, self-skills, 29 tools */

const ErnOS = (() => {
    'use strict';

    // ─── State ───
    let ws = null;
    let currentSessionId = null;
    let isGenerating = false;
    let currentAssistantEl = null;
    let thinkingEl = null;
    let pendingText = '';
    let renderTimer = null;
    let attachedFiles = [];
    let reactIteration = 0;
    let currentView = 'chat';
    let currentAgentId = null;

    // ─── WebSocket ───
    function connect() {
        const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
        ws = new WebSocket(`${protocol}//${location.host}/ws`);
        ws.onopen = () => updateConnectionStatus(true);
        ws.onmessage = (e) => handleMessage(JSON.parse(e.data));
        ws.onclose = () => { updateConnectionStatus(false); setTimeout(connect, 3000); };
        ws.onerror = () => ws.close();
    }

    function updateConnectionStatus(connected) {
        const bar = document.getElementById('status-bar');
        bar.innerHTML = connected
            ? '<span class="status-dot connected"></span>Connected'
            : '<span class="status-dot disconnected"></span>Reconnecting...';
    }

    // ─── Message Router ───
    function handleMessage(msg) {
        switch (msg.type) {
            case 'connected':       onConnected(msg); break;
            case 'ack':             break;
            case 'text_delta':      onTextDelta(msg); break;
            case 'thinking_delta':  onThinkingDelta(msg); break;
            case 'tool_executing':  onToolExecuting(msg); break;
            case 'tool_completed':  onToolCompleted(msg); break;
            case 'audit_running':   onAuditRunning(); break;
            case 'audit_completed': onAuditCompleted(msg); break;
            case 'plan_proposal':   onPlanProposal(msg); break;
            case 'artifact_created': onArtifactCreated(msg); break;
            case 'stopped':         onStopped(); break;
            case 'status':          onStatus(msg); break;
            case 'done':            onDone(); break;
            case 'error':           onError(msg); break;
            case 'autonomy_state':  handleAutonomyResponse(msg); break;
            default: handleNewWsMessages(msg); break;
        }
    }

    // ─── Message Handlers ───
    function onConnected(msg) {
        document.getElementById('model-badge').textContent = msg.model || 'unknown';
        loadSessions();
        checkModelDownloadProgress();
    }

    async function checkModelDownloadProgress() {
        try {
            const resp = await fetch('/api/model/download-progress');
            const data = await resp.json();
            if (data.downloading && data.progress !== undefined) {
                const badge = document.getElementById('model-badge');
                const pct = Math.round(data.progress * 100);
                badge.textContent = `⬇ Downloading ${data.model || 'model'} ${pct}%`;
                // Poll again in 3s if still downloading
                setTimeout(checkModelDownloadProgress, 3000);
            }
        } catch { /* not downloading or endpoint unavailable */ }
    }

    function onTextDelta(msg) {
        if (!currentAssistantEl) return;
        pendingText += msg.content;
        scheduleRender();
    }

    function onThinkingDelta(msg) {
        if (!currentAssistantEl) return;
        if (!thinkingEl) {
            thinkingEl = document.createElement('div');
            thinkingEl.className = 'thinking-block';
            thinkingEl.innerHTML = '<div class="thinking-label">💭 Thinking <span style="font-size:10px;opacity:0.5">(click to expand)</span></div><div class="thinking-content"></div>';
            thinkingEl.onclick = () => thinkingEl.classList.toggle('expanded');
            const content = currentAssistantEl.querySelector('.message-content');
            currentAssistantEl.insertBefore(thinkingEl, content);
        }
        thinkingEl.querySelector('.thinking-content').textContent += msg.content;
    }

    function onToolExecuting(msg) {
        if (!currentAssistantEl) return;
        reactIteration++;
        const chip = document.createElement('div');
        chip.className = 'tool-chip';
        chip.id = `tool-${msg.id}`;
        chip.innerHTML = `<div class="tool-spinner"></div> Running <strong>${escapeHtml(msg.name)}</strong>...`;
        const content = currentAssistantEl.querySelector('.message-content');
        content.before(chip);
        scrollToBottom();
    }

    function onToolCompleted(msg) {
        if (!currentAssistantEl) return;
        const chip = document.getElementById(`tool-${msg.id}`) ||
            currentAssistantEl.querySelector('.tool-chip:last-of-type');
        if (chip) {
            const icon = msg.success ? '✅' : '❌';
            chip.className = `tool-chip ${msg.success ? 'completed' : 'failed'}`;
            chip.innerHTML = `${icon} <strong>${escapeHtml(msg.name)}</strong>`;

            if (msg.result) {
                const output = document.createElement('div');
                output.className = 'tool-output';
                output.textContent = msg.result;
                chip.after(output);
                chip.style.cursor = 'pointer';
                chip.onclick = () => output.classList.toggle('visible');
            }
        }

        // During onboarding interview, auto-complete when agent writes identity.md
        if (onboardingPhase === 3 && msg.name === 'file_write' && msg.success &&
            msg.result && msg.result.includes('identity.md')) {
            finishOnboarding();
            onboardingPhase = 0;
        }
    }

    function onAuditRunning() {
        if (!currentAssistantEl) return;
        removeAuditBadge();
        const badge = document.createElement('div');
        badge.className = 'audit-badge running';
        badge.id = 'audit-badge';
        badge.innerHTML = '<div class="tool-spinner" style="width:10px;height:10px;border-color:var(--info);border-top-color:transparent"></div> Observer auditing...';
        currentAssistantEl.appendChild(badge);
    }

    function onAuditCompleted(msg) {
        removeAuditBadge();
        if (!currentAssistantEl) return;
        const badge = document.createElement('div');
        badge.id = 'audit-badge';
        if (msg.approved) {
            badge.className = 'audit-badge approved';
            badge.textContent = `✅ Approved`;
        } else {
            badge.className = 'audit-badge rejected';
            badge.textContent = `❌ Rejected: ${msg.reason || 'No reason'}`;
        }
        currentAssistantEl.appendChild(badge);
    }

    function removeAuditBadge() {
        const existing = document.getElementById('audit-badge');
        if (existing) existing.remove();
    }

    function onStopped() {
        addSystemMessage('⏹ Generation stopped');
        finishGeneration();
    }

    function onStatus(msg) {
        if (msg.message) {
            // Remove previous status badge if exists
            const prev = document.getElementById('react-status');
            if (prev) prev.remove();
            const container = document.getElementById('messages');
            const el = document.createElement('div');
            el.className = 'message system';
            el.id = 'react-status';
            el.innerHTML = `<div class="message-content" style="display:flex;align-items:center;gap:8px">
                <span>🔄 ${escapeHtml(msg.message)}</span>
                <button id="react-stop-btn" style="
                    background:rgba(255,82,82,0.15);border:1px solid rgba(255,82,82,0.4);
                    color:#ff5252;padding:2px 10px;border-radius:6px;cursor:pointer;
                    font-size:11px;font-weight:600;transition:all 0.2s;
                ">⏹ Stop</button>
            </div>`;
            container.appendChild(el);
            const stopBtn = document.getElementById('react-stop-btn');
            if (stopBtn) {
                stopBtn.onclick = () => {
                    if (ws && ws.readyState === WebSocket.OPEN) {
                        ws.send(JSON.stringify({ type: 'stop_react' }));
                        stopBtn.textContent = '⏳ Stopping...';
                        stopBtn.disabled = true;
                        stopBtn.style.opacity = '0.5';
                    }
                };
            }
            scrollToBottom();
            reactIteration = 0;
        }
    }

    function onDone() {
        // Update ReAct status badge to completed tick + remove stop button
        const reactStatus = document.getElementById('react-status');
        if (reactStatus) {
            const content = reactStatus.querySelector('.message-content');
            const stopBtn = document.getElementById('react-stop-btn');
            if (stopBtn) stopBtn.remove();
            const span = content.querySelector('span');
            if (span) span.textContent = span.textContent.replace('🔄', '✅');
        }
        finishGeneration();

        // Auto-title session after first reply
        maybeAutoTitleSession();
    }

    // ─── Plan Proposal ───
    function onPlanProposal(msg) {
        finishGeneration();
        hideWelcome();
        const container = document.getElementById('messages');
        const el = document.createElement('div');
        el.className = 'message assistant plan-card-wrapper';
        el.setAttribute('role', 'article');

        const renderedPlan = markdownToHtml(msg.plan_markdown || '');
        const sessionId = msg.session_id || currentSessionId || '';
        const revisionBadge = msg.revision > 1 ? `<span class="plan-revision-badge">Rev ${msg.revision}</span>` : '';

        el.innerHTML = `
            <div class="plan-card" id="plan-card-${sessionId}">
                <div class="plan-card-header">
                    <div class="plan-card-title">
                        <span class="plan-icon">📋</span>
                        <span>Implementation Plan</span>
                        ${revisionBadge}
                    </div>
                    <div class="plan-card-meta">~${msg.estimated_turns || '?'} tool turns</div>
                </div>
                <div class="plan-card-title-text">${escapeHtml(msg.title || 'Untitled Plan')}</div>
                <div class="plan-card-content">${renderedPlan}</div>
                <div class="plan-card-notes-section">
                    <textarea id="plan-notes-${sessionId}" class="plan-notes-input" placeholder="Add notes or feedback (optional)..." rows="3"></textarea>
                </div>
                <div class="plan-card-actions">
                    <button class="plan-btn plan-btn-approve" onclick="ErnOS.approvePlan('${sessionId}')">
                        <span>✅</span> Approve & Execute
                    </button>
                    <button class="plan-btn plan-btn-revise" onclick="ErnOS.revisePlan('${sessionId}')">
                        <span>✏️</span> Revise
                    </button>
                    <button class="plan-btn plan-btn-cancel" onclick="ErnOS.cancelPlan('${sessionId}')">
                        <span>❌</span> Cancel
                    </button>
                </div>
            </div>
        `;

        container.appendChild(el);
        addCopyButtons(el.querySelector('.plan-card-content'));
        postProcessContent(el.querySelector('.plan-card-content'));
        scrollToBottom();
    }

    function approvePlan(sessionId) {
        const card = document.getElementById(`plan-card-${sessionId}`);
        if (card) {
            const actions = card.querySelector('.plan-card-actions');
            const notes = card.querySelector('.plan-card-notes-section');
            if (actions) actions.innerHTML = '<div class="plan-status plan-status-approved">✅ Plan Approved — Executing...</div>';
            if (notes) notes.remove();
        }

        // Prepare streaming state for ReAct response
        currentAssistantEl = addMessage('assistant', '');
        pendingText = '';
        thinkingEl = null;
        isGenerating = true;

        const payload = JSON.stringify({
            type: 'plan_decision',
            session_id: sessionId,
            approved: true,
        });

        function trySend(attempt) {
            if (ws && ws.readyState === WebSocket.OPEN) {
                try {
                    ws.send(payload);
                    console.log('[plan] plan_decision sent on attempt', attempt);
                } catch (e) {
                    console.error('[plan] ws.send threw:', e);
                    if (attempt < 3) setTimeout(() => trySend(attempt + 1), 1000);
                }
            } else {
                console.warn('[plan] WS not OPEN (readyState=' + (ws ? ws.readyState : 'null') + '), retry', attempt);
                if (attempt < 5) setTimeout(() => trySend(attempt + 1), 1000);
                else console.error('[plan] Failed to send plan_decision after 5 retries');
            }
        }
        trySend(1);
    }

    function revisePlan(sessionId) {
        const notesEl = document.getElementById(`plan-notes-${sessionId}`);
        const notes = notesEl ? notesEl.value.trim() : '';
        if (!notes) {
            showToast('Please add notes describing what to change', 'warning');
            if (notesEl) notesEl.focus();
            return;
        }

        const card = document.getElementById(`plan-card-${sessionId}`);
        if (card) {
            const actions = card.querySelector('.plan-card-actions');
            if (actions) actions.innerHTML = '<div class="plan-status plan-status-revising">✏️ Revising plan based on your feedback...</div>';
        }

        ws.send(JSON.stringify({
            type: 'plan_decision',
            session_id: sessionId,
            approved: false,
            notes: notes,
        }));
    }

    function cancelPlan(sessionId) {
        const card = document.getElementById(`plan-card-${sessionId}`);
        if (card) {
            const actions = card.querySelector('.plan-card-actions');
            const notes = card.querySelector('.plan-card-notes-section');
            if (actions) actions.innerHTML = '<div class="plan-status plan-status-cancelled">❌ Plan Cancelled</div>';
            if (notes) notes.remove();
        }

        currentAssistantEl = addMessage('assistant', '');
        pendingText = '';

        ws.send(JSON.stringify({
            type: 'plan_decision',
            session_id: sessionId,
            approved: false,
        }));
    }

    // ─── Artifact Cards ───
    function onArtifactCreated(msg) {
        hideWelcome();
        const container = document.getElementById('messages');
        const el = document.createElement('div');
        el.className = 'message assistant artifact-card-wrapper';
        el.setAttribute('role', 'article');

        const typeIcons = { report: '📊', plan: '📋', analysis: '🔍', code: '💻' };
        const typeColors = { report: '#3b82f6', plan: '#10b981', analysis: '#8b5cf6', code: '#f59e0b' };
        const aType = msg.artifact_type || 'report';
        const icon = typeIcons[aType] || '📄';
        const color = typeColors[aType] || '#3b82f6';
        const renderedContent = markdownToHtml(msg.content || '');
        const artifactId = msg.id || '';

        el.innerHTML = `
            <div class="artifact-card" style="--artifact-color: ${color}" id="artifact-${artifactId}">
                <div class="artifact-card-header" onclick="this.parentElement.classList.toggle('collapsed')">
                    <div class="artifact-card-title">
                        <span class="artifact-icon">${icon}</span>
                        <span>${escapeHtml(msg.title || 'Untitled')}</span>
                        <span class="artifact-type-badge">${aType}</span>
                    </div>
                    <div class="artifact-card-controls">
                        <button class="artifact-btn" onclick="event.stopPropagation(); ErnOS.copyArtifact('${artifactId}')" title="Copy">📋</button>
                        <span class="artifact-collapse-icon">▼</span>
                    </div>
                </div>
                <div class="artifact-card-content">${renderedContent}</div>
            </div>
        `;

        container.appendChild(el);
        addCopyButtons(el.querySelector('.artifact-card-content'));
        postProcessContent(el.querySelector('.artifact-card-content'));
        scrollToBottom();
    }

    function copyArtifact(artifactId) {
        const card = document.getElementById(`artifact-${artifactId}`);
        if (!card) return;
        const content = card.querySelector('.artifact-card-content');
        if (!content) return;
        navigator.clipboard.writeText(content.innerText).then(() => {
            showToast('Artifact copied to clipboard', 'success');
        });
    }

    function onError(msg) {
        addSystemMessage(`⚠️ ${msg.message || 'Unknown error'}`);
        finishGeneration();
    }

    // ─── Chat ───
    function sendMessage() {
        const input = document.getElementById('chat-input');
        const sendBtn = document.getElementById('send-btn');
        const text = input.value.trim();
        if (!text || !ws || ws.readyState !== WebSocket.OPEN) return;
        if (sendBtn.disabled) return;
        if (text === '/stop') { stopGeneration(); input.value = ''; return; }

        // Auto-switch to chat view
        if (currentView !== 'chat') switchView('chat');

        hideWelcome();
        addMessage('user', text, attachedFiles.length > 0 ? attachedFiles : null);

        if (!currentSessionId) {
            fetch('/api/sessions', { method: 'POST' })
                .then(r => r.json())
                .then(s => { currentSessionId = s.id; sendChatPayload(text); loadSessions(); });
        } else {
            sendChatPayload(text);
        }

        input.value = '';
        autoResize(input);
        clearAttachments();
        trackInputHistory(text);
    }

    function sendChatPayload(text) {
        isGenerating = true;
        reactIteration = 0;
        document.querySelector('.header-btn.danger').style.display = 'block';
        document.getElementById('send-btn').disabled = true;

        const payload = { type: 'chat', content: text, session_id: currentSessionId || '' };
        if (currentAgentId) payload.agent_id = currentAgentId;
        if (attachedFiles.length > 0) {
            payload.images = attachedFiles.map(f => f.dataUrl);
        }

        // Create bubble BEFORE sending — prevents thinking_delta race condition
        // where backend streams thinking before currentAssistantEl exists
        currentAssistantEl = addMessage('assistant', '');
        pendingText = '';
        thinkingEl = null;
        ws.send(JSON.stringify(payload));
    }

    function stopGeneration() {
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type: 'stop' }));
        }
        finishGeneration();
    }

    function finishGeneration() {
        if (pendingText && currentAssistantEl) renderMarkdown();
        isGenerating = false;
        const stopBtn = document.querySelector('.header-btn.danger');
        if (stopBtn) stopBtn.style.display = 'none';
        document.getElementById('send-btn').disabled = false;
        // Auto-collapse thinking block after generation finishes
        if (thinkingEl && !thinkingEl.classList.contains('expanded')) {
            thinkingEl.classList.remove('expanded');
        }
        currentAssistantEl = null;
        thinkingEl = null;
        pendingText = '';
    }

    let sessionTitled = {};
    /// Auto-generate a session title from the first user message.
    async function maybeAutoTitleSession() {
        if (!currentSessionId) return;
        if (sessionTitled[currentSessionId]) return;
        sessionTitled[currentSessionId] = true;

        // Skip if session already has a custom or auto-generated title (not "New Chat")
        const sidebar = document.querySelector(`.session-item[data-id="${currentSessionId}"] .session-title`);
        if (sidebar) {
            const existingTitle = sidebar.textContent.trim();
            if (existingTitle && existingTitle !== 'New Chat') return;
        }

        // Get the first user message from the DOM
        const userMsgs = document.querySelectorAll('.message.user .message-content');
        if (userMsgs.length === 0) return;
        const firstMsg = userMsgs[0].textContent.trim();
        if (!firstMsg) return;

        // Generate title: first 50 chars of user message, cleaned up
        let title = firstMsg.replace(/\n/g, ' ').substring(0, 50).trim();
        if (firstMsg.length > 50) title += '...';

        try {
            await fetch(`/api/sessions/${currentSessionId}`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ title }),
            });
            loadSessions(); // Refresh sidebar
        } catch (e) { /* ignore */ }
    }

    // ─── Markdown Renderer ───
    function scheduleRender() {
        if (renderTimer) cancelAnimationFrame(renderTimer);
        renderTimer = requestAnimationFrame(renderMarkdown);
    }

    function renderMarkdown() {
        if (!currentAssistantEl) return;
        const content = currentAssistantEl.querySelector('.message-content');
        if (!content) return;
        content.innerHTML = markdownToHtml(pendingText);
        addCopyButtons(content);
        postProcessContent(content);
        scrollToBottom();
    }

    function markdownToHtml(md) {
        if (!md) return '';
        let html = escapeHtml(md);

        // Mermaid diagrams
        html = html.replace(/```mermaid\n([\s\S]*?)```/g, (_, code) => {
            return `<div class="mermaid">${code.trim()}</div>`;
        });

        html = html.replace(/```(\w*)\n([\s\S]*?)```/g, (_, lang, code) => {
            // Diff blocks: line-level red/green coloring
            if (lang === 'diff') {
                const lines = code.trim().split('\n').map(line => {
                    if (line.startsWith('@@')) return `<span class="diff-hunk">${line}</span>`;
                    if (line.startsWith('+'))  return `<span class="diff-add">${line}</span>`;
                    if (line.startsWith('-'))  return `<span class="diff-del">${line}</span>`;
                    return `<span class="diff-ctx">${line}</span>`;
                }).join('\n');
                return `<pre class="diff-block" data-lang="diff"><code>${lines}</code><button class="code-copy-btn" onclick="ErnOS.copyCode(this)">Copy</button></pre>`;
            }
            const langClass = lang ? ` class="language-${lang}"` : '';
            const langLabel = lang ? ` data-lang="${lang}"` : '';
            return `<pre${langLabel}><code${langClass}>${code.trim()}</code><button class="code-copy-btn" onclick="ErnOS.copyCode(this)">Copy</button></pre>`;
        });

        html = html.replace(/`([^`]+)`/g, '<code>$1</code>');
        html = html.replace(/^#### (.+)$/gm, '<h4>$1</h4>');
        html = html.replace(/^### (.+)$/gm, '<h3>$1</h3>');
        html = html.replace(/^## (.+)$/gm, '<h2>$1</h2>');
        html = html.replace(/^# (.+)$/gm, '<h1>$1</h1>');
        html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
        html = html.replace(/\*(.+?)\*/g, '<em>$1</em>');
        html = html.replace(/!\[([^\]]*)\]\(([^)]+)\)/g, '<img src="$2" alt="$1" class="generated-image">');
        html = html.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
        html = html.replace(/^---$/gm, '<hr>');
        html = html.replace(/^&gt; (.+)$/gm, '<blockquote>$1</blockquote>');

        html = html.replace(/^(\|.+\|)\n(\|[\s-:|]+\|)\n((?:\|.+\|\n?)+)/gm, (_, header, sep, body) => {
            const ths = header.split('|').filter(c => c.trim()).map(c => `<th>${c.trim()}</th>`).join('');
            const rows = body.trim().split('\n').map(row => {
                const tds = row.split('|').filter(c => c.trim()).map(c => `<td>${c.trim()}</td>`).join('');
                return `<tr>${tds}</tr>`;
            }).join('');
            return `<table><thead><tr>${ths}</tr></thead><tbody>${rows}</tbody></table>`;
        });

        html = html.replace(/^(\s*)[-*] (.+)$/gm, '$1<li>$2</li>');
        html = html.replace(/((?:<li>.*<\/li>\n?)+)/g, '<ul>$1</ul>');
        html = html.replace(/^\d+\. (.+)$/gm, '<li>$1</li>');

        html = html.replace(/\n\n/g, '</p><p>');
        html = html.replace(/\n/g, '<br>');

        html = html.replace(/<br><\/p>/g, '</p>');
        html = html.replace(/<p><br>/g, '<p>');
        html = html.replace(/<br>(<h[1-4]>)/g, '$1');
        html = html.replace(/(<\/h[1-4]>)<br>/g, '$1');
        html = html.replace(/<br>(<pre)/g, '$1');
        html = html.replace(/(<\/pre>)<br>/g, '$1');
        html = html.replace(/<br>(<table)/g, '$1');
        html = html.replace(/(<\/table>)<br>/g, '$1');
        html = html.replace(/<br>(<ul>)/g, '$1');
        html = html.replace(/(<\/ul>)<br>/g, '$1');
        html = html.replace(/<br>(<hr>)/g, '$1');
        html = html.replace(/(<hr>)<br>/g, '$1');
        html = html.replace(/<br>(<blockquote>)/g, '$1');

        return html;
    }

    function addCopyButtons(container) {
        container.querySelectorAll('pre').forEach(pre => {
            if (!pre.querySelector('.code-copy-btn')) {
                const btn = document.createElement('button');
                btn.className = 'code-copy-btn';
                btn.textContent = 'Copy';
                btn.onclick = () => copyCode(btn);
                pre.appendChild(btn);
            }
        });
    }

    function copyCode(btn) {
        const code = btn.parentElement.querySelector('code');
        if (code) {
            navigator.clipboard.writeText(code.textContent);
            btn.textContent = 'Copied!';
            setTimeout(() => btn.textContent = 'Copy', 1500);
        }
    }

    // ─── Messages ───
    let messageCounter = 0;

    function addMessage(role, text, images) {
        const container = document.getElementById('messages');
        const el = document.createElement('div');
        const idx = messageCounter++;
        el.className = `message ${role}`;
        el.setAttribute('role', 'article');
        el.dataset.index = idx;

        const now = new Date();
        const timeStr = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

        let actionsHtml = '';
        if (role === 'assistant') {
            actionsHtml = `<div class="message-actions">
                <button class="msg-action-btn" onclick="ErnOS.copyMessage(this)" title="Copy">📋</button>
                <button class="msg-action-btn" onclick="ErnOS.regenerateMessage(${idx})" title="Regenerate">🔄</button>
                <button class="msg-action-btn" onclick="ErnOS.forkFromMessage(${idx})" title="Fork">🍴</button>
                <button class="msg-action-btn" onclick="ErnOS.speakMessage(this)" title="Speak">🔊</button>
                <button class="msg-action-btn react-btn" onclick="ErnOS.reactToMessage(${idx}, 'up')" title="Good response">👍</button>
                <button class="msg-action-btn react-btn" onclick="ErnOS.reactToMessage(${idx}, 'down')" title="Bad response">👎</button>
            </div>`;
        } else if (role === 'user') {
            actionsHtml = `<div class="message-actions">
                <button class="msg-action-btn" onclick="ErnOS.copyMessage(this)" title="Copy">📋</button>
                <button class="msg-action-btn" onclick="ErnOS.editMessage(${idx})" title="Edit">✏️</button>
                <button class="msg-action-btn" onclick="ErnOS.deleteMessage(${idx})" title="Delete">🗑️</button>
            </div>`;
        }

        const roleLabel = role === 'assistant' ? 'ERN-OS' : role.toUpperCase();
        const renderedContent = text ? markdownToHtml(text) : '';
        el.innerHTML = `${actionsHtml}<div class="message-header"><span class="message-role">${roleLabel}</span><span class="message-timestamp">${timeStr}</span></div><div class="message-content">${renderedContent}</div>`;

        if (images && images.length > 0) {
            const imgContainer = document.createElement('div');
            imgContainer.style.cssText = 'display:flex;gap:8px;margin:8px 0;flex-wrap:wrap';
            images.forEach(f => {
                const img = document.createElement('img');
                img.src = f.dataUrl || f;
                img.style.cssText = 'max-width:200px;max-height:200px;border-radius:8px;border:1px solid var(--border)';
                imgContainer.appendChild(img);
            });
            el.querySelector('.message-content').before(imgContainer);
        }

        container.appendChild(el);
        scrollToBottom();
        return el;
    }

    function addSystemMessage(text) {
        const container = document.getElementById('messages');
        const el = document.createElement('div');
        el.className = 'message system';
        el.innerHTML = `<div class="message-content">${escapeHtml(text)}</div>`;
        container.appendChild(el);
        scrollToBottom();
    }

    function copyMessage(btn) {
        const msg = btn.closest('.message');
        const content = msg.querySelector('.message-content');
        navigator.clipboard.writeText(content.textContent);
        btn.textContent = '✅';
        setTimeout(() => btn.textContent = '📋', 1500);
    }

    function regenerateMessage(idx) {
        if (!currentSessionId) { showToast('No active session', 'error'); return; }
        if (ws && ws.readyState === WebSocket.OPEN) {
            // Remove the last assistant message from the DOM
            const messages = document.getElementById('messages');
            const lastAssistant = messages.querySelector('.message.assistant:last-of-type');
            if (lastAssistant) lastAssistant.remove();

            ws.send(JSON.stringify({
                type: 'regenerate',
                session_id: currentSessionId,
            }));

            // Prepare streaming state for the new response
            currentAssistantEl = addMessage('assistant', '');
            pendingText = '';
            thinkingEl = null;
            reactIteration = 0;
            document.querySelector('.header-btn.danger').style.display = 'block';
            document.getElementById('send-btn').disabled = true;
            showToast('Regenerating response...', 'info');
        }
    }

    function editMessage(idx) {
        const el = document.querySelector(`.message[data-index="${idx}"]`);
        if (!el) return;
        const contentEl = el.querySelector('.message-content');
        const original = contentEl.textContent;
        contentEl.innerHTML = `<textarea class="edit-message-input">${escapeHtml(original)}</textarea>
            <div class="edit-actions">
                <button class="edit-save-btn" onclick="ErnOS.saveEditedMessage(${idx})">Save & Resend</button>
                <button class="edit-cancel-btn" onclick="ErnOS.cancelEditMessage(${idx}, '${escapeHtml(original).replace(/'/g, "\\'")}')">Cancel</button>
            </div>`;
        const textarea = contentEl.querySelector('textarea');
        textarea.focus();
        textarea.style.height = textarea.scrollHeight + 'px';
    }

    function saveEditedMessage(idx) {
        const el = document.querySelector(`.message[data-index="${idx}"]`);
        if (!el) return;
        const textarea = el.querySelector('.edit-message-input');
        const newContent = textarea.value.trim();
        if (!newContent || !currentSessionId) return;

        // Remove all messages after this one from DOM
        const messages = document.getElementById('messages');
        const allMsgs = [...messages.querySelectorAll('.message')];
        const msgIdx = allMsgs.indexOf(el);
        for (let i = allMsgs.length - 1; i > msgIdx; i--) {
            allMsgs[i].remove();
        }
        el.querySelector('.message-content').innerHTML = markdownToHtml(newContent);

        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({
                type: 'edit_and_resend',
                session_id: currentSessionId,
                message_index: idx,
                content: newContent,
            }));

            // Prepare streaming state for the new response
            currentAssistantEl = addMessage('assistant', '');
            pendingText = '';
            thinkingEl = null;
            reactIteration = 0;
            document.querySelector('.header-btn.danger').style.display = 'block';
            document.getElementById('send-btn').disabled = true;
        }
    }

    function cancelEditMessage(idx, original) {
        const el = document.querySelector(`.message[data-index="${idx}"]`);
        if (!el) return;
        el.querySelector('.message-content').innerHTML = markdownToHtml(original);
    }

    async function forkFromMessage(idx) {
        if (!currentSessionId) return;
        try {
            const resp = await fetch(`/api/sessions/${currentSessionId}/fork/${idx}`, { method: 'POST' });
            const data = await resp.json();
            if (data.ok) {
                showToast(`Forked to "${data.title}"`, 'success');
                switchSession(data.id);
            } else {
                showToast(data.error || 'Fork failed', 'error');
            }
        } catch (e) { showToast('Fork failed', 'error'); }
    }

    async function reactToMessage(idx, reaction) {
        if (!currentSessionId) return;
        try {
            const resp = await fetch(`/api/sessions/${currentSessionId}/messages/${idx}/react`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ reaction }),
            });
            const data = await resp.json();
            if (data.ok) {
                // Visually toggle the reaction button
                const el = document.querySelector(`.message[data-index="${idx}"]`);
                if (el) {
                    const emoji = reaction === 'up' ? '👍' : '👎';
                    showToast(`${emoji} Feedback recorded`, 'success');
                    // Highlight the clicked button
                    const btns = el.querySelectorAll('.react-btn');
                    btns.forEach(b => b.style.opacity = '0.4');
                    const clicked = reaction === 'up' ? btns[0] : btns[1];
                    if (clicked) clicked.style.opacity = '1';
                }
            }
        } catch (e) { showToast('Reaction failed', 'error'); }
    }

    async function deleteMessageAtIndex(idx) {
        if (!currentSessionId) return;
        const ok = await confirmAction('Delete Message', 'Remove this message from the conversation?');
        if (!ok) return;
        try {
            await fetch(`/api/sessions/${currentSessionId}/messages/${idx}`, { method: 'DELETE' });
            showToast('Message deleted', 'success');
            switchSession(currentSessionId); // Reload
        } catch (e) { showToast('Delete failed', 'error'); }
    }

    // ─── TTS (Kokoro) ───
    let ttsAudio = null;
    let ttsAvailable = null;

    async function checkTtsStatus() {
        try {
            const resp = await fetch('/api/tts/status');
            const data = await resp.json();
            ttsAvailable = data.available;
        } catch { ttsAvailable = false; }
    }

    async function speakMessage(btn) {
        const msg = btn.closest('.message');
        const contentEl = msg?.querySelector('.message-content');
        if (!contentEl) return;

        // Stop if already playing
        if (ttsAudio && !ttsAudio.paused) {
            ttsAudio.pause();
            ttsAudio = null;
            btn.textContent = '🔊';
            return;
        }

        // Check TTS status before attempting
        await checkTtsStatus();
        if (!ttsAvailable) {
            showToast('TTS unavailable — Kokoro server is not running or still loading', 'warning');
            return;
        }

        // Pre-clean text for TTS (belt-and-braces — backend sanitiser is authoritative)
        const text = contentEl.textContent
            .substring(0, 4000)
            .replace(/```[\s\S]*?```/g, ' (code block omitted) ')
            .replace(/https?:\/\/\S+/g, '')
            .replace(/[#*_~`|]/g, '')
            .replace(/\s{2,}/g, ' ')
            .trim();
        btn.textContent = '⏳';

        try {
            const resp = await fetch('/api/tts', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text, voice: 'am_michael', speed: 1.0 }),
            });

            if (!resp.ok) {
                // Parse error body safely (may be JSON or binary)
                const ct = resp.headers.get('content-type') || '';
                let errMsg = resp.statusText;
                if (ct.includes('json')) {
                    const errData = await resp.json().catch(() => ({}));
                    errMsg = errData.error || errData.detail || errMsg;
                }
                showToast(`TTS: ${errMsg}`, 'error');
                btn.textContent = '🔊';
                return;
            }

            const blob = await resp.blob();
            if (blob.size === 0) {
                showToast('TTS returned empty audio', 'error');
                btn.textContent = '🔊';
                return;
            }

            // Create audio and set up handlers BEFORE calling play()
            // This keeps play() in the synchronous user-gesture call chain
            const url = URL.createObjectURL(blob);
            ttsAudio = new Audio(url);
            ttsAudio.onerror = () => {
                showToast('Audio playback failed', 'error');
                btn.textContent = '🔊';
                URL.revokeObjectURL(url);
                ttsAudio = null;
            };
            ttsAudio.onended = () => {
                btn.textContent = '🔊';
                URL.revokeObjectURL(url);
                ttsAudio = null;
            };

            // play() must be called within user gesture chain
            const playPromise = ttsAudio.play();
            if (playPromise) {
                playPromise.then(() => {
                    btn.textContent = '⏹️';
                }).catch(err => {
                    showToast('Browser blocked audio playback. Click the page first.', 'warning');
                    btn.textContent = '🔊';
                    URL.revokeObjectURL(url);
                    ttsAudio = null;
                });
            }
        } catch (e) {
            showToast('TTS failed: ' + (e.message || 'unknown error'), 'error');
            btn.textContent = '🔊';
        }
    }

    // ─── Voice/Video Calls ───
    let callWs = null;
    let callType = null; // 'voice' or 'video'
    let mediaStream = null;
    let mediaRecorder = null;
    let callMuted = false;
    let callAudioQueue = [];

    function toggleVoiceCall() {
        if (callType === 'voice') { endCall(); return; }
        startCall('voice');
    }

    function toggleVideoCall() {
        if (callType === 'video') { endCall(); return; }
        startCall('video');
    }

    async function startCall(type) {
        callType = type;
        const overlay = document.getElementById('call-overlay');
        const statusEl = document.getElementById('call-status');
        const videoEl = document.getElementById('call-video-preview');
        const waveform = document.getElementById('call-waveform');
        const transcriptEl = document.getElementById('call-transcript');
        const responseEl = document.getElementById('call-response');

        overlay.style.display = 'flex';
        statusEl.textContent = 'Requesting media access...';
        transcriptEl.textContent = '';
        responseEl.textContent = '';

        // Build waveform bars
        waveform.innerHTML = '';
        for (let i = 0; i < 20; i++) {
            const bar = document.createElement('div');
            bar.className = 'bar';
            bar.style.animationDelay = `${i * 0.05}s`;
            waveform.appendChild(bar);
        }

        // Activate button
        const btn = document.getElementById(type === 'voice' ? 'voice-btn' : 'video-btn');
        btn.classList.add('active');

        try {
            const constraints = { audio: true };
            if (type === 'video') constraints.video = { width: 640, height: 480 };
            mediaStream = await navigator.mediaDevices.getUserMedia(constraints);

            if (type === 'video') {
                videoEl.style.display = 'block';
                videoEl.srcObject = mediaStream;
            } else {
                videoEl.style.display = 'none';
            }

            // Connect WebSocket
            const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${proto}//${location.host}/ws/${type}`;
            callWs = new WebSocket(wsUrl);

            callWs.onopen = () => {
                statusEl.textContent = type === 'voice' ? '🎙️ Voice call active' : '📹 Video call active';
                startRecording();
            };

            callWs.onmessage = async (event) => {
                if (event.data instanceof Blob) {
                    // TTS audio response
                    const url = URL.createObjectURL(event.data);
                    const audio = new Audio(url);
                    audio.onended = () => URL.revokeObjectURL(url);
                    try { await audio.play(); } catch(e) { /* autoplay blocked */ }
                } else {
                    const data = JSON.parse(event.data);
                    if (data.type === 'transcript' || data.type === 'response') {
                        if (data.text) transcriptEl.textContent = 'You: ' + data.text;
                        if (data.response) responseEl.textContent = data.response;
                    } else if (data.type === 'voice_error' || data.type === 'video_error') {
                        showToast(data.error, 'error');
                    }
                }
            };

            callWs.onclose = () => { endCall(); };
            callWs.onerror = () => {
                showToast('Call connection failed', 'error');
                endCall();
            };
        } catch (e) {
            showToast('Media access denied: ' + e.message, 'error');
            endCall();
        }
    }

    function startRecording() {
        if (!mediaStream) return;

        // Record audio in 3-second chunks
        mediaRecorder = new MediaRecorder(mediaStream, { mimeType: 'audio/webm;codecs=opus' });
        let chunks = [];

        mediaRecorder.ondataavailable = (e) => {
            if (e.data.size > 0) chunks.push(e.data);
        };

        mediaRecorder.onstop = async () => {
            if (chunks.length === 0 || callMuted) { chunks = []; return; }
            const blob = new Blob(chunks, { type: 'audio/webm' });
            chunks = [];

            if (callWs && callWs.readyState === WebSocket.OPEN) {
                if (callType === 'video') {
                    // Send video frame + audio together
                    const frame = await captureVideoFrame();
                    const audioB64 = await blobToBase64(blob);
                    callWs.send(JSON.stringify({
                        type: 'video_frame',
                        frame: frame,
                        audio: audioB64,
                    }));
                } else {
                    // Send audio binary directly
                    callWs.send(await blob.arrayBuffer());
                }
            }

            // Continue recording if call still active
            if (callType && mediaRecorder && mediaRecorder.state === 'inactive') {
                try { mediaRecorder.start(); setTimeout(() => {
                    if (mediaRecorder && mediaRecorder.state === 'recording') mediaRecorder.stop();
                }, 3000); } catch(e) {}
            }
        };

        mediaRecorder.start();
        setTimeout(() => {
            if (mediaRecorder && mediaRecorder.state === 'recording') mediaRecorder.stop();
        }, 3000);
    }

    async function captureVideoFrame() {
        const video = document.getElementById('call-video-preview');
        const canvas = document.createElement('canvas');
        canvas.width = 320; canvas.height = 240;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(video, 0, 0, 320, 240);
        return canvas.toDataURL('image/jpeg', 0.6).split(',')[1];
    }

    function blobToBase64(blob) {
        return new Promise((resolve) => {
            const reader = new FileReader();
            reader.onloadend = () => resolve(reader.result.split(',')[1] || '');
            reader.readAsDataURL(blob);
        });
    }

    function toggleCallMute() {
        callMuted = !callMuted;
        const btn = document.getElementById('call-mute');
        btn.textContent = callMuted ? '🔈 Unmute' : '🔇 Mute';
        if (mediaStream) {
            mediaStream.getAudioTracks().forEach(t => t.enabled = !callMuted);
        }
    }

    function endCall() {
        if (callWs && callWs.readyState === WebSocket.OPEN) {
            callWs.send(JSON.stringify({ type: callType === 'video' ? 'video_end' : 'voice_end' }));
            callWs.close();
        }
        callWs = null;

        if (mediaRecorder && mediaRecorder.state !== 'inactive') {
            try { mediaRecorder.stop(); } catch(e) {}
        }
        mediaRecorder = null;

        if (mediaStream) {
            mediaStream.getTracks().forEach(t => t.stop());
            mediaStream = null;
        }

        callType = null;
        callMuted = false;

        document.getElementById('call-overlay').style.display = 'none';
        document.getElementById('call-video-preview').style.display = 'none';
        document.getElementById('voice-btn').classList.remove('active');
        document.getElementById('video-btn').classList.remove('active');
    }

    function showToast(message, type = 'info', duration = 3000) {
        const container = document.getElementById('toast-container');
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        toast.textContent = message;
        container.appendChild(toast);
        setTimeout(() => { toast.classList.add('toast-exit'); setTimeout(() => toast.remove(), 300); }, duration);
    }

    // ─── Confirm Dialog ───
    function confirmAction(title, message, okLabel = 'Delete') {
        return new Promise(resolve => {
            const overlay = document.getElementById('confirm-overlay');
            document.getElementById('confirm-title').textContent = title;
            document.getElementById('confirm-message').textContent = message;
            document.getElementById('confirm-ok').textContent = okLabel;
            overlay.style.display = 'flex';
            const ok = document.getElementById('confirm-ok');
            const cancel = document.getElementById('confirm-cancel');
            const cleanup = (result) => { overlay.style.display = 'none'; ok.replaceWith(ok.cloneNode(true)); cancel.replaceWith(cancel.cloneNode(true)); resolve(result); };
            document.getElementById('confirm-ok').addEventListener('click', () => cleanup(true));
            document.getElementById('confirm-cancel').addEventListener('click', () => cleanup(false));
        });
    }

    // ─── Sessions ───
    let allSessions = [];

    async function loadSessions() {
        try {
            const resp = await fetch('/api/sessions');
            allSessions = await resp.json();
            renderSessionList(allSessions);
        } catch (e) { showToast('Failed to load sessions', 'error'); }
    }

    function renderSessionList(sessions) {
        const list = document.getElementById('sessions-list');
        const filter = (document.getElementById('session-search')?.value || '').toLowerCase();
        const filtered = filter ? sessions.filter(s =>
            s.title.toLowerCase().includes(filter) || (s.preview || '').toLowerCase().includes(filter)
        ) : sessions;

        // Group sessions by date_group
        const groupOrder = ['pinned', 'today', 'yesterday', 'this_week', 'this_month', 'older'];
        const groupLabels = { pinned: '📌 Pinned', today: 'Today', yesterday: 'Yesterday', this_week: 'This Week', this_month: 'This Month', older: 'Older' };
        const groups = {};
        filtered.forEach(s => {
            const g = s.date_group || 'older';
            if (!groups[g]) groups[g] = [];
            groups[g].push(s);
        });

        let html = '';
        for (const group of groupOrder) {
            if (!groups[group] || groups[group].length === 0) continue;
            html += `<div class="session-group-header">${groupLabels[group] || group}</div>`;
            for (const s of groups[group]) {
                const active = s.id === currentSessionId ? 'active' : '';
                const pin = s.pinned ? '<span class="session-pin">📌</span>' : '';
                const preview = s.preview ? escapeHtml(s.preview) : '';
                html += `
                <div class="session-item ${active}" data-id="${s.id}" onclick="ErnOS.switchSession('${s.id}')">
                    <div class="session-info">
                        <span class="session-title" ondblclick="event.stopPropagation(); ErnOS.startRenameSession('${s.id}')">${escapeHtml(s.title)}</span>
                        <div class="session-meta">
                            <span class="session-preview">${preview}</span>
                            <span class="session-time">${pin}${escapeHtml(s.relative_time || '')}</span>
                        </div>
                    </div>
                    <button class="session-ctx-btn" onclick="event.stopPropagation(); ErnOS.showSessionMenu(event, '${s.id}')" title="More">⋮</button>
                </div>`;
            }
        }
        if (!html) html = '<div class="session-group-header" style="text-align:center;padding:20px">No sessions</div>';
        list.innerHTML = html;
    }

    function filterSessionList(query) {
        renderSessionList(allSessions);
    }

    function newChat() {
        currentSessionId = null;
        document.getElementById('messages').innerHTML = welcomeHtml();
        loadSessions();
        if (currentView !== 'chat') switchView('chat');
    }

    async function switchSession(id) {
        currentSessionId = id;
        messageCounter = 0;
        if (currentView !== 'chat') switchView('chat');
        const container = document.getElementById('messages');
        container.innerHTML = '';
        hideWelcome();

        try {
            const resp = await fetch(`/api/sessions/${id}`);
            if (resp.ok) {
                const session = await resp.json();
                if (session.messages && session.messages.length > 0) {
                    session.messages.forEach(m => {
                        const role = m.role || 'user';
                        // Skip tool-call and tool-result messages from display
                        if (role === 'tool' || role === 'system') return;
                        // Handle assistant messages with null content (tool calls)
                        if (m.content === null || m.content === undefined) return;

                        let text = '';
                        if (typeof m.content === 'string') {
                            text = m.content;
                        } else if (Array.isArray(m.content)) {
                            // Multipart content: [{type:'text', text:'...'}, ...]
                            text = m.content
                                .filter(p => p.type === 'text' && p.text)
                                .map(p => p.text)
                                .join('');
                        }
                        if (text) addMessage(role, text);
                    });
                }
            }
        } catch (e) { showToast('Failed to load session', 'error'); }
        loadSessions();
    }

    async function deleteSession(id) {
        const ok = await confirmAction('Delete Session', 'This will permanently delete this conversation. This cannot be undone.');
        if (!ok) return;
        try {
            await fetch(`/api/sessions/${id}`, { method: 'DELETE' });
            showToast('Session deleted', 'success');
            if (currentSessionId === id) newChat();
            loadSessions();
        } catch (e) { showToast('Failed to delete session', 'error'); }
    }

    // Context menu
    let contextMenuSessionId = null;

    function showSessionMenu(event, id) {
        contextMenuSessionId = id;
        const menu = document.getElementById('context-menu');
        const session = allSessions.find(s => s.id === id);

        // Update pin label
        const pinBtn = menu.querySelector('[data-action="pin"]');
        pinBtn.textContent = session?.pinned ? '📌 Unpin' : '📌 Pin';
        const archiveBtn = menu.querySelector('[data-action="archive"]');
        archiveBtn.textContent = session?.archived ? '📦 Unarchive' : '📦 Archive';

        menu.style.display = 'block';
        menu.style.left = Math.min(event.clientX, window.innerWidth - 180) + 'px';
        menu.style.top = Math.min(event.clientY, window.innerHeight - 200) + 'px';

        const close = (e) => { if (!menu.contains(e.target)) { menu.style.display = 'none'; document.removeEventListener('click', close); } };
        setTimeout(() => document.addEventListener('click', close), 0);
    }

    // Wire up context menu actions
    function initContextMenu() {
        document.getElementById('context-menu').addEventListener('click', async (e) => {
            const btn = e.target.closest('[data-action]');
            if (!btn || !contextMenuSessionId) return;
            document.getElementById('context-menu').style.display = 'none';
            const id = contextMenuSessionId;

            switch (btn.dataset.action) {
                case 'rename': startRenameSession(id); break;
                case 'pin': await togglePin(id); break;
                case 'archive': await toggleArchive(id); break;
                case 'export': await exportSession(id); break;
                case 'delete': await deleteSession(id); break;
            }
        });
    }

    function startRenameSession(id) {
        const item = document.querySelector(`.session-item[data-id="${id}"]`);
        if (!item) return;
        const titleEl = item.querySelector('.session-title');
        const currentTitle = titleEl.textContent;
        const input = document.createElement('input');
        input.className = 'session-rename-input';
        input.value = currentTitle;
        titleEl.replaceWith(input);
        input.focus();
        input.select();

        const save = async () => {
            const newTitle = input.value.trim() || currentTitle;
            try {
                await fetch(`/api/sessions/${id}`, { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ title: newTitle }) });
                showToast('Session renamed', 'success');
            } catch (e) { showToast('Rename failed', 'error'); }
            loadSessions();
        };
        input.addEventListener('blur', save);
        input.addEventListener('keydown', (e) => { if (e.key === 'Enter') input.blur(); if (e.key === 'Escape') { input.value = currentTitle; input.blur(); } });
    }

    async function togglePin(id) {
        try {
            await fetch(`/api/sessions/${id}/pin`, { method: 'PUT' });
            loadSessions();
        } catch (e) { showToast('Pin failed', 'error'); }
    }

    async function toggleArchive(id) {
        try {
            await fetch(`/api/sessions/${id}/archive`, { method: 'PUT' });
            showToast('Session archived', 'success');
            if (currentSessionId === id) newChat();
            loadSessions();
        } catch (e) { showToast('Archive failed', 'error'); }
    }

    async function exportSession(id) {
        try {
            const resp = await fetch(`/api/sessions/${id}/export`);
            const data = await resp.json();
            if (data.ok) {
                const blob = new Blob([data.markdown], { type: 'text/markdown' });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url; a.download = `${(data.title || 'session').replace(/[^a-z0-9]/gi, '_')}.md`;
                a.click(); URL.revokeObjectURL(url);
                showToast('Session exported', 'success');
            }
        } catch (e) { showToast('Export failed', 'error'); }
    }

    // ─── Global Search Palette (Cmd+K) ───
    let searchDebounce = null;

    function openSearchPalette() {
        const pal = document.getElementById('search-palette');
        pal.style.display = 'flex';
        const input = document.getElementById('search-palette-input');
        input.value = '';
        document.getElementById('search-palette-results').innerHTML = '<div class="search-palette-empty">Type to search across all conversations</div>';
        input.focus();

        input.oninput = () => {
            clearTimeout(searchDebounce);
            searchDebounce = setTimeout(() => runGlobalSearch(input.value), 250);
        };
    }

    function closeSearchPalette() {
        document.getElementById('search-palette').style.display = 'none';
    }

    async function runGlobalSearch(query) {
        const results = document.getElementById('search-palette-results');
        if (!query.trim()) {
            results.innerHTML = '<div class="search-palette-empty">Type to search across all conversations</div>';
            return;
        }
        try {
            const resp = await fetch(`/api/sessions/search?q=${encodeURIComponent(query)}`);
            const data = await resp.json();
            if (!data.length) {
                results.innerHTML = '<div class="search-palette-empty">No results found</div>';
                return;
            }
            results.innerHTML = data.map(r => `
                <div class="search-result" onclick="ErnOS.closeSearchPalette(); ErnOS.switchSession('${r.session_id}')">
                    <div class="search-result-title">${escapeHtml(r.title)}</div>
                    <div class="search-result-snippet">${escapeHtml(r.snippet)}</div>
                </div>
            `).join('');
        } catch (e) { results.innerHTML = '<div class="search-palette-empty">Search failed</div>'; }
    }

    // ─── View Switching ───
    function switchView(view) {
        currentView = view;
        // Update nav tabs
        document.querySelectorAll('.nav-tab').forEach(t => {
            t.classList.toggle('active', t.dataset.view === view);
        });
        // Update views
        document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
        document.getElementById(`view-${view}`).classList.add('active');
        // Show/hide sessions sidebar section
        const sessions = document.getElementById('sidebar-sessions');
        sessions.style.display = view === 'chat' ? 'flex' : 'none';
        // Load view data
        if (view === 'memory') loadMemoryView();
        if (view === 'tools') loadToolsView();
        if (view === 'training') loadTrainingView();
        if (view === 'interpretability') loadInterpretabilityView();
        if (view === 'steering') loadSteeringView();
        if (view === 'logs') loadLogsView();
        if (view === 'settings') { loadSettingsView().then(() => loadVersionInfo()); loadAutonomyState(); loadSnapshots(); loadModelHub(); }
        if (view === 'identity') loadIdentityView();
        if (view === 'agents') loadAgentsView();
        if (view === 'scheduler') loadSchedulerView();
        if (view === 'codes') loadCodesView();
    }

    // ─── Codes IDE View ───
    let codesLoaded = false;

    async function loadCodesView() {
        if (codesLoaded) return;
        const loading = document.getElementById('codes-loading');
        const iframe = document.getElementById('codes-iframe');
        const dot = document.getElementById('codes-status-dot');
        const text = document.getElementById('codes-status-text');

        try {
            const resp = await fetch('/api/codes/status');
            const data = await resp.json();

            if (data.available) {
                iframe.src = data.url;
                loading.style.display = 'none';
                iframe.style.display = 'block';
                dot.style.background = 'var(--accent)';
                text.textContent = 'VS Code Connected';
                codesLoaded = true;
            } else if (data.enabled) {
                text.textContent = 'VS Code starting...';
                dot.style.background = '#f59e0b';
                // Retry in 3 seconds
                setTimeout(loadCodesView, 3000);
            } else {
                text.textContent = 'VS Code IDE disabled';
                dot.style.background = '#ef4444';
                loading.querySelector('.codes-loading-text').textContent = 'Codes IDE is disabled';
                loading.querySelector('.codes-loading-sub').textContent = 'Enable in ern-os.toml: [codes] enabled = true';
            }
        } catch (e) {
            text.textContent = 'Connection failed';
            dot.style.background = '#ef4444';
        }
    }

    // ─── Memory View ───
    async function loadMemoryView() {
        try {
            const resp = await fetch('/api/memory/stats');
            const stats = await resp.json();
            document.getElementById('memory-stats').innerHTML = `
                <span class="mem-stat"><strong>${stats.timeline}</strong> Timeline</span>
                <span class="mem-stat"><strong>${stats.lessons}</strong> Lessons</span>
                <span class="mem-stat"><strong>${stats.procedures}</strong> Procedures</span>
                <span class="mem-stat"><strong>${stats.scratchpad}</strong> Scratchpad</span>
                <span class="mem-stat"><strong>${stats.synaptic_nodes}</strong> Nodes / <strong>${stats.synaptic_edges}</strong> Edges</span>
                <span class="mem-stat"><strong>${stats.embeddings}</strong> Embeddings</span>
                <span class="mem-stat"><strong>${stats.consolidations}</strong> Consolidations</span>
            `;
        } catch (e) { /* ignore */ }
        loadMemoryTier('timeline', document.querySelector('.tier-tab.active'));
    }

    async function loadMemoryTier(tier, btn) {
        // Update tabs
        document.querySelectorAll('.tier-tab').forEach(t => t.classList.remove('active'));
        if (btn) btn.classList.add('active');

        const container = document.getElementById('tier-content');
        container.innerHTML = '<p class="empty-state">Loading...</p>';

        try {
            const resp = await fetch(`/api/memory/${tier}`);
            const data = await resp.json();

            if (tier === 'timeline') {
                if (!data.entries || data.entries.length === 0) {
                    container.innerHTML = '<p class="empty-state">No timeline entries yet</p>';
                    return;
                }
                container.innerHTML = `<table class="data-table">
                    <thead><tr><th>Time</th><th>Session</th><th>Content</th></tr></thead>
                    <tbody>${data.entries.map(e => `<tr>
                        <td style="white-space:nowrap;font-size:11px">${new Date(e.timestamp).toLocaleString()}</td>
                        <td style="font-size:11px;font-family:'JetBrains Mono',monospace">${escapeHtml(e.session_id).substring(0, 8)}…</td>
                        <td>${escapeHtml(e.transcript).substring(0, 200)}</td>
                    </tr>`).join('')}</tbody></table>`;
            } else if (tier === 'lessons') {
                if (!data.lessons || data.lessons.length === 0) {
                    container.innerHTML = '<p class="empty-state">No lessons learned yet</p>';
                    return;
                }
                container.innerHTML = data.lessons.map(l => `
                    <div class="data-card">
                        <div class="card-title">${escapeHtml(l.rule)}</div>
                        <div class="card-meta">Source: ${escapeHtml(l.source)} · Confidence: ${(l.confidence * 100).toFixed(0)}% · Applied: ${l.times_applied}×</div>
                    </div>
                `).join('');
            } else if (tier === 'procedures') {
                if (!data.procedures || data.procedures.length === 0) {
                    container.innerHTML = '<p class="empty-state">No procedures stored yet</p>';
                    return;
                }
                container.innerHTML = data.procedures.map(p => `
                    <div class="data-card">
                        <div class="card-title">${escapeHtml(p.name)}</div>
                        <div class="card-meta">${p.steps} steps · ${p.success_count} successes${p.last_used ? ' · Last: ' + new Date(p.last_used).toLocaleDateString() : ''}</div>
                        <div class="card-body">${escapeHtml(p.description)}</div>
                    </div>
                `).join('');
            } else if (tier === 'scratchpad') {
                if (!data.entries || data.entries.length === 0) {
                    container.innerHTML = '<p class="empty-state">Scratchpad is empty</p>';
                    return;
                }
                container.innerHTML = `<table class="data-table">
                    <thead><tr><th>Key</th><th>Value</th><th>Pinned</th></tr></thead>
                    <tbody>${data.entries.map(e => `<tr>
                        <td style="font-family:'JetBrains Mono',monospace;font-weight:600">${escapeHtml(e.key)}</td>
                        <td>${escapeHtml(e.value)}</td>
                        <td>${e.pinned ? '📌' : '—'}</td>
                    </tr>`).join('')}</tbody></table>`;
            } else if (tier === 'synaptic') {
                container.innerHTML = `
                    <div style="margin-bottom:14px;font-size:13px;color:var(--text-secondary)">
                        ${data.node_count} nodes · ${data.edge_count} edges · Layers: ${data.layers.join(', ') || 'none'}
                    </div>
                    ${data.nodes.length === 0 ? '<p class="empty-state">No synaptic nodes yet</p>' :
                    data.nodes.map(n => `
                        <div class="data-card">
                            <div class="card-title">${escapeHtml(n.id)}</div>
                            <div class="card-meta">Layer: ${escapeHtml(n.layer)}</div>
                            <div class="card-body" style="font-family:'JetBrains Mono',monospace;font-size:11px">${escapeHtml(JSON.stringify(n.data, null, 2))}</div>
                        </div>
                    `).join('')}
                `;
            }
        } catch (e) {
            container.innerHTML = `<p class="empty-state">Failed to load ${tier}</p>`;
        }
    }

    // ─── Tools View ───
    async function loadToolsView() {
        try {
            const resp = await fetch('/api/tools');
            const data = await resp.json();
            const grid = document.getElementById('tools-grid');
            const disabledTools = JSON.parse(localStorage.getItem('ern-os-disabled-tools') || '[]');

            const renderTools = (tools, layer) => tools.map(t => {
                const fn = t.function || {};
                const name = fn.name || '?';
                const isEnabled = !disabledTools.includes(name);
                return `<div class="tool-card">
                    <div class="tool-card-header">
                        <div class="tool-name">${escapeHtml(name)}</div>
                        <label class="toggle-switch">
                            <input type="checkbox" ${isEnabled ? 'checked' : ''} onchange="ErnOS.toggleTool('${escapeHtml(name)}', this.checked)">
                            <span class="toggle-slider"></span>
                        </label>
                    </div>
                    <div class="tool-desc">${escapeHtml(fn.description || 'No description')}</div>
                    <span class="tool-layer ${layer}">${layer === 'l1' ? 'Layer 1' : 'Layer 2'}</span>
                </div>`;
            }).join('');

            grid.innerHTML = renderTools(data.layer1 || [], 'l1') + renderTools(data.layer2 || [], 'l2');
        } catch (e) {
            document.getElementById('tools-grid').innerHTML = '<p class="empty-state">Failed to load tools</p>';
        }
    }

    function toggleTool(name, enabled) {
        let disabled = JSON.parse(localStorage.getItem('ern-os-disabled-tools') || '[]');
        if (enabled) {
            disabled = disabled.filter(t => t !== name);
        } else {
            if (!disabled.includes(name)) disabled.push(name);
        }
        localStorage.setItem('ern-os-disabled-tools', JSON.stringify(disabled));
    }

    // ─── Training View ───
    async function loadTrainingView() {
        loadTrainingTab('buffers');
    }

    ErnOS.loadTrainingTab = function(tab, btn) {
        if (btn) {
            btn.closest('.tier-tabs').querySelectorAll('.tier-tab').forEach(t => t.classList.remove('active'));
            btn.classList.add('active');
        }
        const container = document.getElementById('training-sections');
        container.innerHTML = '<p class="empty-state">Loading...</p>';
        if (tab === 'buffers') loadBuffersTab(container);
        else if (tab === 'curriculum') loadCurriculumTab(container);
        else if (tab === 'review') loadReviewTab(container);
        else if (tab === 'adapters') loadAdaptersTab(container);
    };

    async function loadBuffersTab(container) {
        try {
            const resp = await fetch('/api/training');
            const data = await resp.json();
            const goldenHtml = (data.golden.entries || []).slice(0, 30).map(e => {
                const input = e.input || e.user_query || e.query || '';
                const output = e.output || e.response || '';
                const score = e.quality_score !== undefined ? e.quality_score : '';
                const method = e.method || '';
                const ts = e.timestamp || '';
                return `<div class="training-pair">
                    <div class="training-col"><div class="col-label">Input</div><div class="col-text">${escapeHtml(input)}</div></div>
                    <div class="training-col"><div class="col-label">Output</div><div class="col-text">${escapeHtml(output)}</div>
                        ${score || method || ts ? `<div class="training-meta">${score ? `Score: ${score}` : ''} ${method ? `· ${method}` : ''} ${ts ? `· ${ts.substring(0, 16)}` : ''}</div>` : ''}
                    </div>
                </div>`;
            }).join('') || '<div class="training-entry"><em style="color:var(--text-muted)">No golden entries yet</em></div>';
            const rejectionHtml = (data.rejections.entries || []).slice(0, 30).map(e => {
                const input = e.input || e.user_query || e.query || '';
                const reason = e.reason || e.rejected_response || '';
                return `<div class="training-pair">
                    <div class="training-col"><div class="col-label">Input</div><div class="col-text">${escapeHtml(input)}</div></div>
                    <div class="training-col"><div class="col-label">Reason</div><div class="col-text">${escapeHtml(reason)}</div></div>
                </div>`;
            }).join('') || '<div class="training-entry"><em style="color:var(--text-muted)">No rejections yet</em></div>';
            container.innerHTML = `
                <div class="training-section"><div class="training-section-header"><span>✅ Golden Buffer</span><span class="training-count">${data.golden.count}</span></div>${goldenHtml}</div>
                <div class="training-section"><div class="training-section-header"><span>❌ Rejection Pairs</span><span class="training-count">${data.rejections.count}</span></div>${rejectionHtml}</div>
            `;
            // Append learning engine stats
            try {
                const ls = await (await fetch('/api/learning/status')).json();
                container.insertAdjacentHTML('beforeend', `
                    <div class="training-section" style="margin-top:16px"><div class="training-section-header"><span>🧬 Learning Engine</span></div>
                    <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:10px;padding:8px 0">
                        <div class="data-card" style="padding:10px"><div style="font-size:20px;font-weight:700;color:var(--accent)">${ls.adapter_count ?? 0}</div><div style="font-size:10px;color:var(--text-muted)">LoRA Adapters</div></div>
                        <div class="data-card" style="padding:10px"><div style="font-size:20px;font-weight:700;color:var(--accent)">${ls.sleep_cycles ?? 0}</div><div style="font-size:10px;color:var(--text-muted)">Sleep Cycles</div></div>
                    </div></div>
                `);
            } catch (_) {}
        } catch (e) {
            container.innerHTML = '<p class="empty-state">Failed to load training data</p>';
        }
    }

    async function loadCurriculumTab(container) {
        try {
            const resp = await fetch('/api/curriculum');
            const data = await resp.json();
            if (!data.courses || data.courses.length === 0) {
                container.innerHTML = `
                    <div class="training-section">
                        <div class="training-section-header"><span>📚 Courses</span><span class="training-count">0</span></div>
                        <div style="padding:20px;text-align:center;color:var(--text-muted)">
                            <p>No courses enrolled yet.</p>
                            <p style="font-size:12px;margin-top:8px">Add courses via the scheduler <strong>attend_class</strong> job or the learning tool.</p>
                        </div>
                    </div>`;
                return;
            }
            const cards = data.courses.map(c => {
                const pct = c.total_lessons > 0 ? Math.round((c.completed_lessons / c.total_lessons) * 100) : 0;
                const levelColor = {'Primary':'#4ade80','Secondary':'#38bdf8','Undergraduate':'#a78bfa','Masters':'#f472b6','Doctoral':'#fbbf24'}[c.level] || 'var(--accent)';
                return `<div class="course-card">
                    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">
                        <div style="font-weight:600;font-size:14px">${escapeHtml(c.title)}</div>
                        <span class="level-badge" style="background:${levelColor}">${c.level}</span>
                    </div>
                    <div style="font-size:12px;color:var(--text-muted);margin-bottom:8px">${escapeHtml(c.subject)} · ${c.completed_lessons}/${c.total_lessons} lessons${c.is_complete ? ' · ✅ Complete' : ''}</div>
                    <div class="course-progress-track"><div class="course-progress-fill" style="width:${pct}%"></div></div>
                    <div style="display:flex;justify-content:space-between;font-size:11px;color:var(--text-muted);margin-top:4px">
                        <span>${pct}% complete</span>
                        <span>Avg: ${(c.average_score * 100).toFixed(0)}%</span>
                    </div>
                </div>`;
            }).join('');
            container.innerHTML = `
                <div class="training-section">
                    <div class="training-section-header"><span>📚 Courses</span><span class="training-count">${data.count}</span></div>
                    <div class="course-grid">${cards}</div>
                </div>`;
        } catch (e) {
            container.innerHTML = '<p class="empty-state">Failed to load curriculum</p>';
        }
    }

    async function loadReviewTab(container) {
        try {
            const resp = await fetch('/api/curriculum/review');
            const stats = await resp.json();
            container.innerHTML = `
                <div class="training-section">
                    <div class="training-section-header"><span>🔁 Spaced Review</span><span class="training-count">${stats.total_cards} cards</span></div>
                    <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:10px;padding:12px 0">
                        <div class="data-card" style="padding:12px">
                            <div style="font-size:24px;font-weight:700;color:${stats.cards_due > 0 ? '#f97316' : 'var(--accent)'}">${stats.cards_due}</div>
                            <div style="font-size:10px;color:var(--text-muted)">Cards Due Now</div>
                        </div>
                        <div class="data-card" style="padding:12px">
                            <div style="font-size:24px;font-weight:700;color:var(--accent)">${stats.total_cards}</div>
                            <div style="font-size:10px;color:var(--text-muted)">Total Cards</div>
                        </div>
                        <div class="data-card" style="padding:12px">
                            <div style="font-size:24px;font-weight:700;color:var(--accent)">${stats.avg_box_level.toFixed(1)}</div>
                            <div style="font-size:10px;color:var(--text-muted)">Avg Box Level</div>
                        </div>
                        <div class="data-card" style="padding:12px">
                            <div style="font-size:24px;font-weight:700;color:var(--accent)">${(stats.retention_rate * 100).toFixed(0)}%</div>
                            <div style="font-size:10px;color:var(--text-muted)">Retention Rate</div>
                        </div>
                    </div>
                    ${stats.total_cards === 0 ? '<div style="padding:12px;text-align:center;color:var(--text-muted);font-size:12px">Complete courses to generate review cards via Leitner spaced repetition.</div>' : ''}
                </div>`;
        } catch (e) {
            container.innerHTML = '<p class="empty-state">Failed to load review stats</p>';
        }
    }

    async function loadAdaptersTab(container) {
        try {
            const resp = await fetch('/api/learning/adapters');
            const ad = await resp.json();
            if (!ad.adapters || ad.adapters.length === 0) {
                container.innerHTML = `
                    <div class="training-section">
                        <div class="training-section-header"><span>🔌 LoRA Adapters</span><span class="training-count">0</span></div>
                        <div style="padding:20px;text-align:center;color:var(--text-muted)">
                            <p>No adapters trained yet.</p>
                            <p style="font-size:12px;margin-top:8px">Adapters are created automatically during sleep cycles when training buffers have sufficient data.</p>
                        </div>
                    </div>`;
                return;
            }
            const cards = ad.adapters.map(a => `
                <div class="data-card" style="margin-bottom:8px">
                    <div class="card-title">${escapeHtml(a.name || a.id)}</div>
                    <div class="card-meta">${a.method || ''} ${a.param_count ? '· ' + a.param_count + ' params' : ''} ${a.created_at ? '· ' + new Date(a.created_at).toLocaleDateString() : ''}</div>
                </div>
            `).join('');
            container.innerHTML = `
                <div class="training-section">
                    <div class="training-section-header"><span>🔌 LoRA Adapters</span><span class="training-count">${ad.adapters.length}</span></div>
                    ${cards}
                </div>`;
        } catch (e) {
            container.innerHTML = '<p class="empty-state">Failed to load adapters</p>';
        }
    }

    // ─── Scheduler View ───
    async function loadSchedulerView() {
        const container = document.getElementById('scheduler-sections');
        container.innerHTML = '<p class="empty-state">Loading...</p>';
        try {
            const resp = await fetch('/api/scheduler');
            const scheduler = await resp.json();

            let jobsHtml = '';
            scheduler.jobs.forEach(j => {
                const badge = j.builtin
                    ? '<span style="font-size:9px;padding:1px 6px;border-radius:4px;background:rgba(68,170,255,0.1);color:var(--info);border:1px solid rgba(68,170,255,0.15)">SYSTEM</span>'
                    : '<span style="font-size:9px;padding:1px 6px;border-radius:4px;background:rgba(0,255,136,0.1);color:var(--accent);border:1px solid rgba(0,255,136,0.15)">USER</span>';
                const delBtn = !j.builtin
                    ? '<button class="sched-del-btn" data-job-id="' + j.id + '" style="background:none;border:1px solid rgba(255,68,102,0.2);color:var(--error);padding:2px 8px;border-radius:6px;font-size:10px;cursor:pointer">✕</button>'
                    : '';
                jobsHtml += '<div class="setting-row" style="border-bottom:1px solid var(--border-subtle);padding:10px 0">'
                    + '<div class="setting-label" style="flex:1">'
                    + '<span class="label-text" style="display:flex;align-items:center;gap:8px">' + escapeHtml(j.name) + ' ' + badge + '</span>'
                    + '<span class="label-desc">' + escapeHtml(j.description) + '</span>'
                    + '<span class="label-desc" style="font-family:\'JetBrains Mono\',monospace;color:var(--text-muted);font-size:10px">'
                    + escapeHtml(j.schedule) + ' · runs: ' + j.run_count + ' · ' + (j.last_result ? 'last: ' + j.last_result : 'never run')
                    + '</span></div>'
                    + '<div style="display:flex;align-items:center;gap:8px">'
                    + '<label class="toggle-switch"><input type="checkbox" ' + (j.enabled ? 'checked' : '') + ' onchange="ErnOS.toggleSchedulerJob(\'' + j.id + '\')"><span class="toggle-slider"></span></label>'
                    + delBtn
                    + '</div></div>';
            });

            container.innerHTML = '<div class="settings-card" style="padding:0">'
                + '<div style="padding:16px 20px;border-bottom:1px solid var(--border-subtle);display:flex;justify-content:space-between;align-items:center">'
                + '<div class="card-title" style="margin:0"><span class="title-icon">⏱️</span> Background Jobs</div>'
                + '<span style="padding:2px 10px;border-radius:10px;font-size:11px;font-weight:700;background:rgba(0,255,136,0.1);color:var(--accent);border:1px solid rgba(0,255,136,0.2)">● ' + scheduler.enabled_jobs + '/' + scheduler.total_jobs + ' active</span>'
                + '</div>'
                + '<div style="padding:12px 20px">' + jobsHtml + '</div>'
                + '<div style="padding:12px 20px;border-top:1px solid var(--border-subtle)">'
                + '<button onclick="ErnOS.showAddJobForm()" style="background:var(--surface);border:1px solid var(--border);color:var(--accent);padding:6px 16px;border-radius:8px;font-size:12px;cursor:pointer;width:100%;font-weight:600">+ Add Job</button>'
                + '<div id="add-job-form" style="display:none;margin-top:12px">'
                + '<div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:8px">'
                + '<input id="job-name" placeholder="Job name" style="background:var(--bg-secondary);border:1px solid var(--border);color:var(--text-primary);padding:6px 10px;border-radius:6px;font-size:12px">'
                + '<input id="job-desc" placeholder="Description" style="background:var(--bg-secondary);border:1px solid var(--border);color:var(--text-primary);padding:6px 10px;border-radius:6px;font-size:12px">'
                + '</div>'
                + '<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;margin-bottom:8px">'
                + '<select id="job-schedule-type" style="background:var(--bg-secondary);border:1px solid var(--border);color:var(--text-primary);padding:6px 10px;border-radius:6px;font-size:12px">'
                + '<option value="interval">Interval (s)</option><option value="cron">Cron expr</option><option value="once">Once (UTC)</option></select>'
                + '<input id="job-schedule-value" placeholder="300" style="background:var(--bg-secondary);border:1px solid var(--border);color:var(--text-primary);padding:6px 10px;border-radius:6px;font-size:12px;font-family:\'JetBrains Mono\',monospace">'
                + '<select id="job-task-type" onchange="document.getElementById(\'custom-cmd-row\').style.display = this.value === \'custom\' ? \'block\' : \'none\'; document.getElementById(\'prompt-row\').style.display = this.value === \'prompt\' ? \'block\' : \'none\';" style="background:var(--bg-secondary);border:1px solid var(--border);color:var(--text-primary);padding:6px 10px;border-radius:6px;font-size:12px">'
                + '<option value="prompt">🧠 Prompt Job</option><option value="health_check">Health Check</option><option value="sleep_cycle">Sleep Cycle</option><option value="lesson_decay">Lesson Decay</option>'
                + '<option value="memory_consolidate">Memory Consolidate</option><option value="snapshot_capture">Snapshot Capture</option><option value="synaptic_prune">Synaptic Prune</option>'
                + '<option value="buffer_flush">Buffer Flush</option><option value="log_rotate">Log Rotate</option><option value="custom">Custom Shell</option></select>'
                + '</div>'
                + '<div id="prompt-row" style="margin-bottom:8px">'
                + '<textarea id="job-prompt" placeholder="Natural language instruction — the agent will use all available tools to fulfill this." rows="3" style="background:var(--bg-secondary);border:1px solid var(--border);color:var(--text-primary);padding:8px 10px;border-radius:6px;font-size:12px;width:100%;resize:vertical;box-sizing:border-box;font-family:inherit"></textarea>'
                + '</div>'
                + '<div id="custom-cmd-row" style="display:none;margin-bottom:8px">'
                + '<input id="job-custom-cmd" placeholder="shell command (e.g. echo hello)" style="background:var(--bg-secondary);border:1px solid var(--border);color:var(--text-primary);padding:6px 10px;border-radius:6px;font-size:12px;width:100%;font-family:\'JetBrains Mono\',monospace;box-sizing:border-box">'
                + '</div>'
                + '<button onclick="ErnOS.createSchedulerJob()" style="background:var(--accent);border:none;color:#000;padding:6px 16px;border-radius:8px;font-size:12px;cursor:pointer;font-weight:700;width:100%">Create Job</button>'
                + '</div></div></div>';

            // Wire up delete buttons
            container.querySelectorAll('.sched-del-btn').forEach(btn => {
                btn.addEventListener('click', () => deleteSchedulerJob(btn.dataset.jobId));
            });

            // Load execution history
            loadSchedulerHistory();
        } catch (e) {
            container.innerHTML = '<p class="empty-state">Failed to load scheduler</p>';
        }
    }

    async function loadSchedulerHistory() {
        try {
            const resp = await fetch('/api/scheduler/history');
            const data = await resp.json();
            if (!data.entries || data.entries.length === 0) return;

            const container = document.getElementById('scheduler-sections');
            const historyHtml = data.entries.slice(0, 25).map(h => {
                const t = new Date(h.started_at);
                const badge = h.success
                    ? '<span style="color:var(--accent);font-size:10px;font-weight:700">✓</span>'
                    : '<span style="color:var(--error);font-size:10px;font-weight:700">✗</span>';
                return '<div style="display:flex;align-items:center;gap:10px;padding:6px 0;border-bottom:1px solid var(--border-subtle);font-size:12px">'
                    + badge
                    + '<span style="min-width:60px;color:var(--text-muted);font-size:10px;font-family:\'JetBrains Mono\',monospace">' + t.toLocaleTimeString() + '</span>'
                    + '<span style="flex:1;font-weight:600">' + escapeHtml(h.job_name) + '</span>'
                    + '<span style="color:var(--text-muted);font-size:10px">' + escapeHtml(h.task) + '</span>'
                    + '<span style="color:var(--text-muted);font-size:10px;min-width:50px;text-align:right">' + h.duration_ms + 'ms</span>'
                    + '</div>';
            }).join('');

            container.insertAdjacentHTML('beforeend',
                '<div class="settings-card" style="margin-top:16px;padding:0">'
                + '<div style="padding:16px 20px;border-bottom:1px solid var(--border-subtle)">'
                + '<div class="card-title" style="margin:0"><span class="title-icon">📜</span> Execution History</div>'
                + '</div>'
                + '<div style="padding:12px 20px">' + historyHtml + '</div>'
                + '</div>'
            );
        } catch (e) {
            // History is optional — fail silently
        }
    }

    // ─── Settings View ───
    async function loadSettingsView() {
        const container = document.getElementById('settings-sections');
        container.innerHTML = '<p class="empty-state">Loading...</p>';

        try {
            const [statusResp, modelsResp, apiKeysResp, platformResp, platformStatusResp] = await Promise.all([
                fetch('/api/status'),
                fetch('/api/models'),
                fetch('/api/api-keys'),
                fetch('/api/platforms/config'),
                fetch('/api/platforms'),
            ]);
            const status = await statusResp.json();
            const modelsData = await modelsResp.json();
            const apiKeysData = await apiKeysResp.json();
            const apiKeys = apiKeysData.keys || {};
            const platformConfig = await platformResp.json();
            const platformStatus = await platformStatusResp.json();
            const model = status.model || {};

            // Load saved preferences
            const fontSize = localStorage.getItem('ern-os-font-size') || '14';
            const msgWidth = localStorage.getItem('ern-os-msg-width') || '760';
            const theme = localStorage.getItem('ern-os-theme') || 'dark';

            const llmModels = (modelsData.models || []).filter(m => !m.is_mmproj);
            const mmprojModels = (modelsData.models || []).filter(m => m.is_mmproj);

            const toggle = (label, desc, checked, onChange) => `
                <div class="setting-row">
                    <div class="setting-label">
                        <span class="label-text">${label}</span>
                        <span class="label-desc">${desc}</span>
                    </div>
                    <label class="toggle-switch">
                        <input type="checkbox" ${checked ? 'checked' : ''} onchange="${onChange}">
                        <span class="toggle-slider"></span>
                    </label>
                </div>`;

            container.innerHTML = `
                <!-- Appearance -->
                <div class="settings-card">
                    <div class="card-title"><span class="title-icon">🎨</span> Appearance</div>
                    <div class="setting-row">
                        <div class="setting-label">
                            <span class="label-text">Theme</span>
                            <span class="label-desc">Switch between dark and light mode</span>
                        </div>
                        <select class="setting-select" onchange="ErnOS.setTheme(this.value)">
                            <option value="dark" ${theme === 'dark' ? 'selected' : ''}>🌙 Dark</option>
                            <option value="light" ${theme === 'light' ? 'selected' : ''}>☀️ Light</option>
                        </select>
                    </div>
                    <div class="setting-row">
                        <div class="setting-label">
                            <span class="label-text">Font Size</span>
                            <span class="label-desc">Message text size in pixels</span>
                        </div>
                        <select class="setting-select" onchange="ErnOS.setFontSize(this.value)">
                            ${[12,13,14,15,16,18].map(s => `<option value="${s}" ${fontSize == s ? 'selected' : ''}>${s}px</option>`).join('')}
                        </select>
                    </div>
                    <div class="setting-row">
                        <div class="setting-label">
                            <span class="label-text">Message Width</span>
                            <span class="label-desc">Maximum width of message bubbles</span>
                        </div>
                        <select class="setting-select" onchange="ErnOS.setMsgWidth(this.value)">
                            ${[600,700,760,900,1000,1200].map(w => `<option value="${w}" ${msgWidth == w ? 'selected' : ''}>${w}px</option>`).join('')}
                        </select>
                    </div>
                </div>

                <!-- Model -->
                <div class="settings-card">
                    <div class="card-title"><span class="title-icon">🤖</span> Model</div>
                    <div class="setting-row">
                        <div class="setting-label">
                            <span class="label-text">Active Model</span>
                            <span class="label-desc">Loaded GGUF model (requires restart)</span>
                        </div>
                        <select class="setting-select" id="model-select" onchange="ErnOS.swapModel(this.value)" title="Select model — will restart server">
                            ${llmModels.map(m => `<option value="${escapeHtml(m.name)}" ${model.name && model.name.includes(m.name.replace('.gguf','')) ? 'selected' : ''}>${escapeHtml(m.name)} (${m.size_gb} GB)</option>`).join('')}
                        </select>
                    </div>
                    <div class="setting-row">
                        <div class="setting-label">
                            <span class="label-text">Vision Projector</span>
                            <span class="label-desc">Multimodal projector for image/video</span>
                        </div>
                        <select class="setting-select" disabled title="Requires restart">
                            <option value="">None</option>
                            ${mmprojModels.map(m => `<option value="${escapeHtml(m.name)}" selected>${escapeHtml(m.name)} (${m.size_gb} GB)</option>`).join('')}
                        </select>
                    </div>
                    <div class="setting-row">
                        <div class="setting-label">
                            <span class="label-text">Context Length</span>
                            <span class="label-desc">Maximum tokens in context window</span>
                        </div>
                        <span style="font-family:'JetBrains Mono',monospace;font-size:12px;color:var(--accent)">${Number(model.context_length || 0).toLocaleString()}</span>
                    </div>
                </div>

                <!-- Capabilities -->
                <div class="settings-card">
                    <div class="card-title"><span class="title-icon">⚡</span> Capabilities</div>
                    ${toggle('Vision', 'Process images and screenshots', model.supports_vision, '')}
                    ${toggle('Video', 'Analyze video content', model.supports_video, '')}
                    ${toggle('Audio', 'Process audio input', model.supports_audio, '')}
                    ${toggle('Tool Calling', 'Execute tools and function calls', model.supports_tool_calling, '')}
                    ${toggle('Thinking', 'Show internal reasoning process', model.supports_thinking, '')}
                </div>

                <!-- Observer -->
                <div class="settings-card">
                    <div class="card-title"><span class="title-icon">👁️</span> Observer</div>
                    ${toggle('Observer Enabled', 'Audit all responses for accuracy and governance compliance', status.observer?.enabled, '')}
                </div>

                <!-- Web Search API Keys -->
                <div class="settings-card">
                    <div class="card-title"><span class="title-icon">🌐</span> Web Search API Keys</div>
                    <div class="setting-row" style="flex-direction:column;align-items:stretch;gap:4px">
                        <span class="label-desc" style="margin-bottom:4px">8-tier waterfall: keys are tried in order. Free tiers (DDG, Google, Wikipedia, News RSS) always work without keys.</span>
                    </div>
                    ${[
                        ['BRAVE_API_KEY', 'Brave Search', 'Tier 1 — brave.com/search/api', 'https://brave.com/search/api/'],
                        ['SERPER_API_KEY', 'Serper.dev', 'Tier 2 — Google SERP results', 'https://serper.dev/'],
                        ['TAVILY_API_KEY', 'Tavily AI', 'Tier 3 — AI-powered search', 'https://tavily.com/'],
                        ['SERPAPI_API_KEY', 'SerpAPI', 'Tier 4 — Multi-engine search', 'https://serpapi.com/'],
                    ].map(([key, label, desc, url]) => {
                        const info = apiKeys[key] || {};
                        return '<div class="setting-row">'
                            + '<div class="setting-label">'
                            + '<span class="label-text">' + label + (info.set ? ' <span style="color:var(--accent);font-size:10px">✓ SET</span>' : '') + '</span>'
                            + '<span class="label-desc">' + desc + ' — <a href="' + url + '" target="_blank" style="color:var(--accent)">Get key</a></span>'
                            + '</div>'
                            + '<div style="display:flex;gap:6px;align-items:center">'
                            + '<input type="password" id="apikey-' + key + '" placeholder="' + (info.set ? info.masked : 'Paste API key…') + '" '
                            + 'style="background:var(--surface-2);border:1px solid var(--border);border-radius:6px;padding:4px 8px;color:var(--text-primary);font-family:JetBrains Mono,monospace;font-size:11px;width:200px" />'
                            + '<button onclick="ErnOS.saveApiKey(\'' + key + '\')" '
                            + 'style="background:var(--accent);border:none;color:#000;padding:4px 10px;border-radius:6px;font-size:11px;cursor:pointer;font-weight:700;white-space:nowrap">Save</button>'
                            + (info.set ? '<button onclick="ErnOS.clearApiKey(\'' + key + '\')" '
                            + 'style="background:transparent;border:1px solid var(--danger,#f44);color:var(--danger,#f44);padding:4px 8px;border-radius:6px;font-size:10px;cursor:pointer">Clear</button>' : '')
                            + '</div></div>';
                    }).join('')}
                </div>

                <!-- Platform Adapters -->
                <div class="settings-card">
                    <div class="card-title"><span class="title-icon">📡</span> Platform Adapters</div>
                    <span class="label-desc" style="display:block;margin-bottom:8px">Connect Discord and Telegram bots to this Ern-OS instance. Tokens are masked after saving.</span>

                    <!-- Discord -->
                    <div style="border:1px solid var(--border);border-radius:8px;padding:10px;margin-bottom:10px">
                        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">
                            <span style="font-weight:700;font-size:13px">🎮 Discord</span>
                            <div style="display:flex;gap:6px;align-items:center">
                                <label class="toggle-switch">
                                    <input type="checkbox" id="discord-enabled" ${platformConfig.discord?.enabled ? 'checked' : ''} onchange="ErnOS.savePlatformConfig()">
                                    <span class="toggle-slider"></span>
                                </label>
                                ${(() => {
                                    const ds = (platformStatus.platforms || []).find(p => p.name === 'Discord');
                                    if (ds && ds.connected) return '<span style="color:var(--accent);font-size:10px;font-weight:700">🟢 CONNECTED</span>';
                                    if (platformConfig.discord?.has_token) return '<button onclick="ErnOS.connectPlatform(\'Discord\')" style="background:var(--accent);border:none;color:#000;padding:3px 8px;border-radius:5px;font-size:10px;cursor:pointer;font-weight:700">Connect</button>';
                                    return '<span style="color:var(--text-secondary);font-size:10px">Not configured</span>';
                                })()}
                            </div>
                        </div>
                        <div class="setting-row">
                            <div class="setting-label"><span class="label-text">Bot Token</span><span class="label-desc">Or set DISCORD_TOKEN env var${platformConfig.discord?.has_token ? ' — <span style="color:var(--accent)">✓ SET</span>' : ''}</span></div>
                            <div style="display:flex;gap:6px;align-items:center">
                                <input type="password" id="discord-token" placeholder="${platformConfig.discord?.token || 'Paste bot token…'}" style="background:var(--surface-2);border:1px solid var(--border);border-radius:6px;padding:4px 8px;color:var(--text-primary);font-family:JetBrains Mono,monospace;font-size:11px;width:200px" />
                                <button onclick="ErnOS.savePlatformConfig()" style="background:var(--accent);border:none;color:#000;padding:4px 10px;border-radius:6px;font-size:11px;cursor:pointer;font-weight:700">Save</button>
                            </div>
                        </div>
                        <div class="setting-row">
                            <div class="setting-label"><span class="label-text">Admin User IDs</span><span class="label-desc">Comma-separated Discord user IDs with full tool access</span></div>
                            <input type="text" id="discord-admin-ids" value="${(platformConfig.discord?.admin_ids || []).join(', ')}" placeholder="123456,789012" onchange="ErnOS.savePlatformConfig()" style="background:var(--surface-2);border:1px solid var(--border);border-radius:6px;padding:4px 8px;color:var(--text-primary);font-family:JetBrains Mono,monospace;font-size:11px;width:200px" />
                        </div>
                        <div class="setting-row">
                            <div class="setting-label"><span class="label-text">Listen Channels</span><span class="label-desc">Comma-separated channel IDs (empty = all)</span></div>
                            <input type="text" id="discord-channels" value="${(platformConfig.discord?.listen_channels || []).join(', ')}" placeholder="All channels" onchange="ErnOS.savePlatformConfig()" style="background:var(--surface-2);border:1px solid var(--border);border-radius:6px;padding:4px 8px;color:var(--text-primary);font-family:JetBrains Mono,monospace;font-size:11px;width:200px" />
                        </div>
                    </div>

                    <!-- Telegram -->
                    <div style="border:1px solid var(--border);border-radius:8px;padding:10px">
                        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">
                            <span style="font-weight:700;font-size:13px">✈️ Telegram</span>
                            <div style="display:flex;gap:6px;align-items:center">
                                <label class="toggle-switch">
                                    <input type="checkbox" id="telegram-enabled" ${platformConfig.telegram?.enabled ? 'checked' : ''} onchange="ErnOS.savePlatformConfig()">
                                    <span class="toggle-slider"></span>
                                </label>
                                ${(() => {
                                    const ts = (platformStatus.platforms || []).find(p => p.name === 'Telegram');
                                    if (ts && ts.connected) return '<span style="color:var(--accent);font-size:10px;font-weight:700">🟢 CONNECTED</span>';
                                    if (platformConfig.telegram?.has_token) return '<button onclick="ErnOS.connectPlatform(\'Telegram\')" style="background:var(--accent);border:none;color:#000;padding:3px 8px;border-radius:5px;font-size:10px;cursor:pointer;font-weight:700">Connect</button>';
                                    return '<span style="color:var(--text-secondary);font-size:10px">Not configured</span>';
                                })()}
                            </div>
                        </div>
                        <div class="setting-row">
                            <div class="setting-label"><span class="label-text">Bot Token</span><span class="label-desc">Or set TELEGRAM_TOKEN env var${platformConfig.telegram?.has_token ? ' — <span style="color:var(--accent)">✓ SET</span>' : ''}</span></div>
                            <div style="display:flex;gap:6px;align-items:center">
                                <input type="password" id="telegram-token" placeholder="${platformConfig.telegram?.token || 'Paste bot token…'}" style="background:var(--surface-2);border:1px solid var(--border);border-radius:6px;padding:4px 8px;color:var(--text-primary);font-family:JetBrains Mono,monospace;font-size:11px;width:200px" />
                                <button onclick="ErnOS.savePlatformConfig()" style="background:var(--accent);border:none;color:#000;padding:4px 10px;border-radius:6px;font-size:11px;cursor:pointer;font-weight:700">Save</button>
                            </div>
                        </div>
                        <div class="setting-row">
                            <div class="setting-label"><span class="label-text">Admin User IDs</span><span class="label-desc">Comma-separated Telegram user IDs with full tool access</span></div>
                            <input type="text" id="telegram-admin-ids" value="${(platformConfig.telegram?.admin_ids || []).join(', ')}" placeholder="123456,789012" onchange="ErnOS.savePlatformConfig()" style="background:var(--surface-2);border:1px solid var(--border);border-radius:6px;padding:4px 8px;color:var(--text-primary);font-family:JetBrains Mono,monospace;font-size:11px;width:200px" />
                        </div>
                        <div class="setting-row">
                            <div class="setting-label"><span class="label-text">Allowed Chats</span><span class="label-desc">Comma-separated chat IDs (empty = all)</span></div>
                            <input type="text" id="telegram-chats" value="${(platformConfig.telegram?.allowed_chats || []).join(', ')}" placeholder="All chats" onchange="ErnOS.savePlatformConfig()" style="background:var(--surface-2);border:1px solid var(--border);border-radius:6px;padding:4px 8px;color:var(--text-primary);font-family:JetBrains Mono,monospace;font-size:11px;width:200px" />
                        </div>
                    </div>
                </div>

                <!-- Engine -->
                <div class="settings-card">
                    <div class="card-title"><span class="title-icon">🔧</span> Engine</div>
                    <div class="setting-row">
                        <div class="setting-label">
                            <span class="label-text">Provider</span>
                            <span class="label-desc">Active inference provider</span>
                        </div>
                        <span style="font-family:'JetBrains Mono',monospace;font-size:12px;color:var(--text-primary)">${escapeHtml(status.provider || '?')}</span>
                    </div>
                    <div class="setting-row">
                        <div class="setting-label">
                            <span class="label-text">Sessions</span>
                            <span class="label-desc">Total active chat sessions</span>
                        </div>
                        <span style="font-family:'JetBrains Mono',monospace;font-size:12px;color:var(--text-primary)">${status.sessions || 0}</span>
                    </div>
                    <div class="setting-row">
                        <div class="setting-label">
                            <span class="label-text">Memory</span>
                            <span class="label-desc">7-tier cognitive memory status</span>
                        </div>
                        <span style="font-family:'JetBrains Mono',monospace;font-size:11px;color:var(--text-secondary);max-width:300px;text-align:right">${escapeHtml(typeof status.memory === 'string' ? status.memory : JSON.stringify(status.memory))}</span>
                    </div>
                </div>

                <!-- System Version -->
                <div class="settings-card">
                    <div class="card-title"><span class="title-icon">🔄</span> System Version</div>
                    <div id="version-info" class="setting-row">
                        <span class="label-desc">Loading version info...</span>
                    </div>
                    <div class="setting-row" style="gap:10px">
                        <button class="save-prompt-btn" onclick="ErnOS.checkForUpdates()" id="check-updates-btn">Check for Updates</button>
                        <button class="save-prompt-btn" onclick="ErnOS.updateNow()" id="update-now-btn" style="display:none;background:var(--accent)">Update Now</button>
                    </div>
                    <div id="update-status" style="display:none;padding:8px 12px;border-radius:8px;margin-top:8px;font-size:13px"></div>
                    <div style="margin-top:16px">
                        <button class="save-prompt-btn" onclick="ErnOS.toggleVersionHistory()" style="background:transparent;border:1px solid var(--border);color:var(--text-secondary)">Version History ▾</button>
                        <div id="version-history" style="display:none;margin-top:10px;max-height:400px;overflow-y:auto"></div>
                    </div>
                </div>

                <!-- Danger Zone -->
                <div class="settings-card danger-zone">
                    <div class="card-title"><span class="title-icon">⚠️</span> Danger Zone</div>
                    <div class="setting-row">
                        <div class="setting-label">
                            <span class="label-text">Factory Reset</span>
                            <span class="label-desc">Clear ALL data — memory, sessions, training buffers. This cannot be undone.</span>
                        </div>
                        <button class="danger-btn" onclick="ErnOS.factoryReset()">Reset All Data</button>
                    </div>
                </div>
            `;

            // Attach delete handlers imperatively (avoids quote-hell in templates)
            container.querySelectorAll('.sched-del-btn').forEach(btn => {
                btn.addEventListener('click', () => {
                    deleteSchedulerJob(btn.dataset.jobId);
                });
            });
        } catch (e) {
            container.innerHTML = '<p class="empty-state">Failed to load settings</p>';
        }
    }

    // ─── Interpretability View ───
    let interpCategoryFilter = 'all';

    async function loadInterpretabilityView() {
        try {
            const [featResp, snapResp, saeResp] = await Promise.all([
                fetch('/api/interpretability/features'),
                fetch('/api/interpretability/snapshots'),
                fetch('/api/interpretability/sae'),
            ]);
            const featData = await featResp.json();
            const snapData = await snapResp.json();
            const saeData = await saeResp.json();

            // SAE Status
            const saeEl = document.getElementById('interp-sae-status');
            if (!saeData.model_loaded && !saeData.feature_count) {
                saeEl.innerHTML = `<div class="sae-status-card" style="background:rgba(255,170,0,0.05);border:1px solid rgba(255,170,0,0.15);padding:20px;border-radius:12px;text-align:center">
                    <div style="font-size:28px;margin-bottom:8px">🧠</div>
                    <div style="font-size:14px;font-weight:600;color:var(--warning);margin-bottom:4px">SAE Weights Not Loaded</div>
                    <div style="font-size:12px;color:var(--text-secondary);line-height:1.5">
                        Place <code style="background:var(--surface-2);padding:1px 4px;border-radius:3px">*.safetensors</code> weights in <code style="background:var(--surface-2);padding:1px 4px;border-radius:3px">models/sae/</code> and restart, or train via the learning pipeline.
                    </div>
                </div>`;
            } else {
                saeEl.innerHTML = `<div class="sae-status-card">
                    <div class="sae-stat"><div class="stat-value">${saeData.feature_count}</div><div class="stat-label">Features</div></div>
                    <div class="sae-stat"><div class="stat-value">${saeData.input_dim?.toLocaleString()}</div><div class="stat-label">Input Dim</div></div>
                    <div class="sae-stat"><div class="stat-value">${saeData.hidden_dim}</div><div class="stat-label">Hidden Dim</div></div>
                    <div class="sae-stat"><div class="stat-value">${Number(saeData.sparsity_coefficient || 0).toFixed(3)}</div><div class="stat-label">λ Sparsity</div></div>
                    <div class="sae-stat"><div class="stat-value">${saeData.model_loaded ? '✅' : '—'}</div><div class="stat-label">Weights</div></div>
                    <div class="sae-stat"><div class="stat-value">${snapData.count}</div><div class="stat-label">Snapshots</div></div>
                </div>`;
            }

            // Category filters
            const categories = [...new Set(featData.features.map(f => f.category))];
            const featureContainer = document.getElementById('interp-features-container');

            const renderFeatures = (filter) => {
                const filtered = filter === 'all' ? featData.features : featData.features.filter(f => f.category === filter);
                return `
                    <div class="feature-filters">
                        <span class="category-badge ${filter === 'all' ? 'active' : ''}" onclick="ErnOS.filterFeatures('all')">All (${featData.features.length})</span>
                        ${categories.map(c => {
                            const count = featData.features.filter(f => f.category === c).length;
                            return `<span class="category-badge ${filter === c ? 'active' : ''}" onclick="ErnOS.filterFeatures('${c}')">${c} (${count})</span>`;
                        }).join('')}
                    </div>
                    <div class="features-grid">
                        ${filtered.map(f => `<div class="feature-chip">
                            <span class="feature-index">#${f.index}</span>
                            <span class="feature-label">${escapeHtml(f.label)}</span>
                            <div class="feature-bar"><div class="feature-bar-fill" style="width:${Math.min(f.baseline_activation * 100, 100)}%"></div></div>
                        </div>`).join('')}
                    </div>`;
            };

            featureContainer.innerHTML = renderFeatures(interpCategoryFilter);
            window._interpRenderFeatures = renderFeatures;
            window._interpContainer = featureContainer;

            // Snapshots
            const snapContainer = document.getElementById('interp-snapshots-container');
            if (snapData.snapshots.length === 0) {
                snapContainer.innerHTML = '<p class="empty-state" style="margin-top:20px">No neural snapshots captured yet</p>';
            } else {
                const snapsHtml = snapData.snapshots.slice(0, 20).map(s => {
                    const data = s.data || {};
                    const div = data.divergence_from_baseline != null ? data.divergence_from_baseline : 0;
                    const divClass = div < 0.3 ? 'low' : div < 0.7 ? 'medium' : 'high';
                    const features = (data.top_features || []).slice(0, 5).map(f => `#${f[0]}`).join(', ');
                    return `<div class="snapshot-card">
                        <div class="snapshot-header">
                            <span class="snapshot-time">${s.file}</span>
                            <span class="snapshot-divergence ${divClass}">${div.toFixed(3)}</span>
                        </div>
                        <div style="font-size:12px;color:var(--text-secondary)">${escapeHtml(data.context_summary || '')}</div>
                        ${features ? `<div style="font-size:11px;color:var(--text-muted);margin-top:4px">Top features: ${features}</div>` : ''}
                    </div>`;
                }).join('');
                snapContainer.innerHTML = `<div class="settings-card" style="padding:0">
                    <div style="padding:16px 20px;border-bottom:1px solid var(--border-subtle)">
                        <div class="card-title" style="margin:0"><span class="title-icon">📸</span> Neural Snapshots</div>
                    </div>
                    <div class="snapshot-timeline" style="padding:12px 20px">${snapsHtml}</div>
                </div>`;
            }
        } catch (e) {
            document.getElementById('interp-features-container').innerHTML = '<p class="empty-state">Failed to load interpretability data</p>';
        }
    }

    function filterFeatures(category) {
        interpCategoryFilter = category;
        if (window._interpRenderFeatures && window._interpContainer) {
            window._interpContainer.innerHTML = window._interpRenderFeatures(category);
        }
    }

    // ─── Steering View ───
    async function loadSteeringView() {
        const container = document.getElementById('steering-container');
        try {
            const resp = await fetch('/api/steering/vectors');
            const data = await resp.json();

            if (data.vectors.length === 0) {
                container.innerHTML = `<div class="steering-empty">
                    <div class="empty-icon">🎯</div>
                    <p style="font-size:14px;margin-bottom:8px">No steering vectors found</p>
                    <p style="font-size:12px">Place <code>.gguf</code> control vectors in <code>data/steering/</code></p>
                </div>`;
                return;
            }

            container.innerHTML = `<div class="steering-grid">${data.vectors.map(v => `
                <div class="steering-card ${v.active ? 'active-vector' : ''}">
                    <div class="tool-card-header">
                        <div class="steering-name">${escapeHtml(v.name)}</div>
                        <label class="toggle-switch">
                            <input type="checkbox" ${v.active ? 'checked' : ''} disabled title="Toggle requires restart">
                            <span class="toggle-slider"></span>
                        </label>
                    </div>
                    <div class="steering-desc">${escapeHtml(v.description)}</div>
                    <div class="steering-controls">
                        <input type="range" class="strength-slider" min="0" max="2" step="0.05"
                            value="${v.strength}" disabled title="Strength requires restart">
                        <span class="strength-value">${v.strength.toFixed(2)}</span>
                    </div>
                </div>
            `).join('')}</div>`;
        } catch (e) {
            container.innerHTML = '<p class="empty-state">Failed to load steering vectors</p>';
        }
    }

    // ─── Logs View ───
    let logLevelFilter = new Set(['INFO', 'WARN', 'ERROR', 'DEBUG']);

    async function loadLogsView() {
        const filtersEl = document.getElementById('log-filters');
        const container = document.getElementById('logs-container');

        // Render filter buttons
        filtersEl.innerHTML = ['INFO', 'DEBUG', 'WARN', 'ERROR'].map(level =>
            `<button class="log-filter-btn ${logLevelFilter.has(level) ? 'active' : ''}"
                data-level="${level}" onclick="ErnOS.toggleLogFilter('${level}')">${level}</button>`
        ).join('') + `<span style="font-size:11px;color:var(--text-muted);margin-left:auto">Auto-refresh: 10s</span>`;

        try {
            const resp = await fetch('/api/logs');
            const data = await resp.json();

            if (data.entries.length === 0) {
                // If no structured JSON logs, try reading plain text
                container.innerHTML = `<div class="log-entries">
                    <div class="log-entry level-INFO">
                        <span class="log-level INFO">INFO</span>
                        <span class="log-message">No structured log entries found. Logs may be in plain text format in data/logs/</span>
                    </div>
                </div>`;
                return;
            }

            const filteredEntries = data.entries.filter(e => {
                const level = (e.level || 'INFO').toUpperCase();
                return logLevelFilter.has(level);
            });

            container.innerHTML = `<div class="log-entries">${filteredEntries.map(e => {
                const level = (e.level || 'INFO').toUpperCase();
                const time = e.timestamp ? e.timestamp.substring(11, 19) : '';
                const target = e.target || e.module || '';
                const msg = e.fields?.message || e.message || JSON.stringify(e);
                return `<div class="log-entry level-${level}">
                    <span class="log-time">${time}</span>
                    <span class="log-level ${level}">${level}</span>
                    <span class="log-target">${escapeHtml(target)}</span>
                    <span class="log-message">${escapeHtml(typeof msg === 'string' ? msg : JSON.stringify(msg))}</span>
                </div>`;
            }).join('')}</div>`;
        } catch (e) {
            container.innerHTML = '<p class="empty-state">Failed to load logs</p>';
        }
        // Also load self-edit audit, checkpoints, and observer history
        loadSelfEditLog();
        loadCheckpoints();
        loadObserverHistory();
    }

    function toggleLogFilter(level) {
        if (logLevelFilter.has(level)) {
            logLevelFilter.delete(level);
        } else {
            logLevelFilter.add(level);
        }
        loadLogsView();
    }

    async function loadSelfEditLog() {
        const container = document.getElementById('self-edit-container');
        if (!container) return;
        try {
            const resp = await fetch('/api/self-edits');
            const data = await resp.json();
            if (data.count === 0) {
                container.innerHTML = '<p class="empty-state">No self-edits recorded yet</p>';
                return;
            }
            container.innerHTML = `<div class="log-entries">${data.entries.map(e => {
                const time = e.timestamp ? e.timestamp.substring(0, 19) : '';
                const action = e.action || 'unknown';
                const path = e.path || '';
                const detail = e.detail || '';
                return `<div class="log-entry level-WARN">
                    <span class="log-time">${escapeHtml(time)}</span>
                    <span class="log-level WARN">${escapeHtml(action.toUpperCase())}</span>
                    <span class="log-target">${escapeHtml(path)}</span>
                    <span class="log-message">${escapeHtml(detail.substring(0, 200))}</span>
                </div>`;
            }).join('')}</div>`;
        } catch {
            container.innerHTML = '<p class="empty-state">Failed to load self-edit log</p>';
        }
    }

    async function loadCheckpoints() {
        const container = document.getElementById('checkpoints-container');
        if (!container) return;
        try {
            const resp = await fetch('/api/checkpoints');
            const data = await resp.json();
            if (data.count === 0) {
                container.innerHTML = '<p class="empty-state">No checkpoints found</p>';
                return;
            }
            container.innerHTML = `<div class="log-entries">${data.checkpoints.map(c => {
                const size = c.size_bytes < 1024 ? `${c.size_bytes}B` : `${(c.size_bytes/1024).toFixed(1)}KB`;
                return `<div class="log-entry level-INFO">
                    <span class="log-time">${escapeHtml(c.modified ? c.modified.substring(0, 19) : '')}</span>
                    <span class="log-level INFO">${size}</span>
                    <span class="log-message">${escapeHtml(c.name)}</span>
                </div>`;
            }).join('')}</div>`;
        } catch {
            container.innerHTML = '<p class="empty-state">Failed to load checkpoints</p>';
        }
    }

    async function loadObserverHistory() {
        const container = document.getElementById('observer-history-container');
        if (!container) return;
        try {
            const resp = await fetch('/api/observer/history');
            const data = await resp.json();
            if (!data.entries || data.entries.length === 0) {
                container.innerHTML = '<p class="empty-state">No observer audits recorded yet</p>';
                return;
            }
            container.innerHTML = `<div class="log-entries">${data.entries.slice(0, 50).map(e => {
                const t = e.timestamp ? new Date(e.timestamp).toLocaleString() : '';
                const verdictClass = e.verdict === 'allowed' || e.approved ? 'INFO' : 'WARN';
                const verdictLabel = e.verdict || (e.approved ? 'PASS' : 'FAIL');
                const conf = e.confidence !== undefined ? (e.confidence * 100).toFixed(0) + '%' : '';
                return `<div class="log-entry level-${verdictClass}">
                    <span class="log-time">${escapeHtml(t)}</span>
                    <span class="log-level ${verdictClass}">${verdictLabel.toUpperCase()}</span>
                    <span class="log-message">${conf ? '[' + conf + '] ' : ''}${escapeHtml(e.category || e.failure_category || '')} ${escapeHtml(e.reason || e.what_went_wrong || '')}</span>
                </div>`;
            }).join('')}</div>`;
        } catch {
            container.innerHTML = '<p class="empty-state">Failed to load observer history</p>';
        }
    }

    // ─── Scheduler Actions ───
    async function toggleSchedulerJob(id) {
        await fetch(`/api/scheduler/jobs/${id}/toggle`, { method: 'PUT' });
        loadSchedulerView();
    }

    async function deleteSchedulerJob(id) {
        if (!confirm('Delete this scheduled job?')) return;
        await fetch(`/api/scheduler/jobs/${id}`, { method: 'DELETE' });
        loadSchedulerView();
    }

    function showAddJobForm() {
        const form = document.getElementById('add-job-form');
        form.style.display = form.style.display === 'none' ? 'block' : 'none';

        // Show/hide custom command field based on task type
        const taskSelect = document.getElementById('job-task-type');
        if (taskSelect) {
            taskSelect.onchange = () => {
                const row = document.getElementById('custom-cmd-row');
                if (row) row.style.display = taskSelect.value === 'custom' ? 'block' : 'none';
            };
        }
    }

    async function createSchedulerJob() {
        const name = document.getElementById('job-name')?.value || 'unnamed';
        const desc = document.getElementById('job-desc')?.value || '';
        const schedType = document.getElementById('job-schedule-type')?.value || 'interval';
        const schedValue = document.getElementById('job-schedule-value')?.value || '300';
        const taskType = document.getElementById('job-task-type')?.value || 'prompt';
        const customCmd = document.getElementById('job-custom-cmd')?.value || 'echo ok';
        const prompt = document.getElementById('job-prompt')?.value || '';

        const body = {
            name, description: desc,
            schedule_type: schedType,
            schedule_value: schedType === 'interval' ? parseInt(schedValue) || 300 : schedValue,
            task_type: taskType,
            custom_command: customCmd,
            prompt: prompt,
        };

        const resp = await fetch('/api/scheduler/jobs', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });

        if (resp.ok) {
            loadSchedulerView();
        } else {
            const err = await resp.json();
            alert('Failed: ' + (err.error || 'Unknown error'));
        }
    }

    async function swapModel(modelName) {
        if (!confirm(`Switching to ${modelName} requires restarting the inference server. Continue?`)) {
            loadSettingsView(); // Reset dropdown to current
            return;
        }

        try {
            const resp = await fetch('/api/settings/model', {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ model: modelName }),
            });
            const data = await resp.json();
            if (data.success) {
                addSystemMessage(`🔄 Switching to **${modelName}** — server restarting...`);
                // Wait for restart then reload
                setTimeout(() => location.reload(), 3000);
            } else {
                addSystemMessage(`❌ Failed to swap model: ${data.error || 'Unknown error'}`);
                loadSettingsView();
            }
        } catch (e) {
            addSystemMessage(`❌ Model swap failed: ${e.message}`);
            loadSettingsView();
        }
    }

    // ─── Settings Actions ───
    function setTheme(theme) {
        document.documentElement.setAttribute('data-theme', theme === 'dark' ? '' : 'light');
        localStorage.setItem('ern-os-theme', theme);
        const btn = document.getElementById('theme-toggle');
        if (btn) btn.textContent = theme === 'light' ? '🌙' : '☀️';
    }

    function setFontSize(size) {
        localStorage.setItem('ern-os-font-size', size);
        document.documentElement.style.setProperty('--msg-font-size', size + 'px');
        document.querySelectorAll('.message').forEach(m => m.style.fontSize = size + 'px');
    }

    function setMsgWidth(width) {
        localStorage.setItem('ern-os-msg-width', width);
        document.querySelectorAll('.message').forEach(m => m.style.maxWidth = width + 'px');
    }

    async function saveApiKey(keyName) {
        const input = document.getElementById('apikey-' + keyName);
        if (!input) return;
        const value = input.value.trim();
        if (!value) { alert('Please enter a key value'); return; }
        try {
            const body = {};
            body[keyName] = value;
            const resp = await fetch('/api/api-keys', {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body),
            });
            const data = await resp.json();
            if (data.ok) {
                input.value = '';
                loadSettingsView(); // Refresh to show masked key
            } else {
                alert('Failed to save: ' + (data.error || 'Unknown error'));
            }
        } catch (e) {
            alert('Failed to save API key: ' + e.message);
        }
    }

    async function clearApiKey(keyName) {
        if (!confirm('Clear ' + keyName + '? This tier will be skipped during web searches.')) return;
        try {
            const body = {};
            body[keyName] = '';
            const resp = await fetch('/api/api-keys', {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body),
            });
            const data = await resp.json();
            if (data.ok) {
                loadSettingsView();
            }
        } catch (e) {
            alert('Failed to clear API key: ' + e.message);
        }
    }

    async function savePlatformConfig() {
        const discordToken = document.getElementById('discord-token')?.value?.trim();
        const telegramToken = document.getElementById('telegram-token')?.value?.trim();
        const payload = {
            discord: {
                enabled: document.getElementById('discord-enabled')?.checked || false,
                admin_ids: (document.getElementById('discord-admin-ids')?.value || '').split(',').map(s => s.trim()).filter(Boolean),
                listen_channels: (document.getElementById('discord-channels')?.value || '').split(',').map(s => s.trim()).filter(Boolean),
            },
            telegram: {
                enabled: document.getElementById('telegram-enabled')?.checked || false,
                admin_ids: (document.getElementById('telegram-admin-ids')?.value || '').split(',').map(s => s.trim()).filter(Boolean).map(Number),
                allowed_chats: (document.getElementById('telegram-chats')?.value || '').split(',').map(s => s.trim()).filter(Boolean).map(Number),
            },
        };
        if (discordToken) payload.discord.token = discordToken;
        if (telegramToken) payload.telegram.token = telegramToken;
        try {
            const resp = await fetch('/api/platforms/config', {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
            });
            const data = await resp.json();
            if (data.success) {
                showToast('Platform config saved');
                loadSettingsView();
            } else {
                showToast('Failed: ' + (data.error || 'Unknown'), 'error');
            }
        } catch (e) {
            showToast('Failed to save platform config: ' + e.message, 'error');
        }
    }

    async function connectPlatform(name) {
        try {
            const resp = await fetch('/api/platforms/' + name + '/connect', { method: 'POST' });
            const data = await resp.json();
            if (data.success) {
                showToast(name + ' connected');
                loadSettingsView();
            } else {
                showToast('Failed: ' + (data.error || 'Unknown'), 'error');
            }
        } catch (e) {
            showToast('Failed to connect ' + name + ': ' + e.message, 'error');
        }
    }

    async function disconnectPlatform(name) {
        try {
            const resp = await fetch('/api/platforms/' + name + '/disconnect', { method: 'POST' });
            const data = await resp.json();
            if (data.success) {
                showToast(name + ' disconnected');
                loadSettingsView();
            } else {
                showToast('Failed: ' + (data.error || 'Unknown'), 'error');
            }
        } catch (e) {
            showToast('Failed to disconnect ' + name + ': ' + e.message, 'error');
        }
    }

    async function factoryReset() {
        if (!confirm('⚠️ FACTORY RESET\n\nThis will permanently delete:\n• All chat sessions\n• All memory (timeline, lessons, procedures, etc.)\n• All training data (golden + rejection buffers)\n\nThis cannot be undone. Continue?')) return;
        if (!confirm('Are you absolutely sure? Type OK to confirm.')) return;

        try {
            const resp = await fetch('/api/factory-reset', { method: 'POST' });
            const data = await resp.json();
            if (data.ok) {
                alert('✅ Factory reset complete. All data cleared. The page will now reload.');
                window.location.reload();
            } else {
                alert('❌ Reset failed: ' + (data.message || 'unknown error'));
            }
        } catch (e) {
            alert('❌ Reset failed: ' + e.message);
        }
    }

    // ─── Multimodal ───
    function handleFilesSelected(fileList) {
        Array.from(fileList).forEach(file => {
            if (file.type.startsWith('image/') || file.type.startsWith('video/')) {
                // Images/video: base64 data URL for inline preview + vision
                const reader = new FileReader();
                reader.onload = (e) => {
                    attachedFiles.push({ name: file.name, dataUrl: e.target.result, type: file.type, isMedia: true });
                    renderFilePreview();
                };
                reader.readAsDataURL(file);
            } else {
                // Documents/other: upload to server, get path back
                const formData = new FormData();
                formData.append('file', file);
                fetch('/api/upload', { method: 'POST', body: formData })
                    .then(r => r.json())
                    .then(data => {
                        if (data.files && data.files.length > 0) {
                            const uploaded = data.files[0];
                            attachedFiles.push({
                                name: uploaded.original_name,
                                path: uploaded.path,
                                type: file.type || 'application/octet-stream',
                                size: uploaded.size,
                                isMedia: false,
                            });
                            renderFilePreview();
                            showToast('File uploaded: ' + uploaded.original_name, 'success');
                        }
                    })
                    .catch(e => showToast('Upload failed: ' + e.message, 'error'));
            }
        });
    }

    function renderFilePreview() {
        const preview = document.getElementById('file-preview');
        preview.innerHTML = '';
        preview.className = attachedFiles.length > 0 ? 'file-preview has-files' : 'file-preview';
        attachedFiles.forEach((f, i) => {
            const item = document.createElement('div');
            item.className = 'file-preview-item';
            if (f.isMedia && f.dataUrl) {
                item.innerHTML = `<img src="${f.dataUrl}" alt="${escapeHtml(f.name)}"><button class="remove-file" onclick="ErnOS.removeFile(${i})">×</button>`;
            } else {
                const ext = f.name.split('.').pop() || '?';
                const sizeStr = f.size ? (f.size > 1024*1024 ? (f.size/1024/1024).toFixed(1)+'MB' : (f.size/1024).toFixed(0)+'KB') : '';
                item.innerHTML = `<div style="display:flex;align-items:center;gap:6px;padding:6px 10px;background:var(--bg-secondary);border-radius:6px;font-size:12px">
                    <span style="font-size:18px">📄</span>
                    <span style="color:var(--text-primary);max-width:150px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${escapeHtml(f.name)}</span>
                    <span style="color:var(--text-tertiary)">${sizeStr}</span>
                    <button class="remove-file" onclick="ErnOS.removeFile(${i})" style="position:static;background:var(--bg-tertiary);width:20px;height:20px;font-size:12px">×</button>
                </div>`;
            }
            preview.appendChild(item);
        });
    }

    function removeFile(index) {
        attachedFiles.splice(index, 1);
        renderFilePreview();
    }

    function clearAttachments() {
        attachedFiles = [];
        renderFilePreview();
    }

    function initDragDrop() {
        const dropOverlay = document.getElementById('drop-overlay');
        let dragCounter = 0;
        document.addEventListener('dragenter', (e) => { e.preventDefault(); dragCounter++; dropOverlay.classList.add('active'); });
        document.addEventListener('dragleave', (e) => { e.preventDefault(); dragCounter--; if (dragCounter === 0) dropOverlay.classList.remove('active'); });
        document.addEventListener('dragover', (e) => e.preventDefault());
        document.addEventListener('drop', (e) => { e.preventDefault(); dragCounter = 0; dropOverlay.classList.remove('active'); if (e.dataTransfer.files.length > 0) handleFilesSelected(e.dataTransfer.files); });
    }

    function initPaste() {
        document.getElementById('chat-input').addEventListener('paste', (e) => {
            const items = e.clipboardData.items;
            for (const item of items) {
                if (item.type.startsWith('image/')) { e.preventDefault(); handleFilesSelected([item.getAsFile()]); return; }
            }
        });
    }

    // ─── Theme ───
    function toggleTheme() {
        const current = document.documentElement.getAttribute('data-theme');
        const next = current === 'light' ? 'dark' : 'light';
        document.documentElement.setAttribute('data-theme', next === 'dark' ? '' : 'light');
        localStorage.setItem('ern-os-theme', next);
        document.getElementById('theme-toggle').textContent = next === 'light' ? '🌙' : '☀️';
    }

    function loadTheme() {
        const saved = localStorage.getItem('ern-os-theme') || 'dark';
        if (saved === 'light') document.documentElement.setAttribute('data-theme', 'light');
        const btn = document.getElementById('theme-toggle');
        if (btn) btn.textContent = saved === 'light' ? '🌙' : '☀️';
    }

    // ─── Sidebar ───
    function toggleSidebar() {
        document.querySelector('.sidebar').classList.toggle('open');
        document.getElementById('overlay').classList.toggle('visible');
    }

    // ─── Keyboard Shortcuts ───
    function initKeyboard() {
        document.addEventListener('keydown', (e) => {
            if ((e.metaKey || e.ctrlKey) && e.key === 'n') { e.preventDefault(); newChat(); }
            if ((e.metaKey || e.ctrlKey) && e.key === 'k') { e.preventDefault(); openSearchPalette(); }
            if (e.key === 'Escape') {
                if (document.getElementById('search-palette').style.display !== 'none') { closeSearchPalette(); return; }
                if (document.getElementById('confirm-overlay').style.display !== 'none') { document.getElementById('confirm-overlay').style.display = 'none'; return; }
                if (document.getElementById('context-menu').style.display !== 'none') { document.getElementById('context-menu').style.display = 'none'; return; }
                if (isGenerating) stopGeneration();
            }
        });
    }

    // ─── Helpers ───
    function escapeHtml(text) {
        const d = document.createElement('div');
        d.textContent = text;
        return d.innerHTML;
    }

    /// Post-process rendered content: syntax highlighting, LaTeX, Mermaid
    function postProcessContent(container) {
        // Syntax highlighting
        if (typeof hljs !== 'undefined') {
            container.querySelectorAll('pre code[class*="language-"]').forEach(block => {
                hljs.highlightElement(block);
            });
        }
        // LaTeX math rendering
        if (typeof renderMathInElement !== 'undefined') {
            try {
                renderMathInElement(container, {
                    delimiters: [
                        { left: '$$', right: '$$', display: true },
                        { left: '$', right: '$', display: false },
                        { left: '\\(', right: '\\)', display: false },
                        { left: '\\[', right: '\\]', display: true },
                    ],
                    throwOnError: false,
                });
            } catch (e) { /* KaTeX parse errors are non-fatal */ }
        }
        // Mermaid diagrams
        if (typeof mermaid !== 'undefined') {
            const mermaidEls = container.querySelectorAll('.mermaid');
            if (mermaidEls.length > 0) {
                mermaid.init(undefined, mermaidEls);
            }
        }
    }

    function scrollToBottom() {
        const c = document.getElementById('messages');
        c.scrollTop = c.scrollHeight;
    }

    // ─── Scroll-to-bottom FAB ───
    function initScrollFab() {
        const messages = document.getElementById('messages');
        const fab = document.getElementById('scroll-fab');
        if (!messages || !fab) return;
        messages.addEventListener('scroll', () => {
            const nearBottom = messages.scrollHeight - messages.scrollTop - messages.clientHeight < 100;
            fab.style.display = nearBottom ? 'none' : 'flex';
        });
    }

    // ─── Input History (up/down arrow) ───
    let inputHistory = [];
    let historyIndex = -1;

    function trackInputHistory(text) {
        if (text && text.trim()) {
            inputHistory.push(text.trim());
            if (inputHistory.length > 50) inputHistory.shift();
        }
        historyIndex = -1;
    }

    /// Fill the chat input with a prompt template and focus it.
    function fillPrompt(text) {
        const input = document.getElementById('chat-input');
        input.value = text;
        autoResize(input);
        input.focus();
    }

    function autoResize(el) {
        el.style.height = 'auto';
        el.style.height = Math.min(el.scrollHeight, 200) + 'px';
    }

    function hideWelcome() {
        const w = document.getElementById('welcome');
        if (w) w.style.display = 'none';
    }

    function welcomeHtml() {
        return `<div class="welcome" id="welcome">
            <div class="welcome-logo">E</div>
            <h2>Welcome to Ern-OS</h2>
            <p>High-performance AI engine with dual-layer inference, 7-tier cognitive memory, 16 tools, observer audit, autonomous learning, and self-skills.</p>
            <div class="welcome-features">
                <div class="feature-card" onclick="ErnOS.switchView('memory')">
                    <span>🧠</span><strong>7-Tier Memory</strong>
                    <small>Timeline, Lessons, Procedures, Scratchpad, Synaptic, Embeddings, Consolidation</small>
                </div>
                <div class="feature-card" onclick="ErnOS.switchView('tools')">
                    <span>🔧</span><strong>16 Tools</strong>
                    <small>Shell, Search, Memory, Files, Learning, Interpretability</small>
                </div>
                <div class="feature-card" onclick="ErnOS.switchView('training')">
                    <span>📊</span><strong>Training</strong>
                    <small>Golden buffer, rejection pairs, preference learning</small>
                </div>
                <div class="feature-card" onclick="ErnOS.switchView('settings')">
                    <span>⚙️</span><strong>System</strong>
                    <small>Model, observer, provider, sessions</small>
                </div>
            </div>
        </div>`;
    }

    function handleKeyDown(e) {
        if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
        const input = document.getElementById('chat-input');
        if (e.key === 'ArrowUp' && input.value === '' && inputHistory.length > 0) {
            e.preventDefault();
            if (historyIndex < 0) historyIndex = inputHistory.length;
            historyIndex = Math.max(0, historyIndex - 1);
            input.value = inputHistory[historyIndex] || '';
        }
        if (e.key === 'ArrowDown' && historyIndex >= 0) {
            e.preventDefault();
            historyIndex = Math.min(inputHistory.length, historyIndex + 1);
            input.value = historyIndex < inputHistory.length ? inputHistory[historyIndex] : '';
            if (historyIndex >= inputHistory.length) historyIndex = -1;
        }
    }

    // ─── Init ───
    async function init() {
        loadTheme();
        applyStoredPreferences();
        connect();
        initDragDrop();
        initPaste();
        initKeyboard();
        initContextMenu();
        initScrollFab();
        initMobileDetection();
        // Init mermaid with dark theme
        if (typeof mermaid !== 'undefined') {
            mermaid.initialize({ startOnLoad: false, theme: 'dark' });
        }
        const fileInput = document.getElementById('file-input');
        if (fileInput) fileInput.addEventListener('change', (e) => handleFilesSelected(e.target.files));
        // Check onboarding status
        await checkOnboarding();
        // Load version badge in sidebar
        loadVersionBadge();
    }

    // ─── Mobile / Android Detection ───
    function initMobileDetection() {
        const ua = navigator.userAgent || '';
        const isAndroidWebView = ua.includes('wv') && ua.includes('Android');
        const isNarrow = window.innerWidth <= 480;
        const isMobile = isAndroidWebView || isNarrow;

        if (isMobile) {
            document.body.classList.add('is-mobile');
        }

        // Listen for resize (e.g. rotating device)
        window.addEventListener('resize', () => {
            if (window.innerWidth <= 480) {
                document.body.classList.add('is-mobile');
            } else if (!isAndroidWebView) {
                document.body.classList.remove('is-mobile');
            }
        });

        // Show/hide platform-specific cards
        const computeCard = document.getElementById('compute-mode-card');
        const companionCard = document.getElementById('mobile-companion-card');
        if (isMobile) {
            if (computeCard) computeCard.style.display = '';
            if (companionCard) companionCard.style.display = 'none';
        } else {
            if (computeCard) computeCard.style.display = 'none';
            if (companionCard) companionCard.style.display = '';
            loadHostAddress();
        }
    }

    // ─── Compute Mode (Mobile) ───
    function setComputeMode(mode) {
        document.querySelectorAll('.compute-mode-card').forEach(c => {
            c.classList.toggle('active', c.dataset.mode === mode);
        });
        localStorage.setItem('ernos-compute-mode', mode);
        showToast(`Compute mode: ${mode}`, 'success');
        // In Android WebView, this would be picked up by EngineService via SharedPreferences bridge
        if (window.Android && window.Android.setComputeMode) {
            window.Android.setComputeMode(mode);
        }
    }

    // ─── Host Address (Desktop Companion Card) ───
    async function loadHostAddress() {
        const el = document.getElementById('host-address');
        if (!el) return;
        try {
            const res = await fetch('/api/health');
            if (res.ok) {
                // Use the current page hostname (works on LAN)
                const host = window.location.hostname;
                const port = window.location.port || '3000';
                el.textContent = `${host}:${port}`;
            }
        } catch {
            el.textContent = 'Unable to detect';
        }
    }

    // ─── Onboarding ───
    let onboardingPhase = 0;

    async function checkOnboarding() {
        try {
            const res = await fetch('/api/onboarding/status');
            const data = await res.json();
            if (!data.complete) {
                document.getElementById('onboarding-overlay').style.display = 'flex';
            }
        } catch (e) { /* silently continue if endpoint not available */ }
    }

    async function submitOnboardingProfile() {
        const name = document.getElementById('onboarding-name').value.trim() || 'User';
        const desc = document.getElementById('onboarding-desc').value.trim();
        try {
            await fetch('/api/onboarding/profile', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name, description: desc }),
            });
        } catch (e) { /* continue */ }

        // Move to Phase 2: Introduction
        onboardingPhase = 2;
        document.getElementById('step-1').classList.remove('active');
        document.getElementById('step-1').classList.add('done');
        document.getElementById('step-2').classList.add('active');
        document.getElementById('onboarding-title').textContent = `Nice to meet you, ${name}`;
        document.getElementById('onboarding-subtitle').textContent = '';
        document.getElementById('onboarding-body').innerHTML = `
            <p style="color:var(--text-secondary);font-size:14px;line-height:1.6;margin-bottom:20px;text-align:left">
                I'm <strong style="color:var(--accent)">Ern-OS</strong> — a local-first AI engine running on your hardware.
                I have persistent memory, tool execution, self-improvement, and an observer audit system.
            </p>
            <p style="color:var(--text-secondary);font-size:14px;line-height:1.6;margin-bottom:24px;text-align:left">
                Right now I'm using a default personality. Would you like to customise my name and identity through a short interview?
            </p>
            <div class="onboarding-btn-row">
                <button class="onboarding-btn secondary" onclick="ErnOS.skipIdentityInterview()">No, keep defaults</button>
                <button class="onboarding-btn" onclick="ErnOS.startIdentityInterview()">Yes, customise →</button>
            </div>
        `;
    }

    async function skipIdentityInterview() {
        await finishOnboarding();
    }

    async function startIdentityInterview() {
        // Mark onboarding complete FIRST — user chose a path, never show again
        await finishOnboarding();

        // Move to Phase 3
        onboardingPhase = 3;
        document.getElementById('step-2').classList.remove('active');
        document.getElementById('step-2').classList.add('done');
        document.getElementById('step-3').classList.add('active');

        // Close the overlay and switch to chat
        document.getElementById('onboarding-overlay').style.display = 'none';
        if (currentView !== 'chat') switchView('chat');
        hideWelcome();

        // Build the interview prompt
        const interviewPrompt = [
            'The user has just completed onboarding and wants to customise your identity.',
            'Conduct a short interview (about 3 questions) to understand what personality,',
            'name, and communication style they want from their AI assistant.',
            'After the interview, use the file_write tool to update the file at',
            'data/prompts/identity.md with a new identity prompt based on their answers.',
            'Keep the markdown format. When done, tell the user you have updated your identity.'
        ].join(' ');

        // Create a session if we don't have one
        if (!currentSessionId) {
            try {
                const res = await fetch('/api/sessions', { method: 'POST' });
                const s = await res.json();
                currentSessionId = s.id;
                loadSessions();
            } catch (e) { /* continue */ }
        }

        // Send via the proper chat flow
        sendChatPayload('[SYSTEM: Identity interview mode] ' + interviewPrompt);
    }

    async function finishOnboarding() {
        try {
            const res = await fetch('/api/onboarding/complete', { method: 'POST' });
            const data = await res.json();
            if (data.ok) {
                console.log('[Onboarding] Marked complete on disk');
            } else {
                console.error('[Onboarding] Server failed to mark complete:', data.error);
            }
        } catch (e) {
            console.error('[Onboarding] Failed to call complete endpoint:', e);
        }
        document.getElementById('onboarding-overlay').style.display = 'none';
    }

    // ─── Identity Editor ───
    async function loadIdentityView() {
        try {
            const [identityRes, coreRes, observerRes] = await Promise.all([
                fetch('/api/prompts/identity'),
                fetch('/api/prompts/core'),
                fetch('/api/prompts/observer'),
            ]);
            const identityData = await identityRes.json();
            const coreData = await coreRes.json();
            const observerData = await observerRes.json();

            const identityEditor = document.getElementById('editor-identity');
            const coreEditor = document.getElementById('editor-core');
            const observerEditor = document.getElementById('editor-observer');
            if (identityData.content) identityEditor.value = identityData.content;
            if (coreData.content) coreEditor.value = coreData.content;
            if (observerData.content) observerEditor.value = observerData.content;
        } catch (e) {
            console.error('Failed to load prompts:', e);
        }
    }

    async function savePrompt(name) {
        const editor = document.getElementById(`editor-${name}`);
        const btn = editor.parentElement.querySelector('.save-prompt-btn');
        const content = editor.value;
        try {
            const res = await fetch(`/api/prompts/${name}`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ content }),
            });
            const data = await res.json();
            if (data.ok) {
                btn.textContent = '✓ Saved';
                btn.classList.add('saved');
                setTimeout(() => {
                    btn.textContent = name === 'identity' ? 'Save Identity' : name === 'core' ? 'Save Core' : 'Save Observer';
                    btn.classList.remove('saved');
                }, 2000);
            }
        } catch (e) {
            console.error('Failed to save prompt:', e);
        }
    }

    function applyStoredPreferences() {
        const fontSize = localStorage.getItem('ern-os-font-size');
        if (fontSize) document.documentElement.style.setProperty('--msg-font-size', fontSize + 'px');
    }

    // ─── Agents View ───
    let agentsList = [];

    async function loadAgentsView() {
        try {
            const [agentsRes, teamsRes] = await Promise.all([
                fetch('/api/agents'),
                fetch('/api/teams'),
            ]);
            const agentsData = await agentsRes.json();
            const teamsData = await teamsRes.json();
            agentsList = agentsData.agents || [];
            renderAgentsList(agentsList);
            renderTeamsList(teamsData.teams || []);
            loadAgentActivity();
            loadDagStatus();
        } catch (e) {
            console.error('Failed to load agents:', e);
        }
    }

    async function loadAgentActivity() {
        const feed = document.getElementById('agent-activity-feed');
        if (!feed) return;
        try {
            const resp = await fetch('/api/agents/activity');
            const data = await resp.json();
            const dot = document.getElementById('activity-live-dot');
            const countEl = document.getElementById('active-agent-count');
            const entries = data.entries || [];
            const activeCount = data.active_agents || 0;

            if (dot) dot.className = 'live-dot' + (activeCount > 0 ? ' active' : '');
            if (countEl) countEl.textContent = activeCount + ' active';

            if (entries.length === 0) {
                feed.innerHTML = '<p class="empty-state">No agent activity yet</p>';
                return;
            }

            feed.innerHTML = entries.slice(0, 30).map(e => {
                const statusClass = e.status || 'pending';
                const icon = {running:'🔄', done:'✅', failed:'❌', pending:'⏳'}[statusClass] || '⏳';
                const t = e.timestamp ? new Date(e.timestamp).toLocaleTimeString() : '';
                return `<div class="activity-entry">
                    <div class="activity-badge ${statusClass}">${icon}</div>
                    <div style="flex:1;min-width:0">
                        <div><span class="activity-agent">🤖 ${escapeHtml(e.agent_name || e.agent_id || 'Agent')}</span> <span class="activity-action">▸ ${escapeHtml(e.action || '')}</span></div>
                        ${e.detail ? '<div class="activity-detail">└─ ' + escapeHtml(e.detail) + '</div>' : ''}
                    </div>
                    <span class="activity-time">${t}</span>
                </div>`;
            }).join('');
        } catch { /* activity not available */ }
    }

    async function loadDagStatus() {
        try {
            const resp = await fetch('/api/planning/status');
            const data = await resp.json();
            const panel = document.getElementById('dag-panel');
            if (!panel) return;

            if (!data.active || !data.dag) {
                panel.style.display = 'none';
                return;
            }

            panel.style.display = 'block';
            const dag = data.dag;
            const nodes = dag.nodes || [];
            const completed = nodes.filter(n => n.status === 'Completed').length;
            const total = nodes.length;
            const pct = total > 0 ? Math.round((completed / total) * 100) : 0;

            document.getElementById('dag-progress-label').textContent = completed + '/' + total;
            document.getElementById('dag-progress-bar').style.width = pct + '%';

            const list = document.getElementById('dag-task-list');
            list.innerHTML = nodes.map(n => {
                const icons = {Pending:'⏳', Ready:'🟡', Running:'🔄', Completed:'✅', Failed:'❌', Blocked:'🚫'};
                const icon = icons[n.status] || '⏳';
                const cls = n.status === 'Completed' ? ' completed' : '';
                const dur = n.completed_at && n.started_at
                    ? ((new Date(n.completed_at) - new Date(n.started_at)) / 1000).toFixed(0) + 's'
                    : n.status === 'Running' ? 'running...' : '';
                return `<div class="dag-task${cls}">
                    <span class="dag-task-icon">${icon}</span>
                    <span class="dag-task-title">${escapeHtml(n.title)}</span>
                    <span class="dag-task-time">${dur}</span>
                </div>` + (n.depends_on && n.depends_on.length && n.status !== 'Completed'
                    ? '<div class="dag-task-dep">└─ blocked by: ' + n.depends_on.join(', ') + '</div>' : '');
            }).join('');
        } catch { /* planning not available */ }
    }

    async function sendAgentDirection() {
        const input = document.getElementById('agent-direction-input');
        if (!input || !input.value.trim()) return;
        const direction = input.value.trim();

        // Send to the first active/running agent
        try {
            const resp = await fetch('/api/agents/activity');
            const data = await resp.json();
            const running = (data.entries || []).find(e => e.status === 'running');
            const agentId = running ? (running.agent_id || 'default') : 'default';

            await fetch(`/api/agents/${agentId}/direct`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ direction }),
            });
            input.value = '';
            showToast('Direction sent to agent', 'success');
        } catch {
            showToast('Failed to send direction', 'error');
        }
    }

    function renderAgentsList(agents) {
        const container = document.getElementById('agents-list');
        if (agents.length === 0) {
            container.innerHTML = `<div class="prompt-editor-card" style="text-align:center;padding:40px;color:var(--text-muted)">
                <p style="font-size:32px;margin-bottom:12px">🤖</p>
                <p>No agents yet. Create one above to get started.</p>
                <p style="font-size:12px;margin-top:8px;opacity:0.6">Each agent is a fully-equipped Ern-OS instance with its own identity, tools, and sessions.</p>
            </div>`;
            return;
        }
        container.innerHTML = agents.map(a => `
            <div class="prompt-editor-card agent-card" id="agent-${a.id}">
                <div class="prompt-editor-header">
                    <h3>🤖 ${escapeHtml(a.name)}</h3>
                    <div style="display:flex;gap:8px;align-items:center">
                        <span class="prompt-tag">${a.tools.length ? a.tools.length + ' tools' : 'All tools'}</span>
                        <span class="prompt-tag" style="${a.observer_enabled ? '' : 'background:var(--error-bg);color:var(--error)'}">
                            ${a.observer_enabled ? '✅ Observer' : '❌ Observer'}
                        </span>
                        <button class="save-prompt-btn" style="margin:0;padding:6px 12px;font-size:11px;background:var(--error);color:white" onclick="ErnOS.deleteAgent('${a.id}')">✕</button>
                    </div>
                </div>
                <p style="color:var(--text-secondary);font-size:13px;margin-bottom:10px">${escapeHtml(a.description || 'No description')}</p>
                <div style="display:flex;gap:8px;flex-wrap:wrap">
                    <button class="save-prompt-btn" style="margin:0;padding:8px 16px;font-size:12px" onclick="ErnOS.chatWithAgent('${a.id}')">💬 Chat</button>
                    <button class="save-prompt-btn secondary-btn" style="margin:0;padding:8px 16px;font-size:12px" onclick="ErnOS.toggleAgentObserver('${a.id}', ${!a.observer_enabled})">
                        ${a.observer_enabled ? 'Disable Observer' : 'Enable Observer'}
                    </button>
                </div>
            </div>
        `).join('');
    }

    function renderTeamsList(teams) {
        const container = document.getElementById('teams-list');
        if (teams.length === 0) {
            container.innerHTML = '<p style="color:var(--text-muted);font-size:13px;text-align:center;padding:16px">No teams yet. Create agents first, then group them into teams.</p>';
            return;
        }
        container.innerHTML = teams.map(t => `
            <div style="display:flex;align-items:center;justify-content:space-between;padding:10px 14px;background:var(--bg-input);border-radius:8px;margin-bottom:8px;border:1px solid var(--border)">
                <div>
                    <strong style="color:var(--text-primary)">${escapeHtml(t.name)}</strong>
                    <span class="prompt-tag" style="margin-left:8px">${t.mode === 'sequential' ? '🔗 Sequential' : '⚡ Parallel'}</span>
                    <span style="color:var(--text-muted);font-size:12px;margin-left:8px">${t.agents.length} agents</span>
                </div>
                <button class="save-prompt-btn" style="margin:0;padding:6px 12px;font-size:11px;background:var(--error);color:white" onclick="ErnOS.deleteTeam('${t.id}')">✕</button>
            </div>
        `).join('');
    }

    async function createAgent() {
        const name = document.getElementById('new-agent-name').value.trim();
        const desc = document.getElementById('new-agent-desc').value.trim();
        if (!name) return;
        try {
            const res = await fetch('/api/agents', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name, description: desc }),
            });
            const data = await res.json();
            if (data.ok) {
                document.getElementById('new-agent-name').value = '';
                document.getElementById('new-agent-desc').value = '';
                loadAgentsView();
            }
        } catch (e) { console.error('Failed to create agent:', e); }
    }

    async function deleteAgent(id) {
        try {
            await fetch(`/api/agents/${id}`, { method: 'DELETE' });
            loadAgentsView();
        } catch (e) { console.error('Failed to delete agent:', e); }
    }

    async function toggleAgentObserver(id, enabled) {
        try {
            await fetch(`/api/agents/${id}`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ observer_enabled: enabled }),
            });
            loadAgentsView();
        } catch (e) { console.error('Failed to toggle observer:', e); }
    }

    async function chatWithAgent(agentId) {
        const agent = agentsList.find(a => a.id === agentId);
        if (!agent) return;

        // Create a new session for this agent
        try {
            const resp = await fetch('/api/sessions', { method: 'POST' });
            const data = await resp.json();
            if (data.id) {
                currentSessionId = data.id;
                currentAgentId = agentId;
                document.getElementById('messages').innerHTML = '';

                // Rename the session to the agent's name
                await fetch(`/api/sessions/${data.id}`, {
                    method: 'PUT',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ title: `🤖 ${agent.name}` }),
                });

                await loadSessions();
                addSystemMessage(`🤖 Now chatting with agent: **${agent.name}**\n\n_${agent.description || 'No description'}_`);
                switchView('chat');
            }
        } catch (e) {
            console.error('Failed to create agent session:', e);
            // Fallback: just switch to chat with agent context
            currentAgentId = agentId;
            addSystemMessage(`🤖 Now chatting with agent: **${agent.name}**`);
            switchView('chat');
        }
    }

    async function createTeam() {
        const name = document.getElementById('new-team-name').value.trim();
        const mode = document.getElementById('new-team-mode').value;
        if (!name) return;
        // Use all current agents by default
        const agentIds = agentsList.map(a => a.id);
        try {
            await fetch('/api/teams', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name, mode, agents: agentIds }),
            });
            document.getElementById('new-team-name').value = '';
            loadAgentsView();
        } catch (e) { console.error('Failed to create team:', e); }
    }

    async function deleteTeam(id) {
        try {
            await fetch(`/api/teams/${id}`, { method: 'DELETE' });
            loadAgentsView();
        } catch (e) { console.error('Failed to delete team:', e); }
    }

    // ─── Version Management ───

    async function loadVersionBadge() {
        try {
            const resp = await fetch('/api/version');
            const v = await resp.json();
            const badge = document.getElementById('version-badge');
            if (badge && v.hash) {
                badge.textContent = v.hash;
                badge.style.cssText = 'font-size:10px;color:var(--text-tertiary);font-family:monospace;margin-left:auto';
            }
            // Auto-check for updates
            setTimeout(autoCheckUpdates, 3000);
        } catch (e) { /* ignore */ }
    }

    async function autoCheckUpdates() {
        try {
            const resp = await fetch('/api/version/check');
            const data = await resp.json();
            if (!data.up_to_date && data.commits_behind > 0) {
                const badge = document.getElementById('settings-update-badge');
                if (badge) {
                    badge.style.display = 'inline';
                    badge.title = data.commits_behind + ' update(s) available';
                }
            }
        } catch (e) { /* ignore */ }
    }

    async function loadVersionInfo() {
        try {
            const resp = await fetch('/api/version');
            const v = await resp.json();
            const el = document.getElementById('version-info');
            if (el) {
                el.innerHTML = `
                    <div style="display:flex;flex-direction:column;gap:4px;width:100%">
                        <div style="display:flex;justify-content:space-between;align-items:center">
                            <span style="font-weight:600;color:var(--text-primary)">Current Version</span>
                            <code style="font-size:12px;background:var(--bg-secondary);padding:2px 8px;border-radius:4px">${escapeHtml(v.hash || '?')}</code>
                        </div>
                        <div style="font-size:12px;color:var(--text-secondary)">${escapeHtml(v.message || '')}</div>
                        <div style="font-size:11px;color:var(--text-tertiary)">${escapeHtml(v.date || '')} · ${escapeHtml(v.branch || '')}${v.dirty ? ' · <span style="color:var(--warning)">uncommitted changes</span>' : ''}</div>
                    </div>`;
            }
        } catch (e) {
            const el = document.getElementById('version-info');
            if (el) el.innerHTML = '<span class="label-desc">Failed to load version info</span>';
        }
    }

    async function checkForUpdates() {
        const btn = document.getElementById('check-updates-btn');
        const status = document.getElementById('update-status');
        const updateBtn = document.getElementById('update-now-btn');
        if (btn) btn.textContent = 'Checking...';
        try {
            const resp = await fetch('/api/version/check');
            const data = await resp.json();
            status.style.display = 'block';
            if (data.up_to_date) {
                status.style.background = 'rgba(34,197,94,0.1)';
                status.style.color = 'var(--success, #22c55e)';
                status.innerHTML = '✅ You are up to date';
                if (updateBtn) updateBtn.style.display = 'none';
                const badge = document.getElementById('settings-update-badge');
                if (badge) badge.style.display = 'none';
            } else {
                status.style.background = 'rgba(59,130,246,0.1)';
                status.style.color = 'var(--accent, #3b82f6)';
                status.innerHTML = `🔄 <strong>${data.commits_behind}</strong> update(s) available`;
                if (data.new_commits) {
                    status.innerHTML += '<div style="margin-top:6px;font-size:11px;max-height:120px;overflow-y:auto">' +
                        data.new_commits.map(c => '<div style="padding:2px 0">' + escapeHtml(c) + '</div>').join('') + '</div>';
                }
                if (updateBtn) updateBtn.style.display = 'inline-block';
            }
        } catch (e) {
            status.style.display = 'block';
            status.style.background = 'rgba(239,68,68,0.1)';
            status.style.color = 'var(--danger, #ef4444)';
            status.innerHTML = '❌ Failed to check: ' + escapeHtml(e.message);
        }
        if (btn) btn.textContent = 'Check for Updates';
    }

    async function updateNow() {
        if (!confirm('⚠️ UPDATE\n\nThis will pull the latest version from GitHub and recompile.\nThe server will restart — you will be disconnected briefly.\n\nContinue?')) return;
        const status = document.getElementById('update-status');
        status.style.display = 'block';
        status.style.background = 'rgba(59,130,246,0.1)';
        status.style.color = 'var(--accent)';
        status.innerHTML = '⏳ Updating... (pulling + recompiling, this takes ~30s)';
        try {
            const resp = await fetch('/api/version/update', { method: 'POST' });
            const data = await resp.json();
            if (data.success) {
                status.innerHTML = '✅ ' + escapeHtml(data.message || 'Update applied') + '<br>Reconnecting...';
                // Server will restart — reconnect loop handles it
            } else {
                status.style.background = 'rgba(239,68,68,0.1)';
                status.style.color = 'var(--danger)';
                status.innerHTML = '❌ ' + escapeHtml(data.error || 'Update failed') + '<br>' + escapeHtml(data.action || '');
            }
        } catch (e) {
            status.innerHTML = '⏳ Server is restarting... reconnecting shortly.';
        }
    }

    async function toggleVersionHistory() {
        const container = document.getElementById('version-history');
        if (container.style.display !== 'none') {
            container.style.display = 'none';
            return;
        }
        container.style.display = 'block';
        container.innerHTML = '<span class="label-desc">Loading history...</span>';
        try {
            const resp = await fetch('/api/version/history');
            const data = await resp.json();
            if (!data.commits || data.commits.length === 0) {
                container.innerHTML = '<span class="label-desc">No history available</span>';
                return;
            }
            container.innerHTML = data.commits.map(c => `
                <div style="display:flex;align-items:center;gap:8px;padding:6px 8px;border-radius:6px;${c.current ? 'background:rgba(59,130,246,0.1)' : ''}">
                    <span style="font-size:12px;color:${c.current ? 'var(--accent)' : 'var(--text-tertiary)'};font-weight:${c.current ? '700' : '400'}">${c.current ? '●' : '○'}</span>
                    <code style="font-size:11px;min-width:60px;color:var(--text-secondary)">${escapeHtml(c.short_hash)}</code>
                    <span style="font-size:12px;color:var(--text-primary);flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${escapeHtml(c.message)}</span>
                    <span style="font-size:10px;color:var(--text-tertiary);min-width:80px">${escapeHtml((c.date || '').substring(0,10))}</span>
                    ${c.current ? '<span style="font-size:10px;color:var(--accent);font-weight:600">current</span>' : 
                        `<button onclick="ErnOS.revertToVersion('${c.hash}','${escapeHtml(c.short_hash)}','${escapeHtml(c.message).replace(/'/g,'')}')" 
                         style="font-size:10px;padding:2px 8px;border-radius:4px;background:var(--bg-tertiary);color:var(--text-secondary);border:1px solid var(--border);cursor:pointer">Revert</button>`}
                </div>
            `).join('');
        } catch (e) {
            container.innerHTML = '<span class="label-desc">Failed to load history</span>';
        }
    }

    async function revertToVersion(hash, shortHash, message) {
        if (!confirm(`⚠️ REVERT\n\nRollback to version ${shortHash}?\n"${message}"\n\nThis will recompile and restart the server.\nLocal changes will be stashed.\n\nContinue?`)) return;
        const status = document.getElementById('update-status');
        if (status) {
            status.style.display = 'block';
            status.style.background = 'rgba(245,158,11,0.1)';
            status.style.color = 'var(--warning, #f59e0b)';
            status.innerHTML = '⏳ Reverting to ' + escapeHtml(shortHash) + '... (recompiling ~30s)';
        }
        try {
            const resp = await fetch('/api/version/rollback', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ hash }),
            });
            const data = await resp.json();
            if (data.success) {
                if (status) status.innerHTML = '✅ Reverted to ' + escapeHtml(shortHash) + '. Reconnecting...';
            } else {
                if (status) {
                    status.style.background = 'rgba(239,68,68,0.1)';
                    status.style.color = 'var(--danger)';
                    status.innerHTML = '❌ ' + escapeHtml(data.error || 'Revert failed');
                }
            }
        } catch (e) {
            if (status) status.innerHTML = '⏳ Server is restarting... reconnecting shortly.';
        }
    }

    // ─── Autonomy Controls ───
    function loadAutonomyState() {
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type: 'get_autonomy' }));
        }
    }

    function setAutonomy(level) {
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type: 'set_autonomy', level }));
            showToast(`Autonomy set to ${level}`, 'success');
        }
    }

    function handleAutonomyResponse(data) {
        const level = data.level || 'supervised';
        const radio = document.querySelector(`input[name="autonomy"][value="${level}"]`);
        if (radio) radio.checked = true;
    }

    // ─── System Snapshots ───
    async function loadSnapshots() {
        const container = document.getElementById('snapshots-container');
        if (!container) return;
        try {
            const resp = await fetch('/api/state-checkpoint');
            if (!resp.ok) {
                container.innerHTML = '<p class="empty-state">Snapshots not available</p>';
                return;
            }
            const data = await resp.json();
            if (!data.checkpoints || data.checkpoints.length === 0) {
                container.innerHTML = '<p class="empty-state">No snapshots yet</p>';
                return;
            }
            container.innerHTML = data.checkpoints.map(c => {
                const time = new Date(c.timestamp).toLocaleString();
                const commit = c.git_commit ? c.git_commit.substring(0, 7) : '—';
                return `<div class="snapshot-entry">
                    <div class="snapshot-meta">
                        <div class="snapshot-label">${escapeHtml(c.label || c.id)}</div>
                        <div class="snapshot-time">${time}</div>
                        <div class="snapshot-commit">commit ${commit}</div>
                    </div>
                    <div class="snapshot-actions-row">
                        <button class="snapshot-btn restore" onclick="ErnOS.restoreSnapshot('${c.id}')">↩ Restore</button>
                        <button class="snapshot-btn delete" onclick="ErnOS.deleteSnapshot('${c.id}')">✕</button>
                    </div>
                </div>`;
            }).join('');
        } catch (e) {
            container.innerHTML = '<p class="empty-state">Failed to load snapshots</p>';
        }
    }

    async function createSnapshot() {
        const labelEl = document.getElementById('snapshot-label');
        const label = labelEl ? labelEl.value.trim() : '';
        try {
            const resp = await fetch('/api/state-checkpoint', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ label: label || 'Manual snapshot' }),
            });
            if (resp.ok) {
                showToast('Snapshot created', 'success');
                if (labelEl) labelEl.value = '';
                loadSnapshots();
            } else {
                const err = await resp.text();
                showToast('Snapshot failed: ' + err, 'error');
            }
        } catch (e) {
            showToast('Snapshot failed: ' + e.message, 'error');
        }
    }

    async function restoreSnapshot(id) {
        if (!confirm('Restore this snapshot? Current state will be overwritten.')) return;
        try {
            const resp = await fetch(`/api/state-checkpoint/${id}/restore`, { method: 'POST' });
            if (resp.ok) {
                showToast('Snapshot restored — reloading...', 'success');
                setTimeout(() => location.reload(), 2000);
            } else {
                showToast('Restore failed', 'error');
            }
        } catch (e) {
            showToast('Restore failed: ' + e.message, 'error');
        }
    }

    async function deleteSnapshot(id) {
        if (!confirm('Delete this snapshot?')) return;
        try {
            const resp = await fetch(`/api/state-checkpoint/${id}`, { method: 'DELETE' });
            if (resp.ok) {
                showToast('Snapshot deleted', 'success');
                loadSnapshots();
            } else {
                showToast('Delete failed', 'error');
            }
        } catch (e) {
            showToast('Delete failed: ' + e.message, 'error');
        }
    }

    // ─── HuggingFace Model Hub ───
    async function loadModelHub() {
        // Load local models
        try {
            const resp = await fetch('/api/models');
            const data = await resp.json();
            const container = document.getElementById('hf-local-models');
            if (container && data.models && data.models.length > 0) {
                container.innerHTML = '<div style="font-size:11px;color:var(--text-muted);margin-bottom:6px;font-weight:600;text-transform:uppercase;letter-spacing:0.5px">Local Models (./models/)</div>'
                    + data.models.filter(m => !m.is_mmproj).map(m => `<div class="hf-model-card">
                        <div class="hf-model-info">
                            <div class="hf-model-name">${escapeHtml(m.name)}</div>
                            <div class="hf-model-meta"><span>${m.size_gb} GB</span></div>
                        </div>
                        <button class="hf-download-btn" onclick="ErnOS.swapModel('${escapeHtml(m.name)}')" style="border-color:var(--border);background:var(--bg-tertiary);color:var(--text-secondary)">Switch</button>
                    </div>`).join('');
            }
        } catch { /* models not available */ }

        // Check download progress
        pollModelDownload();
    }

    async function searchHuggingFace() {
        const input = document.getElementById('hf-search-input');
        const container = document.getElementById('hf-search-results');
        if (!input || !container) return;
        const q = input.value.trim();
        if (!q) return;

        container.innerHTML = '<p class="empty-state">Searching HuggingFace...</p>';

        try {
            const resp = await fetch(`/api/models/search?q=${encodeURIComponent(q)}`);
            const data = await resp.json();
            const models = data.models || [];

            if (models.length === 0) {
                container.innerHTML = '<p class="empty-state">No GGUF models found</p>';
                return;
            }

            container.innerHTML = '<div style="font-size:11px;color:var(--text-muted);margin-bottom:6px;font-weight:600;text-transform:uppercase;letter-spacing:0.5px">Search Results</div>'
                + models.map(m => {
                    const id = m.id || '';
                    const likes = m.likes || 0;
                    const downloads = m.downloads || 0;
                    return `<div class="hf-model-card">
                        <div class="hf-model-info">
                            <div class="hf-model-name">${escapeHtml(id)}</div>
                            <div class="hf-model-meta">
                                <span>❤️ ${likes.toLocaleString()}</span>
                                <span>⬇️ ${downloads.toLocaleString()}</span>
                            </div>
                        </div>
                        <button class="hf-download-btn" onclick="ErnOS.downloadModel('${escapeHtml(id)}')">⬇ Download</button>
                    </div>`;
                }).join('');
        } catch (e) {
            container.innerHTML = '<p class="empty-state">Search failed: ' + e.message + '</p>';
        }
    }

    async function downloadModel(repo) {
        // Guess most common GGUF filename pattern
        const filename = prompt('Enter GGUF filename to download:', repo.split('/').pop() + '-Q4_K_M.gguf');
        if (!filename) return;

        try {
            await fetch('/api/models/download', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ repo, filename }),
            });
            showToast('Download started: ' + filename, 'success');
            pollModelDownload();
        } catch (e) {
            showToast('Download failed: ' + e.message, 'error');
        }
    }

    let downloadPollTimer = null;
    function pollModelDownload() {
        if (downloadPollTimer) clearInterval(downloadPollTimer);
        downloadPollTimer = setInterval(async () => {
            try {
                const resp = await fetch('/api/model/download-progress');
                const data = await resp.json();
                const el = document.getElementById('hf-download-progress');
                if (!el) return;

                if (data.downloading && data.progress !== undefined) {
                    const pct = Math.round(data.progress * 100);
                    const totalGB = data.total_bytes ? (data.total_bytes / 1073741824).toFixed(1) : '?';
                    const dlGB = data.downloaded_bytes ? (data.downloaded_bytes / 1073741824).toFixed(1) : '0';
                    el.style.display = 'block';
                    el.innerHTML = `<div style="font-size:12px;font-weight:600;color:var(--text-primary);margin-bottom:4px">⬇️ ${escapeHtml(data.model || 'model')} — ${pct}%</div>
                        <div class="hf-progress-bar"><div class="hf-progress-fill" style="width:${pct}%"></div></div>
                        <div style="font-size:10px;color:var(--text-muted);margin-top:4px">${dlGB} / ${totalGB} GB</div>`;

                    // Update model badge too
                    const badge = document.getElementById('model-badge');
                    if (badge) badge.textContent = `⬇ ${pct}% ${data.model || ''}`;
                } else {
                    el.style.display = 'none';
                    if (data.complete) {
                        showToast('Model download complete: ' + (data.model || ''), 'success');
                        loadModelHub();
                    }
                    clearInterval(downloadPollTimer);
                    downloadPollTimer = null;
                }
            } catch {
                clearInterval(downloadPollTimer);
                downloadPollTimer = null;
            }
        }, 2000);
    }

    // ─── WS Message Handlers for New Features ───
    function handleNewWsMessages(msg) {
        switch (msg.type) {
            case 'task_progress':
                updateDagTask(msg);
                break;
            case 'agent_activity':
                appendAgentActivity(msg);
                break;
            case 'verification_result':
                showVerificationResult(msg);
                break;
            case 'progress_report':
                updateAutonomyDashboard(msg);
                break;
        }
    }

    function updateDagTask(msg) {
        const panel = document.getElementById('dag-panel');
        if (panel) panel.style.display = 'block';
        loadDagStatus(); // Refresh the full DAG
    }

    function appendAgentActivity(msg) {
        const feed = document.getElementById('agent-activity-feed');
        if (!feed) return;
        const emptyState = feed.querySelector('.empty-state');
        if (emptyState) emptyState.remove();

        const statusClass = msg.status || 'running';
        const icon = {running:'🔄', done:'✅', failed:'❌', pending:'⏳'}[statusClass] || '⏳';
        const t = new Date().toLocaleTimeString();
        const html = `<div class="activity-entry">
            <div class="activity-badge ${statusClass}">${icon}</div>
            <div style="flex:1;min-width:0">
                <div><span class="activity-agent">🤖 ${escapeHtml(msg.agent_name || 'Agent')}</span> <span class="activity-action">▸ ${escapeHtml(msg.action || '')}</span></div>
                ${msg.detail ? '<div class="activity-detail">└─ ' + escapeHtml(msg.detail) + '</div>' : ''}
            </div>
            <span class="activity-time">${t}</span>
        </div>`;
        feed.insertAdjacentHTML('afterbegin', html);
    }

    function showVerificationResult(msg) {
        if (!currentAssistantEl) return;
        const buildIcon = msg.success ? '✅' : '❌';
        const testIcon = msg.tests_passed !== undefined ? (msg.tests_failed === 0 ? '✅' : '❌') : '⏭';
        const html = `<div class="verification-card">
            <div style="font-weight:700;margin-bottom:6px;color:var(--text-primary)">🔍 Verification</div>
            <div class="verification-row"><span class="v-label">Build:</span> <span class="v-icon">${buildIcon}</span> ${msg.errors ? msg.errors + ' errors' : 'Success'} ${msg.warnings ? '(' + msg.warnings + ' warnings)' : ''}</div>
            ${msg.tests_passed !== undefined ? `<div class="verification-row"><span class="v-label">Tests:</span> <span class="v-icon">${testIcon}</span> ${msg.tests_passed} passed, ${msg.tests_failed || 0} failed</div>` : ''}
            ${msg.fix_attempts ? `<div class="verification-row"><span class="v-label">Auto-fix:</span> ${msg.fix_attempts} attempts</div>` : ''}
        </div>`;
        currentAssistantEl.querySelector('.message-content')?.insertAdjacentHTML('beforeend', html);
    }

    function updateAutonomyDashboard(msg) {
        let dash = document.querySelector('.autonomy-dashboard');
        if (!dash) {
            dash = document.createElement('div');
            dash.className = 'autonomy-dashboard';
            document.body.appendChild(dash);
        }
        dash.classList.add('visible');

        const elapsed = msg.elapsed_min !== undefined ? `${msg.elapsed_min}m` : '—';
        const completed = msg.completed !== undefined ? msg.completed : 0;
        const remaining = msg.remaining !== undefined ? msg.remaining : 0;
        const files = msg.files_modified !== undefined ? msg.files_modified : 0;
        const current = msg.current || 'Working...';

        dash.innerHTML = `
            <div style="font-size:14px">🚀</div>
            <div class="autonomy-stat"><span class="autonomy-stat-value">${elapsed}</span><span class="autonomy-stat-label">Elapsed</span></div>
            <div class="autonomy-stat"><span class="autonomy-stat-value">${completed}/${completed + remaining}</span><span class="autonomy-stat-label">Tasks</span></div>
            <div class="autonomy-stat"><span class="autonomy-stat-value">${files}</span><span class="autonomy-stat-label">Files</span></div>
            <div class="autonomy-current">${escapeHtml(current)}</div>
            <button class="autonomy-dashboard-btn danger" onclick="ErnOS.stopGeneration()">Stop</button>
        `;
    }

    // ─── Public API ───
    return {
        init, sendMessage, newChat, switchSession, deleteSession, stopGeneration,
        toggleTheme, toggleSidebar, handleKeyDown, autoResize,
        copyMessage, copyCode, removeFile, handleFilesSelected,
        switchView, loadMemoryTier, toggleTool,
        factoryReset, setTheme, setFontSize, setMsgWidth, swapModel,
        saveApiKey, clearApiKey, savePlatformConfig, connectPlatform, disconnectPlatform,
        filterFeatures, toggleLogFilter,
        toggleSchedulerJob, deleteSchedulerJob, showAddJobForm, createSchedulerJob,
        submitOnboardingProfile, skipIdentityInterview, startIdentityInterview,
        savePrompt,
        createAgent, deleteAgent, toggleAgentObserver, chatWithAgent,
        createTeam, deleteTeam,
        // Phase 1: Session management
        showSessionMenu, startRenameSession, filterSessionList,
        togglePin, toggleArchive, exportSession,
        // Phase 2: Message interactions
        regenerateMessage, editMessage, saveEditedMessage, cancelEditMessage,
        forkFromMessage, reactToMessage, deleteMessage: deleteMessageAtIndex,
        speakMessage,
        // Phase 4: Search
        openSearchPalette, closeSearchPalette,
        // Toast
        showToast,
        // Scroll
        scrollToBottom,
        // Prompt templates
        fillPrompt,
        // Planning mode
        approvePlan, revisePlan, cancelPlan,
        // Artifacts
        copyArtifact,
        // Voice/Video
        toggleVoiceCall, toggleVideoCall, toggleCallMute, endCall,
        // Version management
        checkForUpdates, updateNow, toggleVersionHistory, revertToVersion,
        // Autonomy & Snapshots
        setAutonomy, createSnapshot, restoreSnapshot, deleteSnapshot,
        // Mobile
        setComputeMode,
        // Agent Activity & DAG
        sendAgentDirection, loadAgentActivity, loadDagStatus,
        // HuggingFace Model Hub
        searchHuggingFace, downloadModel,
    };
})();

document.addEventListener('DOMContentLoaded', ErnOS.init);
