#!/bin/bash
# Ern-OS — High-performance, model-neutral Rust AI agent engine
# Created by @mettamazza (github.com/mettamazza)
# License: MIT
# ============================================================
# Hot-Swap Upgrade Script
# ============================================================
# Swaps the running binary with the newly compiled one, monitors
# health, and auto-rolls back if the new binary fails to start.
#
# Invoked by the system_recompile tool after a successful build.
# ============================================================

echo "[UPGRADE] Engaging upgrade sequence..."

ERNOS_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BINARY_NAME="ern-os"
NEXT_BINARY="$ERNOS_DIR/${BINARY_NAME}_next"
RELEASE_BINARY="$ERNOS_DIR/target/release/$BINARY_NAME"
BACKUP_BINARY="$ERNOS_DIR/target/release/${BINARY_NAME}_rollback"
LOG_DIR="$ERNOS_DIR/data/logs"

mkdir -p "$LOG_DIR"

# ── PRE-FLIGHT ──────────────────────────────────────────────────
if [ ! -f "$NEXT_BINARY" ]; then
    echo "[UPGRADE] ❌ No staged binary at $NEXT_BINARY"
    exit 1
fi

# ── BACKUP ──────────────────────────────────────────────────────
if [ -f "$RELEASE_BINARY" ]; then
    cp "$RELEASE_BINARY" "$BACKUP_BINARY"
    echo "[UPGRADE] Backup saved to $BACKUP_BINARY"
fi

# ── SWAP ────────────────────────────────────────────────────────
echo "[UPGRADE] Overwriting release binary..."
cp "$NEXT_BINARY" "$RELEASE_BINARY"
rm -f "$NEXT_BINARY"

PORT=3000

echo "[UPGRADE] Killing old processes..."
for pid in $(pgrep -f "target/release/$BINARY_NAME" 2>/dev/null || true); do
    if [ "$pid" != "$$" ] && [ "$pid" != "$PPID" ]; then
        kill "$pid" 2>/dev/null
    fi
done
# Kill stale llama-server so the new binary starts fresh
for pid in $(pgrep -f "llama-server" 2>/dev/null || true); do
    kill "$pid" 2>/dev/null
done

# Wait for port to free (max 30s)
echo "[UPGRADE] Waiting for port $PORT to release..."
TIMEOUT=30
ELAPSED=0
while [ $ELAPSED -lt $TIMEOUT ]; do
    if ! lsof -i ":$PORT" -t >/dev/null 2>&1; then
        echo "[UPGRADE] ✅ Port released in ${ELAPSED}s"
        break
    fi
    sleep 2
    ELAPSED=$((ELAPSED + 2))
    if [ $ELAPSED -ge 10 ]; then
        lsof -i ":$PORT" -t 2>/dev/null | xargs kill -9 2>/dev/null
    fi
done

# ── LAUNCH ──────────────────────────────────────────────────────
echo "[UPGRADE] Starting new binary..."
cd "$ERNOS_DIR"
./target/release/$BINARY_NAME 2>&1 | tee -a "$LOG_DIR/upgrade.log" &
NEW_PID=$!
echo "[UPGRADE] New process: PID $NEW_PID"

# ── HEALTH WATCHDOG ─────────────────────────────────────────────
TIMEOUT=300
ELAPSED=0
HEALTHY=false

while [ $ELAPSED -lt $TIMEOUT ]; do
    sleep 5
    ELAPSED=$((ELAPSED + 5))

    if ! kill -0 $NEW_PID 2>/dev/null; then
        echo "[UPGRADE] ❌ Process died (PID: $NEW_PID)"
        break
    fi

    if tail -n 50 "$LOG_DIR/upgrade.log" 2>/dev/null | grep -q "WebUI hub listening"; then
        HEALTHY=true
        echo "[UPGRADE] ✅ Healthy in ${ELAPSED}s"
        break
    fi

    echo "[UPGRADE] Waiting for health... (${ELAPSED}s/${TIMEOUT}s)"
done

if [ "$HEALTHY" = false ]; then
    echo "[UPGRADE] ⚠️ Not healthy within ${TIMEOUT}s"

    kill $NEW_PID 2>/dev/null
    sleep 2
    kill -9 $NEW_PID 2>/dev/null

    if [ -f "$BACKUP_BINARY" ]; then
        echo "[UPGRADE] 🔄 ROLLING BACK..."
        cp "$BACKUP_BINARY" "$RELEASE_BINARY"
        rm -f "$ERNOS_DIR/data/resume.json"

        sleep 5
        lsof -i ":$PORT" -t 2>/dev/null | xargs kill -9 2>/dev/null
        sleep 2

        cd "$ERNOS_DIR"
        ./target/release/$BINARY_NAME 2>&1 | tee -a "$LOG_DIR/upgrade.log" &
        echo "[UPGRADE] 🔄 Rollback launched (PID: $!)"

        if command -v osascript &>/dev/null; then
            osascript -e "
            tell application \"Terminal\"
                activate
                do script \"echo '[UPGRADE] ⚠️ ROLLBACK — previous binary restored.' && tail -f '$LOG_DIR/upgrade.log'\"
            end tell
            "
        fi
    else
        echo "[UPGRADE] ❌ No rollback binary — manual intervention required"
    fi
else
    echo "[UPGRADE] ✅ Deployment verified."
    if command -v osascript &>/dev/null; then
        osascript -e "
        tell application \"Terminal\"
            activate
            do script \"echo '[UPGRADE] ✅ Ern-OS upgraded successfully.' && tail -f '$LOG_DIR/upgrade.log'\"
        end tell
        "
    fi
fi

echo "[UPGRADE] Done."
