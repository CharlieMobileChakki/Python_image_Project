#!/bin/bash
# =============================================================================
# Arangtik RTX 3090 GPU Worker & Tunnel Auto-Healer Watchdog
# Keeps Uvicorn and Autossh reverse tunnel alive 24/7 with zero downtime
# =============================================================================

CONTABO_IP="157.173.124.137"
PYTHON_BIN="/opt/conda/envs/qwen21/bin/python3.11"
WORKER_DIR="/root/Python_image_Project/gpu_worker"

echo "[$(date)] Arangtik GPU Watchdog Active..." >> /root/watchdog.log

while true; do
    # 1. Check if Uvicorn (Port 8001) is running
    if ! pgrep -f "uvicorn server:app" > /dev/null; then
        echo "[$(date)] [AUTO-HEAL] Uvicorn is DOWN! Auto-restarting now..." >> /root/watchdog.log
        cd "$WORKER_DIR" || cd /root
        nohup "$PYTHON_BIN" -m uvicorn server:app --host 0.0.0.0 --port 8001 >> /root/uvicorn_gpu.log 2>&1 &
        sleep 5
    fi

    # 2. Check if Autossh reverse tunnel is running
    if ! pgrep -f "autossh.*8001:127.0.0.1:8001" > /dev/null; then
        echo "[$(date)] [AUTO-HEAL] Autossh tunnel is DOWN! Auto-restarting now..." >> /root/watchdog.log
        pkill -f autossh 2>/dev/null
        nohup autossh -M 0 \
            -o "ServerAliveInterval 15" \
            -o "ServerAliveCountMax 3" \
            -o "StrictHostKeyChecking=no" \
            -o "ExitOnForwardFailure=yes" \
            -N -R 8001:127.0.0.1:8001 root@"$CONTABO_IP" >> /root/autossh.log 2>&1 &
        sleep 3
    fi

    sleep 8
done
