#!/bin/bash
# Song Generator - Startup Script
# Launches the YuE Song Generator web application

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PORT=${PORT:-8080}
CONDA_PYTHON="/home/usnmp/miniforge3/envs/yue/bin/python"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                 🎵 Song Generator - YuE 🎵                   ║"
echo "║                                                              ║"
echo "║  Starting on: http://localhost:${PORT}                         ║"
echo "║  Press Ctrl+C to stop                                       ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

"${CONDA_PYTHON}" app.py
