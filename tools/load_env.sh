#!/bin/bash
# .envファイルを読み込むヘルパースクリプト
# 
# Usage:
#   source tools/load_env.sh
#   または
#   . tools/load_env.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${PROJECT_ROOT}/.env"

if [[ -f "$ENV_FILE" ]]; then
    echo "[INFO] Loading environment variables from .env"
    # .envファイルから環境変数を読み込む（コメントと空行をスキップ）
    set -a
    source <(grep -v '^#' "$ENV_FILE" | grep -v '^$' | sed 's/^export //')
    set +a
    echo "[INFO] Environment variables loaded"
else
    echo "[WARN] .env file not found at ${ENV_FILE}"
    echo "[INFO] Copy .env.example to .env and configure it"
    echo "[INFO]   cp .env.example .env"
fi
