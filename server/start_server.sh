#!/usr/bin/env bash

set -euo pipefail

# 切到项目根目录（假设本脚本位于 server/ 下）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

echo "Starting Broker knob server ..."
exec python -m server.server

