#!/usr/bin/env bash
set -euo pipefail

echo "Installing uv..."
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

echo "Fixing .venv volume ownership..."
sudo chown -R vscode:vscode /workspace/.venv

echo "Syncing dependencies (dev + full CPU ML stack: diarize, align, local)..."
cd /workspace
uv sync --group dev --extra diarize --extra align --extra local --extra cpu --reinstall-package tolka

echo ""
echo "Ready. Try:"
echo "  uv run pytest"
echo "  TOLKA_API_TOKENS=dev TOLKA_ENGINE=fake uv run uvicorn tolka.main:create_app --factory --host 0.0.0.0"
