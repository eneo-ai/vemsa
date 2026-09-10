#!/usr/bin/env bash
set -euo pipefail

echo "Installing uv..."
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

echo "Fixing .venv volume ownership..."
sudo chown -R vscode:vscode /workspace/.venv

echo "Syncing dependencies (dev + full CPU ML stack: diarize, align, local)..."
cd /workspace
uv sync --group dev --extra diarize --extra align --extra local --extra cpu --reinstall-package vemsa

echo "Building the operator dashboard (ui/ -> src/vemsa/ops/static)..."
npm ci --prefix ui --no-audit --no-fund
npm run --prefix ui build

echo ""
echo "Ready. Try:"
echo "  uv run pytest   # includes postgres store tests against the compose postgres"
echo "  uv run start    # API + in-process worker: postgres database, fake engine, token 'dev'"
echo "  open http://localhost:8000/ops         # operator dashboard (rebuild: npm run --prefix ui build)"
echo "  npm run --prefix ui dev                # dashboard with hot reload on :5173, proxied to :8000"
