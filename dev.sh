#!/usr/bin/env bash
# Start the ML service locally with the settings from .env.local.sh:  bash dev.sh
set -e
cd "$(dirname "$0")"
node ../scripts/local-setup.mjs   # regenerate from the root .env every start, so .env is the only file you edit
source .env.local.sh
exec python -m uvicorn app.main:app --port 7860
