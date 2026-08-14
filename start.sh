#!/usr/bin/env bash
# Startup script for Render (or any host with limited memory).
#
# Prefers a pre-built vector database committed to the repo
# (data/chroma_db/) over rebuilding it live on every boot. This exists
# because live rebuilding -- embedding the whole corpus fresh on every
# restart -- was OOM-killing the deploy on Render's free tier (512MB
# RAM). A pre-built database sidesteps that: the heavy embedding
# computation happens once, locally, where it already works reliably,
# and Render just loads the (small) result.
#
# Falls back to rebuilding from scratch if no pre-built database is
# found, so this still works on a fresh clone that hasn't committed
# one yet -- this fallback still builds BOTH collections (FastAPI and
# Node.js), even while the currently-committed database is
# FastAPI-only, so it stays correct once Node.js gets added back.
#
# To update the deployed corpus later: re-run ingest.py locally,
# re-commit data/chroma_db/, push.
set -e

if [ -d "data/chroma_db" ] && [ -n "$(ls -A data/chroma_db 2>/dev/null)" ]; then
    echo "=== Using pre-built vector store committed to the repo ==="
else
    echo "=== No pre-built vector store found -- building from scratch ==="
    echo "=== Building FastAPI docs vector store ==="
    python fetch_docs.py
    python ingest.py

    echo "=== Building Node.js docs vector store ==="
    python fetch_nodejs_docs.py
    python ingest.py --docs_dir data/nodejs_docs --collection nodejs_docs
fi

echo "=== Starting server ==="
uvicorn api:app --host 0.0.0.0 --port "$PORT"