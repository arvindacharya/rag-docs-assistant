#!/usr/bin/env bash
# Startup script for Render (or any host with no persistent disk on the
# free tier). Rebuilds both vector databases from scratch every time
# this runs -- not just on first deploy, but every time the service
# wakes up from being asleep, since Render's free web services get a
# fresh filesystem on each restart. Costs ~30-60 extra seconds on
# startup; costs nothing in dollars, since it avoids paying for
# persistent storage entirely.
set -e  # stop immediately if any step fails, rather than starting a
        # server with an empty or partially-built vector store

echo "=== Building FastAPI docs vector store ==="
python fetch_docs.py
python ingest.py

echo "=== Building Node.js docs vector store ==="
python fetch_nodejs_docs.py
python ingest.py --docs_dir data/nodejs_docs --collection nodejs_docs

echo "=== Starting server ==="
# Render assigns a dynamic port via $PORT and expects the app to bind
# to 0.0.0.0, not localhost/127.0.0.1 -- binding to localhost is a
# common first-deploy mistake that makes Render's health check fail
# and the service never come online, with no obvious error message.
uvicorn api:app --host 0.0.0.0 --port "$PORT"
