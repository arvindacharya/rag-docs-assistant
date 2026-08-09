"""
Pulls down the English FastAPI documentation (markdown) as the example corpus.
Swap this out for your own docs later -- ingest.py just needs a folder of .md files.

Usage:
    python fetch_docs.py
"""
import shutil
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/tiangolo/fastapi.git"
CLONE_DIR = Path("_fastapi_repo_tmp")
DOCS_SRC = CLONE_DIR / "docs" / "en" / "docs"
DOCS_DEST = Path("data/docs")


def main():
    if DOCS_DEST.exists() and any(DOCS_DEST.glob("**/*.md")):
        print(f"{DOCS_DEST} already has markdown files. Delete it if you want to re-fetch.")
        return

    if CLONE_DIR.exists():
        shutil.rmtree(CLONE_DIR)

    print("Cloning FastAPI repo (shallow, docs only take a moment)...")
    subprocess.run(
        ["git", "clone", "--depth", "1", REPO_URL, str(CLONE_DIR)],
        check=True,
    )

    if not DOCS_SRC.exists():
        print(f"Could not find {DOCS_SRC} -- the docs folder layout may have changed upstream.")
        sys.exit(1)

    DOCS_DEST.mkdir(parents=True, exist_ok=True)
    md_files = list(DOCS_SRC.glob("**/*.md"))
    for f in md_files:
        rel = f.relative_to(DOCS_SRC)
        dest = DOCS_DEST / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(f, dest)

    shutil.rmtree(CLONE_DIR)
    print(f"Copied {len(md_files)} markdown files into {DOCS_DEST}")


if __name__ == "__main__":
    main()
