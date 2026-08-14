"""
Pulls down Node.js's official API documentation (markdown) as the second
corpus for the multi-source router. Uses a sparse, blobless checkout so
we don't have to clone the entire (huge) nodejs/node source repo just to
get the doc/api folder.

Usage:
    python fetch_nodejs_docs.py
"""
import shutil
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/nodejs/node.git"
CLONE_DIR = Path("_nodejs_repo_tmp")
DOCS_SRC = CLONE_DIR / "doc" / "api"
DOCS_DEST = Path("data/nodejs_docs")


def main():
    if DOCS_DEST.exists() and any(DOCS_DEST.glob("**/*.md")):
        print(f"{DOCS_DEST} already has markdown files. Delete it if you want to re-fetch.")
        return

    if CLONE_DIR.exists():
        shutil.rmtree(CLONE_DIR)

    print("Cloning nodejs/node (sparse -- doc/api only, not the whole runtime source)...")
    subprocess.run(
        ["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse", "-q", REPO_URL, str(CLONE_DIR)],
        check=True,
    )
    subprocess.run(["git", "sparse-checkout", "set", "doc/api"], cwd=CLONE_DIR, check=True)

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
