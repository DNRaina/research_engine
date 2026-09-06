"""
run_me.py
---------
Bootstrap script. Creates required directories, pre-builds the knowledge base
index from books/ into scans/, then launches the Streamlit app.
"""

import os
import subprocess
import sys

from dotenv import load_dotenv

load_dotenv()

# Ensure required directories exist
for folder in ["past_chats", "books", "scans", "past_sessions"]:
    os.makedirs(folder, exist_ok=True)

# Pre-build knowledge base index (skipped if already up-to-date)
print("=" * 60)
print("Research Agent — startup checks")
print("=" * 60)

try:
    from knowledge_base import build_knowledge_base_if_needed
    status = build_knowledge_base_if_needed(
        books_dir="books",
        scans_dir="scans",
    )
    print(f"Knowledge base status: {status}")
except Exception as e:
    print(f"Knowledge base build skipped: {e}")

print("=" * 60)
print("Launching Streamlit app...")
print("=" * 60)

# Launch the Streamlit UI
subprocess.run(
    [sys.executable, "-m", "streamlit", "run", "app.py"],
    check=True,
)