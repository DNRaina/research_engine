"""
run_me.py
---------
Production bootstrap and launcher script.
Initializes runtime environment, verifies configurations, runs startup checks,
and launches the Streamlit research interface.
"""

import sys
import subprocess
from dotenv import load_dotenv

load_dotenv()

from config import settings
from logger import app_logger

# Ensure all runtime storage paths are created
settings.ensure_directories()

print("=" * 65)
print(f"  {settings.app_name} v{settings.app_version} — Startup Sequence")
print("=" * 65)

# Diagnostic CLI flag support
if "--check" in sys.argv or "--diagnostics" in sys.argv:
    from health import run_diagnostics
    sys.exit(0 if run_diagnostics() else 1)

try:
    from knowledge_base import build_knowledge_base_if_needed
    from retriever import build_memory_index

    kb_status = build_knowledge_base_if_needed(
        books_dir=settings.books_dir,
        scans_dir=settings.scans_dir,
    )
    print(f"Knowledge Base Index : {kb_status}")

    mem_status = build_memory_index(
        folder_path=settings.past_chats_dir,
        scans_dir=settings.scans_dir,
    )
    print(f"Memory Index Status  : {mem_status}")

except Exception as exc:
    app_logger.warning(f"Startup index build error: {exc}")

print("=" * 65)
print("Launching Streamlit Application...")
print("=" * 65)

cmd = [
    sys.executable,
    "-m",
    "streamlit",
    "run",
    "app.py",
    "--browser.gatherUsageStats=false",
    "--server.fileWatcherType=none",
    *sys.argv[1:],
]

try:
    subprocess.run(cmd, check=True)
except KeyboardInterrupt:
    print("\n[JANE] Application stopped by user. Goodbye!")
except Exception as exc:
    app_logger.error(f"Failed to launch Streamlit application: {exc}")
    sys.exit(1)