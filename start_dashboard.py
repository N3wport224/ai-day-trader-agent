#!/usr/bin/env python3
"""
One command to run the trading dashboard and open it in your browser:

    python start_dashboard.py

Then follow the Get Started tab. The dashboard only listens on this computer
(127.0.0.1); keys and live-trading settings can only be changed from here.
Stop it with Ctrl+C. A running bot keeps going on its own; stop it from the
dashboard first if you want it to stop too.
"""

from __future__ import annotations

import os
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from dotenv import load_dotenv  # noqa: E402

from core.env_store import env_path  # noqa: E402

load_dotenv(env_path())


def main() -> int:
    import uvicorn

    port = int(os.getenv("API_PORT", "8000"))
    url = f"http://127.0.0.1:{port}/dashboard"

    def open_browser() -> None:
        for _ in range(50):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/api/health", timeout=1)
                break
            except Exception:
                time.sleep(0.2)
        print(f"\n  Dashboard: {url}\n")
        if os.getenv("NO_BROWSER", "").lower() not in {"1", "true", "yes"}:
            webbrowser.open(url)

    threading.Thread(target=open_browser, daemon=True).start()
    from core.keep_awake import keep_awake

    keep_awake()
    uvicorn.run("config.api.server:app", host="127.0.0.1", port=port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
