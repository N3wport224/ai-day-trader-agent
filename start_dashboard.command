#!/bin/bash
# AI Day Trader - double-click to start the dashboard on macOS.
# First run creates a private Python environment in .venv and installs packages.
cd "$(dirname "$0")" || exit 1

pause_exit() { echo; read -r -p "Press Return to close this window." _; exit "${1:-1}"; }

if ! command -v python3 >/dev/null 2>&1; then
  echo "Python is not installed. Install Python 3.12 from https://www.python.org/downloads/macos/ and try again."
  pause_exit 1
fi

if [ ! -x .venv/bin/python ]; then
  if ! python3 -c 'import sys; sys.exit(sys.version_info < (3, 10))'; then
    echo "Your python3 is $(python3 --version 2>&1); 3.10 or newer is needed."
    echo "Install Python 3.12 from https://www.python.org/downloads/macos/ and try again."
    pause_exit 1
  fi
  echo "First run: setting up. This takes a few minutes..."
  python3 -m venv .venv \
    && .venv/bin/python -m pip install --upgrade pip \
    && .venv/bin/python -m pip install -r requirements.txt \
    || { echo "Setup failed - see the messages above. Delete the .venv folder and try again."; pause_exit 1; }
fi

if ! .venv/bin/python -c 'import lightgbm' >/dev/null 2>&1; then
  echo "The ML library (LightGBM) needs OpenMP on macOS. Install Homebrew from https://brew.sh,"
  echo "then run:  brew install libomp   and double-click this file again."
  pause_exit 1
fi

echo "Starting the dashboard. Keep this window open while you trade; close it to stop the dashboard."
.venv/bin/python start_dashboard.py
pause_exit 0
