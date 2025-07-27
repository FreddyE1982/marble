#!/usr/bin/env python
"""Simple cross-platform installer for MARBLE.

Creates a virtual environment in ``env`` and installs all
requirements. Run with ``python install.py``. Activate the environment
using ``source env/bin/activate`` on Unix or ``env\\Scripts\\activate`` on
Windows.
"""

import os
import subprocess
import sys
import venv

ENV_DIR = "env"


def main() -> None:
    if not os.path.isdir(ENV_DIR):
        print("Creating virtual environment...")
        venv.EnvBuilder(with_pip=True).create(ENV_DIR)
    pip = os.path.join(ENV_DIR, "bin", "pip") if os.name != "nt" else os.path.join(ENV_DIR, "Scripts", "pip.exe")
    print("Installing requirements...")
    subprocess.check_call([pip, "install", "-r", "requirements.txt"])
    print("Installation complete. Activate the environment and enjoy MARBLE!")


if __name__ == "__main__":
    main()
