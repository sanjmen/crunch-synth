#!/usr/bin/env python3
"""
CLI Utility to submit production code to CrunchDAO Synth.
Enforces minimal packaging rules: only main.py and requirements.txt (plus resources if existing).
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description="Submit minimal codebase to CrunchDAO Synth")
    parser.add_argument("-m", "--message", required=True, help="Submission commit message")
    parser.add_argument("--dry", action="store_true", help="Perform a dry run without uploading")
    args = parser.parse_args()

    os.chdir(REPO_ROOT)

    # 1. Validate requirements.txt contains crunch-synth
    req_file = REPO_ROOT / "requirements.txt"
    if not req_file.exists():
        print("ERROR: requirements.txt not found!")
        sys.exit(1)

    content = req_file.read_text()
    if "crunch-synth" not in content:
        print("ERROR: crunch-synth must be in requirements.txt!")
        sys.exit(1)

    # 2. Validate main.py exists
    main_file = REPO_ROOT / "main.py"
    if not main_file.exists():
        print("ERROR: main.py not found!")
        sys.exit(1)

    cmd = ["crunch", "push", "-m", args.message]
    if args.dry:
        cmd.append("--dry")

    print("=" * 60)
    print("CrunchDAO Synth Submission Packaging")
    print(f"Message: {args.message}")
    print(f"Target Project: necessary-marsupial (synth)")
    print(f"Dry Run: {args.dry}")
    print("=" * 60)

    res = subprocess.run(cmd)
    sys.exit(res.returncode)


if __name__ == "__main__":
    main()
