from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from voxhands.check_intel import main as check_main


def main() -> int:
    parser = argparse.ArgumentParser(description="Print Intel OpenVINO hardware presence and honesty status")
    parser.parse_args()
    return check_main()


if __name__ == "__main__":
    raise SystemExit(main())