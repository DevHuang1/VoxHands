from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from voxhands.imageio import encode_bmp


def _camera_reader(use_mujoco: bool) -> Any | None:
    try:
        if use_mujoco:
            from voxhands.assets.scene_dinner import DinnerConfig
            from voxhands.mujoco_workcell import DinnerTableWorkcell

            env = DinnerTableWorkcell(DinnerConfig(seed=0), oracle=True)

            def mujoco_reader(width: int = 640, height: int = 480):
                frame = env.render(camera="overhead", width=width, height=height)
                if frame is None:
                    return None, None
                return np.asarray(frame, dtype=np.uint8), None

            return mujoco_reader
        import cv2  # type: ignore

        capture = cv2.VideoCapture(0)
        if not capture.isOpened():
            capture.release()
            return None

        def cv2_reader(width: int = 640, height: int = 480):
            ok, frame = capture.read()
            if not ok:
                return None, None
            return np.asarray(frame, dtype=np.uint8), None

        return cv2_reader
    except Exception:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Record RGB-D demo frames over the overhead camera")
    parser.add_argument("--out", type=Path, default=Path("data/demos"))
    parser.add_argument("--frames", type=int, default=1)
    parser.add_argument("--mujoco", action="store_true", help="render the MuJoCo overhead camera instead of a webcam")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    reader = _camera_reader(use_mujoco=args.mujoco)
    if reader is None:
        placeholder = args.out / "camera_unavailable.txt"
        placeholder.write_text(
            json.dumps(
                {
                    "status": "camera_unavailable",
                    "note": "No camera device opened; placeholder directory created instead.",
                    "out_dir": str(args.out),
                },
                indent=2,
            )
        )
        print(f"No camera available; wrote placeholder {placeholder}")
        return 0

    saved = 0
    for i in range(args.frames):
        frame, depth = reader()
        if frame is None:
            continue
        (args.out / f"rgb_{i:03d}.bmp").write_bytes(encode_bmp(frame))
        if depth is not None:
            np.save(args.out / f"depth_{i:03d}.npy", np.asarray(depth, dtype=np.float32))
        saved += 1
    print(f"Recorded {saved}/{args.frames} RGB frames to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())