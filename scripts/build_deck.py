#!/usr/bin/env python3
"""Build the VoxHands slide deck PDF and cover image.

The slide artwork lives in the Remotion project (``video/src/Deck.tsx``) so the
deck and the demo video share one visual language. This script renders the
slides to PNG (``video/scripts/render-deck.mjs``) and assembles them into a
16:9 PDF, plus the standalone cover image the hackathon form asks for.

    python3.13 scripts/build_deck.py
    python3.13 scripts/build_deck.py --skip-render   # reuse existing stills
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VIDEO = ROOT / "video"
SUBMISSION = ROOT / "submission"
RENDER_SCRIPT = VIDEO / "scripts" / "render-deck.mjs"

# 1920x1080 pixels at 144 dpi is a 960x540 pt page: exactly 16:9, matching the
# deck that was already produced for the project page.
PDF_DPI = 144


def render_stills(out_dir: Path) -> list[Path]:
    subprocess.run(["node", str(RENDER_SCRIPT), str(out_dir)], cwd=str(VIDEO), check=True)
    return sorted(out_dir.glob("slide-*.png"))


def assemble(frames: list[Path], pdf_path: Path, cover_path: Path) -> None:
    from PIL import Image

    if not frames:
        raise SystemExit("No slide PNGs found; run without --skip-render first.")

    pages = [Image.open(frame).convert("RGB") for frame in frames]
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    # First page is the cover: reuse it verbatim as the 16:9 cover image.
    pages[0].save(cover_path, "PNG")
    pages[0].save(
        pdf_path,
        "PDF",
        save_all=True,
        append_images=pages[1:],
        resolution=PDF_DPI,
    )
    print(f"wrote {pdf_path.relative_to(ROOT)} ({len(pages)} pages, {pdf_path.stat().st_size // 1024} KB)")
    print(f"wrote {cover_path.relative_to(ROOT)}")


def copy_demo_shots() -> None:
    source = VIDEO / "public" / "demo"
    if not source.is_dir():
        print("note: no video/public/demo screenshots; run video/scripts/capture-demo.mjs")
        return
    target = SUBMISSION / "demo"
    target.mkdir(parents=True, exist_ok=True)
    for shot in sorted(source.glob("*.png")):
        shutil.copy2(shot, target / shot.name)
        print(f"wrote {(target / shot.name).relative_to(ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-render", action="store_true", help="reuse stills in the work directory")
    parser.add_argument("--pdf", type=Path, default=SUBMISSION / "VoxHands-deck.pdf")
    parser.add_argument("--cover", type=Path, default=SUBMISSION / "cover.png")
    args = parser.parse_args()

    if args.skip_render:
        workdir = Path(tempfile.gettempdir()) / "voxhands-deck"
        frames = sorted(workdir.glob("slide-*.png"))
        if not frames:
            raise SystemExit(f"--skip-render requested but {workdir} has no slide PNGs")
    else:
        workdir = Path(tempfile.gettempdir()) / "voxhands-deck"
        workdir.mkdir(parents=True, exist_ok=True)
        for stale in workdir.glob("slide-*.png"):
            stale.unlink()
        frames = render_stills(workdir)

    assemble(frames, args.pdf, args.cover)
    copy_demo_shots()


if __name__ == "__main__":
    main()
