"""Tiny image encoders for the camera feed.

Uses Pillow when installed (JPEG), and falls back to an uncompressed BMP
written with the standard library so the pipeline has no hard dependencies.
"""

from __future__ import annotations

import struct
from typing import Any

import numpy as np


def encode_jpeg(frame: np.ndarray, quality: int = 82) -> bytes:
    """Encode an HxWx3 uint8 frame as JPEG.

    Raises ``ImportError`` (from the caller's perspective) when Pillow is not
    installed; callers should fall back to :func:`encode_bmp`.
    """
    from PIL import Image  # type: ignore  # (optional dependency)

    image = Image.fromarray(np.asarray(frame, dtype=np.uint8), mode="RGB")
    buffer = __import__("io").BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    return buffer.getvalue()


def encode_bmp(frame: np.ndarray) -> bytes:
    """Encode an HxWx3 uint8 frame as an uncompressed 24-bit BMP (stdlib)."""
    frame = np.asarray(frame, dtype=np.uint8)
    height, width = frame.shape[:2]
    pixels = frame[:, :, ::-1]  # BGR order for BMP
    row_size = width * 3
    padding = (4 - row_size % 4) % 4
    stride = row_size + padding
    data_size = stride * height
    file_size = 14 + 40 + data_size
    header = struct.pack("<2sIHHI", b"BM", file_size, 0, 0, 14 + 40)
    info = struct.pack("<IiiHHIIiiII", 40, width, height, 1, 24, 0, data_size, 2835, 2835, 0, 0)
    rows = []
    for y in range(height - 1, -1, -1):
        rows.append(pixels[y].tobytes())
        rows.append(b"\x00" * padding)
    return header + info + b"".join(rows)