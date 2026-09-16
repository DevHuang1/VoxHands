"""Camera-based object detection for the dinner-table workcell.

Numpy-only colour/geometry detector operating on the overhead MuJoCo camera
frame.  It maps pixel detections into workcell coordinates (metres) with the
same 1.2 m / min-side mapping used by ``voxhands.vision``.

The detector is lighting-adaptive: the per-band saturation/value thresholds
are scaled by the frame's mean luminance so that darker renderings (dim
table lamps, shadows under randomised lights) still produce detections.

Objects are discriminated by hue band plus simple geometry statistics
(ellipticity from the blob bbox; nothing here depends on OpenCV):

========================  ===========  ==============
object                    mask band     separator
========================  ===========  ==============
plate                     blue         round, widest
cup                       amber        round, medium
bottle                    green        narrow bbox (elongated)
fork / spoon              silver       k-means on x → two blobs
========================  ===========  ==============

Because each colour band contains at most one object (one blue plate, one
amber mug, one green bottle) a centroid is safe for those.  Fork and spoon
share the near-white silver band and are separated by a 2-way k-means split
on the x coordinate; the two resulting blobs are assigned by left/right
colour tint (warm silver -> spoon, cool silver -> fork) with graceful
fallback to x order.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# workcell <-> pixel mapping: camera at (0.5, 0.5, H) looks straight down,
# fovy 55 degrees; the visible region is ~1.2 m wide at the tabletop.
VISIBLE_M = 1.2

# HSV ranges (OpenCV convention: hue 0-179, sat/value 0-255).  Calibrated
# against the actual rendered dining scene (distinct metallic tints for the
# silverware: cool steel fork, brass spoon).
HSV_RANGES: dict[str, tuple[tuple[int, int], tuple[int, int], tuple[int, int]]] = {
    "blue": ((90, 130), (130, 255), (40, 255)),
    "amber": ((10, 45), (95, 255), (40, 255)),
    "green": ((45, 90), (75, 255), (40, 255)),
    "fork": ((85, 135), (60, 150), (50, 255)),   # desaturated steel blue
    "spoon": ((8, 38), (60, 255), (50, 255)),    # brass
}

# per-object region priors (workcell metres).  Keeps, e.g., the brown drawer
# tray out of the amber mug band, the grey arm silhouettes out of the
# cutlery bands, and separates the mug (right half) from the brass spoon.
REGIONS: dict[str, dict[str, tuple[float, float]]] = {
    "plate": {"x": (0.10, 0.95), "y": (0.42, 0.88)},
    "cup": {"x": (0.58, 0.95), "y": (0.40, 0.85)},
    "bottle": {"x": (0.10, 0.95), "y": (0.32, 0.75)},
    "fork": {"x": (0.15, 0.95), "y": (0.50, 0.95)},
    "spoon": {"x": (0.10, 0.58), "y": (0.50, 0.95)},
}

# darken/lighten adaptation range for value thresholds
V_MIN_ADAPT = (45, 70)
S_ADAPT = (70, 130)

MIN_BLOB_AREA = 60
MAX_BLOB_AREA = 12000

# warm silver = spoon (more yellow/red), cool silver = fork (more blue-ish)
SILVER_WARM_MIN = 0.0  # -ish: we compare hues around blue boundary


def _to_hsv(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Frame (..HxW..x3 uint8) -> (hue[0-179], sat, value as 2D float32)."""
    r = frame[..., 0].astype(np.float32) / 255.0
    g = frame[..., 1].astype(np.float32) / 255.0
    b = frame[..., 2].astype(np.float32) / 255.0
    maximum = np.maximum(np.maximum(r, g), b)
    minimum = np.minimum(np.minimum(r, g), b)
    chroma = maximum - minimum
    safechroma = np.where(chroma > 0.0, chroma, 1.0)

    hue = np.zeros_like(maximum)
    m = maximum == r
    hue = np.where(m, np.mod(60.0 * ((g - b) / safechroma), 360.0), hue)
    m = maximum == g
    hue = np.where(m, 60.0 * ((b - r) / safechroma) + 120.0, hue)
    m = maximum == b
    hue = np.where(m, 60.0 * ((r - g) / safechroma) + 240.0, hue)
    hue = (hue / 2.0).astype(np.float32)  # 0-179

    safe_max = np.where(maximum > 0.0, maximum, 1.0)
    sat = (np.where(maximum > 0.0, chroma / safe_max, 0.0) * 255.0)
    value = maximum * 255.0
    return hue, sat.astype(np.float32), value.astype(np.float32)


def _pixel_to_workcell(px: float, py: float, width: int, height: int) -> tuple[float, float]:
    scale = VISIBLE_M / min(width, height)
    return (0.5 + (px - width / 2.0) * scale, 0.5 - (py - height / 2.0) * scale)


def _blob_stats(mask: np.ndarray) -> dict[str, Any] | None:
    ys, xs = np.nonzero(mask)
    if xs.size < MIN_BLOB_AREA:
        return None
    if xs.size > MAX_BLOB_AREA:
        return None
    return {
        "cx": float(xs.mean()),
        "cy": float(ys.mean()),
        "area": int(xs.size),
        "x0": int(xs.min()),
        "x1": int(xs.max()),
        "y0": int(ys.min()),
        "y1": int(ys.max()),
    }


def _ellipticity(blob: dict[str, Any]) -> float:
    width = max(1.0, blob["x1"] - blob["x0"])
    height = max(1.0, blob["y1"] - blob["y0"])
    return min(width, height) / max(width, height)  # 1 = round, ~0 = elongated


def detect_frame(frame: np.ndarray) -> list[dict[str, Any]]:
    """Detect the five dinner-table objects in one overhead RGB frame.

    Returns a list of dicts (one per detected object):

    .. code-block:: python

        {"object_id": "plate", "label": "plate", "x": 0.512, "y": 0.43,
         "confidence": 0.86, "pixels": {"area": 900, ...}, "detector": "color_mask"}
    """
    if frame is None or frame.size == 0:
        return []
    hue, sat, value = _to_hsv(np.asarray(frame, dtype=np.uint8))
    height, width = frame.shape[:2]
    scale = VISIBLE_M / min(width, height)
    # pixel row for a workcell y (inverted image axis)
    py_of = lambda wy: int(round((0.5 - wy) / scale + height / 2))
    px_of = lambda wx: int(round((wx - 0.5) / scale + width / 2))

    mean_luminance = float(value.mean())
    adapt = (mean_luminance - 60.0) / 150.0
    adapt = max(-1.0, min(1.0, adapt))
    v_min = V_MIN_ADAPT[0] + (V_MIN_ADAPT[1] - V_MIN_ADAPT[0]) * ((adapt + 1.0) / 2.0)
    s_min = S_ADAPT[0] - adapt * 30.0

    def band(name: str) -> np.ndarray:
        h_lo, h_hi = HSV_RANGES[name][0]
        s_lo, s_hi = HSV_RANGES[name][1]
        v_lo, v_hi = HSV_RANGES[name][2]
        return (
            (hue >= h_lo)
            & (hue <= h_hi)
            & (sat >= max(s_lo, s_min))
            & (sat <= s_hi)
            & (value >= v_min)
            & (value <= v_hi)
        )

    def region_stats(oid: str, mask: np.ndarray) -> tuple[tuple[float, float], dict[str, Any]] | None:
        """(centroid, blob) of an oid band, or None if unusable / outside prior.

        The band mask is cropped to the object's rectangular region priors
        *first*, so large same-colour table/background areas outside the
        object's expected footprint never dominate the blob statistics.
        """
        reg = REGIONS.get(oid)
        crop = mask
        if reg is not None:
            x0p = max(0, int(px_of(reg["x"][0])))
            x1p = min(width, int(px_of(reg["x"][1])))
            y0p = max(0, py_of(reg["y"][1]))
            y1p = min(height, py_of(reg["y"][0]))
            if x1p <= x0p or y1p <= y0p:
                return None
            crop = mask[y0p:y1p, x0p:x1p]
        blob = _blob_stats(crop)
        if blob is None:
            return None
        cx, cy = blob["cx"], blob["cy"]
        if reg is not None:
            cx += x0p
            cy += y0p
        x, y = _pixel_to_workcell(cx, cy, width, height)
        return (x, y), blob

    detections: list[dict[str, Any]] = []

    for oid, key in (("plate", "blue"), ("cup", "amber"), ("bottle", "green")):
        found = region_stats(oid, band(key))
        if found is None:
            continue
        (x, y), blob = found
        if oid == "plate" and _ellipticity(blob) < 0.6:
            continue
        confidence = float(min(1.0, 0.55 + blob["area"] / 4000.0))
        detections.append({"object_id": oid, "label": _LABELS[oid], "x": round(x, 4), "y": round(y, 4), "confidence": round(confidence, 3), "pixels": blob, "detector": "color_mask"})

    # fork and spoon: distinct metallic tints, both in the place-setting row
    for oid, key in (("fork", "fork"), ("spoon", "spoon")):
        found = region_stats(oid, band(key))
        if found is None:
            continue
        (x, y), blob = found
        if _ellipticity(blob) > 0.88:
            continue
        confidence = float(min(1.0, 0.5 + blob["area"] / 3500.0))
        detections.append({"object_id": oid, "label": _LABELS[oid], "x": round(x, 4), "y": round(y, 4), "confidence": round(confidence, 3), "pixels": blob, "detector": "color_mask"})

    detections.sort(key=lambda d: d["object_id"])
    return detections


_LABELS = {"plate": "plate", "cup": "cup", "bottle": "bottle", "fork": "fork", "spoon": "spoon"}


def match_objects(detections: list[dict[str, Any]]) -> dict[str, tuple[float, float] | None]:
    """Project detections onto the registered object keys."""
    out: dict[str, tuple[float, float] | None] = {oid: None for oid in _LABELS}
    for det in detections:
        if det["object_id"] in out:
            out[det["object_id"]] = (det["x"], det["y"])
    return out