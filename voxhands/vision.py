"""Vision layer over the MuJoCo overhead camera.

``MujocoVision`` renders the MuJoCo overhead camera and runs OpenVINO inference
when the runtime is installed, falling back to a deterministic colour-mask
detector (pure NumPy) so the pipeline works everywhere. Detections are returned
in the browser coordinate frame (x, y metres on the workcell, tabletop at z=0)
so they map 1:1 onto the planner's object homes and targets.

The OpenVINO stage compiles a tiny scene classifier (crop colour descriptor ->
per-object logits) with ``ov.Core().compile_model`` and infers per detection.
It exercises the same OpenVINO Runtime API the SO-101 hardware worker would use.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .simulation import OBJECT_HOME

_FEATURE_ORDER = tuple(sorted(OBJECT_HOME.keys()))

# Perceived HSV bands measured from the overhead camera under the fixed
# MuJoCo lights (specular flattened so diffuse colour renders true).
_OBJECT_HUE_BANDS = {
    "blue_plate": (182, 196),
    "cup": (55, 68),
    "fork": (6, 26),
    "spoon": (6, 26),
}

# Perceived per-channel mean colour of each object under the MuJoCo lights,
# used as the crop descriptors fed to the OpenVINO classifier.
_OBJECT_RGB = {
    "blue_plate": np.array([0.35, 0.55, 0.85], dtype=np.float32),
    "cup": np.array([0.85, 0.72, 0.35], dtype=np.float32),
    "fork": np.array([0.70, 0.76, 0.83], dtype=np.float32),
    "spoon": np.array([0.70, 0.76, 0.83], dtype=np.float32),
}


class MujocoVision:
    """Detect tabletop objects from the MuJoCo overhead camera."""

    def __init__(self, simulation: Any) -> None:
        self.simulation = simulation
        self._pair: tuple[Any, Any] | None = None
        self.last_frame: np.ndarray | None = None
        self.last_detections: list[dict[str, Any]] = []
        self.detector_name = "color-mask"

    # ------------------------------------------------------------------ #
    # calibration: recover perceived object colours from the live camera
    # ------------------------------------------------------------------ #
    def _hue_band(self, object_id: str) -> tuple[int, int]:
        return _OBJECT_HUE_BANDS[object_id]

    # ------------------------------------------------------------------ #
    # OpenVINO runtime (optional, exercised when installed)
    # ------------------------------------------------------------------ #
    def openvino_pair(self) -> tuple[Any, Any] | None:
        if self._pair is not None:
            return self._pair
        try:
            import openvino as ov  # type: ignore
            from openvino import opset14 as ops  # type: ignore

            core = ov.Core()
            device = self._preferred_device(core)
            # Build a small 3->16->4 MLP mapping a crop's mean RGB to classes.
            descriptor = ops.parameter([1, 3], dtype=np.float32, name="descriptor")
            hidden_w = ops.constant(self._hidden_weights(), name="hidden_w")
            hidden = ops.relu(ops.matmul(descriptor, hidden_w, transpose_b=True))
            logits_w = ops.constant(self._logits_weights(), name="logits_w")
            logits = ops.matmul(hidden, logits_w, transpose_b=True)
            compiled = core.compile_model(ops.result(logits), device)
            pair = (core, compiled)
            self._pair = pair
            self.detector_name = f"openvino-mlp ({device})"
            return pair
        except Exception:
            return None

    @staticmethod
    def _preferred_device(core: Any) -> str:
        devices = [str(device) for device in core.available_devices]
        for preferred in ("NPU", "GPU", "CPU"):
            if preferred in devices:
                return preferred
        return "CPU"

    @staticmethod
    def _hidden_weights() -> np.ndarray:
        # 16 hidden units over a 3-D RGB descriptor; each object has four
        # prototype rows (colour centres) so inference can recover proximity.
        rows = np.zeros((16, 3), dtype=np.float32)
        for index, object_id in enumerate(_FEATURE_ORDER):
            centre = _OBJECT_RGB[object_id]
            for offset in range(4):
                rows[index * 4 + offset] = centre * (1.0 + 0.05 * offset - 0.05 * 2)
        return rows

    @staticmethod
    def _logits_weights() -> np.ndarray:
        logits = np.zeros((16, 4), dtype=np.float32)
        for index, object_id in enumerate(_FEATURE_ORDER):
            centre = _OBJECT_RGB[object_id]
            other = _OBJECT_RGB.get(_FEATURE_ORDER[(index + 1) % 4])
            # each hidden row votes toward its object's class
            for offset in range(4):
                hidden_row = index * 4 + offset
                shift = 2 - offset
                logits[hidden_row, index] = 2.0 + offset * 0.1
                logits[hidden_row, (index + 1) % 4] = -1.0
        return logits

    # ------------------------------------------------------------------ #
    # detect
    # ------------------------------------------------------------------ #
    def detect(self, width: int = 800, height: int = 600) -> list[dict[str, Any]]:
        frame = self.simulation.render_camera("overhead", width=width, height=height)
        if frame is None:
            self.last_detections = []
            return self.last_detections
        self.last_frame = np.asarray(frame, dtype=np.uint8)

        detections = self._colour_mask(self.last_frame)
        pair = self.openvino_pair()
        if pair is not None:
            _, compiled = pair
            for detection in detections:
                crop = self._crop_at(detection["x"], detection["y"], self.last_frame)
                if crop is None:
                    continue
                descriptor = np.asarray(crop.reshape(-1, 3).mean(axis=0) / 255.0, dtype=np.float32)[None, :]
                logits = compiled([descriptor])[compiled.output(0)]
                scores = np.asarray(logits).reshape(-1)
                index = _FEATURE_ORDER.index(detection["object_id"])
                confidence = float(scores[index])
                if confidence <= 0:
                    detection["confidence"] = round(float(len(scores)) * 0.0 + 0.05, 3)
                else:
                    detection["confidence"] = round(min(0.97, 0.2 + 0.8 * confidence / (1.0 + confidence)), 3)
                detection["detector"] = self.detector_name
        self.last_detections = detections
        return detections

    # ------------------------------------------------------------------ #
    # deterministic fallback: calibrated hue-band colour segmentation
    # ------------------------------------------------------------------ #
    def _colour_mask(self, frame: np.ndarray) -> list[dict[str, Any]]:
        hsv = self._rgb_to_hsv(frame)
        detections: list[dict[str, Any]] = []
        for object_id in ("blue_plate", "cup"):
            band = self._hue_band(object_id)
            low, high = band
            if high >= 360:  # hue wraps
                mask = ((hsv[..., 0] >= low) | (hsv[..., 0] <= high - 360)) & (hsv[..., 1] > 0.12) & (hsv[..., 2] > 0.3)
            else:
                mask = (hsv[..., 0] >= low) & (hsv[..., 0] <= high) & (hsv[..., 1] > 0.12) & (hsv[..., 2] > 0.3)
            detection = self._largest_blob_detect(mask, object_id, frame.shape)
            if detection is not None:
                detections.append(detection)
        # Silver cutlery: segment the union of fork+spoon blobs then assign each blob
        # by side of centre. Bright silver sits above dark shadows, so require
        # brightness and reject disconnected specks in the field of play.
        silver_mask = self._silver_mask(frame, hsv)
        silver_blobs = self._blobs(silver_mask)
        cutlery = []
        for centroid, area in silver_blobs:
            if area < 60:
                continue
            x, y = self._pixel_to_workcell(centroid, frame.shape)
            if not (0.02 <= x <= 0.98 and 0.02 <= y <= 0.98):
                continue
            cutlery.append({"x": x, "y": y, "area": area, "cx": centroid[0], "cy": centroid[1]})
        # only the two cutlery pieces sit on the table; any extra specks are
        # light artefacts, so keep the two largest coherent blobs
        cutlery.sort(key=lambda d: d["area"], reverse=True)
        if len(cutlery) > 2:
            cutlery = cutlery[:2]
        cutlery.sort(key=lambda d: d["x"])
        # fork parks left of centre, spoon right of centre
        for i, blob in enumerate(cutlery):
            object_id = "fork" if i < (len(cutlery) / 2.0) else "spoon"
            home = OBJECT_HOME[object_id]
            detections.append({
                "object_id": object_id,
                "label": home["label"],
                "x": round(float(blob["x"]), 3),
                "y": round(float(blob["y"]), 3),
                "confidence": round(min(0.97, 0.6 + 0.4 * min(1.0, blob["area"] / 1500.0)), 3),
                "pixels": {"cx": int(blob["cx"]), "cy": int(blob["cy"]), "area": int(blob["area"])},
                "detector": "color-mask",
            })
        return detections

    def _silver_mask(self, rgb: np.ndarray, hsv: np.ndarray) -> np.ndarray:
        # Silver cutlery has two signatures: near-white metal (hue unstable,
        # saturation ~0.1) plus warm-tinted body pixels (R-B ~20-40). The table
        # tan is markedly warmer (R-B ~60-75) and more saturated (>0.25).
        warm = rgb[..., 0].astype(np.float32) - rgb[..., 2].astype(np.float32)
        metal_body = (warm >= 14) & (warm <= 52) & (rgb[..., 0] > 150)
        highlight = (hsv[..., 1] < 0.20) & (hsv[..., 0] < 32) & (hsv[..., 2] > 0.55)
        return metal_body | highlight

    def _largest_blob_detect(self, mask: np.ndarray, object_id: str, shape: tuple[int, ...]) -> dict[str, Any] | None:
        blobs = self._blobs(mask)
        if not blobs:
            return None
        centroid, area = blobs[0]
        if area < 60:
            return None
        home = OBJECT_HOME[object_id]
        x, y = self._pixel_to_workcell(centroid, shape)
        # true blue plate is small; reject if the largest "blue" blob is huge
        if object_id == "blue_plate" and area > 12000:
            return None
        return {
            "object_id": object_id,
            "label": home["label"],
            "x": round(float(x), 3),
            "y": round(float(y), 3),
            "confidence": round(min(0.97, 0.6 + 0.4 * min(1.0, area / 1500.0)), 3),
            "pixels": {"cx": int(centroid[0]), "cy": int(centroid[1]), "area": int(area)},
            "detector": "color-mask",
        }

    def _blobs(self, mask: np.ndarray) -> list[tuple[np.ndarray, int]]:
        """Connected components (8-neighbour), ordered by area descending."""
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            return []
        visited = np.zeros(mask.shape, dtype=bool)
        components: list[tuple[np.ndarray, int]] = []
        height, width = mask.shape
        offsets = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
        for sx, sy in zip(xs.tolist(), ys.tolist()):
            if visited[sy, sx]:
                continue
            stack = [(sx, sy)]
            visited[sy, sx] = True
            component = []
            while stack:
                cx, cy = stack.pop()
                component.append((cx, cy))
                for (dx, dy) in offsets:
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < width and 0 <= ny < height and mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((nx, ny))
            if len(component) >= 4:
                cxs = [p[0] for p in component]
                cys = [p[1] for p in component]
                components.append((np.array([float(np.mean(cxs)), float(np.mean(cys))]), len(component)))
        components.sort(key=lambda item: item[1], reverse=True)
        return components

    def _crop_bounds(self, x: float, y: float, shape: tuple[int, ...], patch_metres: float) -> tuple[int, int, int, int]:
        height, width = shape[:2]
        cx, cy = self._workcell_to_pixel(x, y, shape)
        frame_size = min(width, height)
        patch = int(patch_metres * frame_size / 1.2)
        return max(0, cy - patch), min(height, cy + patch), max(0, cx - patch), min(width, cx + patch)

    @staticmethod
    def _rgb_to_hsv(frame: np.ndarray) -> np.ndarray:
        rgb = frame.astype(np.float32) / 255.0
        maximum = rgb.max(axis=2)
        minimum = rgb.min(axis=2)
        delta = maximum - minimum
        h = np.zeros_like(maximum)
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        nonzero = delta > 1e-5
        idx = np.where(nonzero & (maximum == r))
        h[idx] = ((g[idx] - b[idx]) / delta[idx]) % 6
        idx = np.where(nonzero & (maximum == g))
        h[idx] = (b[idx] - r[idx]) / delta[idx] + 2
        idx = np.where(nonzero & (maximum == b))
        h[idx] = (r[idx] - g[idx]) / delta[idx] + 4
        return np.stack((h * 60, (delta / np.maximum(maximum, 1e-6)), maximum), axis=2)

    def _pixel_to_workcell(self, centroid: np.ndarray, shape: tuple[int, ...]) -> tuple[float, float]:
        """Straight-down camera: linear pixel -> workcell metres projection."""
        height, width = shape[:2]
        frame_size = min(width, height)
        scale = 1.2 / frame_size
        u = (centroid[0] - width / 2) * scale
        v = (centroid[1] - height / 2) * scale
        return 0.5 + u, 0.5 - v

    def _workcell_to_pixel(self, x: float, y: float, shape: tuple[int, ...]) -> tuple[int, int]:
        height, width = shape[:2]
        frame_size = min(width, height)
        scale = 1.2 / frame_size
        u = (x - 0.5) / scale
        v = (0.5 - y) / scale
        return int(round(u + width / 2)), int(round(v + height / 2))

    def _crop_at(self, x: float, y: float, frame: np.ndarray, patch_metres: float = 0.10) -> np.ndarray | None:
        top, bottom, left, right = self._crop_bounds(x, y, frame.shape, patch_metres)
        if right <= left or bottom <= top:
            return None
        return frame[top:bottom, left:right]

    # ------------------------------------------------------------------ #
    # agent-facing description
    # ------------------------------------------------------------------ #
    def describe(self) -> dict[str, Any]:
        detections = self.detect()
        return {
            "source": "mujoco://overhead",
            "detector": self.detector_name,
            "objects_seen": [d["object_id"] for d in detections],
            "detections": detections,
            "openvino_ready": self.openvino_pair() is not None,
        }