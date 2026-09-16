from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

REQUIRED_STATUS_KEYS = ("installed", "devices", "device_used", "verified", "message")


def _import_openvino():
    import openvino as ov

    return ov


def _activation(ops: Any, node: Any, name: str) -> Any:
    if name == "tanh":
        return ops.tanh(node)
    if name == "sigmoid":
        return ops.sigmoid(node)
    return ops.relu(node)


def build_openvino_model(ops: Any, model_weights: dict[str, Any]) -> tuple[Any, Any]:
    """Build an ``openvino.Model`` graph for the MLP weights.

    Returns ``(result, input_parameter)`` so callers can wrap them with
    ``ov.Model(result, [input_parameter], name)``.  Kept separate so it can be
    unit tested against a real ``openvino.Model`` without file IO.
    """
    weights = [np.asarray(w, dtype=np.float32) for w in model_weights["weights"]]
    biases = [np.asarray(b, dtype=np.float32) for b in model_weights["biases"]]
    activation = str(model_weights.get("activation", "relu"))
    input_dim = int(weights[0].shape[0])

    x = ops.parameter([1, input_dim], dtype=np.float32, name="descriptor")
    node = x
    for index, (weight, bias) in enumerate(zip(weights, biases)):
        dense = ops.add(
            ops.matmul(node, ops.constant(weight, name=f"w{index}"), False, False),
            ops.constant(bias, name=f"b{index}"),
        )
        if index < len(weights) - 1:
            node = _activation(ops, dense, activation)
        else:
            node = dense
    return ops.result(node, name="logits"), x


class OpenVINOAdapter:
    """Honest Intel OpenVINO readiness adapter.

    Two export paths are offered:

    * :meth:`to_ir` builds the MLP graph with ``openvino.opset14`` and saves a
      genuine OpenVINO IR (``.xml`` + ``.bin``) via ``openvino.save_model``.
      This is the artifact ``ov.Core().compile_model`` loads and the CPU
      benchmark / optional NNCF INT8 quantization consume.
    * :meth:`convert` remains a numpy-only fallback that stores the weights as
      a ``.npz`` plus a JSON manifest; :meth:`run` executes a pure-numpy forward
      pass with no device involvement.  It does *not* emit an IR.

    GPU/NPU/INT8 capability is gated behind a real device-exposure check:
    ``ov.Core().available_devices``. When only ``CPU`` is exposed (Apple
    hardware, no Intel GPU/NPU), ``status()`` returns ``verified=False``,
    ``device_used="CPU"``, and no GPU/NPU benchmark numbers are fabricated.
    INT8 post-training quantization is still real on CPU and is reported as
    such.

    ``model_weights`` follows the voxhands.policy dict convention: a mapping
    with ``input_dim``, ``hidden_dims``, ``output_dim``, ``activation`` and
    ``weights`` / ``biases`` lists of numpy arrays (the on-disk JSON produced
    by ``voxhands.policy.MLPPolicy.save``, decoded through
    ``load_policy_weights``).
    """

    def __init__(self, model_weights: dict[str, Any] | None = None) -> None:
        self.model_weights = model_weights
        self.artifact_dir: Path | None = None

    def available(self) -> bool:
        try:
            _import_openvino()
            return True
        except Exception:
            return False

    def status(self) -> dict[str, Any]:
        try:
            ov = _import_openvino()
        except Exception as exc:
            return {
                "installed": False,
                "devices": [],
                "device_used": "none",
                "verified": False,
                "message": f"openvino is not importable on this host: {exc}",
            }
        devices = [str(device) for device in ov.Core().available_devices]
        accelerators = [device for device in devices if device == "GPU" or device.startswith("GPU") or device == "NPU"]
        if not accelerators:
            return {
                "installed": True,
                "devices": devices,
                "device_used": "CPU",
                "verified": False,
                "message": f"openvino {getattr(ov, '__version__', 'installed')} is installed but only CPU is exposed ({devices}); GPU/NPU acceleration is blocked/unverified on this host (INT8 post-training quantization still runs on CPU).",
            }
        device_used = accelerators[0]
        return {
            "installed": True,
            "devices": devices,
            "device_used": device_used,
            "verified": True,
            "message": f"openvino {getattr(ov, '__version__', 'installed')} is installed and reports {devices}; acceleration verified on {device_used}.",
        }

    def convert(self, model_weights: dict[str, Any], model_dir: Path) -> Path:
        """Produce an ONNX-free numeric artifact from raw numpy weights.

        IMPORTANT (honesty note): this does NOT emit a real OpenVINO IR
        ``.xml``/``.bin``. A genuine IR export requires building the compute
        graph as an ``openvino.Model`` (via ``opset`` or a framework exporter).
        This method instead writes ``weights.npz`` (the numpy layer arrays) and
        ``manifest.json`` (layer-shape metadata), which is a faithful
        serialisation of the same numpy MLP the policy actually runs, and
        :meth:`run` mirrors the exact forward pass the policy uses.
        """
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        weights = [np.asarray(w, dtype=np.float32) for w in model_weights["weights"]]
        biases = [np.asarray(b, dtype=np.float32) for b in model_weights["biases"]]
        sizes = [int(weights[0].shape[0])] + [int(w.shape[1]) for w in weights]

        arrays: dict[str, np.ndarray] = {}
        for i, (w, b) in enumerate(zip(weights, biases)):
            arrays[f"w{i}"] = w
            arrays[f"b{i}"] = b
        np.savez(model_dir / "weights.npz", **arrays)

        manifest = {
            "format": "numpy-mlp (ONNX-free; not OpenVINO IR)",
            "input_dim": int(sizes[0]),
            "hidden_dims": [int(s) for s in sizes[1:-1]],
            "output_dim": int(sizes[-1]),
            "sizes": sizes,
            "activation": model_weights.get("activation", "relu"),
            "origin": "voxhands.policy MLP weights -> numpy npz + JSON manifest",
            "note": "Stored weights load back through numpy; no true OpenVINO IR blob was produced.",
        }
        (model_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        self.model_weights = model_weights
        self.artifact_dir = model_dir
        return model_dir

    def to_ir(self, model_weights: dict[str, Any], model_dir: Path, model_name: str = "policy_mlp") -> Path:
        """Build and save a genuine OpenVINO IR (``.xml`` + ``.bin``).

        Unlike :meth:`convert` (which stores a numpy ``.npz``), this writes a
        real OpenVINO Runtime artifact via ``openvino.save_model``.  The float32
        IR is the export path the SO-101 deployment would ship; the CPU
        benchmark and the optional INT8 quantization both consume it.
        """
        try:
            import openvino as ov
            from openvino import opset14 as ops
        except Exception as exc:  # pragma: no cover - host without openvino
            raise RuntimeError(f"openvino is not importable: {exc}") from exc

        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        result, parameter = build_openvino_model(ops, model_weights)
        model = ov.Model(result, [parameter], model_name)

        xml_path = model_dir / f"{model_name}.xml"
        ov.save_model(model, str(xml_path))
        self.model_weights = model_weights
        self.artifact_dir = model_dir
        return xml_path

    def run(self, inputs: np.ndarray) -> np.ndarray:
        """CPU-inference fallback: pure numpy MLP forward pass over stored weights.

        Requires no device; never routed through the OpenVINO runtime here.
        Operates on the weights given at construction time or produced by
        :meth:`convert` (or loaded from an artifact dir via :meth:`load`).
        """
        if self.model_weights is None:
            raise ValueError("OpenVINOAdapter has no stored weights; call convert() or load() first.")
        x = np.asarray(inputs, dtype=np.float32)
        activation = self.model_weights.get("activation", "relu")
        for i, (w, b) in enumerate(zip(self.model_weights["weights"], self.model_weights["biases"])):
            x = x @ np.asarray(w, dtype=np.float32) + np.asarray(b, dtype=np.float32)
            is_last = i == len(self.model_weights["biases"]) - 1
            if is_last:
                continue
            if activation == "tanh":
                x = np.tanh(x)
            elif activation == "sigmoid":
                x = 1.0 / (1.0 + np.exp(-np.clip(x, -500.0, 500.0)))
            else:
                x = np.maximum(0.0, x)
        return x

    def load(self, model_dir: Path) -> None:
        """Load weights back from a :meth:`convert` artifact (manifest + npz)."""
        model_dir = Path(model_dir)
        manifest = json.loads((model_dir / "manifest.json").read_text())
        with np.load(model_dir / "weights.npz") as data:
            n_layers = len(manifest["sizes"]) - 1
            weights = [np.asarray(data[f"w{i}"]) for i in range(n_layers)]
            biases = [np.asarray(data[f"b{i}"]) for i in range(n_layers)]
        self.model_weights = {
            "weights": weights,
            "biases": biases,
            "activation": manifest.get("activation", "relu"),
        }
        self.artifact_dir = model_dir


def load_policy_weights(path: str | Path) -> dict[str, Any]:
    """Load policy weights following the voxhands.policy save() convention.

    Accepts either the JSON produced by ``MLPPolicy.save`` (``weights`` /
    ``biases`` as nested lists) or a ``.npz`` produced by
    :meth:`OpenVINOAdapter.convert` (keys ``w0..wn``, ``b0..bn`` plus a
    manifest). Returns the dict convention OpenVINOAdapter expects.
    """
    path = Path(path)
    if path.suffix == ".npz":
        adapter = OpenVINOAdapter()
        adapter.load(path.parent)
        return adapter.model_weights
    data = json.loads(path.read_text())
    return {
        "input_dim": data.get("input_dim"),
        "hidden_dims": data.get("hidden_dims"),
        "output_dim": data.get("output_dim"),
        "activation": data.get("activation", "relu"),
        "weights": [np.asarray(w, dtype=np.float32) for w in data["weights"]],
        "biases": [np.asarray(b, dtype=np.float32) for b in data["biases"]],
    }


def quantize_ir(
    xml_path: str | Path,
    out_dir: str | Path,
    calibration: np.ndarray,
    subset_size: int = 128,
    model_name: str = "policy_mlp_int8",
) -> Path:
    """INT8 post-training quantization of an OpenVINO IR via NNCF.

    Runs NNCF's PTQ on the CPU with a calibration set sampled from the policy's
    input distribution and saves a genuine quantized IR (``.xml`` + ``.bin``).
    Quantization is a real acceleration technique on CPU; NPU/GPU target
    graphs are gated on hardware and are *not* claimed here.
    """
    try:
        import nncf
        import openvino as ov
    except Exception as exc:  # pragma: no cover - host without nncf/openvino
        raise RuntimeError(f"nncf/openvino not importable: {exc}") from exc

    xml_path = Path(xml_path)
    calibration = np.asarray(calibration, dtype=np.float32)
    if calibration.ndim != 2:
        raise ValueError("calibration must be a 2D array of shape (samples, input_dim)")
    if calibration.shape[0] == 0:
        raise ValueError("calibration must contain at least one sample")

    model = ov.Core().read_model(str(xml_path))
    input_name = model.inputs[0].get_any_name()
    dataset = nncf.Dataset(
        calibration,
        lambda sample: {input_name: np.asarray(sample, dtype=np.float32)[None, :]},
    )
    quantized = nncf.quantize(model, dataset, subset_size=min(subset_size, calibration.shape[0]))

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    xml_out = out_dir / f"{model_name}.xml"
    ov.save_model(quantized, str(xml_out))
    return xml_out


def make_mlp_weights(
    sizes: list[int],
    seed: int = 0,
    scale: float = 0.02,
    activation: str = "relu",
) -> dict[str, Any]:
    """Deterministic synthetic MLP weights for a layer-size sweep.

    ``sizes`` is ``[input_dim, hidden..., output_dim]``.  Random-but-seeded so
    the sweep is reproducible across hosts without shipping model files.
    """
    if len(sizes) < 2:
        raise ValueError("sizes must contain at least input and output dims")
    rng = np.random.default_rng(seed)
    weights: list[np.ndarray] = []
    biases: list[np.ndarray] = []
    for i in range(len(sizes) - 1):
        weights.append((rng.standard_normal((sizes[i], sizes[i + 1])) * scale).astype(np.float32))
        biases.append(np.zeros(sizes[i + 1], dtype=np.float32))
    return {
        "input_dim": int(sizes[0]),
        "hidden_dims": [int(s) for s in sizes[1:-1]],
        "output_dim": int(sizes[-1]),
        "activation": activation,
        "weights": weights,
        "biases": biases,
    }


def _compile_config(ov: Any, performance_mode: str | None, num_streams: int | None) -> dict[str, Any]:
    config: dict[str, Any] = {}
    if performance_mode == "throughput":
        config[ov.properties.hint.performance_mode] = ov.properties.hint.PerformanceMode.THROUGHPUT
    elif performance_mode == "latency":
        config[ov.properties.hint.performance_mode] = ov.properties.hint.PerformanceMode.LATENCY
    if num_streams:
        config[ov.properties.streams.num] = int(num_streams)
    return config


def benchmark_inference(
    xml_path: str | Path,
    iterations: int = 2000,
    device: str = "CPU",
    warmup: int = 50,
    sample: np.ndarray | None = None,
    performance_mode: str | None = None,
    num_streams: int | None = None,
    repeats: int = 1,
) -> dict[str, Any]:
    """Measure real OpenVINO compiled-model latency for an IR.

    Returns p50/p95/mean latency (ms), throughput (inferences/s), the model's
    on-disk size, the runtime inference precision, and the device actually used.
    ``performance_mode`` selects OpenVINO's latency/throughput hint and
    ``num_streams`` sets the CPU stream count.  With ``repeats > 1`` the reported
    latency/throughput are the median across repeats so a single noisy pass
    cannot mislead.  No GPU/NPU numbers are produced — the caller passes the
    device under test.
    """
    try:
        import openvino as ov
    except Exception as exc:  # pragma: no cover - host without openvino
        raise RuntimeError(f"openvino not importable: {exc}") from exc

    import statistics
    import time

    xml_path = Path(xml_path)
    core = ov.Core()
    config = _compile_config(ov, performance_mode, num_streams)
    compiled = core.compile_model(str(xml_path), device, config) if config else core.compile_model(str(xml_path), device)
    input_port = compiled.input(0)
    input_dim = int(input_port.partial_shape[1].get_length())
    if sample is None:
        sample = np.zeros((1, input_dim), dtype=np.float32)
    sample = np.asarray(sample, dtype=np.float32)

    size_bytes = xml_path.stat().st_size
    bin_path = xml_path.with_suffix(".bin")
    bin_bytes = bin_path.stat().st_size if bin_path.exists() else 0
    size_bytes += bin_bytes
    try:
        precision = str(compiled.get_property("INFERENCE_PRECISION_HINT"))
    except Exception:
        precision = "unknown"

    passes: list[dict[str, float]] = []
    for _ in range(max(1, repeats)):
        for _ in range(max(0, warmup)):
            compiled([sample])
        latencies: list[float] = []
        for _ in range(max(1, iterations)):
            start = time.perf_counter()
            compiled([sample])
            latencies.append((time.perf_counter() - start) * 1000.0)
        latencies.sort()
        mean = statistics.fmean(latencies)
        passes.append(
            {
                "mean": mean,
                "p50": latencies[len(latencies) // 2],
                "p95": latencies[min(len(latencies) - 1, int(round(0.95 * (len(latencies) - 1))))],
            }
        )

    mean = statistics.median(p["mean"] for p in passes)
    p50 = statistics.median(p["p50"] for p in passes)
    p95 = statistics.median(p["p95"] for p in passes)
    return {
        "model": xml_path.name,
        "device": device,
        "iterations": iterations,
        "repeats": repeats,
        "performance_mode": performance_mode or "default",
        "num_streams": num_streams,
        "latency_ms_mean": round(mean, 5),
        "latency_ms_p50": round(p50, 5),
        "latency_ms_p95": round(p95, 5),
        "throughput_infers_per_s": round(1000.0 / mean, 1) if mean > 0 else None,
        "model_size_bytes": size_bytes,
        "model_size_kib": round(size_bytes / 1024.0, 2),
        "bin_size_bytes": bin_bytes,
        "bin_size_kib": round(bin_bytes / 1024.0, 2),
        "inference_precision": precision,
    }


def benchmark_throughput(
    xml_path: str | Path,
    requests: int = 4000,
    jobs: int = 8,
    device: str = "CPU",
    num_streams: int | None = None,
    warmup: int = 200,
    sample: np.ndarray | None = None,
    performance_mode: str | None = "throughput",
) -> dict[str, Any]:
    """Aggregate throughput via ``ov.AsyncInferQueue`` (real concurrent load).

    A single sequential call chain cannot benefit from multiple streams; queuing
    ``jobs`` async requests lets the CPU run streams concurrently and measures
    true inferences/second.  ``performance_mode`` defaults to OpenVINO's
    throughput hint, which sizes the stream pool to the core count; pass
    ``None`` to keep the default (latency-sized) compilation, which avoids the
    multi-process stream pool the CPU plugin uses on Apple silicon.
    """
    try:
        import openvino as ov
    except Exception as exc:  # pragma: no cover - host without openvino
        raise RuntimeError(f"openvino not importable: {exc}") from exc

    import time

    xml_path = Path(xml_path)
    core = ov.Core()
    config = _compile_config(ov, performance_mode, num_streams)
    compiled = core.compile_model(str(xml_path), device, config) if config else core.compile_model(str(xml_path), device)
    input_dim = int(compiled.input(0).partial_shape[1].get_length())
    if sample is None:
        sample = np.zeros((1, input_dim), dtype=np.float32)
    sample = np.asarray(sample, dtype=np.float32)

    queue = ov.AsyncInferQueue(compiled, max(1, jobs))
    for _ in range(max(0, warmup)):
        queue.start_async([sample])
    queue.wait_all()

    start = time.perf_counter()
    for _ in range(max(1, requests)):
        queue.start_async([sample])
    queue.wait_all()
    elapsed = time.perf_counter() - start
    return {
        "model": xml_path.name,
        "device": device,
        "jobs": max(1, jobs),
        "num_streams": num_streams,
        "requests": max(1, requests),
        "elapsed_s": round(elapsed, 4),
        "throughput_infers_per_s": round(max(1, requests) / elapsed, 1) if elapsed > 0 else None,
    }