import importlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from voxhands.openvino_adapter import (
    OpenVINOAdapter,
    REQUIRED_STATUS_KEYS,
    benchmark_inference,
    benchmark_throughput,
    build_openvino_model,
    load_policy_weights,
    make_mlp_weights,
    quantize_ir,
)

try:
    importlib.import_module("openvino")
    OPENVINO_AVAILABLE = True
    OPENVINO_SKIP_REASON = None
except Exception as exc:
    OPENVINO_AVAILABLE = False
    OPENVINO_SKIP_REASON = f"openvino not importable: {exc}"

try:
    importlib.import_module("nncf")
    NNCF_AVAILABLE = OPENVINO_AVAILABLE
    NNCF_SKIP_REASON = None
except Exception as exc:
    NNCF_AVAILABLE = False
    NNCF_SKIP_REASON = f"nncf not importable: {exc}"


def _make_policy_dict(seed: int = 42) -> dict[str, object]:
    from voxhands.policy import MLPPolicy

    policy = MLPPolicy(input_dim=8, hidden_dims=[16, 16], output_dim=8)
    flat = policy.params()
    weights = []
    biases = []
    for i in range(0, len(flat), 2):
        weights.append(flat[i])
        biases.append(flat[i + 1])
    return {
        "input_dim": 8,
        "hidden_dims": [16, 16],
        "output_dim": 8,
        "activation": "relu",
        "weights": weights,
        "biases": biases,
    }


@unittest.skipUnless(OPENVINO_AVAILABLE, OPENVINO_SKIP_REASON or "openvino not importable")
class OpenVINOAdapterGatedTests(unittest.TestCase):
    def test_status_returns_valid_json_dict(self) -> None:
        status = OpenVINOAdapter().status()
        self.assertIsInstance(status, dict)
        for key in REQUIRED_STATUS_KEYS:
            self.assertIn(key, status)
        json.dumps(status)
        self.assertTrue(status["installed"])
        self.assertIsInstance(status["devices"], list)
        self.assertIsInstance(status["verified"], bool)

    def test_available_returns_bool(self) -> None:
        self.assertTrue(OpenVINOAdapter().available())

    def test_cpu_only_reports_verified_false_and_device_cpu(self) -> None:
        status = OpenVINOAdapter().status()
        devices = status["devices"]
        has_accelerator = any(
            d in {"GPU", "NPU"} or d.startswith("GPU") for d in devices
        )
        self.assertIsNotNone(status["device_used"])
        if not has_accelerator:
            self.assertFalse(status["verified"])
            self.assertEqual(status["device_used"], "CPU")
            self.assertIn("blocked/unverified", status["message"])

    def test_convert_and_run_roundtrip(self) -> None:
        weights = _make_policy_dict(seed=1)
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            adapter = OpenVINOAdapter()
            out = adapter.convert(weights, model_dir)
            self.assertTrue((out / "weights.npz").exists())
            self.assertTrue((out / "manifest.json").exists())
            manifest = json.loads((out / "manifest.json").read_text())
            self.assertEqual(manifest["sizes"], [8, 16, 16, 8])

            reloaded = OpenVINOAdapter()
            reloaded.load(out)
            inputs = np.random.default_rng(2).normal(size=(1, 8)).astype(np.float32)
            output = reloaded.run(inputs)
            self.assertEqual(output.shape, (1, 8))

    def test_load_policy_weights_from_json(self) -> None:
        from voxhands.policy import MLPPolicy

        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "policy.json"
            policy = MLPPolicy(input_dim=8, hidden_dims=[16, 16], output_dim=8)
            policy.save(str(target))
            model_weights = load_policy_weights(target)
            self.assertIn("weights", model_weights)
            self.assertIn("biases", model_weights)
            self.assertEqual(len(model_weights["weights"]), 3)


@unittest.skipUnless(OPENVINO_AVAILABLE, OPENVINO_SKIP_REASON or "openvino not importable")
class OpenVINOIRExportTests(unittest.TestCase):
    def test_to_ir_writes_real_xml_and_bin(self) -> None:
        import openvino as ov

        weights = _make_policy_dict()
        with tempfile.TemporaryDirectory() as tmp:
            xml = OpenVINOAdapter().to_ir(weights, Path(tmp) / "ir")
            self.assertEqual(xml.suffix, ".xml")
            self.assertTrue(xml.exists())
            self.assertTrue(xml.with_suffix(".bin").exists())

            core = ov.Core()
            model = core.read_model(str(xml))
            self.assertEqual(len(model.inputs), 1)
            compiled = core.compile_model(str(xml), "CPU")
            output = np.asarray(compiled([np.zeros((1, 8), np.float32)])[compiled.output(0)])
            self.assertEqual(output.shape, (1, 8))

    def test_ir_matches_numpy_reference(self) -> None:
        import openvino as ov

        weights = _make_policy_dict()
        with tempfile.TemporaryDirectory() as tmp:
            xml = OpenVINOAdapter().to_ir(weights, Path(tmp) / "ir")
            compiled = ov.Core().compile_model(str(xml), "CPU")
            adapter = OpenVINOAdapter(model_weights=weights)
            probe = np.random.default_rng(5).normal(size=(1, 8)).astype(np.float32)
            ir = np.asarray(compiled([probe])[compiled.output(0)]).reshape(-1)
            ref = adapter.run(probe).reshape(-1)
            np.testing.assert_allclose(ir, ref, rtol=1e-2, atol=1e-2)

    def test_build_openvino_model_returns_result_and_input(self) -> None:
        from openvino import opset14 as ops

        weights = _make_policy_dict()
        result, parameter = build_openvino_model(ops, weights)
        self.assertIsNotNone(result)
        self.assertIsNotNone(parameter)


@unittest.skipUnless(NNCF_AVAILABLE, NNCF_SKIP_REASON or "nncf not importable")
class OpenVINOQuantizationTests(unittest.TestCase):
    def test_int8_ir_roundtrip_and_parity(self) -> None:
        import openvino as ov

        weights = _make_policy_dict()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            fp32_xml = OpenVINOAdapter().to_ir(weights, tmp_path / "ir")
            calibration = np.random.default_rng(0).normal(size=(64, 8)).astype(np.float32)
            int8_xml = quantize_ir(fp32_xml, tmp_path / "ir_int8", calibration, subset_size=32)
            self.assertEqual(int8_xml.suffix, ".xml")
            self.assertTrue(int8_xml.exists())
            self.assertTrue(int8_xml.with_suffix(".bin").exists())

            core = ov.Core()
            fp32_compiled = core.compile_model(str(fp32_xml), "CPU")
            int8_compiled = core.compile_model(str(int8_xml), "CPU")
            probe = calibration[:1]
            fp32_out = np.asarray(fp32_compiled([probe])[fp32_compiled.output(0)]).reshape(-1)
            int8_out = np.asarray(int8_compiled([probe])[int8_compiled.output(0)]).reshape(-1)
            np.testing.assert_allclose(int8_out, fp32_out, rtol=0.1, atol=0.1)


@unittest.skipUnless(OPENVINO_AVAILABLE, OPENVINO_SKIP_REASON or "openvino not importable")
class OpenVINOSweepTests(unittest.TestCase):
    def test_make_mlp_weights_shapes_and_determinism(self) -> None:
        sizes = [8, 32, 16, 4]
        weights = make_mlp_weights(sizes, seed=3)
        self.assertEqual(weights["input_dim"], 8)
        self.assertEqual(weights["hidden_dims"], [32, 16])
        self.assertEqual(weights["output_dim"], 4)
        self.assertEqual(len(weights["weights"]), 3)
        self.assertEqual(weights["weights"][0].shape, (8, 32))
        self.assertEqual(weights["weights"][-1].shape, (16, 4))
        self.assertTrue(np.allclose(weights["weights"][0], make_mlp_weights(sizes, seed=3)["weights"][0]))

    def test_benchmark_repeats_and_mode_keys(self) -> None:
        weights = make_mlp_weights([8, 16, 16, 8], seed=1)
        with tempfile.TemporaryDirectory() as tmp:
            xml = OpenVINOAdapter().to_ir(weights, Path(tmp) / "ir")
            report = benchmark_inference(xml, iterations=15, warmup=2, repeats=3, performance_mode="latency")
            self.assertEqual(report["repeats"], 3)
            self.assertEqual(report["performance_mode"], "latency")
            self.assertIn("bin_size_bytes", report)
            self.assertGreater(report["latency_ms_mean"], 0)

    def test_benchmark_throughput_positive(self) -> None:
        weights = make_mlp_weights([8, 16, 16, 8], seed=2)
        with tempfile.TemporaryDirectory() as tmp:
            xml = OpenVINOAdapter().to_ir(weights, Path(tmp) / "ir")
            report = benchmark_throughput(xml, requests=200, jobs=4, warmup=20)
            self.assertGreater(report["throughput_infers_per_s"], 0)
            self.assertEqual(report["requests"], 200)


@unittest.skipUnless(OPENVINO_AVAILABLE, OPENVINO_SKIP_REASON or "openvino not importable")
class OpenVINOBenchmarkTests(unittest.TestCase):
    def test_benchmark_reports_openvino_metrics(self) -> None:
        weights = _make_policy_dict()
        with tempfile.TemporaryDirectory() as tmp:
            xml = OpenVINOAdapter().to_ir(weights, Path(tmp) / "ir")
            report = benchmark_inference(xml, iterations=20, warmup=2)
            for key in (
                "latency_ms_mean",
                "latency_ms_p50",
                "latency_ms_p95",
                "throughput_infers_per_s",
                "model_size_bytes",
                "device",
            ):
                self.assertIn(key, report)
            self.assertGreater(report["latency_ms_mean"], 0)
            self.assertGreater(report["model_size_bytes"], 0)


class OpenVINOAdapterStatusSchemaTests(unittest.TestCase):
    def test_status_schema_even_when_not_installed(self) -> None:
        status = OpenVINOAdapter().status()
        for key in REQUIRED_STATUS_KEYS:
            self.assertIn(key, status)
        json.dumps(status)
        self.assertIsInstance(status["installed"], bool)
        self.assertIsInstance(status["verified"], bool)


if __name__ == "__main__":
    unittest.main()