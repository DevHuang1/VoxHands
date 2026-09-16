import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from voxhands import intel_runtime  # noqa: E402


def _openvino_installed() -> bool:
    try:
        import openvino  # type: ignore # noqa: F401

        return True
    except Exception:
        return False


class HostCpuTests(unittest.TestCase):
    def test_host_cpu_model_is_a_non_empty_string(self) -> None:
        model = intel_runtime.host_cpu_model()
        self.assertIsInstance(model, str)
        self.assertTrue(model.strip())

    def test_status_always_reports_the_host_cpu(self) -> None:
        status = intel_runtime.self_test_status()
        self.assertIn("host_cpu", status)
        self.assertIn("installable", status)
        self.assertIn("available_devices", status)


class RuntimeProbeTests(unittest.TestCase):
    def tearDown(self) -> None:
        intel_runtime.runtime_probe(refresh=True)

    def test_probe_without_openvino_reports_a_reason(self) -> None:
        saved = sys.modules.get("openvino", "missing")
        sys.modules["openvino"] = None  # type: ignore[assignment]
        try:
            probe = intel_runtime.runtime_probe(refresh=True)
        finally:
            if saved == "missing":
                sys.modules.pop("openvino", None)
            else:
                sys.modules["openvino"] = saved  # type: ignore[assignment]
        self.assertFalse(probe["installable"])
        self.assertIn("openvino", probe["reason"])
        self.assertEqual(probe["devices"], [])

    def test_start_self_test_degrades_without_openvino(self) -> None:
        saved = sys.modules.get("openvino", "missing")
        sys.modules["openvino"] = None  # type: ignore[assignment]
        try:
            intel_runtime.runtime_probe(refresh=True)
            state = intel_runtime.start_self_test()
            status = intel_runtime.self_test_status()
        finally:
            if saved == "missing":
                sys.modules.pop("openvino", None)
            else:
                sys.modules["openvino"] = saved  # type: ignore[assignment]
            intel_runtime.runtime_probe(refresh=True)
        self.assertEqual(state, "unavailable")
        self.assertEqual(status["state"], "unavailable")
        self.assertFalse(status["available"])
        self.assertIn("openvino", status["error"])

    def test_openvino_status_entry_points_at_the_host_cpu(self) -> None:
        from voxhands.integrations import openvino_status

        status = openvino_status()
        self.assertEqual(status["name"], "OpenVINO")
        self.assertIn("host_cpu", status)
        self.assertIn("self_test", status)
        self.assertIsInstance(status["devices"], list)


@unittest.skipUnless(_openvino_installed(), "openvino runtime not installed")
class SelfTestTests(unittest.TestCase):
    def test_self_test_measures_a_real_compiled_model(self) -> None:
        report = intel_runtime.run_self_test(iterations=5, repeats=1, requests=20, jobs=2)
        self.assertEqual(report["state"], "ready")
        self.assertTrue(report["available"])
        self.assertEqual(report["device"], "CPU")
        self.assertIn("policy", report["models"])
        self.assertIn("workcell-large", report["models"])
        for name, entry in report["models"].items():
            self.assertGreater(entry["params"], 0, msg=name)
            self.assertGreater(entry["latency_ms_mean"], 0.0, msg=name)
            self.assertGreater(entry["model_size_bytes"], 0, msg=name)
        self.assertIsNotNone(report["openvino_version"])
        self.assertLess(report["duration_ms"], 120_000.0)


if __name__ == "__main__":
    unittest.main()
