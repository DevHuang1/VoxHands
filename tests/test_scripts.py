import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"

SCRIPT_NAMES = [
    "train_policy.py",
    "convert_openvino.py",
    "quantize_openvino.py",
    "benchmark_openvino.py",
    "run_mujoco_eval.py",
    "capture_demo.py",
    "check_intel.py",
]


class ScriptImportTests(unittest.TestCase):
    def test_each_script_imports_cleanly(self) -> None:
        for name in SCRIPT_NAMES:
            with self.subTest(script=name):
                code = (
                    "import sys, importlib; "
                    f"sys.path.insert(0, r'{SCRIPTS}'); "
                    f"importlib.import_module({name[:-3]!r})"
                )
                result = subprocess.run(
                    [sys.executable, "-c", code],
                    cwd=str(ROOT),
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode, 0, msg=f"import {name} failed: {result.stderr}")

    def test_each_script_help_exits_zero(self) -> None:
        for name in SCRIPT_NAMES:
            with self.subTest(script=name):
                result = subprocess.run(
                    [sys.executable, str(SCRIPTS / name), "--help"],
                    cwd=str(ROOT),
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode, 0, msg=f"--help for {name} failed: {result.stderr}")
                self.assertIn("usage:", result.stdout.lower())

    def test_check_intel_script_imports_module_main(self) -> None:
        result = subprocess.run(
            [sys.executable, str(SCRIPTS / "check_intel.py"), "--help"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("usage:", result.stdout.lower())


if __name__ == "__main__":
    unittest.main()