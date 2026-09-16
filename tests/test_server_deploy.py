import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Render installs only requirements.txt (the standard library), so MuJoCo,
# OpenVINO, and Pillow are absent at runtime. Block those imports and confirm
# the server still boots and degrades gracefully.
PROBE = """
import json, sys, threading, time, urllib.error, urllib.request
for name in ("mujoco", "openvino", "PIL"):
    sys.modules[name] = None
from voxhands.server import VoxHandsServer, _MUJOCO_AVAILABLE

server = VoxHandsServer(("127.0.0.1", 0))
port = server.server_address[1]
threading.Thread(target=server.serve_forever, daemon=True).start()
time.sleep(0.25)


def hit(path):
    url = f"http://127.0.0.1:{port}{path}"
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            return response.status, json.loads(response.read().decode())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read().decode())


report = {
    "mujoco": _MUJOCO_AVAILABLE,
    "sim": type(server.simulation).__name__,
    "vision": server.vision,
    "health": hit("/api/health"),
    "camera": hit("/api/camera"),
    "vision_endpoint": hit("/api/vision"),
}
server.shutdown()
print("RESULT:" + json.dumps(report))
"""


class RenderDeployTests(unittest.TestCase):
    def test_server_boots_without_optional_backends(self) -> None:
        result = subprocess.run(
            [sys.executable, "-c", PROBE],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        line = next(
            (item for item in result.stdout.splitlines() if item.startswith("RESULT:")),
            None,
        )
        self.assertIsNotNone(line, msg=f"probe produced no result: {result.stdout}\n{result.stderr}")
        report = json.loads(line[len("RESULT:"):])

        self.assertFalse(report["mujoco"], msg="probe must simulate a backend-free host")
        self.assertEqual(report["sim"], "TableSettingSimulation")
        self.assertIsNone(report["vision"])

        self.assertEqual(report["health"][0], 200)
        self.assertTrue(report["health"][1]["ok"])

        self.assertEqual(report["camera"][0], 404)
        self.assertIn("unavailable", report["camera"][1]["error"])

        self.assertEqual(report["vision_endpoint"][0], 404)


if __name__ == "__main__":
    unittest.main()
