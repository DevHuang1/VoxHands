from __future__ import annotations

import argparse
import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .simulation import TableSettingSimulation


ROOT = Path(__file__).resolve().parent.parent
FRONTEND = ROOT / "frontend"


class VoxHandsHandler(BaseHTTPRequestHandler):
    server_version = "VoxHands/0.1"

    @property
    def simulation(self) -> TableSettingSimulation:
        return self.server.simulation  # type: ignore[attr-defined]

    def _headers(self, content_type: str, length: int) -> None:
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")

    def _json(self, payload: Any, status: int = 200) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self._headers("application/json; charset=utf-8", len(body))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict[str, Any]:
        try:
            length = min(int(self.headers.get("Content-Length", "0")), 32_000)
            data = json.loads(self.rfile.read(length) or b"{}")
            return data if isinstance(data, dict) else {}
        except (ValueError, json.JSONDecodeError):
            return {}

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()

    def do_GET(self) -> None:
        if self.path == "/api/state":
            self._json(self.simulation.snapshot())
            return
        if self.path == "/api/health":
            self._json({"ok": True, "service": "voxhands", "version": "0.1.0"})
            return
        if self.path in {"/", "/index.html"}:
            self._serve_file(FRONTEND / "index.html")
            return
        self._json({"error": "Not found"}, 404)

    def do_POST(self) -> None:
        if self.path == "/api/command":
            payload = self._read_json()
            text = str(payload.get("text", "")).strip()
            if not text:
                self._json({"error": "Command text is required."}, 400)
                return
            self._json(self.simulation.submit_command(text))
            return
        if self.path == "/api/reset":
            self.simulation.reset()
            self._json(self.simulation.snapshot())
            return
        self._json({"error": "Not found"}, 404)

    def _serve_file(self, path: Path) -> None:
        if not path.is_file():
            self._json({"error": "Frontend file missing."}, 500)
            return
        body = path.read_bytes()
        content_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        self.send_response(200)
        self._headers(content_type, len(body))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        print(f"[voxhands] {self.address_string()} - {format % args}")


class VoxHandsServer(ThreadingHTTPServer):
    def __init__(self, address: tuple[str, int]) -> None:
        super().__init__(address, VoxHandsHandler)
        self.simulation = TableSettingSimulation()
        self.daemon_threads = True


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the VoxHands MVP server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    server = VoxHandsServer((args.host, args.port))
    print(f"VoxHands running at http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping VoxHands.")
    finally:
        server.server_close()

