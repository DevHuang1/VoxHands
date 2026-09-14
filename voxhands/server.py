from __future__ import annotations

import argparse
import json
import mimetypes
import os
import signal
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .simulation import TableSettingSimulation
from .planner import build_plan
from .safety import validate_plan
from .groq import GroqClient, ai_build_plan


ROOT = Path(__file__).resolve().parent.parent
FRONTEND = ROOT / "frontend"


class VoxHandsHandler(BaseHTTPRequestHandler):
    server_version = "VoxHands/0.2"

    @property
    def simulation(self) -> TableSettingSimulation:
        return self.server.simulation  # type: ignore[attr-defined]

    @property
    def groq(self) -> GroqClient:
        return self.server.groq  # type: ignore[attr-defined]

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
        length = int(self.headers.get("Content-Length", "0"))
        if not 0 <= length <= 32_000:
            raise ValueError("Request body must be at most 32000 bytes.")
        data = json.loads(self.rfile.read(length) or b"{}")
        if not isinstance(data, dict):
            raise ValueError("Expected a JSON object.")
        return data

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
            self._json({"ok": True, "service": "voxhands", "version": "0.2.0"})
            return
        if self.path.startswith("/vendor/"):
            name = self.path.removeprefix("/vendor/")
            if name not in {"three.module.js", "rapier.es.js", "THREE-LICENSE.txt", "RAPIER-LICENSE.txt"}:
                self._json({"error": "Not found"}, 404)
                return
            self._serve_file(FRONTEND / "vendor" / name)
            return
        if self.path in {"/", "/index.html"}:
            self._serve_file(FRONTEND / "index.html")
            return
        self._json({"error": "Not found"}, 404)

    def do_POST(self) -> None:
        try:
            self._post()
        except (ValueError, UnicodeDecodeError) as error:
            self._json({"error": str(error)}, 400)

    def _post(self) -> None:
        if self.path == "/api/command":
            payload = self._read_json()
            text = payload.get("text", "")
            if not isinstance(text, str):
                raise ValueError("Command text must be a string.")
            text = text.strip()
            if not text:
                self._json({"error": "Command text is required."}, 400)
                return
            style = payload.get("style") if isinstance(payload.get("style"), str) else None
            gesture = payload.get("gesture") if isinstance(payload.get("gesture"), str) else None
            plan, reply = ai_build_plan(text, self.groq, style=style, gesture=gesture)
            if plan.mode == "conversation":
                self._json(self.simulation.reply(text, reply, plan.llm_provider, suggestions=plan.suggestions))
                return
            result = self.simulation.submit_command(text, plan=plan, reply=reply)
            self._json(result, 409 if "error" in result else 200)
            return
        if self.path == "/api/chat":
            payload = self._read_json()
            text = payload.get("text", "")
            if not isinstance(text, str):
                raise ValueError("Message text must be a string.")
            text = text.strip()
            if not text:
                self._json({"error": "Message text is required."}, 400)
                return
            style = payload.get("style") if isinstance(payload.get("style"), str) else None
            gesture = payload.get("gesture") if isinstance(payload.get("gesture"), str) else None
            plan, reply = ai_build_plan(text, self.groq, style=style, gesture=gesture)
            result = self.simulation.reply(text, reply, plan.llm_provider, suggestions=plan.suggestions)
            self._json(result)
            return
        if self.path == "/api/plan":
            payload = self._read_json()
            text = payload.get("text", "")
            if not isinstance(text, str):
                raise ValueError("Command text must be a string.")
            text = text.strip()
            if not text:
                self._json({"error": "Command text is required."}, 400)
                return
            plan = build_plan(text)
            issues = validate_plan(plan)
            self._json({"valid": not issues, "safety_issues": issues, "plan": plan.to_dict()})
            return
        if self.path == "/api/physics-event":
            self._json(self.simulation.record_physics_event(self._read_json()))
            return
        if self.path == "/api/control":
            action = str(self._read_json().get("action", "")).strip().lower()
            handlers = {"pause": self.simulation.pause, "resume": self.simulation.resume, "stop": self.simulation.stop}
            handler = handlers.get(action)
            if handler is None:
                self._json({"error": "Control action must be pause, resume, or stop."}, 400)
                return
            self._json(handler())
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
        content_type = "text/javascript" if path.suffix == ".js" else mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        self.send_response(200)
        self._headers(content_type, len(body))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        if self.path in {"/api/state", "/api/physics-event"} and args and str(args[1]) == "200":
            return
        print(f"[voxhands] {self.address_string()} - {format % args}")


class VoxHandsServer(ThreadingHTTPServer):
    def __init__(self, address: tuple[str, int]) -> None:
        super().__init__(address, VoxHandsHandler)
        self.simulation = TableSettingSimulation()
        self.groq = GroqClient()
        self.daemon_threads = True


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the VoxHands MVP server")
    parser.add_argument("--host", default=os.environ.get("HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    args = parser.parse_args()
    server = VoxHandsServer((args.host, args.port))
    print(f"VoxHands running at http://{args.host}:{args.port}")

    def _stop(_signum: int, _frame: Any) -> None:
        print("\nStopping VoxHands.")
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
