"""
Local HTTP + SSE server for the ta-LBFGS web dashboard.

Serves ui.html and streams optimizer snapshots to the browser.
"""

from __future__ import annotations

import json
import os
import queue
import threading
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse


def _json_bytes(payload: Dict[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=True).encode("utf-8")


@dataclass
class _Client:
    q: queue.Queue


class DashboardServer:
    """Hosts the dashboard UI and pushes state updates via SSE."""

    def __init__(self, host: str = "127.0.0.1", port: int = 7860):
        self.host = host
        self.port = port
        self._state_lock = threading.Lock()
        self._clients_lock = threading.Lock()
        self._latest_state: Dict[str, Any] = self._empty_state()
        self._clients: List[_Client] = []
        self._httpd: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @staticmethod
    def _empty_state() -> Dict[str, Any]:
        return {
            "run": {
                "status": "initializing",
                "val_loss": 0.0,
                "outer_step": 0,
                "max_outer_steps": 1,
                "mean_kappa": 1.0,
                "topology_valid_pct": 0.0,
            },
            "heatmap": {
                "num_layers": 0,
                "heads_per_layer": 0,
                "kappa": [],
                "valid_mask": [],
            },
            "head_buffers": {},
            "experts": {
                "rows": [],
                "active_count": 0,
                "expired_ttl": 0,
            },
            "trajectory": {
                "loss": [],
                "pivot_steps": [],
                "spectral_guard_steps": [],
            },
            "chain": {
                "segments": {
                    "reasoning": 0.25,
                    "pivot": 0.1,
                    "answer": 0.45,
                    "verify": 0.2,
                },
                "status_rows": {
                    "current_segment": "reasoning",
                    "pivot_index": 0,
                    "reasoning_tokens": 0,
                    "answer_tokens": 0,
                    "verify_tokens": 0,
                    "topology_valid": False,
                },
            },
            "events": [],
        }

    def start(self) -> None:
        if self._httpd is not None:
            return

        server = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_GET(self) -> None:
                parsed = urlparse(self.path)
                path = parsed.path

                if path in {"/", "/ui.html"}:
                    self._serve_ui()
                    return
                if path == "/events":
                    self._serve_sse()
                    return
                if path == "/state":
                    self._serve_state()
                    return

                self.send_error(HTTPStatus.NOT_FOUND, "Not Found")

            def do_POST(self) -> None:
                parsed = urlparse(self.path)
                if parsed.path == "/select_head":
                    self._select_head()
                    return
                self.send_error(HTTPStatus.NOT_FOUND, "Not Found")

            def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
                return

            def _serve_ui(self) -> None:
                ui_path = os.path.join(os.path.dirname(__file__), "ui.html")
                try:
                    with open(ui_path, "rb") as f:
                        body = f.read()
                except OSError:
                    self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, "ui.html missing")
                    return

                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _serve_state(self) -> None:
                with server._state_lock:
                    body = _json_bytes(server._latest_state)
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _serve_sse(self) -> None:
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "keep-alive")
                self.end_headers()

                q: queue.Queue = queue.Queue(maxsize=16)
                client = _Client(q=q)
                with server._clients_lock:
                    server._clients.append(client)

                try:
                    with server._state_lock:
                        snapshot = json.dumps(server._latest_state, ensure_ascii=True)
                    self.wfile.write(f"data: {snapshot}\n\n".encode("utf-8"))
                    self.wfile.flush()

                    while True:
                        try:
                            payload = q.get(timeout=15.0)
                            self.wfile.write(f"data: {payload}\n\n".encode("utf-8"))
                        except queue.Empty:
                            self.wfile.write(b": keepalive\n\n")
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass
                finally:
                    with server._clients_lock:
                        if client in server._clients:
                            server._clients.remove(client)

            def _select_head(self) -> None:
                content_len = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(content_len) if content_len > 0 else b"{}"
                try:
                    req = json.loads(raw.decode("utf-8"))
                except json.JSONDecodeError:
                    req = {}

                layer = int(req.get("layer", -1))
                head = int(req.get("head", -1))
                key = f"{layer}:{head}"

                with server._state_lock:
                    head_buffers = server._latest_state.get("head_buffers", {})
                    detail = head_buffers.get(key)

                if detail is None:
                    detail = {
                        "layer": layer,
                        "head": head,
                        "pairs": [],
                    }

                body = _json_bytes(detail)
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self._httpd = ThreadingHTTPServer((self.host, self.port), Handler)
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._httpd is None:
            return
        self._httpd.shutdown()
        self._httpd.server_close()
        self._httpd = None
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def publish(self, payload: Dict[str, Any]) -> None:
        with self._state_lock:
            self._latest_state = payload
        encoded = json.dumps(payload, ensure_ascii=True)
        with self._clients_lock:
            clients = list(self._clients)
        for client in clients:
            try:
                client.q.put_nowait(encoded)
            except queue.Full:
                try:
                    _ = client.q.get_nowait()
                except queue.Empty:
                    pass
                try:
                    client.q.put_nowait(encoded)
                except queue.Full:
                    pass
