"""Local OpenAI-compatible proxy for hosted endpoints (stdlib only).

MiniMax Code talks to ``http://127.0.0.1:PORT/v1``; the proxy forwards to ``--upstream`` with
the API key, merges ``--inject`` into every chat-completions body (default: thinking off, which
hosted Qwen-template models otherwise spend the whole output budget on) and streams the reply
back unchanged. One JSON line per call goes to ``--log`` (latency, status, usage if present).

    python arc_runner/llm_proxy.py --upstream https://host/v1 --api-key-file .secrets/key --port 8012
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def make_handler(upstream: str, key: str, inject: dict, log_path: str):
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a) -> None:  # quiet
            pass

        def _forward(self, method: str) -> None:
            t0 = time.time()
            n = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(n) if n else b""
            path = self.path[3:] if self.path.startswith("/v1") else self.path
            if method == "POST" and path.startswith("/chat/completions") and body:
                try:
                    d = json.loads(body)
                    for k, v in inject.items():
                        if isinstance(v, dict) and isinstance(d.get(k), dict):
                            d[k] = {**d[k], **v}
                        else:
                            d[k] = v
                    body = json.dumps(d).encode()
                except json.JSONDecodeError:
                    pass
            req = urllib.request.Request(upstream.rstrip("/") + path, data=body if method == "POST" else None, method=method)
            req.add_header("Authorization", f"Bearer {key}")
            req.add_header("Content-Type", "application/json")
            status, usage, size = 0, None, 0
            try:
                resp = urllib.request.urlopen(req, timeout=900)
                status = resp.status
            except urllib.error.HTTPError as e:
                resp, status = e, e.code
            except Exception as e:  # network failure: report as 502
                msg = json.dumps({"error": {"message": f"proxy: {type(e).__name__}: {e}"}}).encode()
                self.send_response(502)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(msg)))
                self.end_headers()
                self.wfile.write(msg)
                return
            self.send_response(status)
            ctype = resp.headers.get("Content-Type", "application/json")
            self.send_header("Content-Type", ctype)
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            tail = b""
            try:
                while True:
                    chunk = resp.read1(65536) if hasattr(resp, "read1") else resp.read(65536)
                    if not chunk:
                        break
                    size += len(chunk)
                    tail = (tail + chunk)[-4096:]
                    self.wfile.write(f"{len(chunk):x}\r\n".encode() + chunk + b"\r\n")
                    self.wfile.flush()
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            for line in reversed(tail.decode(errors="replace").splitlines()):
                if '"usage"' in line:
                    try:
                        usage = json.loads(line.removeprefix("data: ")).get("usage")
                    except json.JSONDecodeError:
                        pass
                    break
            with lock, open(log_path, "a") as f:
                f.write(json.dumps({"t": round(t0, 1), "path": path, "status": status, "s": round(time.time() - t0, 2),
                                    "bytes": size, "usage": usage}) + "\n")

        def do_POST(self) -> None:
            self._forward("POST")

        def do_GET(self) -> None:
            self._forward("GET")

    return Handler


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--upstream", required=True)
    p.add_argument("--api-key-file", required=True)
    p.add_argument("--port", type=int, default=8012)
    p.add_argument("--inject", default='{"chat_template_kwargs": {"enable_thinking": false}}')
    p.add_argument("--log", default="llm_proxy.jsonl")
    a = p.parse_args()
    key = Path(a.api_key_file).read_text().strip()
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(a.upstream, key, json.loads(a.inject), a.log))
    print(f"proxy on http://127.0.0.1:{a.port}/v1 -> {a.upstream}", file=sys.stderr, flush=True)
    srv.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
