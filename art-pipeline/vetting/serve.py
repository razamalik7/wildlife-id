"""Serve the reference-photo vetting UI.

    python art-pipeline/vetting/serve.py   ->  http://localhost:8777

Hotkeys in the UI: Enter/RightArrow accept, DownArrow next candidate,
UpArrow previous candidate, LeftArrow previous species, S skip species.
Accepted picks land in picks.json next to this script.
"""

import json
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

ROOT = Path(__file__).parent


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(ROOT), **kw)

    def do_POST(self):
        if self.path.rstrip("/").endswith("save_masters"):
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            json.loads(body)
            (ROOT / "picks_masters.json").write_bytes(body)
            self.send_response(200); self.end_headers(); self.wfile.write(b"ok")
        elif self.path.rstrip("/").endswith("save"):
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            json.loads(body)  # validate before writing
            (ROOT / "picks.json").write_bytes(body)
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")
        else:
            self.send_error(404)

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    print("vetting UI at http://localhost:8777  (Ctrl+C to stop)")
    HTTPServer(("127.0.0.1", 8777), Handler).serve_forever()
