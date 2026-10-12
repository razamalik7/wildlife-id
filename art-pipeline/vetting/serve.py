"""Serve the art-pipeline review UIs.

    python art-pipeline/vetting/serve.py   ->  http://localhost:8777

Pages: index.html (reference-photo vetting), curate.html (training-set
curation; ?set=core for the core-only pass), style.html (style labels),
axes.html (empirical style axes + families).
Each page POSTs its state to /save_<name>; ROUTES maps that to a file here.
"""

import json
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

ROOT = Path(__file__).parent

ROUTES = {
    "save": "picks.json",
    "save_masters": "picks_masters.json",
    "save_core": "picks_core.json",
    "save_style": "style_labels_reviewed.json",
    "save_axes": "axis_targets.json",
}


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(ROOT), **kw)

    def do_POST(self):
        target = ROUTES.get(self.path.strip("/").rsplit("/", 1)[-1])
        if target is None:
            self.send_error(404)
            return
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        json.loads(body)  # validate before writing
        (ROOT / target).write_bytes(body)
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"ok")

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    print("vetting UI at http://localhost:8777  (Ctrl+C to stop)")
    HTTPServer(("127.0.0.1", 8777), Handler).serve_forever()
