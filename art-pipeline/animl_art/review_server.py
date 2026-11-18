"""Stdlib HTTP review dashboard backend for large-scale animl_art batch QA & candidate swapping."""

import json
import mimetypes
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from animl_art import VETTING_DIR


def _read_json(path: Path, default):
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return default
    return default


def _build_batch_payload(work_dir: Path) -> dict:
    meta = _read_json(work_dir / "metadata.json", {})
    metrics = _read_json(work_dir / "metrics.json", {"summary": {}, "rows": []})
    qa_summary = _read_json(work_dir / "qa_summary.json", {"total": 0, "clean_count": 0, "flagged_count": 0, "flagged": []})
    decisions = _read_json(work_dir / "review_decisions.json", {})
    errors = _read_json(work_dir / "errors.json", [])

    row_by_stem = {r["stem"]: r for r in metrics.get("rows", [])}
    stems = sorted(set(meta.keys()) | set(row_by_stem.keys()))
    items = []
    for stem in stems:
        m = meta.get(stem, {})
        r = row_by_stem.get(stem, {})
        cands_p = work_dir / "cands_top3" / stem / "cands.json"
        cands = _read_json(cands_p, [])
        items.append(
            {
                "stem": stem,
                "common_name": m.get("common_name") or r.get("common") or stem,
                "scientific_name": m.get("scientific_name", ""),
                "class": m.get("class") or r.get("class") or "",
                "subtype": m.get("subtype", ""),
                "pose_quality_score": m.get("pose_quality_score"),
                "cutout_repair_needed": m.get("cutout_repair_needed", "none"),
                "has_perch_stub": m.get("has_perch_stub", False),
                "visual_traits": m.get("visual_traits", ""),
                "photo_id": m.get("photo_id", ""),
                "metrics": r,
                "qa_flags": r.get("qa_flags", []),
                "cands": cands,
                "decision": decisions.get(stem, {"action": "pending"}),
                "has_eq": (work_dir / "eq_out" / f"{stem}.png").exists(),
                "has_raw": (work_dir / "raw_stylized" / f"{stem}.png").exists(),
                "has_cut": (work_dir / "cutouts_final" / f"{stem}.png").exists() or (work_dir / "cutouts_1024" / f"{stem}.png").exists(),
            }
        )
    return {
        "batch": work_dir.name,
        "summary": metrics.get("summary", {}),
        "qa_summary": qa_summary,
        "errors": errors,
        "items": items,
    }


def make_handler(work_dir: Path, html_path: Path):
    """Create a thread-safe request handler bound to `work_dir` and `html_path`."""

    class ReviewHandler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args) -> None:  # noqa: A002
            return

        def _send_json(self, status: int, payload: dict) -> None:
            raw = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path in ("/", "/index.html"):
                body = html_path.read_bytes()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            if parsed.path == "/api/batch":
                self._send_json(HTTPStatus.OK, _build_batch_payload(work_dir))
                return

            if parsed.path.startswith("/files/"):
                rel = unquote(parsed.path[len("/files/") :]).lstrip("/")
                target = (work_dir / rel).resolve()
                root = work_dir.resolve()
                if root not in target.parents or not target.is_file():
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                ctype = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
                data = target.read_bytes()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return

            self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path not in ("/api/decision", "/api/approve-unflagged"):
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length).decode("utf-8")) if length > 0 else {}
            dec_path = work_dir / "review_decisions.json"
            decisions = _read_json(dec_path, {})

            if parsed.path == "/api/decision":
                stem = str(body["stem"])
                decisions[stem] = {
                    "action": body.get("action", "approve"),
                    "swap_rank": body.get("swap_rank"),
                    "unet_only": bool(body.get("unet_only", False)),
                }
            elif parsed.path == "/api/approve-unflagged":
                payload = _build_batch_payload(work_dir)
                for it in payload["items"]:
                    cur_act = decisions.get(it["stem"], {}).get("action", "pending")
                    if cur_act == "pending" and not it["qa_flags"] and it["has_eq"]:
                        decisions[it["stem"]] = {"action": "approve", "swap_rank": None, "unet_only": False}

            dec_path.write_text(json.dumps(decisions, indent=2), encoding="utf-8")
            self._send_json(HTTPStatus.OK, {"ok": True, "decisions": decisions})

    return ReviewHandler


def serve_batch_review(work_dir: Path, port: int = 8765) -> None:
    """Start local interactive QA dashboard for `work_dir` until Ctrl+C."""
    html_path = VETTING_DIR / "batch_review.html"
    if not html_path.exists():
        raise FileNotFoundError(f"Missing dashboard template: {html_path}")
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(work_dir, html_path))
    print(f"Batch Review UI running at http://127.0.0.1:{port}/  (batch={work_dir.name}, Ctrl+C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped Batch Review UI.", flush=True)
    finally:
        server.server_close()
