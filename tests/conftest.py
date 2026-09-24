import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
import torch

from scout.model import Scout


def unchanged(model):
    before = {key: value.clone() for key, value in model.state_dict().items()}
    yield model
    after = model.state_dict()
    assert all(torch.equal(after[key], value) for key, value in before.items()), "a test changed a shared model"


@pytest.fixture(scope="session")
def full_model():
    torch.manual_seed(0)
    yield from unchanged(Scout())


class Server:
    def __init__(self, files, honour_ranges=True, truncate_to=None, cut_first=0, range_on_416=True):
        self.files = files
        self.requests = []
        self.cuts_left = cut_first
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                outer.requests.append((self.path, self.headers.get("Range")))
                body = outer.files.get(self.path.lstrip("/"))
                if body is None:
                    self.send_error(404)
                    return
                extra = {}
                if isinstance(body, tuple):
                    body, extra = body
                header = self.headers.get("Range")
                if header and honour_ranges:
                    start = int(header.removeprefix("bytes=").rstrip("-"))
                    if start >= len(body):
                        self.send_response(416)
                        if range_on_416:
                            self.send_header("Content-Range", f"bytes */{len(body)}")
                        self.send_header("Content-Length", "0")
                        self.end_headers()
                        return
                    part = body[start:]
                    self.send_response(206)
                    self.send_header("Content-Range", f"bytes {start}-{len(body) - 1}/{len(body)}")
                else:
                    part = body
                    self.send_response(200)
                self.send_header("Content-Length", str(len(part)))
                for name, value in extra.items():
                    self.send_header(name, value)
                self.end_headers()
                limit = truncate_to
                if outer.cuts_left and len(part) > 1:
                    outer.cuts_left -= 1
                    limit = len(part) // 2
                self.wfile.write(part[:limit] if limit is not None else part)

            def log_message(self, *args):
                pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}/"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def serve():
    servers = []

    def start(files, **kwargs):
        servers.append(Server(files, **kwargs))
        return servers[-1]

    yield start
    for server in servers:
        server.close()
