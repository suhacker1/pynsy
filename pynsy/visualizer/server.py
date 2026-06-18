# Copyright 2023 The pynsy Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""HTTP server for the pynsy analysis visualizer.

Serves the single-page HUD at ``/`` and the captured trace at ``/api/trace``.

  python -m pynsy.visualizer.server [--port 8080] [--trace PATH]
"""

import argparse
import functools
import http.server
import os
import webbrowser

from pynsy.analyses import util

HERE = os.path.dirname(os.path.abspath(__file__))
INDEX_HTML = os.path.join(HERE, "index.html")


def default_trace_path():
  return str(util.get_output_path("visualizer", "visualizer_trace.json"))


class Handler(http.server.BaseHTTPRequestHandler):

  def __init__(self, *args, trace_path=None, **kwargs):
    self.trace_path = trace_path
    super().__init__(*args, **kwargs)

  def log_message(self, *args):  # quieter console
    pass

  def _send(self, body, content_type, status=200):
    if isinstance(body, str):
      body = body.encode("utf-8")
    self.send_response(status)
    self.send_header("Content-Type", content_type)
    self.send_header("Content-Length", str(len(body)))
    self.end_headers()
    self.wfile.write(body)

  def do_GET(self):
    if self.path in ("/", "/index.html"):
      with open(INDEX_HTML, "r") as f:
        self._send(f.read(), "text/html; charset=utf-8")
    elif self.path == "/api/trace":
      if not os.path.exists(self.trace_path):
        self._send(
            '{"error": "trace not found; run the analysis first"}',
            "application/json",
            status=404,
        )
        return
      with open(self.trace_path, "r") as f:
        self._send(f.read(), "application/json")
    else:
      self._send("Not found", "text/plain", status=404)


def main():
  parser = argparse.ArgumentParser(description="pynsy visualizer server")
  parser.add_argument("--port", type=int, default=8080)
  parser.add_argument("--trace", default=None, help="path to visualizer_trace.json")
  parser.add_argument("--no-open", action="store_true", help="don't open a browser")
  args = parser.parse_args()

  trace_path = args.trace or default_trace_path()
  handler = functools.partial(Handler, trace_path=trace_path)
  server = http.server.ThreadingHTTPServer(("localhost", args.port), handler)
  url = f"http://localhost:{args.port}"
  print(f"Serving pynsy visualizer at {url}")
  print(f"Trace: {trace_path}")
  if not args.no_open:
    webbrowser.open(url)
  try:
    server.serve_forever()
  except KeyboardInterrupt:
    print("\nShutting down.")
    server.shutdown()


if __name__ == "__main__":
  main()
