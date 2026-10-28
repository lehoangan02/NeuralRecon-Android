import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import trimesh

VIEWER_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'viewer')

CONTENT_TYPES = {'.html': 'text/html; charset=utf-8', '.js': 'application/javascript', '.css': 'text/css'}


class ViewerState(object):
    """Thread-safe data shared between the reconstruction code and the HTTP viewer."""

    def __init__(self):
        self.lock = threading.Lock()
        self.mesh_bytes = b''
        self.mesh_version = 0
        self.latest_jpg = b''
        self.info = {'hud': 'waiting...', 'frames': 0, 'trajectory': [], 'frustum': None}

    def set_mesh(self, mesh_bytes):
        with self.lock:
            self.mesh_bytes = mesh_bytes
            self.mesh_version += 1

    def set_frame(self, jpg_bytes):
        with self.lock:
            self.latest_jpg = jpg_bytes

    def update(self, **kwargs):
        with self.lock:
            self.info.update(kwargs)

    def snapshot(self):
        with self.lock:
            out = dict(self.info)
            out['mesh_version'] = self.mesh_version
            return out


def prepare_display_mesh(mesh, min_faces=300, smooth_iters=5):
    """Return a cleaned copy of a marching-cubes mesh for display: drop small floating pieces, then Taubin-smooth.

    Only meant for the viewer. The raw mesh from the network is what gets saved as the result.
    """
    mesh = mesh.copy()
    if min_faces > 1 and len(mesh.faces) > 0:
        comps = trimesh.graph.connected_components(mesh.face_adjacency, min_len=min_faces)
        if len(comps) > 0:
            mask = np.zeros(len(mesh.faces), dtype=bool)
            mask[np.concatenate(comps)] = True
            mesh.update_faces(mask)
            mesh.remove_unreferenced_vertices()
    if smooth_iters > 0 and len(mesh.faces) > 0:
        trimesh.smoothing.filter_taubin(mesh, iterations=smooth_iters)
    return mesh


def mesh_to_ply_bytes(mesh):
    return mesh.export(file_type='ply')


def _make_handler(state):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, body, ctype, status=200):
            self.send_response(status)
            self.send_header('Content-Type', ctype)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = self.path.split('?')[0]
            if path == '/state.json':
                self._send(json.dumps(state.snapshot()).encode(), 'application/json')
            elif path == '/mesh.ply':
                with state.lock:
                    body = state.mesh_bytes
                self._send(body, 'application/octet-stream')
            elif path == '/latest.jpg':
                with state.lock:
                    body = state.latest_jpg
                self._send(body, 'image/jpeg')
            else:
                rel = 'index.html' if path in ('/', '') else path.lstrip('/')
                full = os.path.normpath(os.path.join(VIEWER_DIR, rel))
                if not full.startswith(VIEWER_DIR + os.sep) or not os.path.isfile(full):
                    self._send(b'not found', 'text/plain', 404)
                    return
                with open(full, 'rb') as f:
                    self._send(f.read(), CONTENT_TYPES.get(os.path.splitext(full)[1], 'application/octet-stream'))

    return Handler


def serve(state, port=8000):
    """Start the viewer HTTP server on localhost in a background thread."""
    server = ThreadingHTTPServer(('127.0.0.1', port), _make_handler(state))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server
