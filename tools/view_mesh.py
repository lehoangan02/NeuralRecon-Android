import argparse
import time
import webbrowser

import trimesh

from tools.live_viewer import ViewerState, serve, prepare_display_mesh, mesh_to_ply_bytes

parser = argparse.ArgumentParser(description='Open a NeuralRecon mesh in the browser viewer')
parser.add_argument('ply', help='mesh file, e.g. results/scene_demo_checkpoints_fusion_eval_47/<scene>.ply')
parser.add_argument('--port', type=int, default=8000)
parser.add_argument('--min-faces', type=int, default=300, help='drop floating pieces smaller than this many faces (1 keeps all)')
parser.add_argument('--smooth', type=int, default=5, help='Taubin smoothing iterations (0 disables)')
parser.add_argument('--no-open', action='store_true', help='do not open a browser window')
args = parser.parse_args()

mesh = trimesh.load(args.ply, process=False)
shown = prepare_display_mesh(mesh, args.min_faces, args.smooth)

state = ViewerState()
state.set_mesh(mesh_to_ply_bytes(shown))
state.update(hud='{}\n{} vertices, {} faces (raw: {} faces)'.format(
    args.ply.split('/')[-1], len(shown.vertices), len(shown.faces), len(mesh.faces)))
serve(state, args.port)

url = 'http://127.0.0.1:{}/'.format(args.port)
print('Viewer: ' + url + '  (Ctrl-C to stop)')
if not args.no_open:
    webbrowser.open(url)
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    pass
