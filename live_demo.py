import argparse
import io
import os
import queue
import shutil
import socket
import struct
import subprocess
import threading
import time
import webbrowser

import numpy as np
import torch
from loguru import logger
from PIL import Image
from torch.utils.data import default_collate
from transforms3d.quaternions import quat2mat

from config import cfg, update_config
from datasets import transforms
from models import NeuralRecon
from tools.kp_reproject import rotx
from tools.live_viewer import ViewerState, serve, prepare_display_mesh, mesh_to_ply_bytes
from utils import SaveScene

# One packet per frame from the ArCoreRecorder app:
#   magic 'NRP1', idx, timestamp_ns, tx ty tz, qx qy qz qw, fx fy cx cy, img_w, img_h, jpeg_len, then the JPEG bytes.
PACKET = struct.Struct('<4sIq11fHHI')
CSV_HEADER = 'idx,timestamp_ns,tx,ty,tz,qx,qy,qz,qw,fx,fy,cx,cy,img_w,img_h,intr_w,intr_h\n'

parser = argparse.ArgumentParser(description='NeuralRecon live preview: phone stream -> incremental mesh -> browser')
parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)
parser.add_argument('--phone-port', type=int, default=8777, help='TCP port the phone connects to (via adb reverse)')
parser.add_argument('--http-port', type=int, default=8000, help='port of the browser viewer')
parser.add_argument('--min-faces', type=int, default=300, help='viewer only: drop floating pieces smaller than this')
parser.add_argument('--smooth', type=int, default=5, help='viewer only: Taubin smoothing iterations (0 disables)')
parser.add_argument('--save-dir', default='results/live', help='where streamed captures and final meshes are written')
parser.add_argument('--no-adb', action='store_true', help='do not run adb reverse')
parser.add_argument('--no-open', action='store_true', help='do not open a browser window')
parser.add_argument('opts', help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER)
args = parser.parse_args()
update_config(cfg, args)


def arcore_pose_to_matrix(t, q):
    """Camera-to-world pose from ARCore's Camera.getPose() (OpenGL camera, Y-up world), converted the same way as
    tools/kp_reproject.load_camera_pose does for ARKit data."""
    rot = quat2mat([q[3], q[0], q[1], q[2]])
    rot = rot.dot(np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]))
    rot = rotx(np.pi / 2) @ rot
    pose = np.eye(4)
    pose[:3, :3] = rot
    pose[:3, 3] = rotx(np.pi / 2) @ np.array(t, dtype=float)
    return pose


def recv_exact(conn, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise EOFError
        buf.extend(chunk)
    return bytes(buf)


class Session(object):
    """One recording = one phone connection. Also saves the stream in the ArCoreRecorder folder layout."""

    def __init__(self, root):
        self.name = 'live_' + time.strftime('%Y%m%d_%H%M%S')
        self.dir = os.path.join(root, self.name)
        os.makedirs(os.path.join(self.dir, 'images'))
        self.csv = open(os.path.join(self.dir, 'poses.csv'), 'w')
        self.csv.write(CSV_HEADER)
        self.last_pose = None
        self.group = []
        self.n_frames = 0
        self.n_keyframes = 0
        self.n_fragments = 0
        self.trajectory = []

    def save_frame(self, idx, ts, t, q, k, w, h, jpg):
        with open(os.path.join(self.dir, 'images', '{:06d}.jpg'.format(idx)), 'wb') as f:
            f.write(jpg)
        self.csv.write('{},{},{:.9f},{:.9f},{:.9f},{:.9f},{:.9f},{:.9f},{:.9f},{:.6f},{:.6f},{:.6f},{:.6f},{},{},{},{}\n'.format(
            idx, ts, *t, *q, *k, w, h, w, h))
        self.csv.flush()

    def is_keyframe(self, pose, min_angle=15, min_distance=0.1):
        """Same rule as tools/process_arkit_data.py."""
        if self.last_pose is None:
            return True
        cos = ((np.linalg.inv(pose[:3, :3]) @ self.last_pose[:3, :3] @ np.array([0, 0, 1]).T) *
               np.array([0, 0, 1])).sum()
        angle = np.arccos(np.clip(cos, -1, 1))
        dis = np.linalg.norm(pose[:3, 3] - self.last_pose[:3, 3])
        return angle > (min_angle / 180) * np.pi or dis > min_distance

    def close(self):
        self.csv.close()


def frustum_points(pose, k, w, h, depth=0.25):
    corners = []
    for x, y in ((0, 0), (w, 0), (w, h), (0, h)):
        c = np.array([(x - k[2]) / k[0] * depth, (y - k[3]) / k[1] * depth, depth])
        corners.append((pose[:3, :3] @ c + pose[:3, 3]).tolist())
    return [pose[:3, 3].tolist()] + corners


def listen_for_phone(port):
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(('127.0.0.1', port))
    srv.listen(1)
    return srv


def phone_server(srv, state, frag_queue, n_views, save_root):
    while True:
        conn, _ = srv.accept()
        sess = Session(save_root)
        logger.info('phone connected, session {}'.format(sess.name))
        state.update(hud='recording {}\nwaiting for keyframes...'.format(sess.name), frames=0, trajectory=[],
                     frustum=None)
        t0 = time.time()
        try:
            while True:
                magic, idx, ts, tx, ty, tz, qx, qy, qz, qw, fx, fy, cx, cy, w, h, n = PACKET.unpack(
                    recv_exact(conn, PACKET.size))
                if magic != b'NRP1':
                    raise ValueError('bad packet magic')
                jpg = recv_exact(conn, n)
                t, q, k = (tx, ty, tz), (qx, qy, qz, qw), (fx, fy, cx, cy)
                sess.save_frame(idx, ts, t, q, k, w, h, jpg)
                sess.n_frames += 1
                state.set_frame(jpg)

                pose = arcore_pose_to_matrix(t, q)
                state.update(frames=sess.n_frames, frustum=frustum_points(pose, k, w, h))
                if sess.is_keyframe(pose):
                    sess.last_pose = pose
                    img = Image.open(io.BytesIO(jpg)).convert('RGB')
                    kmat = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=float)
                    if (w, h) != (640, 480):
                        img = img.resize((640, 480), Image.BILINEAR)
                        kmat[0, :] *= 640 / w
                        kmat[1, :] *= 480 / h
                    sess.group.append((img, kmat, pose))
                    sess.n_keyframes += 1
                    sess.trajectory.append(pose[:3, 3].tolist())
                    state.update(trajectory=list(sess.trajectory))
                    if len(sess.group) == n_views:
                        frag_queue.put(('fragment', sess, sess.n_fragments, sess.group))
                        sess.n_fragments += 1
                        sess.group = []
        except (EOFError, ConnectionError, ValueError, struct.error) as e:
            logger.info('phone disconnected ({} frames, {} keyframes, {:.0f}s): {}'.format(
                sess.n_frames, sess.n_keyframes, time.time() - t0, repr(e)))
        finally:
            conn.close()
            sess.close()
            frag_queue.put(('end', sess, None, None))


def load_model():
    logger.info('Initializing the model on GPU...')
    model = NeuralRecon(cfg).cuda().eval()
    model = torch.nn.DataParallel(model, device_ids=[0])
    saved = [fn for fn in os.listdir(cfg.LOGDIR) if fn.endswith('.ckpt')]
    saved = sorted(saved, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    ckpt = os.path.join(cfg.LOGDIR, saved[-1])
    logger.info('Resuming from ' + ckpt)
    model.load_state_dict(torch.load(ckpt)['model'], strict=False)
    return model


def build_transform():
    return transforms.Compose([
        transforms.ResizeImage((640, 480)),
        transforms.ToTensor(),
        transforms.RandomTransformSpace(cfg.MODEL.N_VOX, cfg.MODEL.VOXEL_SIZE, random_rotation=False,
                                        random_translation=False, paddingXY=0, paddingZ=0,
                                        max_epoch=cfg.TRAIN.EPOCHS),
        transforms.IntrinsicsPoseToProjection(cfg.TEST.N_VIEWS, 4)])


def run_fragment(model, transform, sess, frag_id, group):
    poses = []
    for _, _, pose in group:
        shifted = pose.copy()
        shifted[2, 3] += 1.5  # same shift as process_arkit_data.py: the model expects ScanNet's floor height
        poses.append(shifted)
    items = {
        'imgs': [g[0] for g in group],
        'intrinsics': np.stack([g[1] for g in group]),
        'extrinsics': np.stack(poses),
        'scene': sess.name,
        'fragment': '{}_{}'.format(sess.name, frag_id),
        'epoch': [0],
        'vol_origin': np.array([0, 0, 0]),
    }
    sample = default_collate([transform(items)])
    with torch.no_grad():
        outputs, _ = model(sample, True)
    if 'scene_name' not in outputs or sess.name not in outputs['scene_name']:
        return None
    i = outputs['scene_name'].index(sess.name)
    tsdf = outputs['scene_tsdf'][i].data.cpu().numpy()
    origin = outputs['origin'][i].data.cpu().numpy()
    origin[2] -= 1.5  # back to the unshifted pose frame, matching the trajectory shown in the viewer
    if (tsdf == 1).all():
        return None
    try:
        return SaveScene.tsdf2mesh(cfg.MODEL.VOXEL_SIZE, origin, tsdf)
    except ValueError:
        return None


def main():
    state = ViewerState()
    serve(state, args.http_port)
    url = 'http://127.0.0.1:{}/'.format(args.http_port)
    logger.info('Viewer: ' + url)

    model = load_model()
    transform = build_transform()
    n_views = cfg.TEST.N_VIEWS

    frag_queue = queue.Queue()
    os.makedirs(args.save_dir, exist_ok=True)
    srv = listen_for_phone(args.phone_port)  # bound here so a port clash fails loudly
    threading.Thread(target=phone_server, daemon=True,
                     args=(srv, state, frag_queue, n_views, args.save_dir)).start()

    if not args.no_adb:
        adb = shutil.which('adb') or os.path.expanduser('~/Android/Sdk/platform-tools/adb')
        port = 'tcp:{}'.format(args.phone_port)
        try:
            out = subprocess.run([adb, 'reverse', port, port], capture_output=True, text=True, timeout=15)
            logger.info('adb reverse {}: {}'.format(port, (out.stdout + out.stderr).strip() or 'ok'))
        except (OSError, subprocess.TimeoutExpired) as e:
            logger.warning('adb reverse failed ({}); run it yourself: adb reverse {} {}'.format(e, port, port))

    state.update(hud='ready. Start recording in the phone app.')
    if not args.no_open:
        webbrowser.open(url)
    logger.info('Ready. Start recording in the phone app.')

    last_mesh = {}
    try:
        while True:
            try:
                kind, sess, frag_id, group = frag_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if kind == 'fragment':
                t0 = time.time()
                raw = run_fragment(model, transform, sess, frag_id, group)
                ms = (time.time() - t0) * 1000
                if raw is not None:
                    last_mesh[sess.name] = raw
                    shown = prepare_display_mesh(raw, args.min_faces, args.smooth)
                    state.set_mesh(mesh_to_ply_bytes(shown))
                    hud = 'recording {}\n{} frames, {} keyframes, {} fragments\nlast fragment {:.0f} ms, {} faces'.format(
                        sess.name, sess.n_frames, sess.n_keyframes, frag_id + 1, ms, len(shown.faces))
                else:
                    hud = 'recording {}\n{} keyframes, {} fragments (no surface yet)'.format(
                        sess.name, sess.n_keyframes, frag_id + 1)
                state.update(hud=hud)
                logger.info('fragment {} done in {:.0f} ms'.format(frag_id, ms))
            else:
                raw = last_mesh.pop(sess.name, None)
                if raw is not None:
                    raw.export(os.path.join(sess.dir, 'mesh_raw.ply'))
                    prepare_display_mesh(raw, args.min_faces, args.smooth).export(os.path.join(sess.dir, 'mesh.ply'))
                    msg = 'finished {}\nmesh saved to {}'.format(sess.name, sess.dir)
                else:
                    msg = 'finished {}\nno mesh produced (too few keyframes?)'.format(sess.name)
                state.update(hud=msg)
                logger.info(msg.replace('\n', ' - '))
    except KeyboardInterrupt:
        logger.info('bye')


if __name__ == '__main__':
    main()
