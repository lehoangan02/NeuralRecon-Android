import argparse
import csv
import os
import pickle
import shutil

import numpy as np
from PIL import Image
from tqdm import tqdm

from tools.kp_reproject import load_camera_pose


def process_arcore_data(rec_dir, out_dir, window_size=9, min_angle=15, min_distance=0.1, size=(640, 480)):
    """Convert a recording from the ArCoreRecorder Android app (poses.csv + images/) into the layout
    demo.py expects (images/, SyncedPoses.txt, fragments.pkl).

    ARCore's Camera.getPose() uses the same OpenGL camera convention and Y-up world as ARKit, so the poses
    go through the same loader as the ARKit data (tools/kp_reproject.py, data_source='ARKit').
    The keyframe selection and fragment generation mirror tools/process_arkit_data.py.
    """
    rec_dir = os.path.normpath(rec_dir)
    out_dir = os.path.normpath(out_dir)
    os.makedirs(os.path.join(out_dir, 'images'), exist_ok=True)

    with open(os.path.join(rec_dir, 'poses.csv'), 'r') as f:
        rows = list(csv.DictReader(f))
    assert len(rows) > 0, "poses.csv is empty"

    # images -> 640x480, intrinsics follow whatever is done to the image
    intrinsics = {}
    pose_lines = []
    for r in tqdm(rows, desc='Processing images and intrinsics...'):
        fid = str(int(r['idx'])).zfill(5)
        w, h = int(r['img_w']), int(r['img_h'])
        assert (w, h) == (int(r['intr_w']), int(r['intr_h'])), \
            "image size {}x{} does not match intrinsics size for frame {}".format(w, h, fid)
        fx, fy, cx, cy = (float(r[k]) for k in ('fx', 'fy', 'cx', 'cy'))

        img = Image.open(os.path.join(rec_dir, 'images', '{}.jpg'.format(str(int(r['idx'])).zfill(6))))
        # center crop to the target aspect ratio (no-op for 4:3 images such as ARCore's 640x480)
        target_w = int(round(h * size[0] / size[1]))
        if target_w < w:
            left = (w - target_w) // 2
            img = img.crop((left, 0, left + target_w, h))
            cx -= left
            w = target_w
        elif target_w > w:
            target_h = int(round(w * size[1] / size[0]))
            top = (h - target_h) // 2
            img = img.crop((0, top, w, top + target_h))
            cy -= top
            h = target_h
        if (w, h) != tuple(size):
            fx, cx = fx * size[0] / w, cx * size[0] / w
            fy, cy = fy * size[1] / h, cy * size[1] / h
            img = img.resize(size, Image.BILINEAR)
        img.save(os.path.join(out_dir, 'images', '{}.jpg'.format(fid)), quality=95)

        intrinsics[fid] = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=float)
        pose_lines.append(' '.join([fid] + [r[k] for k in ('tx', 'ty', 'tz', 'qx', 'qy', 'qz', 'qw')]) + '\n')

    # demo.py skips pre-processing when SyncedPoses.txt exists, so it has to be written
    synced_path = os.path.join(out_dir, 'SyncedPoses.txt')
    with open(synced_path, 'w') as f:
        f.writelines(pose_lines)
    cam_pose_dict = load_camera_pose(synced_path, data_source='ARKit')

    # keyframe selection and fragments, same as process_arkit_data.py
    fragments = []
    all_ids = []
    ids = []
    count = 0
    last_pose = None
    for id in tqdm(intrinsics.keys(), desc='Keyframes selection...'):
        cam_pose = cam_pose_dict[id]
        if count == 0:
            ids.append(id)
            last_pose = cam_pose
            count += 1
        else:
            cos = ((np.linalg.inv(cam_pose[:3, :3]) @ last_pose[:3, :3] @ np.array([0, 0, 1]).T) * np.array(
                [0, 0, 1])).sum()
            angle = np.arccos(np.clip(cos, -1, 1))
            dis = np.linalg.norm(cam_pose[:3, 3] - last_pose[:3, 3])
            if angle > (min_angle / 180) * np.pi or dis > min_distance:
                ids.append(id)
                last_pose = cam_pose
                count += 1
                if count == window_size:
                    all_ids.append(ids)
                    ids = []
                    count = 0

    scene = os.path.basename(out_dir)
    for i, ids in enumerate(all_ids):
        poses = []
        ks = []
        for id in ids:
            # Moving down the X-Y plane in the ARKit coordinate to meet the training settings in ScanNet.
            cam_pose_dict[id][2, 3] += 1.5
            poses.append(cam_pose_dict[id])
            ks.append(intrinsics[id])
        fragments.append({
            'scene': scene,
            'fragment_id': i,
            'image_ids': ids,
            'extrinsics': poses,
            'intrinsics': ks
        })

    with open(os.path.join(out_dir, 'fragments.pkl'), 'wb') as f:
        pickle.dump(fragments, f)

    n_key = sum(len(i) for i in all_ids)
    print("{} frames -> {} keyframes -> {} fragments (of {} views)".format(len(rows), n_key, len(fragments), window_size))
    if len(fragments) == 0:
        print("WARNING: no fragments produced. Move the phone more (>= {} keyframes needed).".format(window_size))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert an ArCoreRecorder capture for NeuralRecon demo.py')
    parser.add_argument('--rec_dir', required=True, help='folder pulled from the phone (poses.csv + images/)')
    parser.add_argument('--out_dir', default=None, help='output folder, default: <rec_dir>_neucon')
    args = parser.parse_args()
    process_arcore_data(args.rec_dir, args.out_dir or os.path.normpath(args.rec_dir) + '_neucon')
