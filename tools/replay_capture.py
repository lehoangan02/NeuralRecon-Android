import argparse
import csv
import os
import socket
import struct
import time

PACKET = struct.Struct('<4sIq11fHHI')

parser = argparse.ArgumentParser(description='Stream a saved ArCoreRecorder capture to live_demo.py as if it were the phone')
parser.add_argument('--rec_dir', required=True, help='capture folder with poses.csv and images/')
parser.add_argument('--port', type=int, default=8777)
parser.add_argument('--fps', type=float, default=12.0, help='replay rate, frames per second (0 = as fast as possible)')
args = parser.parse_args()

with open(os.path.join(args.rec_dir, 'poses.csv'), 'r') as f:
    rows = list(csv.DictReader(f))

sock = socket.create_connection(('127.0.0.1', args.port))
print('connected, sending {} frames'.format(len(rows)))
for r in rows:
    idx = int(r['idx'])
    with open(os.path.join(args.rec_dir, 'images', '{:06d}.jpg'.format(idx)), 'rb') as f:
        jpg = f.read()
    sock.sendall(PACKET.pack(
        b'NRP1', idx, int(r['timestamp_ns']),
        *(float(r[k]) for k in ('tx', 'ty', 'tz', 'qx', 'qy', 'qz', 'qw', 'fx', 'fy', 'cx', 'cy')),
        int(r['img_w']), int(r['img_h']), len(jpg)) + jpg)
    if args.fps > 0:
        time.sleep(1.0 / args.fps)
sock.close()
print('done')
