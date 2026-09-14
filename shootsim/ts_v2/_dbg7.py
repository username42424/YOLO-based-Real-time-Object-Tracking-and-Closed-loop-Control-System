# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, r'C:\Users\12951\Desktop\12323\shootsim')
sys.path.insert(0, r'C:\Users\12951\Desktop\12323\shootsim\ts_v2')
from logparse import parse_rows

path = r'C:\Users\12951\Desktop\12323\logs\aim_20260906_003353.log'
rows = parse_rows(path)


def split_indices(rows):
    idxs = []
    pf = pt = None
    for i, row in enumerate(rows):
        f = int(row.get("frame", 0))
        t = float(row.get("capture_t", 0.0))
        if pf is not None and (f <= pf or t <= pt):
            idxs.append((i, f, t, pf, pt))
        pf, pt = f, t
    return idxs


mine = split_indices(rows)
print("my splits:", len(mine))
for m in mine[:8]:
    print("  split at row %d frame %s t %.6f (prev f %s t %.6f)" % m)


def replay_style(rows):
    idxs = []
    previous_frame = previous_capture = None
    for i, row in enumerate(rows):
        frame = int(row.get("frame", 0))
        capture_t = float(row.get("capture_t", 0.0))
        if previous_frame is not None and (
                frame <= previous_frame or capture_t <= previous_capture):
            idxs.append((i, frame, capture_t, previous_frame, previous_capture))
        previous_frame, previous_capture = frame, capture_t
    return idxs


rs = replay_style(rows)
print("replay-style splits:", len(rs))
for m in rs[:8]:
    print("  split at row %d frame %s t %.6f (prev f %s t %.6f)" % m)
