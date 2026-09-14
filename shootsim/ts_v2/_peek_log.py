# -*- coding: utf-8 -*-
"""Peek at the head of a replay log: config snapshot + first frame lines."""
import io
import os
import sys

p = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\12951\Desktop\12323\logs\aim_20260906_003353.log"
n = int(sys.argv[2]) if len(sys.argv) > 2 else 40
print("size:", os.path.getsize(p))
enc = None
for e in ("utf-8", "gbk", "utf-8-sig"):
    try:
        with io.open(p, encoding=e) as f:
            f.read(4096)
        enc = e
        break
    except UnicodeDecodeError:
        continue
print("encoding guess:", enc)
with io.open(p, encoding=enc or "utf-8", errors="replace") as f:
    for i, line in enumerate(f):
        if i >= n:
            break
        print(i, "|", line.rstrip()[:400])
