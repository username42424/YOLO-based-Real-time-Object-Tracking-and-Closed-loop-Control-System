# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, r'C:\Users\12951\Desktop\12323\shootsim')
sys.path.insert(0, r'C:\Users\12951\Desktop\12323\shootsim\ts_v2')
import replay
from logparse import parse_rows, group_episodes

path = r'C:\Users\12951\Desktop\12323\logs\aim_20260906_003353.log'
text = replay._read_text(path)
rows_replay = []
for raw in text.splitlines():
    if "[瞄准观测] " not in raw:
        continue
    try:
        row = json.loads(raw.split("[瞄准观测] ", 1)[1]) if (json := __import__("json")) else None
    except Exception:
        continue
    if isinstance(row, dict) and "frame" in row and "capture_t" in row:
        rows_replay.append(row)
rows_a = parse_rows(path)
print("replay rows:", len(rows_replay), "logparse rows:", len(rows_a))
for i, (a, b) in enumerate(zip(rows_replay, rows_a)):
    if a != b:
        print("first diff at row", i)
        print("replay  :", {k: a.get(k) for k in ("frame", "capture_t", "target")})
        print("logparse:", {k: b.get(k) for k in ("frame", "capture_t", "target")})
        break
else:
    print("rows identical up to", min(len(rows_replay), len(rows_a)))
# group both ways
def gp(rows):
    groups, cur, pf, pt = [], [], None, None
    for row in rows:
        f = int(row.get("frame", 0)); t = float(row.get("capture_t", 0.0))
        if cur and (f <= pf or t <= pt):
            groups.append(cur); cur = []
        cur.append(row); pf, pt = f, t
    if cur: groups.append(cur)
    return groups
ga = gp(rows_replay); gb = gp(rows_a)
print("groups:", len(ga), len(gb))
print("ga sizes:", [len(g) for g in ga][:25])
print("gb sizes:", [len(g) for g in gb][:25])
