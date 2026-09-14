# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, r'C:\Users\12951\Desktop\12323\shootsim')
sys.path.insert(0, r'C:\Users\12951\Desktop\12323\shootsim\ts_v2')
import replay
from logparse import parse_rows

path = r'C:\Users\12951\Desktop\12323\logs\aim_20260906_003353.log'
rows = parse_rows(path)
print('rows:', len(rows))
eps = replay._structured_episodes(path, rows, 0.2)
print('structured episodes:', len(eps))
print('sizes:', [len(e['points']) for e in eps][:25])

# now trace parse_log_file's own row extraction
text = replay._read_text(path)
structured_rows = []
marker = "[瞄准观测] "
for raw in text.splitlines():
    if marker not in raw:
        continue
    try:
        row = replay.json.loads(raw.split(marker, 1)[1])
    except Exception as exc:
        print('JSON FAIL:', repr(exc), raw[:80])
        continue
    if isinstance(row, dict) and "frame" in row and "capture_t" in row:
        structured_rows.append(row)
print('parse_log_file-style rows:', len(structured_rows))
eps2 = replay.parse_log_file(path, source_chest_ratio=0.2)
print('parse_log_file episodes:', len(eps2))
