# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, r'C:\Users\12951\Desktop\12323\shootsim')
sys.path.insert(0, r'C:\Users\12951\Desktop\12323\shootsim\ts_v2')
import replay
from logparse import parse_rows

path = r'C:\Users\12951\Desktop\12323\logs\aim_20260906_003353.log'
eps = replay.parse_log_file(path, source_chest_ratio=0.2)
print('parse_log_file groups:', len(eps))
rows_b = parse_rows(path)
print('logparse rows:', len(rows_b))

# replay row stream: reconstruct from its episodes
rows_a = []
for e in eps:
    rows_a.extend(e['points'])
print('replay total points:', len(rows_a))
# compare (frame, capture_t) sequences
keys_b = [(int(r['frame']), float(r['capture_t'])) for r in rows_b]
keys_a = [(p['frame'], p['t_rel']) for p in rows_a]  # t_rel is relative!
# only compare lengths and frame sequences
fb = [k[0] for k in keys_b]
fa = [k[0] for k in keys_a]
print('frame seq equal:', fa == fb, len(fa), len(fb))
for i, (x, y) in enumerate(zip(fa, fb)):
    if x != y:
        print('first frame diff at', i, x, y)
        print('replay ctx:', fa[max(0, i-3):i+3])
        print('logp  ctx:', fb[max(0, i-3):i+3])
        break
