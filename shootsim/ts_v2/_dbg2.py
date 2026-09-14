# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, r'C:\Users\12951\Desktop\12323\shootsim')
sys.path.insert(0, r'C:\Users\12951\Desktop\12323\shootsim\ts_v2')
import replay
from logparse import load_log
eps = replay.parse_log_file(r'C:\Users\12951\Desktop\12323\logs\aim_20260906_003353.log', source_chest_ratio=0.2)
print('replay groups:', len(eps))
data = load_log(r'C:\Users\12951\Desktop\12323\logs\aim_20260906_003353.log')
print('logparse groups:', len(data['episodes']))
sizes_a = [len(e['points']) for e in eps]
sizes_b = [len(e['points']) for e in data['episodes']]
print('replay sizes:', sizes_a[:20])
print('logparse sizes:', sizes_b[:20])
n = min(len(eps), len(data['episodes']))
ok = all(abs(a['t_start'] - b['points'][0]['t'] * 1000) < 1 for a, b in zip(eps, data['episodes'][:n]))
print('first %d t_starts match:' % n, ok)
