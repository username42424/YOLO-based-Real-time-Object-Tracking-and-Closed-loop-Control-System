# -*- coding: utf-8 -*-
import json
d = json.load(open(r'C:\Users\12951\Desktop\12323\shootsim\results'
                   r'\tracking_sweep_20260906_v2\baseline_metrics.json',
                   encoding='utf-8'))
t = d['splits']['train']
keys = ('no_entry_rate', 'strafe_dir_wrong_rate', 'err_p75', 'quiet_cmd_p95',
        'jump_count_total', 'strafe_xlag_p75', 'dwell_inner60', 'timeout_rate',
        'pred_active_rate', 'pred_unready_rate', 'pred_blocked_rate',
        'cap_saturation_rate', 'first_inner60_median', 'first_inner60_p75',
        'err_p50', 'err_p90', 'strafe_err_p75', 'quiet_cmd_p75')
for k in keys:
    print(k, '=', t.get(k))
