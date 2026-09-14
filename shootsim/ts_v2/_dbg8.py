# -*- coding: utf-8 -*-
"""Find the first divergence frame per episode and inspect the mechanism."""
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from harness import build_sim_cfg, load_production_cfg, run_episode
from dataset import load_all_episodes, build_replay_episodes
from fidelity import raw_log_groups, log_rows_for_group, alignment

PRIMARY = "aim_20260906_003353.log"
groups = [g for g in raw_log_groups(PRIMARY)
          if any(p["detected"] for p in g["points"])]
metas = [m for m in load_all_episodes([PRIMARY]) if m["log"] == PRIMARY]
eps = build_replay_episodes(metas, apply_theta=1.0)
sim_cfg = build_sim_cfg([PRIMARY], apply_theta=1.0)
prod = load_production_cfg()

onsets = []
for k in range(min(8, len(groups))):
    if k not in (2, 4, 5):
        continue
    g, m, ep = groups[k], metas[k], eps[k]
    n_drop, pre = alignment(g)
    rec = run_episode(sim_cfg, prod, ep, output_phase_s=m["output_phase_s"])
    rows = log_rows_for_group(g)
    onset = None
    for j, s in enumerate(rec.obs):
        li = j - pre + n_drop
        if li < 0 or li >= len(rows):
            continue
        L = rows[li]
        d = max(abs(round(s["cmd"][0]) - round(L["cmd"][0])),
                abs(round(s["cmd"][1]) - round(L["cmd"][1])))
        if d > 2:
            onset = (j, li, d)
            break
    onsets.append(onset)
    if onset is None:
        continue
    print("ep %d: n_obs=%d first |Δcmd|>2 at sim_idx=%s (Δ=%s)" %
          (k, len(rec.obs), onset[0] if onset else None,
           onset[2] if onset else None))
    j, li, _ = onset
    for jj in range(max(0, j - 6), min(j + 2, len(rec.obs))):
        s = rec.obs[jj]
        ljj = jj - pre + n_drop
        L = rows[ljj] if 0 <= ljj < len(rows) else None
        print("  [%02d] sim t=%.3f tgt=(%6.1f,%6.1f) lr=%-22s r=%-16s "
              "err=(%6.1f,%6.1f) cmd=(%7.1f,%6.1f) vraw=(%7.0f,%6.0f) lead=(%5.1f,%5.1f) h=%.1f"
              % (jj, s["t"], s["target"][0], s["target"][1],
                 str(s["lock_reason"])[:22], str(s["reason"])[:16],
                 s["obs_err"][0], s["obs_err"][1], s["cmd"][0], s["cmd"][1],
                 s["vel_raw"][0], s["vel_raw"][1], s["lead"][0], s["lead"][1],
                 s["horizon_ms"]))
        if L:
            print("       log     tgt=(%6.1f,%6.1f) lr=%-22s r=%-16s "
                  "err=(%6.1f,%6.1f) cmd=(%7.1f,%6.1f) vraw=(%7.0f,%6.0f) lead=(%5.1f,%5.1f) h=%.1f"
                  % (L["target"][0], L["target"][1],
                     str(L["lock_reason"])[:22], str(L["reason"])[:16],
                     L["err"][0], L["err"][1], L["cmd"][0], L["cmd"][1],
                     L["vel_raw"][0], L["vel_raw"][1], L["lead"][0], L["lead"][1],
                     L["horizon_ms"]))
