# -*- coding: utf-8 -*-
"""Debug: run one episode, dump aligned sim vs log rows."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from harness import build_sim_cfg, load_production_cfg, run_episode
from dataset import load_all_episodes, build_replay_episodes
from fidelity import raw_log_groups, log_rows_for_group, alignment

PRIMARY = "aim_20260906_003353.log"
groups_all = raw_log_groups(PRIMARY)
groups = [g for g in groups_all if any(p["detected"] for p in g["points"])]
metas = [m for m in load_all_episodes([PRIMARY]) if m["log"] == PRIMARY]
assert len(groups) == len(metas), (len(groups), len(metas))
eps = build_replay_episodes(metas, apply_theta=1.0)
sim_cfg = build_sim_cfg([PRIMARY], apply_theta=1.0)
prod = load_production_cfg()
g, m, ep = groups[0], metas[0], eps[0]
n_drop, pre = alignment(g)
print("n_drop=%d pre_frames=%d phase=%.4fs" % (n_drop, pre, m["output_phase_s"]))
rec = run_episode(sim_cfg, prod, ep, output_phase_s=m["output_phase_s"])
rows = log_rows_for_group(g)
print("sim obs:", len(rec.obs), "log rows:", len(rows),
      "summary:", {k: rec.summary[k] for k in
                   ("first_inner60_time", "dwell_reddot", "replay_observed_seconds",
                    "replay_exhausted", "true_timeout")})
for j, s in enumerate(rec.obs[:30]):
    li = j - pre + n_drop
    L = rows[li] if 0 <= li < len(rows) else None
    ls = ("t=%.3f det=%d tgt=(%6.1f,%6.1f) lr=%-26s reason=%-18s "
          "err=(%6.1f,%6.1f) cmd=(%7.1f,%6.1f) vraw=(%7.0f,%6.0f)"
          % (s["t"], s["detected"], s["target"][0], s["target"][1],
             str(s["lock_reason"])[:26], str(s["reason"])[:18],
             s["obs_err"][0], s["obs_err"][1], s["cmd"][0], s["cmd"][1],
             s["vel_raw"][0], s["vel_raw"][1])) if s["target"] else \
         ("t=%.3f det=%d tgt=None" % (s["t"], s["detected"]))
    lo = ""
    if L:
        lo = ("|| log t=%.3f det=%d tgt=(%6.1f,%6.1f) lr=%-26s reason=%-18s "
              "err=(%6.1f,%6.1f) cmd=(%7.1f,%6.1f) vraw=(%7.0f,%6.0f)"
              % (0.0, L["detected"], L["target"][0] if L["target"] else -1,
                 L["target"][1] if L["target"] else -1,
                 str(L["lock_reason"])[:26], str(L["reason"])[:18],
                 L["err"][0], L["err"][1], L["cmd"][0], L["cmd"][1],
                 L["vel_raw"][0], L["vel_raw"][1])) if L["target"] else \
             ("|| log det=%d tgt=None" % L["detected"])
    print("[%02d]" % j, ls, lo)
