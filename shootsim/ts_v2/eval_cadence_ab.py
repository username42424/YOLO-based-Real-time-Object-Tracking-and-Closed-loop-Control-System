# -*- coding: utf-8 -*-
"""Closed-loop sim A/B (alpha=0.2 vs robust) on the 30ms-cadence logs."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "shootsim"))

from logparse import load_log, fill_world  # noqa: E402
from dataset import fit_grid_phase  # noqa: E402
from eval3 import evaluate_v3  # noqa: E402
from harness import build_sim_cfg  # noqa: E402
from run_v4 import robust_cfg  # noqa: E402
from eval_newlogs_v4 import to_replay_episode  # noqa: E402

LOG_DIR = os.path.join(ROOT, "logs")
LOGS = ["aim_20260906_135713.log", "aim_20260906_140550.log",
        "aim_20260906_141313.log"]


def main():
    from eval3 import episode_eligibility
    episodes, eligs, phases = [], [], []
    for name in LOGS:
        data = load_log(os.path.join(LOG_DIR, name), 0.44, 0.51)
        phase, _, _ = fit_grid_phase(os.path.join(LOG_DIR, name))
        for ep in data["episodes"]:
            fill_world(ep)
            pts = ep["points"]
            if not any(p["detected"] for p in pts):
                continue
            if sum(1 for p in pts if p["box"] is not None) < 3:
                continue
            try:
                episodes.append(to_replay_episode(pts, os.path.join(LOG_DIR, name)))
                eligs.append(episode_eligibility(pts))
                phases.append(phase or 0.0075)
            except ValueError:
                continue
    sim_cfg = build_sim_cfg(LOGS)
    base_cfg = json.load(open(os.path.join(
        ROOT, "config_backup_alpha020_20260906.json"), encoding="utf-8"))
    out = {
        "episodes": len(episodes),
        "baseline_alpha020": evaluate_v3(sim_cfg, base_cfg, episodes, eligs,
                                         phases=phases),
        "robust_applied": evaluate_v3(sim_cfg, robust_cfg(), episodes, eligs,
                                      phases=phases),
    }
    os.makedirs(os.path.join(ROOT, "日志V4", "cadence"), exist_ok=True)
    text = json.dumps(out, ensure_ascii=False, indent=1)
    for b in (os.path.join(ROOT, "日志V4", "cadence"),
              os.path.join(os.path.dirname(HERE), "results",
                           "tracking_logic_v4", "cadence")):
        os.makedirs(b, exist_ok=True)
        with open(os.path.join(b, "cadence30_sim_ab.json"), "w",
                  encoding="utf-8", newline="") as f:
            f.write(text)

    def m(integ, key):
        v = integ.get(key)
        return None if v is None else v["mean"]
    for k in ("baseline_alpha020", "robust_applied"):
        it = out[k]
        print("%-18s xlag=%.2f err=%.2f serr=%.2f in60=%s dwell=%.3f to@0.5=%.3f"
              % (k, m(it, "strafe_xlag_p75"), m(it, "err_p75"),
                 m(it, "strafe_err_p75"), fmt(m(it, "first_inner60_median")),
                 m(it, "dwell_inner60"), m(it, "timeout_rate@0.5")))
    bx = out["baseline_alpha020"]["strafe_xlag_p75"]["per_phase_values"]
    rx = out["robust_applied"]["strafe_xlag_p75"]["per_phase_values"]
    print("xlag phases base  :", [round(x, 2) for x in bx])
    print("xlag phases robust:", [round(x, 2) for x in rx],
          " better %d/9" % sum(1 for a, b in zip(rx, bx) if a < b))
    bs = out["baseline_alpha020"]["stable_cmd_p95"]["per_phase_values"]
    rs = out["robust_applied"]["stable_cmd_p95"]["per_phase_values"]
    print("stable_p95 base  :", [round(x, 1) for x in bs])
    print("stable_p95 robust:", [round(x, 1) for x in rs])


def fmt(v):
    return "null" if v is None else "%.3f" % v


if __name__ == "__main__":
    main()
