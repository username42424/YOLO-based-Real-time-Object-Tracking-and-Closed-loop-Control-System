# -*- coding: utf-8 -*-
"""Evaluate the user's new live logs (robust config, alpha=0.18) with the
corrected v4 metrics, and run a closed-loop sim A/B on the same world
trajectories: baseline (alpha=0.2) vs robust.

Outputs: 日志V4/newlogs/newlogs_v4.json (+ mirror)."""
import json
import math
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "shootsim"))

from logparse import load_log, fill_world  # noqa: E402
from dataset import fit_grid_phase  # noqa: E402
from eval3 import (evaluate_v3, episode_eligibility, percentile,
                   STRAFE_VX)  # noqa: E402
from harness import build_sim_cfg  # noqa: E402
from run_v4 import robust_cfg, make_plan_hook, write_both  # noqa: E402

LOG_DIR = os.path.join(ROOT, "logs")
NEW_LOGS = ["aim_20260906_125713.log", "aim_20260906_130338.log",
            "aim_20260906_131131.log"]
REF_LOG_A02 = ["aim_20260906_003353.log"]   # live alpha=0.2 reference (old build)
BASELINE_CONFIG = os.path.join(ROOT, "config_backup_alpha020_20260906.json")
import replay as replay_mod  # noqa: E402


def to_replay_episode(pts, src):
    rows = []
    for p in pts:
        rows.append({
            "frame": 0, "t_rel": p["t"], "lat_s": p["lat_s"],
            "box": p["box"],
            "cx": p["bcx"] if p["bcx"] is not None else 0.0,
            "cy": p["bcy"] if p["bcy"] is not None else 0.0,
            "w": (p["box"][2] - p["box"][0]) if p["box"] else 0.0,
            "h": (p["box"][3] - p["box"][1]) if p["box"] else 0.0,
            "cls": p["cls"] if p["cls"] is not None else 0,
            "conf": p["conf"] if p["conf"] is not None else 0.0,
            "net": p["net"], "ref": p["cx_ref"], "err": p["obs_err"],
            "lead": p["lead"], "ema": p["target"] or (0.0, 0.0),
            "target": p["target"] or (0.0, 0.0), "sent": p["cmd"],
            "detected": p["detected"],
            "detections": [{"cls": d["cls"], "conf": d["conf"],
                            "bbox": d["bbox"]} for d in p["dets"]],
            "dot_status": p["dot_status"],
        })
    ep = {"src": src, "t_start": 0.0, "t_end": 0.0, "points": rows,
          "motion_already_observed": True}
    return replay_mod.ReplayEpisode(ep, 0, (0.44, 0.51), 160.0, 3, 0, 1.0)


def live_metrics(points):
    """Corrected live-observed metrics from a log's own rows."""
    errs, strafe_errs, strafe_lags = [], [], []
    speeds = []
    locked_n = 0
    reasons = {}
    for p in points:
        if p["lock_reason"] and str(p["lock_reason"]).startswith("locked"):
            locked_n += 1
        reasons[p["reason"]] = reasons.get(p["reason"], 0) + 1
        if not (p["lock_reason"] and str(p["lock_reason"]).startswith("locked")):
            continue
        if p["bcx"] is None:
            continue
        vx, vy = p["vx"] or 0.0, p["vy"] or 0.0
        sp = math.hypot(vx, vy)
        speeds.append(sp)
        err = math.hypot(*p["obs_err"])
        if sp > 100.0:
            errs.append(err)
        if abs(vx) >= STRAFE_VX and abs(vx) >= 1.5 * abs(vy) and sp > 1.0:
            strafe_errs.append(err)
            strafe_lags.append((p["obs_err"][0] * vx + p["obs_err"][1] * vy) / sp)
    return {
        "locked_obs": locked_n,
        "movement_samples": len(errs),
        "err_p50": percentile(errs, .5), "err_p75": percentile(errs, .75),
        "strafe_samples": len(strafe_errs),
        "strafe_err_p75": percentile(strafe_errs, .75),
        "strafe_xlag_p75": percentile(strafe_lags, .75),
        "speed_p90": percentile(speeds, .9),
        "reason_counts": reasons,
    }


def main():
    out = {"new_logs": NEW_LOGS, "live": {}, "sim_ab": {}}
    # ── 1. live-observed metrics for the new (robust) logs ──────────────
    all_new_metas = []
    for name in NEW_LOGS:
        path = os.path.join(LOG_DIR, name)
        data = load_log(path, 0.44, 0.51)
        lm_all = []
        for ep in data["episodes"]:
            fill_world(ep)
            pts = ep["points"]
            if not any(p["detected"] for p in pts):
                continue
            if sum(1 for p in pts if p["box"] is not None) < 3:
                continue
            lm = live_metrics(pts)
            lm["ep_points"] = len(pts)
            lm["duration_s"] = round(pts[-1]["t"] - pts[0]["t"], 3)
            lm_all.append(lm)
            all_new_metas.append({"log": name, "points": pts,
                                  "elig": episode_eligibility(pts),
                                  "live": lm})
        agg = {}
        for key in ("err_p75", "strafe_err_p75", "strafe_xlag_p75"):
            pool = [lm[key] for lm in lm_all if lm[key] is not None]
            pool = [x for lm in lm_all for x in [lm[key]] if lm[key] is not None]
            # pool the underlying samples instead: recompute pooled percentiles
            agg[key] = None
        out["live"][name] = {
            "episodes": len(lm_all),
            "locked_obs": sum(lm["locked_obs"] for lm in lm_all),
            "pooled": None,
        }
    # pooled live metrics across each log's episodes (recompute from points)
    def pooled_live(pts_lists):
        errs, strafe_errs, strafe_lags, speeds = [], [], [], []
        for pts in pts_lists:
            for p in pts:
                if not (p["lock_reason"] and str(p["lock_reason"]).startswith("locked")):
                    continue
                if p["bcx"] is None:
                    continue
                vx, vy = p["vx"] or 0.0, p["vy"] or 0.0
                sp = math.hypot(vx, vy)
                speeds.append(sp)
                err = math.hypot(*p["obs_err"])
                if sp > 100.0:
                    errs.append(err)
                if abs(vx) >= STRAFE_VX and abs(vx) >= 1.5 * abs(vy) and sp > 1.0:
                    strafe_errs.append(err)
                    strafe_lags.append((p["obs_err"][0] * vx
                                        + p["obs_err"][1] * vy) / sp)
        return {
            "movement_samples": len(errs),
            "err_p50": percentile(errs, .5), "err_p75": percentile(errs, .75),
            "strafe_samples": len(strafe_errs),
            "strafe_err_p75": percentile(strafe_errs, .75),
            "strafe_xlag_p50": percentile(strafe_lags, .5),
            "strafe_xlag_p75": percentile(strafe_lags, .75),
            "speed_p90": percentile(speeds, .9),
        }

    new_pts = [m["points"] for m in all_new_metas]
    out["live"]["pooled_robust_newlogs"] = pooled_live(new_pts)

    # reference: live alpha=0.2 log (old build) with the same corrected metrics
    ref_path = os.path.join(LOG_DIR, REF_LOG_A02[0])
    ref_data = load_log(ref_path, 0.44, 0.51)
    ref_pts = []
    for ep in ref_data["episodes"]:
        fill_world(ep)
        if any(p["detected"] for p in ep["points"]):
            ref_pts.append(ep["points"])
    out["live"]["pooled_alpha020_reference"] = pooled_live(ref_pts)
    out["live"]["pooled_alpha020_reference"]["log"] = REF_LOG_A02[0]
    out["live"]["pooled_alpha020_reference"]["caveat"] = (
        "old engine build (pre 0143 X-reversal change), different session/map")

    # ── 2. closed-loop sim A/B on the new logs' world trajectories ──────
    metas = all_new_metas
    episodes, eligs, phases = [], [], []
    for m in metas:
        try:
            episodes.append(to_replay_episode(m["points"],
                                              os.path.join(LOG_DIR, m["log"])))
            eligs.append(m["elig"])
            phase, _, _ = fit_grid_phase(os.path.join(LOG_DIR, m["log"]))
            phases.append(phase if phase is not None else 0.0075)
        except ValueError:
            continue
    out["sim_ab"]["episodes_replayed"] = len(episodes)
    sim_cfg = build_sim_cfg(NEW_LOGS)
    base_cfg = json.load(open(BASELINE_CONFIG, encoding="utf-8"))
    rob = robust_cfg()
    out["sim_ab"]["baseline_alpha020"] = evaluate_v3(
        sim_cfg, base_cfg, episodes, eligs, phases=phases)
    out["sim_ab"]["robust_applied"] = evaluate_v3(
        sim_cfg, rob, episodes, eligs, phases=phases)

    os.makedirs(os.path.join(ROOT, "日志V4", "newlogs"), exist_ok=True)
    text = json.dumps(out, ensure_ascii=False, indent=1)
    for base in (os.path.join(ROOT, "日志V4", "newlogs"),
                 os.path.join(os.path.dirname(HERE), "results",
                              "tracking_logic_v4", "newlogs")):
        os.makedirs(base, exist_ok=True)
        with open(os.path.join(base, "newlogs_v4.json"), "w",
                  encoding="utf-8", newline="") as f:
            f.write(text)

    # ── 3. console summary ───────────────────────────────────────────────
    def m(integ, key):
        v = integ.get(key)
        return None if v is None else v["mean"]

    print("== 新日志规模 ==")
    print("episodes:", len(episodes), " locked_obs(robust live):",
          sum(lm["locked_obs"] for lm in
              [m["live"] for m in []]) if False else
          out["live"]["pooled_robust_newlogs"]["movement_samples"], "movement samples")
    print()
    print("== 实测（日志自身 observed_error / velocity_raw，修正口径）")
    for tag in ("pooled_alpha020_reference", "pooled_robust_newlogs"):
        d = out["live"][tag]
        print(" %-26s err_p75=%.2f strafe_err_p75=%.2f strafe_xlag_p75=%.2f "
              "(n=%d, strafe n=%d, speed_p90=%.0f)" % (
                  tag, d["err_p75"], d["strafe_err_p75"], d["strafe_xlag_p75"],
                  d["movement_samples"], d["strafe_samples"], d["speed_p90"]))
    print()
    print("== 闭环模拟 A/B（同一目标轨迹，9 相位均值）")
    for k in ("baseline_alpha020", "robust_applied"):
        it = out["sim_ab"][k]
        print(" %-20s xlag=%.2f err=%.2f serr=%.2f in60=%s dwell=%.3f "
              "to@0.5=%.3f" % (
              k, m(it, "strafe_xlag_p75"), m(it, "err_p75"),
              m(it, "strafe_err_p75"), m(it, "first_inner60_median"),
              m(it, "dwell_inner60"), m(it, "timeout_rate@0.5")))
    bx = out["sim_ab"]["baseline_alpha020"]["strafe_xlag_p75"]["per_phase_values"]
    rx = out["sim_ab"]["robust_applied"]["strafe_xlag_p75"]["per_phase_values"]
    print(" xlag phases baseline:", [round(x, 2) for x in bx])
    print(" xlag phases robust  :", [round(x, 2) for x in rx],
          " better %d/9" % sum(1 for a, b in zip(rx, bx) if a < b))
    bs = out["sim_ab"]["baseline_alpha020"]["stable_cmd_p95"]["per_phase_values"]
    rs = out["sim_ab"]["robust_applied"]["stable_cmd_p95"]["per_phase_values"]
    print(" stable_p95 baseline:", [round(x, 1) for x in bs])
    print(" stable_p95 robust  :", [round(x, 1) for x in rs])


if __name__ == "__main__":
    main()
