# -*- coding: utf-8 -*-
"""Simulator fidelity calibration: closed-loop baseline on the primary log,
sweeping the camera-mixing theta and reconstruction scale; compares the sim's
per-frame detections/commands/states against the live log rows."""
import json
import math
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from harness import (build_sim_cfg, load_production_cfg, patch_cfg,
                     run_episode, percentile)  # noqa: E402
from dataset import load_all_episodes, build_replay_episodes  # noqa: E402
from logparse import load_log as parse_log_raw, fill_world  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(HERE), "results", "tracking_sweep_20260906_v2")
PRIMARY = "aim_20260906_003353.log"


def raw_log_groups(log_name):
    """Full per-session [瞄准观测] row groups straight from the raw JSON lines."""
    path = os.path.join(os.path.dirname(os.path.dirname(HERE)), "logs", log_name)
    data = parse_log_raw(path)
    return data["episodes"]


def log_rows_for_group(group):
    """Per-row comparison fields from one raw session group (raw JSON fields)."""
    rows = []
    for p in group["points"]:
        dets = p["dets"]
        chosen_center = None
        if dets:
            tgt = p["target"]
            if tgt is not None:
                chest = float(group["snap"].get("chest_ratio") or 0.2)

                def ad(d):
                    x1, y1, x2, y2 = d["bbox"]
                    ax = (x1 + x2) * 0.5
                    ay = (y1 + y2) * 0.5 if d["cls"] == 1 else y1 + (y2 - y1) * chest
                    return (ax - tgt[0]) ** 2 + (ay - tgt[1]) ** 2
                chosen = min(dets, key=ad)
            else:
                chosen = dets[0]
            b = chosen["bbox"]
            chosen_center = ((b[0] + b[2]) * 0.5, (b[1] + b[3]) * 0.5)
        rows.append({
            "detected": p["detected"],
            "center": chosen_center,
            "target": p["target"],
            "reason": p["reason"],
            "lock_reason": p["lock_reason"],
            "cmd": p["cmd"],
            "err": p["obs_err"],
            "vel_raw": p["vel_raw"],
            "lead": p["lead"],
            "horizon_ms": p["horizon_ms"],
        })
    return rows


def alignment(group):
    """Return (n_dropped_leading, pre_frames) mapping sim obs j → log row index."""
    pts = group["points"]
    n_drop = 0
    for p in pts:
        if p["detected"]:
            break
        n_drop += 1
    f0 = pts[n_drop]["frame"] if n_drop < len(pts) else pts[-1]["frame"]
    pre_frames = max(0, int(f0) - 1)
    return n_drop, pre_frames


def compare(rec, log_rows, n_drop=0, pre_frames=0, cap_counts=280.3026):
    """Frame-aligned sim-vs-log comparison over one episode.

    sim obs j (j >= pre_frames) maps to log row (j - pre_frames + n_drop);
    sim obs j < pre_frames are synthesized pre-detection frames (skipped).
    """
    det_errs = []
    cmd_int_diffs = []
    dir_match = dir_total = 0
    state_match = state_total = 0
    err_diffs = []
    vel_diffs = []
    horizon_diffs = []
    # pre-divergence window (first N controls) — isolates structural fidelity
    # from chaotic amplification of sub-tick timing races
    dir_match5 = dir_total5 = 0
    state_match5 = state_total5 = 0
    cmd_diffs5 = []
    controls_seen = 0
    for j, s in enumerate(rec.obs):
        li = j - pre_frames + n_drop
        if li < 0 or li >= len(log_rows):
            continue
        L = log_rows[li]
        in_window = controls_seen < 5
        if s["cmd"] != (0.0, 0.0):
            controls_seen += 1
        if L["detected"] and s["detected"] and s["dets"]:
            sd = max(s["dets"], key=lambda d: (d[1][2] - d[1][0]) * (d[1][3] - d[1][1]))
            lc = L["center"]
            if sd and lc:
                det_errs.append(math.hypot((sd[1][0] + sd[1][2]) * 0.5 - lc[0],
                                           (sd[1][1] + sd[1][3]) * 0.5 - lc[1]))
        sc = s["cmd"]
        lcmd = L["cmd"]
        if sc != (0.0, 0.0) or lcmd != (0.0, 0.0):
            dx = round(sc[0]) - round(lcmd[0])
            dy = round(sc[1]) - round(lcmd[1])
            cdiff = max(abs(dx), abs(dy))
            cmd_int_diffs.append(cdiff)
            if abs(lcmd[0]) >= 1 or abs(sc[0]) >= 1:
                dir_total += 1
                if (sc[0] > 0) == (lcmd[0] > 0):
                    dir_match += 1
                if in_window:
                    dir_total5 += 1
                    if (sc[0] > 0) == (lcmd[0] > 0):
                        dir_match5 += 1
            if in_window:
                cmd_diffs5.append(cdiff)
        state_total += 1
        st_ok = ((s["target"] is not None) == (L["target"] is not None)
                 and str(s["reason"]) == str(L["reason"])
                 and _lock_class(s["lock_reason"]) == _lock_class(L["lock_reason"]))
        if st_ok:
            state_match += 1
        if in_window:
            state_total5 += 1
            if st_ok:
                state_match5 += 1
        if L["err"] != (0.0, 0.0) and s["obs_err"] != (0.0, 0.0):
            err_diffs.append(math.hypot(s["obs_err"][0] - L["err"][0],
                                        s["obs_err"][1] - L["err"][1]))
        if L["vel_raw"] != (0.0, 0.0) and s["vel_raw"] != (0.0, 0.0):
            vel_diffs.append(math.hypot(s["vel_raw"][0] - L["vel_raw"][0],
                                        s["vel_raw"][1] - L["vel_raw"][1]))
        if L["reason"] and str(L["reason"]).startswith("arrival") \
                and s["horizon_ms"] and L["horizon_ms"] > 0:
            horizon_diffs.append(abs(s["horizon_ms"] - L["horizon_ms"]))
    return {
        "frames_compared": state_total,
        "det_center_err_median": statistics.median(det_errs) if det_errs else None,
        "det_center_err_p75": percentile(det_errs, .75) if det_errs else None,
        "det_center_err_p95": percentile(det_errs, .95) if det_errs else None,
        "det_err_within2px_rate": (sum(1 for v in det_errs if v <= 2.0) / len(det_errs)
                                   if det_errs else None),
        "cmd_int_mae": (statistics.mean(cmd_int_diffs) if cmd_int_diffs else None),
        "cmd_int_median": (statistics.median(cmd_int_diffs) if cmd_int_diffs else None),
        "dir_match_rate": dir_match / dir_total if dir_total else None,
        "state_match_rate": state_match / state_total if state_total else None,
        "err_diff_p75": percentile(err_diffs, .75) if err_diffs else None,
        "vel_diff_p75": percentile(vel_diffs, .75) if vel_diffs else None,
        "horizon_diff_median": (statistics.median(horizon_diffs)
                                if horizon_diffs else None),
        "first5_dir_match_rate": dir_match5 / dir_total5 if dir_total5 else None,
        "first5_state_match_rate": state_match5 / state_total5
        if state_total5 else None,
        "first5_cmd_median": (statistics.median(cmd_diffs5) if cmd_diffs5 else None),
    }


def _lock_class(reason):
    r = str(reason or "")
    if r.startswith("locked"):
        return "locked"
    if r.startswith(("hold_", "switched_")):
        return "hold"
    if r.startswith("predict_missing"):
        return "missing"
    if r == "no_detections":
        return "none"
    return "other"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    prod = load_production_cfg()
    # raw groups (full rows) for the primary log, only those with detections
    groups_all = raw_log_groups(PRIMARY)
    groups = [g for g in groups_all
              if any(p["detected"] for p in g["points"])]
    metas = [m for m in load_all_episodes([PRIMARY]) if m["log"] == PRIMARY]
    assert len(groups) == len(metas), (len(groups), len(metas))
    results = []
    for theta in (1.0, 0.75, 0.5, 0.25):
        for sx in (0.40, 0.44, 0.48):
            eps = build_replay_episodes(metas, apply_theta=theta,
                                        scale_x=sx, scale_y=0.51)
            sim_cfg = build_sim_cfg([PRIMARY], apply_theta=theta,
                                    scale_x=sx, scale_y=0.51)
            agg = {}
            for g, m, ep in zip(groups, metas, eps):
                rec = run_episode(sim_cfg, prod, ep, output_phase_s=m["output_phase_s"])
                log_rows = log_rows_for_group(g)
                n_drop, pre_frames = alignment(g)
                cmp_res = compare(rec, log_rows, n_drop, pre_frames)
                for k, v in cmp_res.items():
                    if k == "frames_compared" or v is None:
                        continue
                    agg.setdefault(k, []).append(v)
            row = {"theta": theta, "scale_x": sx}
            for k, vals in agg.items():
                row[k] = round(statistics.mean(vals), 4)
            results.append(row)
            print(json.dumps(row, ensure_ascii=False))
    with open(os.path.join(OUT_DIR, "calibration_theta_scale.json"), "w",
              encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
