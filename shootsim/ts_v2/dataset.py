# -*- coding: utf-8 -*-
"""Dataset construction: episode loading, output-grid phase fitting,
stratification stats, and the fixed train/val/holdout split."""
import io
import json
import math
import os
import random
import re
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

from logparse import load_log, fill_world, read_text, config_snapshot  # noqa: E402
from harness import GROUP_A_LOGS, GROUP_B_LOGS, STRAFE_VX  # noqa: E402
import replay as replay_mod  # noqa: E402

LOG_DIR = os.path.join(ROOT, "logs")
OUT_T_RE = re.compile(r"输出t=([0-9.]+)")


def fit_grid_phase(path, period_s=0.015):
    """Fit the global drift-free 15ms output grid phase from [鼠标] 输出t."""
    phases = []
    for line in io.open(path, encoding="utf-8", errors="replace"):
        if "输出t=" not in line:
            continue
        m = OUT_T_RE.search(line)
        if not m:
            continue
        t = float(m.group(1))
        phases.append((t / period_s) % 1.0)  # in [0,1) of a period
    if not phases:
        return None, 0
    # circular mean on the unit circle, then map back to seconds
    sx = sum(math.cos(2 * math.pi * p) for p in phases)
    sy = sum(math.sin(2 * math.pi * p) for p in phases)
    ang = math.atan2(sy, sx) / (2 * math.pi)  # in [-0.5, 0.5)
    if ang < 0:
        ang += 1.0
    # concentration check
    r = math.hypot(sx, sy) / len(phases)
    return ang * period_s, len(phases), r


def load_all_episodes(log_names=None):
    """Load episodes for all sweep logs; return list of dicts with metadata."""
    if log_names is None:
        log_names = list(GROUP_A_LOGS) + list(GROUP_B_LOGS)
    all_eps = []
    for name in log_names:
        path = os.path.join(LOG_DIR, name)
        group = "A" if name in GROUP_A_LOGS else "B"
        meta = GROUP_A_LOGS.get(name) or GROUP_B_LOGS.get(name) or {}
        phase, n_ticks, conc = fit_grid_phase(path)
        ep_list = replay_mod.parse_log_file(
            path, source_chest_ratio=meta.get("chest_ratio", 0.2))
        ep_list = [e for e in ep_list if e["points"]]
        for ep in ep_list:
            all_eps.append({
                "log": name,
                "group": group,
                "chest_ratio": meta.get("chest_ratio", 0.2),
                "output_phase_s": phase,
                "phase_ticks": n_ticks,
                "phase_concentration": conc,
                "_raw": ep,
            })
    return all_eps


def build_replay_episodes(ep_metas, apply_theta=1.0, scale_x=0.44, scale_y=0.51):
    """Convert raw episode dicts to replay.ReplayEpisode objects."""
    out = []
    for m in ep_metas:
        ep = replay_mod.ReplayEpisode(
            m["_raw"], 0, (scale_x, scale_y), 160.0, 3, 0, apply_theta)
        ep.src = os.path.join(LOG_DIR, m["log"])
        out.append(ep)
    return out


def episode_features(ep_meta, apply_theta=1.0, scale_x=0.44, scale_y=0.51):
    """Behavioural features for stratification: duration, strafe share, speed."""
    try:
        ep = replay_mod.ReplayEpisode(
            ep_meta["_raw"], 0, (scale_x, scale_y), 160.0, 3, 0, apply_theta)
    except ValueError:
        return {"log": ep_meta["log"], "group": ep_meta["group"],
                "duration_s": 0.0, "n_points": 0, "speed_segments": 0,
                "strafe_share": 0.0, "abs_vx_p75": 0.0, "abs_vx_max": 0.0,
                "mean_speed": 0.0, "invalid": True}
    pts = ep.points
    duration = pts[-1]["t_rel"] - pts[0]["t_rel"]
    vxs, vys = [], []
    strafe_pts = 0
    total = 0
    for i in range(1, len(pts)):
        a, b = pts[i - 1], pts[i]
        dt = max(1e-3, b["t_rel"] - a["t_rel"])
        if a["box"] is None or b["box"] is None:
            continue
        # world motion between captures (ReplayEpisode.wx/wy carry cum scroll)
        jx = (ep.wx[i] - ep.wx[i - 1]) / dt
        jy = (ep.wy[i] - ep.wy[i - 1]) / dt
        vxs.append(jx)
        vys.append(jy)
        total += 1
        if abs(jx) >= STRAFE_VX and abs(jx) >= 1.5 * abs(jy):
            strafe_pts += 1
    ax = [abs(v) for v in vxs]
    return {
        "log": ep_meta["log"],
        "group": ep_meta["group"],
        "duration_s": round(duration, 3),
        "n_points": len(pts),
        "speed_segments": total,
        "strafe_share": strafe_pts / total if total else 0.0,
        "abs_vx_p75": round(sorted(ax)[round(0.75 * (len(ax) - 1))], 1) if ax else 0.0,
        "abs_vx_max": round(max(ax), 1) if ax else 0.0,
        "mean_speed": round(sum(math.hypot(x, y) for x, y in zip(vxs, vys))
                            / len(vxs), 1) if vxs else 0.0,
    }


def stratify_split(features, seed=20260906, train=0.6, val=0.2):
    """Episode-level split stratified by (log, strafe class)."""
    rng = random.Random(seed)
    strata = {}
    for i, f in enumerate(features):
        if f["strafe_share"] >= 0.30:
            cls = "strafe_heavy"
        elif f["strafe_share"] >= 0.10:
            cls = "strafe_mid"
        else:
            cls = "slow"
        key = (f["log"], cls)
        strata.setdefault(key, []).append(i)
    assign = [None] * len(features)
    for key in sorted(strata):
        idxs = sorted(strata[key])
        rng.shuffle(idxs)
        n = len(idxs)
        n_train = int(round(train * n))
        n_val = int(round(val * n))
        for j, i in enumerate(idxs):
            if j < n_train:
                assign[i] = "train"
            elif j < n_train + n_val:
                assign[i] = "val"
            else:
                assign[i] = "holdout"
    return assign
