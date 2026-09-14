# -*- coding: utf-8 -*-
"""v3 dataset: all seven logs, per-log metadata, fresh train/validation split,
group-B logs reserved as the external (never-in-search) holdout, baseline
config pinning with SHA256.

Config families (from log headers, not comments):
  group A — current production family: chest_ratio 0.2, default vertical
            recoil OFF, lock-filter feature present.
  group B — older build: chest_ratio 0.3, default vertical recoil ON, no
            lock-filter header line.  Only X-focused metrics are trusted on
            these; flagged in reports.
"""
import hashlib
import json
import math
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

from logparse import load_log, fill_world  # noqa: E402
from dataset import fit_grid_phase  # noqa: E402
from eval3 import episode_eligibility  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(HERE))
LOG_DIR = os.path.join(ROOT, "logs")
OUT_DIR = os.path.join(os.path.dirname(HERE), "results", "tracking_logic_v3")

BASELINE_CONFIG_PATH = os.path.join(ROOT, "config_backup_alpha020_20260906.json")
SPLIT_SEED = 20260906
STRAFE_VX = 400.0

LOG_META = {
    "aim_20260906_003353.log": dict(group="A", chest_ratio=0.2,
                                    lock_alpha=0.20, vertical_recoil=False),
    "aim_20260906_001842.log": dict(group="A", chest_ratio=0.2,
                                    lock_alpha=0.65, vertical_recoil=False),
    "aim_20260906_000134.log": dict(group="A", chest_ratio=0.2,
                                    lock_alpha="adaptive", vertical_recoil=False),
    "aim_20260905_232312.log": dict(group="B", chest_ratio=0.3,
                                    lock_alpha=None, vertical_recoil=True),
    "aim_20260905_231714.log": dict(group="B", chest_ratio=0.3,
                                    lock_alpha=None, vertical_recoil=True),
    "aim_20260905_230935.log": dict(group="B", chest_ratio=0.3,
                                    lock_alpha=None, vertical_recoil=True),
    "aim_20260905_230244.log": dict(group="B", chest_ratio=0.3,
                                    lock_alpha=None, vertical_recoil=True),
}


def baseline_config():
    with open(BASELINE_CONFIG_PATH, encoding="utf-8") as f:
        return json.load(f)


def baseline_sha256():
    h = hashlib.sha256()
    with open(BASELINE_CONFIG_PATH, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def load_all_episodes(log_names=None, scale=(0.44, 0.51)):
    """Parse logs → per-episode metadata + logparse point groups.

    Returns list of dicts: {log, group, chest_ratio, vertical_recoil,
    output_phase_s, points (logparse group), eligibility}.
    Invalid episodes (no detections) are dropped.
    """
    import replay as replay_mod
    if log_names is None:
        log_names = list(LOG_META)
    out = []
    for name in log_names:
        meta = LOG_META[name]
        path = os.path.join(LOG_DIR, name)
        phase, n_ticks, conc = fit_grid_phase(path)
        data = load_log(path, scale[0], scale[1])
        for ep in data["episodes"]:
            fill_world(ep)
            pts = ep["points"]
            if not any(p["detected"] for p in pts):
                continue
            if sum(1 for p in pts if p["box"] is not None) < 3:
                continue  # too few observations to replay
            elig = episode_eligibility(pts)
            # behaviour features for stratification (world motion)
            vxs, vys, strafe_pts, total = [], [], 0, 0
            for i in range(1, len(pts)):
                a, b = pts[i - 1], pts[i]
                if a["bcx"] is None or b["bcx"] is None:
                    continue
                dt = max(1e-3, b["t"] - a["t"])
                jx = (b["wx"] - a["wx"]) / dt
                jy = (b["wy"] - a["wy"]) / dt
                vxs.append(jx)
                vys.append(jy)
                total += 1
                if abs(jx) >= STRAFE_VX and abs(jx) >= 1.5 * abs(jy):
                    strafe_pts += 1
            strafe_share = strafe_pts / total if total else 0.0
            out.append({
                "log": name,
                "group": meta["group"],
                "chest_ratio": meta["chest_ratio"],
                "vertical_recoil": meta["vertical_recoil"],
                "output_phase_s": phase,
                "phase_ticks": n_ticks,
                "phase_concentration": conc,
                "duration_s": round(pts[-1]["t"] - pts[0]["t"], 3),
                "n_points": len(pts),
                "strafe_share": round(strafe_share, 3),
                "elig": elig,
                "points": pts,
            })
    return out


def replay_episodes(ep_metas):
    """Convert logparse point groups to replay.ReplayEpisode objects (the
    sim's world-trajectory format)."""
    import replay as replay_mod
    out = []
    for m in ep_metas:
        rows = []
        for p in m["points"]:
            rows.append({
                "frame": 0,  # placeholder; ReplayEpisode re-sorts anyway
                "t_rel": p["t"],
                "lat_s": p["lat_s"],
                "box": p["box"],
                "cx": p["bcx"] if p["bcx"] is not None else 0.0,
                "cy": p["bcy"] if p["bcy"] is not None else 0.0,
                "w": (p["box"][2] - p["box"][0]) if p["box"] else 0.0,
                "h": (p["box"][3] - p["box"][1]) if p["box"] else 0.0,
                "cls": p["cls"] if p["cls"] is not None else 0,
                "conf": p["conf"] if p["conf"] is not None else 0.0,
                "net": p["net"],
                "ref": p["cx_ref"],
                "err": p["obs_err"],
                "lead": p["lead"],
                "ema": p["target"] or (0.0, 0.0),
                "target": p["target"] or (0.0, 0.0),
                "sent": p["cmd"],
                "detected": p["detected"],
                "detections": [
                    {"cls": d["cls"], "conf": d["conf"], "bbox": d["bbox"]}
                    for d in p["dets"]],
                "dot_status": p["dot_status"],
            })
        ep = {
            "src": os.path.join(LOG_DIR, m["log"]),
            "t_start": 0.0, "t_end": 0.0, "points": rows,
            "motion_already_observed": True,
        }
        out.append(replay_mod.ReplayEpisode(ep, 0, (0.44, 0.51), 160.0, 3, 0, 1.0))
    return out


def stratified_split(ep_metas, train_frac=0.6, seed=SPLIT_SEED):
    """Group-A episodes → train/validation (60/40, stratified by log × strafe
    class).  Group-B episodes → external holdout (never used in search)."""
    rng = random.Random(seed)
    assign = [None] * len(ep_metas)
    strata = {}
    for i, m in enumerate(ep_metas):
        if m["group"] != "A":
            assign[i] = "holdout_ext"
            continue
        cls = ("strafe_heavy" if m["strafe_share"] >= 0.30
               else "strafe_mid" if m["strafe_share"] >= 0.10 else "slow")
        strata.setdefault((m["log"], cls), []).append(i)
    for key in sorted(strata):
        idxs = sorted(strata[key])
        rng.shuffle(idxs)
        n_train = int(round(train_frac * len(idxs)))
        for j, i in enumerate(idxs):
            assign[i] = "train" if j < n_train else "validation"
    return assign


def build_split_manifest():
    metas = load_all_episodes()
    assign = stratified_split(metas)
    idx = {k: [i for i, a in enumerate(assign) if a == k]
           for k in ("train", "validation", "holdout_ext")}
    manifest = {
        "seed": SPLIT_SEED,
        "baseline_config_path": BASELINE_CONFIG_PATH,
        "baseline_config_sha256": baseline_sha256(),
        "logs": LOG_META,
        "splits": {k: [{"index": i,
                        "log": metas[i]["log"],
                        "group": metas[i]["group"],
                        "duration_s": metas[i]["duration_s"],
                        "n_points": metas[i]["n_points"],
                        "strafe_share": metas[i]["strafe_share"],
                        "max_detected_streak_s":
                            metas[i]["elig"]["max_detected_streak_s"],
                        "started_outside_inner60":
                            metas[i]["elig"]["started_outside_inner60"]}
                       for i in idx[k]] for k in idx},
        "note": ("train/validation come from group-A logs only; group-B logs "
                 "are the external holdout (never in search/selection; "
                 "different chest_ratio, vertical recoil ON, older build)."),
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "dataset_split_v3.json"), "w",
              encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    return metas, assign, idx
