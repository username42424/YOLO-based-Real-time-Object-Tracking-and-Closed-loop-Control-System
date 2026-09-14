# -*- coding: utf-8 -*-
"""Closed-loop trajectory reconstruction model selection.

For candidate models (sign, delay-theta, scale) of how mouse counts move the
camera, reconstruct the target's physical (camera-free) trajectory from a live
log and score it by constant-velocity one-step prediction residual.  The
correct model minimises the residual; wrong sign/delay injects camera-scroll
correlated spikes into the reconstructed motion.
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from logparse import load_log, fill_world

ROOT = r"C:\Users\12951\Desktop\12323\logs"


def _reconstruct(pts, axis, scale, sign, theta):
    """World positions under a candidate camera model for one axis.

    scroll(k) = sum_{j<=k} (theta*net_j + (1-theta)*net_{j-1}) * scale
    world(k)  = cam(k) + sign*scroll(k)   [sign=+1: counts>0 pans right →
    scene shifts left on screen, so world = screen + scroll]
    """
    w = []
    prev_net = 0.0
    acc = 0.0
    for p in pts:
        net = float(p["net"][axis])
        acc += (theta * net + (1.0 - theta) * prev_net) * scale
        prev_net = net
        cam = (p["bcx"] if axis == 0 else p["bcy"])
        if cam is None:
            w.append(None)
        else:
            w.append(cam + sign * acc)
    return w


def _residual_stats(w, ts):
    """Constant-velocity one-step prediction residual on the world path."""
    errs = []
    v_prev = None
    for i in range(2, len(w)):
        if w[i] is None or w[i - 1] is None or w[i - 2] is None:
            v_prev = None
            continue
        dt1 = max(1e-3, ts[i - 1] - ts[i - 2])
        dt2 = max(1e-3, ts[i] - ts[i - 1])
        v = (w[i - 1] - w[i - 2]) / dt1
        pred = w[i - 1] + (v * dt2 if v_prev is None else v * dt2)
        errs.append(abs(w[i] - pred))
        v_prev = v
    if not errs:
        return None
    errs.sort()
    n = len(errs)
    return {
        "mean": sum(errs) / n,
        "p50": errs[int(0.50 * (n - 1))],
        "p75": errs[int(0.75 * (n - 1))],
        "p90": errs[int(0.90 * (n - 1))],
        "n": n,
    }


def _smoothness(w, ts):
    """Second-difference energy (jerk proxy) — robust when motion accelerates."""
    acc = []
    for i in range(2, len(w)):
        if w[i] is None or w[i - 1] is None or w[i - 2] is None:
            continue
        dt1 = max(1e-3, ts[i - 1] - ts[i - 2])
        dt2 = max(1e-3, ts[i] - ts[i - 1])
        v1 = (w[i - 1] - w[i - 2]) / dt1
        v2 = (w[i] - w[i - 1]) / dt2
        acc.append(abs(v2 - v1))
    if not acc:
        return None
    acc.sort()
    return acc[int(0.75 * (len(acc) - 1))]


def fit_axis(pts, axis, base_scale, sign_grid=(1.0, -1.0),
             theta_grid=(1.0, 0.75, 0.5, 0.25, 0.0),
             scale_grid=None):
    ts = [p["t"] for p in pts]
    scales = scale_grid or [round(base_scale * f, 4)
                            for f in (0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0,
                                      1.05, 1.10, 1.15, 1.20, 1.25, 1.30)]
    best = None
    rows = []
    for sign in sign_grid:
        for theta in theta_grid:
            for sc in scales:
                w = _reconstruct(pts, axis, sc, sign, theta)
                st = _residual_stats(w, ts)
                sm = _smoothness(w, ts)
                if st is None:
                    continue
                score = st["p75"]
                rows.append({"sign": sign, "theta": theta, "scale": sc,
                             "res_p50": round(st["p50"], 3),
                             "res_p75": round(st["p75"], 3),
                             "res_mean": round(st["mean"], 3),
                             "jerk_p75": round(sm, 1) if sm else None})
                if best is None or score < best["res_p75"]:
                    best = rows[-1]
    rows.sort(key=lambda r: r["res_p75"])
    return best, rows[:8]


def analyse_log(path, view_scale=0.44, view_scale_y=0.51, restrict_group=True):
    data = load_log(path, view_scale, view_scale_y)
    out = {"src": os.path.basename(path), "episodes": len(data["episodes"]),
           "snap": {k: v for k, v in data["snap"].items()
                    if k in ("lock_box_filter_mode", "lock_box_smoothing_alpha",
                             "chest_ratio", "recoil_line")}}
    axes = {}
    for axis, base in ((0, view_scale), (1, view_scale_y)):
        pts_all = []
        for ep in data["episodes"]:
            fill_world(ep)
            pts_all.extend(ep["points"])
        best, top = fit_axis(pts_all, axis, base)
        base_w = _reconstruct(pts_all, axis, base, 1.0, 1.0)
        ts = [p["t"] for p in pts_all]
        base_st = _residual_stats(base_w, ts)
        wrong_sign = _residual_stats(
            _reconstruct(pts_all, axis, base, -1.0, 1.0), ts)
        axes[f"axis{axis}"] = {
            "base_scale": base,
            "baseline_model_res_p75": round(base_st["p75"], 3),
            "wrong_sign_res_p75": round(wrong_sign["p75"], 3),
            "best": best,
            "top_models": top,
        }
    out["axes"] = axes
    return out


def main():
    logs = [
        "aim_20260906_003353.log",
        "aim_20260906_001842.log",
        "aim_20260906_000134.log",
    ]
    for name in logs:
        res = analyse_log(os.path.join(ROOT, name))
        print("=" * 90)
        print(name, res["snap"])
        for ax in ("axis0", "axis1"):
            a = res["axes"][ax]
            print("  %s: baseline res_p75=%.2fpx  wrong-sign res_p75=%.2fpx  best=%s"
                  % (ax, a["baseline_model_res_p75"], a["wrong_sign_res_p75"],
                     a["best"]))
            for r in a["top_models"][:4]:
                print("      ", r)


if __name__ == "__main__":
    main()
