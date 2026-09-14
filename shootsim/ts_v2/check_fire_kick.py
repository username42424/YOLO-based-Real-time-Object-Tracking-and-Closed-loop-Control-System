# -*- coding: utf-8 -*-
"""检验假设：开火抖动被误读为换向。

把锁定帧分成 开火起始后0-150ms / 开火中(>150ms) / 近1.5s无开火(基线) 三窗，
对比垂直速度尖峰、横向符号翻转率、锁定切换、径向误差。
"""
import json
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
MARKER = "[瞄准观测] "
FIRE_DOWN = re.compile(r"\[开火\] RMB→LMB down t=([\d.]+)")
FIRE_UP = re.compile(r"\[开火\] RMB up t=([\d.]+)")

WINDOW_ONSET = 0.150   # 开火起始后
WINDOW_BASE = 1.5      # 距最近开火起始超过该值视为基线


def percentile(values, q):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[round(q * (len(ordered) - 1))]


def parse(path):
    rows, fires = [], []
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if MARKER in line:
                try:
                    rows.append(json.loads(line.split(MARKER, 1)[1]))
                except (ValueError, TypeError):
                    pass
                continue
            m = FIRE_DOWN.search(line)
            if m:
                fires.append((float(m.group(1)), "down"))
                continue
            m = FIRE_UP.search(line)
            if m:
                fires.append((float(m.group(1)), "up"))
    return rows, fires


def analyze(path):
    rows, fires = parse(path)
    events = iter(fires)
    pending = list(fires)
    fire_state = {"down_t": None}
    di = 0
    out = {"onset": {}, "during": {}, "base": {}}
    samples = {k: [] for k in out}
    prev_vx = None
    hist = []
    for row in rows:
        t = float(row.get("capture_t", 0.0))
        while di < len(fires) and fires[di][0] <= t:
            when, kind = fires[di]
            if kind == "down":
                fire_state["down_t"] = when
            else:
                fire_state["down_t"] = None
            di += 1
        lock = str(row.get("lock_reason", ""))
        if not (lock.startswith("locked") or lock.startswith("switched")):
            prev_vx = None
            hist = []
            continue
        vx, vy = row.get("velocity_raw") or (0.0, 0.0)
        ex, ey = row.get("observed_error") or (0.0, 0.0)
        down_t = fire_state["down_t"]
        if down_t is not None and 0.0 <= t - down_t <= WINDOW_ONSET:
            bucket = "onset"
        elif down_t is not None:
            bucket = "during"
        elif not any(k == "down" and 0.0 <= t - when <= WINDOW_BASE
                     for when, k in fires):
            bucket = "base"
        else:
            bucket = "during"
        flip = False
        if prev_vx is not None and vx * prev_vx < 0.0 \
                and abs(vx) > 400.0 and abs(prev_vx) > 400.0:
            flip = True
        kick_prev = any(abs(hv) > 800.0 and abs(hv) > 2.0 * abs(hx)
                        for hx, hv in hist[-2:])
        prev_vx = vx
        reason = str(row.get("lock_reason", ""))
        switchy = ("switch" in reason or "ambiguous" in reason
                   or "reacquire" in str(row.get("dot_status", "")))
        samples[bucket].append({
            "abs_vx": abs(vx), "abs_vy": abs(vy),
            "err": math.hypot(ex, ey), "flip": flip, "kick_prev": kick_prev,
            "switchy": switchy,
            "speed": math.hypot(vx, vy),
        })
        hist.append((vx, vy))
        if len(hist) > 3:
            hist.pop(0)
    return samples


def summarize(samples):
    result = {}
    for bucket, items in samples.items():
        if not items:
            result[bucket] = None
            continue
        moving = [it for it in items if it["speed"] > 100.0]
        flips = [it for it in moving if it["flip"]]
        pairs = sum(1 for it in moving) - 1
        result[bucket] = {
            "n": len(items),
            "moving": len(moving),
            "abs_vy_p75": percentile([it["abs_vy"] for it in moving], .75),
            "abs_vy_p90": percentile([it["abs_vy"] for it in moving], .90),
            "abs_vx_p75": percentile([it["abs_vx"] for it in moving], .75),
            "flip_rate": round(len(flips) / pairs, 4) if pairs > 0 else None,
            "flips": len(flips),
            "flip_kick_preceded_rate": round(
                sum(1 for it in flips if it["kick_prev"]) / len(flips), 4)
            if flips else None,
            "kick_frame_rate": round(
                sum(1 for it in moving if it["kick_prev"]) / len(moving), 4)
            if moving else None,
            "switch_rate": round(
                sum(1 for it in items if it["switchy"]) / len(items), 4),
            "err_p75": percentile([it["err"] for it in moving], .75),
        }
    return result


def main():
    logs = sys.argv[1:] or [
        "22ms _1.log", "22ms_1.log", "25ms_1.log", "25ms_2.log",
        "25ms_3.log", "aim_20260906_143059.log", "aim_20260906_222229.log",
        "aim_20260906_135713.log", "aim_20260906_140550.log",
        "aim_20260906_141313.log",
    ]
    pooled = {k: [] for k in ("onset", "during", "base")}
    for name in logs:
        path = os.path.join(ROOT, "logs", name)
        if not os.path.exists(path):
            print("missing:", name)
            continue
        for bucket, items in analyze(path).items():
            pooled[bucket].extend(items)
    summary = summarize(pooled)
    print("%-8s %6s %6s %9s %9s %9s %9s %8s %8s" % (
        "窗口", "n", "moving", "vy_p75", "vy_p90", "vx_p75", "flip率",
        "switch率", "err_p75"))
    for bucket in ("onset", "during", "base"):
        s = summary[bucket]
        if s is None:
            print("%-8s  (无样本)" % bucket)
            continue
        print("%-8s %6d %6d %9.1f %9.1f %9.1f %9s %8s %8.1f" % (
            bucket, s["n"], s["moving"], s["abs_vy_p75"], s["abs_vy_p90"],
            s["abs_vx_p75"],
            "%.3f" % s["flip_rate"] if s["flip_rate"] is not None else "-",
            "%.3f" % s["switch_rate"] if s["switch_rate"] is not None else "-",
            s["err_p75"]))
        print("         翻转事件=%d  其中前2帧有垂直尖峰的比例=%s  "
              "全场垂直尖峰帧占比=%s" % (
                  s["flips"],
                  s["flip_kick_preceded_rate"],
                  s["kick_frame_rate"]))
    detail = {b: summarize({b: pooled[b]})[b] for b in pooled}
    out_dir = os.path.join(ROOT, "日志V4", "cadence")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "fire_kick_check.json"), "w",
              encoding="utf-8", newline="") as handle:
        json.dump({"logs": logs, "pooled": detail}, handle,
                  ensure_ascii=False, indent=1)
    print("saved 日志V4/cadence/fire_kick_check.json")


if __name__ == "__main__":
    main()
