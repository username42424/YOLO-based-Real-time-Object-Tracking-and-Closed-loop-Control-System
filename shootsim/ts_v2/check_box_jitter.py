# -*- coding: utf-8 -*-
"""统计一次锁定流程内，选中目标框的帧间变化有多大。

框配对：每帧取最接近 target 的原始检测框；相邻锁定帧之间
- raw jump   = 框中心位移
- explained  = 滤波速度 × 帧间隔 (目标真实运动的估计)
- residual   = raw − explained  (无法用运动解释的瞬移 = 检测/选框噪声)
"""
import json
import math
import os
import sys

MARKER = "[瞄准观测] "
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))


def percentile(values, q):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[round(q * (len(ordered) - 1))]


def load_flows(path):
    rows = []
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if MARKER not in line:
                continue
            try:
                rows.append(json.loads(line.split(MARKER, 1)[1]))
            except (ValueError, TypeError):
                pass
    flows, flow = [], []
    for row in rows:
        lock = str(row.get("lock_reason", ""))
        if lock.startswith("locked") or lock.startswith("switched"):
            flow.append(row)
        else:
            if len(flow) >= 3:
                flows.append(flow)
            flow = []
    if len(flow) >= 3:
        flows.append(flow)
    return flows


def matched_box(row, fallback):
    boxes = []
    for det in row.get("detections") or []:
        bbox = det.get("bbox")
        if not bbox or len(bbox) < 4:
            continue
        x1, y1, x2, y2 = map(float, bbox[:4])
        boxes.append(((x1 + x2) / 2.0, (y1 + y2) / 2.0,
                      x2 - x1, y2 - y1, int(det.get("cls", -1))))
    if not boxes:
        return None
    ref = row.get("target") or fallback
    if not ref:
        return boxes[0]
    return min(boxes, key=lambda box:
               (box[0] - ref[0]) ** 2 + (box[1] - ref[1]) ** 2)


def main():
    logs = sys.argv[1:] or [
        "22ms _1.log", "22ms_1.log", "25ms_1.log", "25ms_2.log",
        "25ms_3.log", "aim_20260906_143059.log", "aim_20260906_222229.log",
        "aim_20260906_135713.log", "aim_20260906_140550.log",
        "aim_20260906_141313.log",
    ]
    pairs, flow_max, flow_len = [], [], []
    cls_switch = 0
    for name in logs:
        path = os.path.join(ROOT, "logs", name)
        if not os.path.exists(path):
            print("missing:", name)
            continue
        for flow in load_flows(path):
            flow_len.append(len(flow))
            prev_box, prev_t = None, None
            flow_peaks = []
            for row in flow:
                box = matched_box(row, prev_box[:2] if prev_box else None)
                if box is None:
                    continue
                if prev_box is not None:
                    t = float(row.get("capture_t", 0.0))
                    dt = max(1e-3, t - prev_t)
                    vx, vy = row.get("velocity_filtered") or (0.0, 0.0)
                    dx = box[0] - prev_box[0]
                    dy = box[1] - prev_box[1]
                    raw = math.hypot(dx, dy)
                    resid = math.hypot(dx - vx * dt, dy - vy * dt)
                    lock = str(row.get("lock_reason", ""))
                    switchy = "switch" in lock or "grace" in lock
                    pairs.append({
                        "raw": raw, "resid": resid, "dt": dt,
                        "d_h": abs(box[3] - prev_box[3]),
                        "rel_h": abs(box[3] - prev_box[3]) / max(1.0, prev_box[3]),
                        "cls_switch": int(box[4] != prev_box[4]),
                        "switchy": switchy,
                    })
                    if box[4] != prev_box[4]:
                        cls_switch += 1
                    flow_peaks.append(resid)
                prev_box, prev_t = box, float(row.get("capture_t", 0.0))
            if flow_peaks:
                flow_max.append(max(flow_peaks))

    def dist(values, label):
        print("  %-10s p50=%5.1f  p75=%5.1f  p90=%5.1f  p99=%6.1f  max=%6.1f" % (
            label, percentile(values, .5), percentile(values, .75),
            percentile(values, .9), percentile(values, .99), max(values)))

    print("锁定流程: %d 段, 锁定帧 %d, 中位流程长度 %d 帧" % (
        len(flow_len), sum(flow_len), percentile(flow_len, .5)))
    print("\n帧间配对统计 (n=%d):" % len(pairs))
    dist([p["raw"] for p in pairs], "中心位移")
    dist([p["resid"] for p in pairs], "无解释残差")
    n = len(pairs)
    for cut in (10.0, 20.0, 40.0):
        over = sum(1 for p in pairs if p["resid"] > cut)
        print("  残差>%4.0fpx: %5.1f%% (%d)" % (cut, 100.0 * over / n, over))
    print("  类别切换(头/身): %.1f%% (%d)" % (
        100.0 * sum(p["cls_switch"] for p in pairs) / n,
        sum(p["cls_switch"] for p in pairs)))
    print("  框高变化 p50=%.1fpx  p75=%.1fpx  (相对 p75=%.0f%%)" % (
        percentile([p["d_h"] for p in pairs], .5),
        percentile([p["d_h"] for p in pairs], .75),
        100.0 * percentile([p["rel_h"] for p in pairs], .75)))
    sw = [p for p in pairs if p["switchy"]]
    if sw:
        print("\n按锁定原因拆分:")
        for label, subset in (("switch/grace帧", sw),
                              ("普通locked帧", [p for p in pairs if not p["switchy"]])):
            m = len(subset)
            dist([p["resid"] for p in subset], label)
            print("      n=%d  残差>20px: %.1f%%" % (
                m, 100.0 * sum(1 for p in subset if p["resid"] > 20.0) / m))
    print("\n每段锁定流程内的最大残差:")
    dist(flow_max, "流程最大残差")
    print("  含>20px瞬移的流程: %.1f%%  含>40px瞬移: %.1f%%" % (
        100.0 * sum(1 for v in flow_max if v > 20.0) / len(flow_max),
        100.0 * sum(1 for v in flow_max if v > 40.0) / len(flow_max)))

    out = {"pairs": [{k: round(v, 2) if isinstance(v, float) else v
                      for k, v in p.items()} for p in pairs[:2000]],
           "flow_max_px": [round(v, 1) for v in flow_max]}
    out_dir = os.path.join(ROOT, "日志V4", "cadence")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "box_jitter.json"), "w",
              encoding="utf-8", newline="") as handle:
        json.dump(out, handle, ensure_ascii=False)
    print("saved 日志V4/cadence/box_jitter.json")


if __name__ == "__main__":
    main()
