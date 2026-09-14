# -*- coding: utf-8 -*-
"""把锁定帧的类别翻转拆账：可用性导致 / 两框都在仍换选 / 同框几何抖动。

类别(cls)来自 [鼠标] 行的 锁定=[..] cls=C conf=F (滤波后锁框实际选中的类别)。
"""
import json
import math
import os
import re
import sys

MARKER = "[瞄准观测] "
MOUSE_RE = re.compile(r"帧=(\d+).*?锁定=\[([^\]]+)\] cls=(\d+) conf=([\d.]+)")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))


def percentile(values, q):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[round(q * (len(ordered) - 1))]


def parse(path):
    rows = []
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if MARKER in line:
                try:
                    rows.append(json.loads(line.split(MARKER, 1)[1]))
                except (ValueError, TypeError):
                    pass
                continue
            m = MOUSE_RE.search(line)
            if m and rows:
                frame = int(m.group(1))
                for back in range(min(4, len(rows))):
                    row = rows[-1 - back]
                    if int(row.get("frame", -1)) == frame \
                            and "sel_cls" not in row:
                        row["sel_cls"] = int(m.group(3))
                        row["sel_conf"] = float(m.group(4))
                        break
    return rows


def boxes_of(row, cls=None):
    out = []
    for det in row.get("detections") or []:
        bbox = det.get("bbox")
        if not bbox or len(bbox) < 4:
            continue
        if cls is not None and int(det.get("cls", -1)) != cls:
            continue
        x1, y1, x2, y2 = map(float, bbox[:4])
        out.append(((x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1,
                    int(det.get("cls", -1)), float(det.get("conf", 0.0))))
    return out


def main():
    logs = sys.argv[1:] or [
        "22ms _1.log", "22ms_1.log", "25ms_1.log", "25ms_2.log",
        "25ms_3.log", "aim_20260906_143059.log", "aim_20260906_222229.log",
        "aim_20260906_135713.log", "aim_20260906_140550.log",
        "aim_20260906_141313.log",
    ]
    avail = {"both": 0, "body_only": 0, "head_only": 0, "none": 0}
    both_sel = {"body": 0, "head": 0}
    pairs = {"flip_avail": 0, "flip_both": 0, "same": 0}
    resid_same, jump_flip = [], []
    same_conf, flip_conf = [], []
    sel_conf_all = []
    for name in logs:
        path = os.path.join(ROOT, "logs", name)
        if not os.path.exists(path):
            print("missing:", name)
            continue
        rows = parse(path)
        prev = None
        for row in rows:
            lock = str(row.get("lock_reason", ""))
            if not (lock.startswith("locked") or lock.startswith("switched")):
                prev = None
                continue
            dets = row.get("detections") or []
            has_body = any(int(d.get("cls", -1)) == 0 for d in dets)
            has_head = any(int(d.get("cls", -1)) == 1 for d in dets)
            if has_body and has_head:
                avail["both"] += 1
            elif has_body:
                avail["body_only"] += 1
            elif has_head:
                avail["head_only"] += 1
            else:
                avail["none"] += 1
            sel = row.get("sel_cls")
            if sel is not None:
                sel_conf_all.append(row.get("sel_conf", 0.0))
                if has_body and has_head:
                    both_sel["head" if sel == 1 else "body"] += 1
            if prev is None:
                prev = row
                continue
            sel_prev = prev.get("sel_cls")
            if sel is not None and sel_prev is not None:
                both_now = has_body and has_head
                prev_dets = prev.get("detections") or []
                pb = any(int(d.get("cls", -1)) == 0 for d in prev_dets)
                ph = any(int(d.get("cls", -1)) == 1 for d in prev_dets)
                both_before = pb and ph
                if sel != sel_prev:
                    if both_now and both_before:
                        pairs["flip_both"] += 1
                    else:
                        pairs["flip_avail"] += 1
                    mb = boxes_of(row, sel)
                    if mb:
                        jump_flip.append(math.hypot(
                            mb[0][0] - (prev.get("target") or [0, 0])[0],
                            mb[0][1] - (prev.get("target") or [0, 0])[1]))
                        flip_conf.append(mb[0][5])
                else:
                    pairs["same"] += 1
                    t = float(row.get("capture_t", 0.0))
                    pt = float(prev.get("capture_t", 0.0))
                    dt = max(1e-3, t - pt)
                    vx, vy = row.get("velocity_filtered") or (0.0, 0.0)
                    boxes_now = boxes_of(row, sel)
                    boxes_prev = boxes_of(prev, sel)
                    if boxes_now and boxes_prev:
                        ref_prev = min(boxes_prev, key=lambda b: (
                            (b[0] - (prev.get("target") or [0, 0])[0]) ** 2
                            + (b[1] - (prev.get("target") or [0, 0])[1]) ** 2))
                        ref_now = min(boxes_now, key=lambda b: (
                            (b[0] - ref_prev[0]) ** 2
                            + (b[1] - ref_prev[1]) ** 2))
                        dx = ref_now[0] - ref_prev[0]
                        dy = ref_now[1] - ref_prev[1]
                        resid_same.append(math.hypot(
                            dx - vx * dt, dy - vy * dt))
                        same_conf.append(ref_now[5])
            prev = row

    total = sum(avail.values())
    print("锁定帧检测可用性 (n=%d):" % total)
    for key, label in (("both", "头+身都在"), ("body_only", "只有身"),
                       ("head_only", "只有头"), ("none", "无检测")):
        print("  %-8s %5.1f%%" % (label, 100.0 * avail[key] / total))
    both_n = both_sel["body"] + both_sel["head"]
    if both_n:
        print("头+身都在时选中: 身 %.1f%%  头 %.1f%%" % (
            100.0 * both_sel["body"] / both_n,
            100.0 * both_sel["head"] / both_n))
    pn = sum(pairs.values())
    print("\n相邻锁定帧对 (n=%d):" % pn)
    print("  类别不变          %5.1f%%" % (100.0 * pairs["same"] / pn))
    print("  类别翻转-可用性变化 %5.1f%%  (该帧检测只给了一个类别 → 识别侧)"
          % (100.0 * pairs["flip_avail"] / pn))
    print("  类别翻转-两帧双框   %5.1f%%  (头身都在仍换选 → 选框/打分侧)"
          % (100.0 * pairs["flip_both"] / pn))
    print("\n同类别框的几何残差 (扣除真实运动):")
    print("  p50=%5.1f p75=%5.1f p90=%5.1f p99=%6.1f" % (
        percentile(resid_same, .5), percentile(resid_same, .75),
        percentile(resid_same, .9), percentile(resid_same, .99)))
    print("  残差>5px: %.1f%%  >10px: %.1f%%" % (
        100.0 * sum(1 for v in resid_same if v > 5) / len(resid_same),
        100.0 * sum(1 for v in resid_same if v > 10) / len(resid_same)))
    if jump_flip:
        print("类别翻转时的目标点跳变: p50=%5.1f p75=%5.1f p90=%5.1f" % (
            percentile(jump_flip, .5), percentile(jump_flip, .75),
            percentile(jump_flip, .9)))
    print("\n选中框置信度: p10=%.2f p50=%.2f" % (
        percentile(sel_conf_all, .1) or -1, percentile(sel_conf_all, .5) or -1))
    if same_conf:
        print("同类框置信度: p10=%.2f p50=%.2f" % (
            percentile(same_conf, .1), percentile(same_conf, .5)))
    if flip_conf:
        print("翻转后新框置信度: p10=%.2f p50=%.2f" % (
            percentile(flip_conf, .1), percentile(flip_conf, .5)))


if __name__ == "__main__":
    main()
