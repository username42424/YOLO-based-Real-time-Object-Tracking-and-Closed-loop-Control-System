# -*- coding: utf-8 -*-
"""预标注：用 640 YOLO 模型给 7 张目标图各标出 cls0(身)/cls1(头) 框。

与模拟器渲染一致：把目标按 160px 高（保持宽高比）贴到 640x640 深色画布中央，
再跑 YOLO 640，得到的就是模拟器实际"看到"的框。框归一化到目标本地 [0,1]，
输出到 targets_gt.json —— 运行时作为"真实目标"（打中该框才算击中；框外=miss）。
"""
import os
import sys
import json

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import main as aim_main

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MODEL = os.path.join(ROOT, "Dawan_0121_v11s_640_optimized.onnx")
CONF = 0.28
REF_H = 160
CANVAS = 640

IMAGES = [
    "Snipaste_2026-08-13_16-44-35.png",
    "Snipaste_2026-08-13_17-23-14.png",
    "Snipaste_2026-08-13_17-24-28.png",
    "Snipaste_2026-08-13_17-24-42.png",
    "Snipaste_2026-08-15_17-36-55.png",
    "Snipaste_2026-08-15_17-37-24.png",
    "Snipaste_2026-08-15_17-38-18.png",
]


def main():
    det = aim_main._create_detector(MODEL, CONF, 0.7)
    out = {}
    for name in IMAGES:
        img = cv2.imread(os.path.join(HERE, name), cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"{name}: 读取失败")
            continue
        if img.shape[2] == 4:
            b, g, r, a = cv2.split(img)
            img = cv2.merge([b, g, r])
        h, w = img.shape[:2]
        aspect = w / float(h)
        ref_w = int(round(REF_H * aspect))
        if ref_w < 2:
            ref_w = 2
        resized = cv2.resize(img, (ref_w, REF_H), interpolation=cv2.INTER_AREA)
        # 与模拟器一致：深色画布 + 网格，目标放中央
        canvas = np.full((CANVAS, CANVAS, 3), (40, 34, 32), dtype=np.uint8)
        for g in range(0, CANVAS, 100):
            cv2.line(canvas, (g, 0), (g, CANVAS), (62, 52, 48), 1)
            cv2.line(canvas, (0, g), (CANVAS, g), (62, 52, 48), 1)
        x0 = (CANVAS - ref_w) // 2
        y0 = (CANVAS - REF_H) // 2
        canvas[y0:y0 + REF_H, x0:x0 + ref_w] = resized
        try:
            dets = det.detect(canvas)
        except Exception as e:
            print(f"{name}: 检测异常 {e}")
            continue
        best = {}
        for d in dets:
            if d["cls"] not in (0, 1):
                continue
            key = f"cls{d['cls']}"
            if key not in best or d["conf"] > best[key]["conf"]:
                best[key] = d
        entry = {"cls0": None, "cls1": None}
        for key, d in best.items():
            bx1, by1, bx2, by2 = d["bbox"]
            # 目标本地归一化坐标
            entry[key] = [round((bx1 - x0) / ref_w, 4), round((by1 - y0) / REF_H, 4),
                          round((bx2 - x0) / ref_w, 4), round((by2 - y0) / REF_H, 4)]
        out[name] = entry
        print(f"{name}  aspect={aspect:.2f}  {ref_w}x{REF_H}  ->  "
              f"cls0={'有' if entry['cls0'] else '无'}(conf={best['cls0']['conf'] if 'cls0' in best else '-'})  "
              f"cls1={'有' if entry['cls1'] else '无'}(conf={best['cls1']['conf'] if 'cls1' in best else '-'})  "
              f"box={entry}")

    path = os.path.join(HERE, "targets_gt.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n已写入: {path}")


if __name__ == "__main__":
    main()
