# -*- coding: utf-8 -*-
"""YOLO 推理基准：单进程跑 N 次 detect，输出总耗时与单次延迟。"""
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import main as aim_main

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MODEL = os.path.join(ROOT, "Dawan_0121_v11s_640_optimized.onnx")
N = 200


def main():
    det = aim_main._create_detector(MODEL, 0.28, 0.7)
    # 用一张真实目标图渲染成 640x640，接近模拟器实际输入
    img = cv2.imread(os.path.join(HERE, "Snipaste_2026-08-13_16-44-35.png"))
    frame = np.full((640, 640, 3), (40, 34, 32), dtype=np.uint8)
    h, w = img.shape[:2]
    aspect = w / h
    bh = 200
    bw = int(round(bh * aspect))
    if bw % 2:
        bw += 1
    resized = cv2.resize(img, (bw, bh), interpolation=cv2.INTER_AREA)
    x0, y0 = (640 - bw) // 2, (640 - bh) // 2
    frame[y0:y0 + bh, x0:x0 + bw] = resized
    # warmup
    for _ in range(5):
        det.detect(frame)
    t0 = time.perf_counter()
    for _ in range(N):
        det.detect(frame)
    t1 = time.perf_counter()
    dt = t1 - t0
    print(f"PID={os.getpid()}  N={N}  总耗时={dt:.2f}s  单次={dt / N * 1000:.2f}ms", flush=True)


if __name__ == "__main__":
    main()
