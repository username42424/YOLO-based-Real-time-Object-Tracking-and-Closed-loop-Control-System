# -*- coding: utf-8 -*-
"""实测各分辨率 yolo 模型的真实推理耗时（同一 640x640 输入）。"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from perception import _get_aim_main_and_detector

HERE = os.path.dirname(os.path.abspath(__file__))

img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
for name in ["640", "416", "320", "256"]:
    model = os.path.normpath(os.path.join(HERE, "..", f"Dawan_0121_v11s_{name}_optimized.onnx"))
    if not os.path.exists(model):
        print(f"{name}: 文件不存在, 跳过")
        continue
    try:
        det = _get_aim_main_and_detector(model, 0.28, [0, 1], 0.7)
    except Exception as e:
        print(f"{name}: 加载失败 {e}")
        continue
    for _ in range(5):
        det.detect(img)
    t0 = time.perf_counter()
    N = 100
    for _ in range(N):
        det.detect(img)
    el = time.perf_counter() - t0
    print(f"{name}模型: {el/N*1000:.2f} ms/次  (100次均值, 输入640x640)")
