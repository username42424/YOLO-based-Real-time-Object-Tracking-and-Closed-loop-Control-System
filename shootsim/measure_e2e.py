# -*- coding: utf-8 -*-
"""实测完整端到端延迟：截图 + 推理 + 游戏鼠标响应(1帧)。

建模：bot 截图那一刻游戏在 t_src；截图+推理耗 compute_ms 期间游戏独立推进
（推进 compute_ms/dt 帧）；鼠标发出去后游戏下一帧才生效（+1 帧）。
端到端延迟 = compute_ms + 1 帧。
"""
import os
import sys
import time
import statistics

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import load_config, get
from env import Env
from movers import create_mover
from shooters import Shooter
from perception import create_perceiver
from renderer import Renderer

HERE = os.path.dirname(os.path.abspath(__file__))


def measure_e2e(model_rel, n=200):
    cfg = load_config(os.path.join(HERE, "config.json"))
    cfg["perception"]["yolo_model"] = model_rel
    env = Env(cfg)
    env.mover = lambda idx: create_mover(get(cfg, "target.mover"), cfg["target"],
                                         env.world_w, env.world_h, env.rng)
    env.spawn(1)
    env.shooter = Shooter(cfg["shooter"], env.rng)
    renderer = Renderer(cfg, headless=True)
    renderer.start()
    perceiver = create_perceiver("yolo", cfg["perception"], env.rng, renderer.grab_frame_bgr)
    perceiver.reset()
    cs = int(get(cfg, "perception.capture_size", 640))
    dt = env.dt

    # 预热
    for _ in range(20):
        renderer.draw(env)
        img = renderer.grab_frame_bgr()
        h, w = img.shape[:2]
        x0, y0 = int(w / 2 - cs / 2), int(h / 2 - cs / 2)
        perceiver.detector.detect(img[y0:y0 + cs, x0:x0 + cs])
        env.step(0.0, 0.0)

    compute_ms_list, delay_ms_list = [], []
    for _ in range(n):
        renderer.draw(env)           # 游戏渲染当前帧
        t_src = env.t                # 截图那一刻的游戏时间
        t0 = time.perf_counter()
        img = renderer.grab_frame_bgr()
        h, w = img.shape[:2]
        x0, y0 = int(w / 2 - cs / 2), int(h / 2 - cs / 2)
        perceiver.detector.detect(img[y0:y0 + cs, x0:x0 + cs])
        t1 = time.perf_counter()
        compute_ms = (t1 - t0) * 1000
        # 游戏在 bot 推理期间独立推进 compute_ms/dt 帧
        for _ in range(round(compute_ms / 1000.0 / dt)):
            env.step(0.0, 0.0)
        # 鼠标发出后，游戏下一帧才生效
        env.step(0.0, 0.0)
        t_effect = env.t
        compute_ms_list.append(compute_ms)
        delay_ms_list.append((t_effect - t_src) * 1000)
    renderer.close()
    return statistics.median(compute_ms_list), statistics.median(delay_ms_list)


if __name__ == "__main__":
    fps = 60
    frame_ms = 1000.0 / fps
    print(f"=== 完整端到端延迟实测 (游戏在推理期间独立推进, {fps}fps 帧={frame_ms:.1f}ms) ===")
    print(f"{'模型':<5}{'截图+推理':>11}{'端到端延迟':>12}{'≈obs_delay帧':>12}")
    for name in ["640", "416", "320", "256"]:
        model = f"../Dawan_0121_v11s_{name}_optimized.onnx"
        if not os.path.exists(os.path.join(HERE, model)):
            print(f"{name}: 模型文件不存在")
            continue
        comp, delay = measure_e2e(model)
        obs = round(delay / frame_ms)
        print(f"{name:<5}{comp:>10.2f}ms{delay:>11.2f}ms{obs:>11}帧")
