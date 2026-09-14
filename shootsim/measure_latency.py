# -*- coding: utf-8 -*-
"""实测模拟器真实回路各阶段耗时：渲染(draw) / 截图(grab) / 推理(detect)。

真实回路延迟 ≈ 截图 + 推理 + 鼠标在游戏里生效的下一帧响应。
本脚本按实际渲染画面逐阶段计时，用于推导 obs_delay_frames，替代写死值。
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


def measure(model_rel, n=300):
    cfg = load_config(os.path.join(HERE, "config.json"))
    cfg["perception"]["yolo_model"] = model_rel
    env = Env(cfg)

    def mf(idx):
        return create_mover(get(cfg, "target.mover"), cfg["target"],
                            env.world_w, env.world_h, env.rng)
    env.mover = mf
    env.spawn(1)
    env.shooter = Shooter(cfg["shooter"], env.rng)
    renderer = Renderer(cfg, headless=True)
    renderer.start()
    perceiver = create_perceiver("yolo", cfg["perception"], env.rng, renderer.grab_frame_bgr)
    perceiver.reset()

    for _ in range(20):  # 预热（模型加载、GPU 编译）
        renderer.draw(env)
        perceiver.perceive(env, 0)
        env.step(0.0, 0.0)

    draw_t, grab_t, det_t = [], [], []
    cs = int(get(cfg, "perception.capture_size", 640))
    for _ in range(n):
        env.update_red_dot()
        t0 = time.perf_counter()
        renderer.draw(env)
        t1 = time.perf_counter()
        img = renderer.grab_frame_bgr()
        t2 = time.perf_counter()
        h, w = img.shape[:2]
        x0, y0 = int(w / 2 - cs / 2), int(h / 2 - cs / 2)
        crop = img[y0:y0 + cs, x0:x0 + cs]
        perceiver.detector.detect(crop)
        t3 = time.perf_counter()
        draw_t.append((t1 - t0) * 1000)
        grab_t.append((t2 - t1) * 1000)
        det_t.append((t3 - t2) * 1000)
        env.step(0.0, 0.0)
    renderer.close()
    return draw_t, grab_t, det_t


if __name__ == "__main__":
    fps = 60
    frame_ms = 1000.0 / fps
    print(f"=== 实测回路耗时 (每阶段中位数, {fps}fps 帧间隔={frame_ms:.1f}ms) ===")
    print(f"{'模型':<5}{'渲染draw':>10}{'截图grab':>10}{'推理detect':>11}{'截图+推理':>11}{'≈obs_delay帧':>12}")
    for name in ["640", "416", "320", "256"]:
        model = f"../Dawan_0121_v11s_{name}_optimized.onnx"
        if not os.path.exists(os.path.join(HERE, model)):
            print(f"{name}: 模型文件不存在")
            continue
        d, g, det = measure(model)
        md, mg, mdet = (statistics.median(x) for x in (d, g, det))
        loop = mg + mdet  # 截图+推理（bot 侧回路，不含游戏自身渲染）
        obs = round(loop / frame_ms)
        print(f"{name:<5}{md:>9.2f}ms{mg:>9.2f}ms{mdet:>10.2f}ms{loop:>10.2f}ms{obs:>11}帧")
