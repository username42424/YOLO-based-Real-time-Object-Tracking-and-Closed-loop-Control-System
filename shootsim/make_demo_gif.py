# -*- coding: utf-8 -*-
"""离线生成 460px 大目标的模拟实况：逐帧 PNG + 动画 GIF。

用法:
    python make_demo_gif.py [seed] [输出目录]

不弹 pygame 窗口，用 dummy SDL 离线渲染每一帧，再合成 GIF，便于直接查看。
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import load_config, get
from env import Env
from movers import create_mover
from shooters import Shooter
from trackers import create_tracker
from perception import create_perceiver
from renderer import Renderer


def run_episode_gif(cfg, seed, out_dir, fixed_bh=None):
    os.makedirs(out_dir, exist_ok=True)
    env = Env(cfg)
    if fixed_bh is not None:
        env.fixed_bh = float(fixed_bh)

    def _mover_factory(img_idx):
        m = create_mover(get(cfg, "target.mover"), cfg["target"],
                         env.world_w, env.world_h, env.rng)
        return m
    env.mover = _mover_factory
    env.spawn(seed)
    env.shooter = Shooter(cfg["shooter"], env.rng)

    renderer = Renderer(cfg, headless=True)
    renderer.start()
    grab_fn = renderer.grab_frame_bgr
    perceiver = create_perceiver(get(cfg, "perception.mode"), cfg["perception"],
                                 env.rng, grab_fn)

    def _apply_mouse(dx, dy):
        env.apply_mouse(dx, dy)

    tracker = create_tracker(get(cfg, "tracker.strategy"), cfg["tracker"],
                             env.sw, env.sh, env.rng,
                             send_fn=_apply_mouse)
    tracker.reset()
    perceiver.reset()

    obs_delay = int(get(cfg, "tracker.obs_delay_frames", 2))
    ctrl_interval = max(1, int(get(cfg, "tracker.control_interval_frames", 1)))
    ht_cfg = cfg.get("human_track", {}) or {}
    ht_enabled = bool(ht_cfg.get("enabled", False))
    ht_gain = float(ht_cfg.get("gain", 0.35))
    ht_jitter = float(ht_cfg.get("jitter", 0.2))
    ht_margin = float(ht_cfg.get("margin", 60.0))
    fps = int(get(cfg, "view.fps", 60))

    tg = env.targets[0]
    print(f"[演示] seed={seed} 图={tg['img_idx']} 框高={env.bh:.0f}px 宽={env.bw:.0f}px "
          f"cls_type={tg['cls_type']}")

    frames = []
    frame_idx = 0
    last_yolo_boxes = None
    while True:
        # 红点作为瞄准参考点
        env.update_red_dot()
        tracker.set_reference(*env.red_dot())

        mouse_dx = mouse_dy = 0.0
        if frame_idx % ctrl_interval == 0:
            boxes = perceiver.perceive(env, obs_delay)
            last_yolo_boxes = boxes
            if boxes and get(cfg, "tracker.strategy") != "main":
                best = None
                best_d2 = None
                for b in boxes:
                    bcx = (b[0] + b[2]) / 2.0
                    bcy = (b[1] + b[3]) / 2.0
                    d2 = (bcx - env.cx) ** 2 + (bcy - env.cy) ** 2
                    if best_d2 is None or d2 < best_d2:
                        best_d2 = d2
                        best = b
                boxes = best[:4] if best is not None else None
            tdx, tdy = tracker.update(boxes, env.dt)
            mouse_dx += tdx
            mouse_dy += tdy
            if ht_enabled and boxes is None:
                cs = min(env.crop_size, env.sw, env.sh)
                x0 = (env.sw - cs) / 2.0
                y0 = (env.sh - cs) / 2.0
                for rect in env.gt_boxes(0):
                    cx = (rect[0] + rect[2]) / 2.0
                    cy = (rect[1] + rect[3]) / 2.0
                    k = ht_gain * max(0.0, 1.0 + env.rng.gauss(0, ht_jitter))
                    if cx < x0 + ht_margin:
                        mouse_dx -= (x0 + ht_margin - cx) * k
                    elif cx > x0 + cs - ht_margin:
                        mouse_dx += (cx - (x0 + cs - ht_margin)) * k
                    if cy < y0 + ht_margin:
                        mouse_dy -= (y0 + ht_margin - cy) * k
                    elif cy > y0 + cs - ht_margin:
                        mouse_dy += (cy - (y0 + cs - ht_margin)) * k

        running = env.step(mouse_dx, mouse_dy)

        hud = {
            "t": env.t,
            "hp": sum(max(0, t["hp"]) for t in env.targets),
            "shots": env.shots, "hits": env.hits,
            "acc": env.hits / env.shots if env.shots else 0.0,
            "rh": env.region_hits, "kill_time": env.kill_time, "timeout": env.timed_out(),
            "alive": sum(1 for t in env.targets if not t["dead"]),
        }
        renderer.draw(env, hud, yolo_boxes=last_yolo_boxes)
        renderer.tick(fps)
        # 逐帧保存 PNG
        path = os.path.join(out_dir, f"frame_{frame_idx:04d}.png")
        renderer.save_frame(path)
        frames.append(path)
        frame_idx += 1
        if not running:
            break

    renderer.close()
    sm = env.summary()
    print(f"[结束] {'击杀' if sm['kill_time'] is not None else '超时'} "
          f"击杀={sm['kill_time']}s 射击={sm['shots']} 命中={sm['hits']} 区域={sm['region_hits']}")
    return frames, sm


def make_gif(frames, out_path, fps=8):
    """把 PNG 序列合成动画 GIF。"""
    try:
        from PIL import Image
    except Exception as e:
        print(f"[跳过] 未安装 Pillow，无法合成 GIF: {e}")
        return None
    imgs = [Image.open(p).convert("P", palette=Image.ADAPTIVE, colors=128)
            for p in frames]
    if not imgs:
        return None
    # 等比缩小，控制 GIF 体积
    w, h = imgs[0].size
    scale = 640 / w
    imgs = [im.resize((int(w * scale), int(h * scale)), Image.BILINEAR) for im in imgs]
    imgs[0].save(out_path, save_all=True, append_images=imgs[1:],
                 duration=int(1000 / fps), loop=0)
    return out_path


def main():
    import argparse
    ap = argparse.ArgumentParser(description="离线生成模拟实况 GIF（随机尺寸/位置/图片）")
    ap.add_argument("--count", type=int, default=1, help="生成多少个 GIF（随机 seed）")
    ap.add_argument("--size", type=float, default=None, help="强制目标框高 px（默认随机）")
    ap.add_argument("--out", default="demo_random", help="输出根目录")
    ap.add_argument("--fps", type=int, default=8)
    args = ap.parse_args()

    cfg = load_config(os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json"))
    for i in range(args.count):
        seed = int(time.time() * 1000) % 100000 + i * 977
        sub = os.path.join(args.out, f"seed_{seed}")
        frames, sm = run_episode_gif(cfg, seed, sub, fixed_bh=args.size)
        gif = make_gif(frames, os.path.join(sub, "demo.gif"), fps=args.fps)
        print(f"[{i + 1}/{args.count}] seed={seed} 击杀={sm['kill_time']}s 尺寸scale={sm['scale']} "
              f"命中={sm['hits']}/{sm['shots']} 区域={sm['region_hits']} -> {gif}")


if __name__ == "__main__":
    main()
