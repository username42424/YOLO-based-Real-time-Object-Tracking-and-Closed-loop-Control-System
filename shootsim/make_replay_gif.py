# -*- coding: utf-8 -*-
"""离线生成"实战日志回放"模拟实况 GIF（不弹窗口, dummy SDL 渲染）。

用法:
    python make_replay_gif.py [数量=10] [输出目录=demo_replay_gif]

每个 GIF 对应一个实战日志回合（在全部有效回合中均匀取样）。
画面元素: 黄框=回放检测框(实战 YOLO 所见)  绿框=回放真实框  青框=头部判定区
          红点=子弹落点参考  蓝十字=准星  黄线=射击轨迹  HUD=时间/血量/命中。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import load_config, get
from env import Env
from shooters import Shooter
from trackers import create_tracker
from perception import create_perceiver
from renderer import Renderer
import replay as replay_mod

HERE = os.path.dirname(os.path.abspath(__file__))


def make_gif(frames, out_path, duration_ms):
    from PIL import Image
    imgs = [Image.open(p).convert("P", palette=Image.ADAPTIVE, colors=128) for p in frames]
    if not imgs:
        return None
    imgs[0].save(out_path, save_all=True, append_images=imgs[1:],
                 duration=int(duration_ms), loop=0, optimize=True)
    return out_path


def run_one(cfg, ep, main_cfg_path, out_dir, every=2):
    env = Env(cfg)
    player = replay_mod.ReplayPlayer(ep, max_seconds=float(cfg["episode"]["max_seconds"]),
                                     on_end=str(cfg["replay"].get("on_end", "hold")))
    env.spawn_replay(player, 1000 + ep.idx)
    env.shooter = Shooter(cfg["shooter"], env.rng)

    renderer = Renderer(cfg, headless=True)
    renderer.start()
    perceiver = create_perceiver(get(cfg, "perception.mode"), cfg["perception"], env.rng, None)

    def _apply_mouse(dx, dy):
        env.apply_mouse(dx, dy)

    tracker = create_tracker(get(cfg, "tracker.strategy"), cfg["tracker"],
                             env.sw, env.sh, env.rng,
                             send_fn=_apply_mouse, main_cfg_path=main_cfg_path)
    tracker.reset()
    perceiver.reset()

    obs_delay = int(get(cfg, "tracker.obs_delay_frames", 0))
    ctrl_interval = max(1, int(get(cfg, "tracker.control_interval_frames", 1)))
    strategy = get(cfg, "tracker.strategy")
    fps = int(get(cfg, "view.fps", 100))

    frames = []
    dt_sum = 0.0
    dt_n = 0
    frame_idx = 0
    last_boxes = None
    while True:
        # 与 run_sim 相同：鼠标输出时钟独立于日志观测节拍。
        if hasattr(tracker, "advance_to"):
            tracker.advance_to(env.t)
        env.update_red_dot()
        tracker.set_reference(*env.red_dot())
        mouse_dx = mouse_dy = 0.0
        if frame_idx % ctrl_interval == 0:
            boxes = perceiver.perceive(env, obs_delay)
            last_boxes = boxes
            if boxes and strategy not in ("main", "main_real"):
                # 其他策略只喂离准星最近单框(main 系策略自带多目标选择)
                best, best_d2 = None, None
                for b in boxes:
                    bcx = (b[0] + b[2]) / 2.0
                    bcy = (b[1] + b[3]) / 2.0
                    d2 = (bcx - env.cx) ** 2 + (bcy - env.cy) ** 2
                    if best_d2 is None or d2 < best_d2:
                        best_d2, best = d2, b
                boxes = best[:4] if best is not None else None
            tdx, tdy = tracker.update(boxes, env.dt)
            mouse_dx += tdx
            mouse_dy += tdy

        running = env.step(
            mouse_dx, mouse_dy,
            advance_to=(tracker.advance_to if hasattr(tracker, "advance_to") else None))

        hud = {
            "t": env.t,
            "hp": sum(max(0, t["hp"]) for t in env.targets),
            "shots": env.shots, "hits": env.hits,
            "acc": env.hits / env.shots if env.shots else 0.0,
            "rh": env.region_hits, "kill_time": env.kill_time, "timeout": env.timed_out(),
        }
        renderer.draw(env, hud, yolo_boxes=last_boxes)
        renderer.tick(fps)
        if frame_idx % every == 0:
            p = os.path.join(out_dir, "f%05d.png" % frame_idx)
            renderer.save_frame(p)
            frames.append(p)
            dt_sum += env.dt
            dt_n += 1
        frame_idx += 1
        if not running or frame_idx > 400:
            break
    renderer.close()
    sm = env.summary()

    gif_path = os.path.join(out_dir, "ep%02d.gif" % ep.idx)
    duration_ms = 1000.0 * every * (dt_sum / max(1, dt_n))
    make_gif(frames, gif_path, duration_ms)
    for p in frames:  # 清理中间 PNG, 只留 GIF
        try:
            os.remove(p)
        except OSError:
            pass
    return sm, gif_path


def main():
    import argparse
    ap = argparse.ArgumentParser(description="实战日志回放模拟 GIF")
    ap.add_argument("--count", type=int, default=10)
    ap.add_argument("--out", default="demo_replay_gif")
    args = ap.parse_args()

    cfg = load_config(os.path.join(HERE, "config.json"))
    eps_all = replay_mod.load_episodes(cfg["replay"])
    # 演示用过滤: 剔除时长超上限/首检过晚/点太少的回合(如扫视 5s 才见目标的回合)
    eps = [e for e in eps_all
           if e.duration <= 3.2 and e.t_first <= 1.0 and e.n_points >= 5]
    if not eps:
        print("无可用回合")
        return
    main_cfg = os.path.abspath(os.path.join(HERE, "..", "config.json"))
    out_root = os.path.join(HERE, args.out)
    os.makedirs(out_root, exist_ok=True)

    n = min(args.count, len(eps))
    idxs = [round(i * (len(eps) - 1) / max(1, n - 1)) for i in range(n)]
    print("生成 %d 个 GIF(有效回合 %d/%d, 演示过滤后): %s" % (n, len(eps), len(eps_all), idxs))
    for k, ei in enumerate(idxs):
        ep = eps[ei]
        sub = os.path.join(out_root, "ep%02d" % ep.idx)
        os.makedirs(sub, exist_ok=True)
        sm, gif = run_one(cfg, ep, main_cfg, sub)
        size_kb = os.path.getsize(gif) / 1024.0 if os.path.exists(gif) else 0
        print("[%d/%d] ep%02d %s dur=%.2fs -> %s  淘汰=%s 射=%d 命中=%d 停留占比=%.0f%%  (%.0f KB)"
              % (k + 1, n, ep.idx, os.path.basename(ep.src)[:22], ep.duration,
                 gif, sm["kill_time"], sm["shots"], sm["hits"],
                 sm["dwell_reddot"] * 100, size_kb))


if __name__ == "__main__":
    main()
