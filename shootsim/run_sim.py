# -*- coding: utf-8 -*-
"""可视化运行入口。

用法:
    python run_sim.py                         # 默认 auto 模式（追踪器自动瞄准+射击）
    python run_sim.py --mode human            # 人手模式：物理鼠标=视角，自动射击
    python run_sim.py --mode auto --episodes 5 --seed 1
    python run_sim.py --screenshot-dir shots  # 每隔若干帧保存画面（调试/验证渲染）
    python run_sim.py --config my_config.json

mouse 物理位移控制视角（准星固定，场景/目标相对移动）；ESC 退出。
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import load_config, get
from env import Env
from movers import create_mover
from trackers import create_tracker
from shooters import Shooter
from perception import create_perceiver
import replay as replay_mod
from logger import EpisodeLogger
from renderer import Renderer


def episode_seeds(base_seed, episodes):
    """Return deterministic per-episode seeds from one base seed."""
    return [int(base_seed) + i for i in range(max(0, int(episodes)))]


def observation_dt(current_t, last_control_t, default_dt):
    """Return the actual interval between adjacent control observations."""
    if last_control_t is None:
        return max(1e-6, float(default_dt))
    return max(1e-6, float(current_t) - float(last_control_t))


def run_episode(cfg, mode, seed, renderer=None, logger=None, screenshot_dir=None,
                quiet=False, main_cfg_path=None, send_hook=None, demo=False, fixed_bh=None,
                mouse_log=None):
    env = Env(cfg)
    if fixed_bh is not None:
        env.fixed_bh = float(fixed_bh)

    if get(cfg, "perception.mode") == "replay":
        # ── 实战日志回放: 目标世界轨迹取自日志, 不再随机生成 ──
        episodes = replay_mod.load_episodes(cfg["replay"])
        if not episodes:
            raise RuntimeError("replay 模式未解析到有效回合, 请检查 config replay.logs")
        ep = episodes[seed % len(episodes)]
        player = replay_mod.ReplayPlayer(
            ep, max_seconds=float(cfg["episode"]["max_seconds"]),
            on_end=str(cfg["replay"].get("on_end", "hold")))
        env.spawn_replay(player, seed)
        if mouse_log is not None:
            mouse_log.episode(seed, "ep=%d/%d dur=%.2fs points=%d" %
                              (seed % len(episodes), len(episodes), ep.duration, ep.n_points))
    else:
        # mover 工厂：spawn 内部为每个目标创建 mover（img_idx 决定图片索引）
        def _mover_factory(img_idx):
            m = create_mover(get(cfg, "target.mover"), cfg["target"],
                             env.world_w, env.world_h, env.rng)
            return m
        env.mover = _mover_factory
        env.spawn(seed)
    env.shooter = Shooter(cfg["shooter"], env.rng)
    if env.shooter.is_oracle and not quiet:
        print("[警告] oracle_on_aim：开火条件读取真实命中框，accuracy仅作演示，"
              "不参与算法比较/参数排序")

    # yolo 感知需要渲染截图：headless 时用 dummy SDL 渲染器（不出窗口，仅截图）
    own_renderer = None
    if get(cfg, "perception.mode") == "yolo" and renderer is None:
        own_renderer = Renderer(cfg, headless=True)
        own_renderer.start()
        renderer = own_renderer

    grab_fn = renderer.grab_frame_bgr if renderer is not None else None
    perceiver = create_perceiver(get(cfg, "perception.mode"), cfg["perception"],
                                 env.rng, grab_fn)
    if get(cfg, "perception.mode") == "yolo" and grab_fn is None:
        print("[警告] yolo 感知需要渲染截图，当前无法截图，感知不到目标")
    def _apply_mouse(dx, dy):
        """main 策略的物理发送：外部统计由追踪器按整条控制指令回调。"""
        env.apply_mouse(dx, dy)

    tracker = create_tracker(get(cfg, "tracker.strategy"), cfg["tracker"],
                             env.sw, env.sh, env.rng,
                             send_fn=_apply_mouse, main_cfg_path=main_cfg_path,
                             mouse_log=mouse_log, command_hook=send_hook)
    tracker.reset()
    # Optional: align the 15ms output grid to the live log's real tick phase
    # (set per episode by the fidelity harness; <0 keeps the default t=0 grid).
    if get(cfg, "perception.mode") == "replay" and hasattr(tracker, "set_output_phase"):
        _phase = float((cfg.get("replay") or {}).get("output_phase_s", -1.0))
        if _phase >= 0.0:
            tracker.set_output_phase(_phase)
    perceiver.reset()

    if logger is None:
        logger = EpisodeLogger(cfg["logging"]["dir"],
                               cfg["logging"]["per_frame"],
                               cfg["logging"]["frame_interval"])
    logger.start(seed, cfg, env)

    obs_delay = int(get(cfg, "tracker.obs_delay_frames", 2))
    ctrl_interval = max(1, int(get(cfg, "tracker.control_interval_frames", 1)))
    # 人手跟枪（实战修正）：玩家不会把目标钉在准星上，而是"保持目标在视野内"。
    # 实战日志显示目标常偏离准星 100~260px——人手只在目标靠近/越出截图区边缘
    # （margin 内）时才把视角拉回，目标在画面内自由漂移。
    ht_cfg = cfg.get("human_track", {}) or {}
    ht_enabled = bool(ht_cfg.get("enabled", False))
    # replay 模式下实战日志轨迹不含"人手回拉"成分, 禁用以免引入假移动
    if get(cfg, "perception.mode") == "replay":
        ht_enabled = False
    ht_gain = float(ht_cfg.get("gain", 0.35))
    ht_jitter = float(ht_cfg.get("jitter", 0.2))
    ht_margin = float(ht_cfg.get("margin", 60.0))
    fps = int(get(cfg, "view.fps", 60))
    shot_dir = screenshot_dir
    frame_idx = 0
    last_shot_count = 0
    last_yolo_boxes = None
    last_control_t = None
    if not quiet:
        print(f"[回合] seed={seed} 模式={mode} 追踪={get(cfg, 'tracker.strategy')} "
              f"射击={get(cfg, 'shooter.policy')} 瞄准点高度比={get(cfg, 'tracker.aim_ratio')}")

    while True:
        # ── 输入 ──
        mouse_dx = mouse_dy = 0.0
        quit_flag = False
        if renderer is not None:
            rdx, rdy, quit_flag = renderer.poll_events()
            if mode == "human":
                mouse_dx, mouse_dy = rdx, rdy
        if quit_flag:
            break

        # Keep the mouse output clock independent from observation cadence.
        # In replay mode env.step also advances it to the next capture instant,
        # so all 20 ms packets land before that capture is evaluated.
        if hasattr(tracker, "advance_to"):
            tracker.advance_to(env.t)

        # ── 红点（子弹落点）：实战回放优先使用日志记录值 ──
        env.update_red_dot()
        if hasattr(tracker, "set_reference"):
            tracker.set_reference(*env.red_dot())

        # ── 追踪（auto 模式）──
        if mode == "auto" and frame_idx % ctrl_interval == 0:
            boxes = perceiver.perceive(env, obs_delay)
            last_yolo_boxes = boxes
            control_dt = observation_dt(env.t, last_control_t, env.dt)
            # Replay fidelity: use the log's own per-frame processing latency
            # for the arrival-prediction horizon instead of a constant.
            if get(cfg, "perception.mode") == "replay" and hasattr(tracker, "set_live_latency"):
                tracker.set_live_latency(getattr(env, "replay_latency", None))
            # 多目标：main / main_real 策略接收完整列表（各自做目标选择）；其他策略传离准星最近的单框（剥离 cls）
            if boxes and get(cfg, "tracker.strategy") not in (
                    "main", "main_real", "other_test", "obs_auto_aim"):
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
            tdx, tdy = tracker.update(boxes, control_dt)
            mouse_dx += tdx
            mouse_dy += tdy
            if hasattr(env.shooter, "set_tracker_status"):
                tracker_status = (
                    tracker.status() if hasattr(tracker, "status") else None
                )
                env.shooter.set_tracker_status(tracker_status, control_dt)
            if get(cfg, "perception.mode") != "replay":
                # 当前观测先评分，再由 env.step 应用本次新控制。
                env.score_observation(boxes, control_dt)
            last_control_t = env.t
            # 人手跟枪：bot 本帧失明时，人手保持各目标在截图区内（不居中，
            # 只在目标靠近边缘时拉回，目标可在画面内漂移，复刻实战偏离量）
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

        # ── 推进环境（命中按预标注真实部位框判定，与运行时 yolo 无关）──
        advance_to = (tracker.advance_to
                      if get(cfg, "perception.mode") == "replay"
                      and hasattr(tracker, "advance_to") else None)
        running = env.step(mouse_dx, mouse_dy, advance_to=advance_to)

        # ── 逐发子弹：日志 + 命中特效 ──
        if len(env.shot_log) > last_shot_count:
            new_shots = env.shot_log[last_shot_count:]
            last_shot_count = len(env.shot_log)
            for rec in new_shots:
                logger.shot(rec)
                if rec["hit"] and renderer is not None:
                    bx = rec["box"]
                    hx = (bx[0] + bx[2]) / 2 if bx else env.cx
                    renderer.add_hit(rec["region"], rec["dmg"], hx, env.cy)

        # ── 日志（可选逐帧）──
        if cfg["logging"]["per_frame"] and frame_idx % cfg["logging"]["frame_interval"] == 0:
            logger.frame(env, yolo_boxes=last_yolo_boxes)

        # ── 渲染 ──
        if renderer is not None:
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
            if demo and not renderer.headless:
                # 慢速演示：每帧额外等待，把动画放慢约 4 倍（不改变模拟逻辑）
                time.sleep(0.05)
            if shot_dir and frame_idx % 10 == 0:
                os.makedirs(shot_dir, exist_ok=True)
                renderer.save_frame(os.path.join(shot_dir, f"frame_{frame_idx:05d}.png"))

        frame_idx += 1
        if not running:
            break

    summary = env.summary()
    logger.end(env)
    if mouse_log is not None:
        mouse_log.end("kill=%s shots=%d hits=%d dwell_rd=%.1f%% dwell_ch=%.1f%%" %
                      (summary["kill_time"], summary["shots"], summary["hits"],
                       summary["dwell_reddot"] * 100.0, summary["dwell_cross"] * 100.0))
    if own_renderer is not None:
        own_renderer.close()
    if not quiet:
        status = ("击杀" if env.kill_time is not None else
                  "日志结束" if summary.get("replay_exhausted") else "超时")
        print(f"[回合结束] {status}  击杀时间={summary['kill_time']}s  射击={summary['shots']} "
              f"命中={summary['hits']} 命中率={summary['accuracy']:.1%} "
              f"区域={summary['region_hits']}")
    return summary


def main():
    ap = argparse.ArgumentParser(description="ShootSim 可视化运行")
    ap.add_argument("--config", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json"))
    ap.add_argument("--mode", choices=["auto", "human"], default="auto")
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tracker", default=None,
                    help="覆盖 tracker.strategy，例如 other_test")
    ap.add_argument("--headless", action="store_true", help="无窗口（SDL dummy，调试用）")
    ap.add_argument("--screenshot-dir", default=None)
    ap.add_argument("--demo", action="store_true", help="慢速演示：放慢动画帧率便于观察")
    ap.add_argument("--size", type=float, default=None, help="强制目标框高 px（演示大目标用）")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.tracker:
        cfg.setdefault("tracker", {})["strategy"] = args.tracker
    base_seed = (args.seed if args.seed else
                 max(1, int(time.time() * 1000) % 1000000000))
    seeds = episode_seeds(base_seed, args.episodes)
    print(f"[种子] base_seed={base_seed}")
    renderer = None if args.headless else Renderer(cfg)
    if renderer is not None:
        renderer.start()

    try:
        for i, seed in enumerate(seeds):
            print(f"[种子] base_seed={base_seed} episode_index={i} episode_seed={seed}")
            run_episode(cfg, args.mode, seed, renderer=renderer,
                        screenshot_dir=args.screenshot_dir, demo=args.demo,
                        fixed_bh=args.size)
            if i < args.episodes - 1 and not args.headless:
                time.sleep(2.5 if args.demo else 1.0)
    finally:
        if renderer is not None:
            renderer.close()


if __name__ == "__main__":
    main()
