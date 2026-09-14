# -*- coding: utf-8 -*-
"""Closed-loop replay harness for the 20260906 v2 tracking sweep.

Drives shootsim Env + the real MainEngine tracker per episode (mirroring
run_sim.run_episode's replay loop), records every engine observation and every
integer mouse tick, and computes the full metric set from the sweep spec:
first-inner60 times, dwell, moving error percentiles, X-lag for fast strafes,
prediction state ratios, jitter/overshoot stability metrics, plus a
frame-aligned fidelity comparison against the source log rows.
"""
import copy
import json
import math
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SHOOTSIM = os.path.dirname(HERE)
ROOT = os.path.dirname(SHOOTSIM)
sys.path.insert(0, SHOOTSIM)

from env import Env  # noqa: E402
from perception import create_perceiver  # noqa: E402
from trackers import create_tracker  # noqa: E402
import replay as replay_mod  # noqa: E402

LOG_DIR = os.path.join(ROOT, "logs")

GROUP_A_LOGS = {
    "aim_20260906_003353.log": dict(chest_ratio=0.2),
    "aim_20260906_001842.log": dict(chest_ratio=0.2),
    "aim_20260906_000134.log": dict(chest_ratio=0.2),
}
GROUP_B_LOGS = {
    "aim_20260905_232312.log": dict(chest_ratio=0.3),
    "aim_20260905_231714.log": dict(chest_ratio=0.3),
    "aim_20260905_230935.log": dict(chest_ratio=0.3),
    "aim_20260905_230244.log": dict(chest_ratio=0.3),
}

VIEW_SCALE_X = 0.44
VIEW_SCALE_Y = 0.51

STRAFE_VX = 400.0        # px/s, fast-strafe subset threshold
PRED_BLOCK_REASONS = ("hold_", "switched_", "predict_missing")


def build_sim_cfg(logs, apply_theta=1.0, scale_x=VIEW_SCALE_X,
                  scale_y=VIEW_SCALE_Y, max_seconds=3.0):
    return {
        "world": {"width": 4000, "height": 4000, "spawn_margin": 200},
        "screen": {"width": 640, "height": 640},
        "target": {"box_w": 80.0, "box_h": 160.0, "hp": 150,
                   "min_count": 1, "max_count": 1},
        "weapon": {"fire_rate_hz": 11.6667, "damage": {"head": 40, "mid": 20},
                   "red_dot_offset_px": 0.0, "red_dot_jitter_px": 0.0,
                   "red_dot_recoil_max_px": 0.0, "red_dot_recoil_per_shot_px": 0.0,
                   "red_dot_recoil_recover_px": 0.0},
        "view": {"sensitivity": scale_x, "sensitivity_y": scale_y,
                 "apply_delay_frames": 0, "apply_theta": apply_theta, "fps": 100},
        "perception": {"mode": "replay", "capture_size": 320},
        "replay": {
            "logs": [os.path.join(LOG_DIR, n) for n in logs],
            "scale_x": scale_x, "scale_y": scale_y,
            "capture_center": 160.0, "min_points": 3, "send_lag": 0,
            "on_end": "stop", "tracking_only": True,
            "head_zone_ratio": 0.25, "dwell_h_ratio": 0.6,
            "use_logged_crosshair": True,
            "apply_theta": apply_theta,
            "source_chest_ratio": 0.2,
            "source_chest_ratio_by_log": {
                name: float(cfg["chest_ratio"]) for name, cfg in GROUP_B_LOGS.items()
            },
        },
        "tracker": {"strategy": "main_real", "view_scale": scale_x,
                    "view_scale_y": scale_y, "processing_latency_ms": 15.0,
                    "obs_delay_frames": 0, "control_interval_frames": 1,
                    "control_every_update": True},
        "shooter": {"policy": "always", "auto_fire": True},
        "human_track": {"enabled": False},
        "episode": {"max_seconds": max_seconds, "seed": 0},
        "logging": {"dir": "results", "per_frame": False, "frame_interval": 10},
    }


def load_production_cfg():
    with open(os.path.join(ROOT, "config.json"), encoding="utf-8") as f:
        return json.load(f)


def patch_cfg(main_cfg, values):
    cfg = copy.deepcopy(main_cfg)
    for dotted, value in (values or {}).items():
        node = cfg
        parts = dotted.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return cfg


def percentile(values, q):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(q * (len(ordered) - 1)))]


# ── episode driver ───────────────────────────────────────────────────────

class EpisodeRecord:
    __slots__ = ("obs", "ticks", "summary", "ep_id")

    def __init__(self, ep_id):
        self.ep_id = ep_id
        self.obs = []
        self.ticks = []
        self.summary = {}


def run_episode(sim_cfg, main_cfg, ep, ep_id=0, output_phase_s=None,
                collect_dets=True, on_tracker=None, engine_factory=None):
    """Replay one prebuilt episode (replay.ReplayEpisode); returns EpisodeRecord.

    on_tracker(tracker) is called right after tracker creation (before the
    loop) so callers can instrument/replace internals without touching this
    driver.  engine_factory(main_cfg) may return a replacement engine; it is
    installed after tracker construction with view scaling carried over.
    """
    env = Env(sim_cfg)
    player = replay_mod.ReplayPlayer(
        ep, max_seconds=float(sim_cfg["episode"]["max_seconds"]),
        on_end=str(sim_cfg["replay"].get("on_end", "stop")))
    env.spawn_replay(player, 0)

    rec = EpisodeRecord(ep_id)

    def send_fn(dx, dy):
        rec.ticks.append({"t": tracker._t, "counts": (int(dx), int(dy))})
        env.apply_mouse(dx, dy)

    tracker = create_tracker(
        "main_real", sim_cfg["tracker"], env.sw, env.sh, env.rng,
        send_fn=send_fn, main_cfg_path=main_cfg)
    tracker.reset()
    if on_tracker is not None:
        on_tracker(tracker)
    if engine_factory is not None:
        new_engine = engine_factory(main_cfg)
        new_engine.view_scale = tracker.engine.view_scale
        new_engine.view_scale_y = tracker.engine.view_scale_y
        tracker.engine = new_engine
    phase = sim_cfg["replay"].get("output_phase_s", -1.0)
    if output_phase_s is not None:
        phase = output_phase_s
    if phase is not None and phase >= 0.0:
        tracker.set_output_phase(phase)

    orig_compute = tracker.engine.compute

    def recording_compute(dets, crosshair, dt, *args, **kwargs):
        out = orig_compute(dets, crosshair, dt, *args, **kwargs)
        rec.obs.append({
            "t": env.t,
            "dt": env.dt,
            "detected": env.replay_det,
            "dets": ([(d["cls"], tuple(d["bbox"])) for d in dets]
                     if collect_dets else None),
            "target": out.get("target"),
            "lock_reason": tracker.engine.diag_reason,
            "locked_box": (list(tracker.engine.locked_box[0])
                           if tracker.engine.locked_box[0] is not None else None),
            "obs_err": tuple(out.get("observed_error") or (0.0, 0.0)),
            "vel_raw": tuple(out.get("velocity_raw") or (0.0, 0.0)),
            "vel_f": tuple(out.get("velocity_filtered") or (0.0, 0.0)),
            "reason": out.get("prediction_reason"),
            "horizon_ms": out.get("prediction_horizon_ms", 0.0),
            "lead": tuple(out.get("lead") or (0.0, 0.0)),
            "cmd": (out.get("dx", 0.0), out.get("dy", 0.0)),
            "can_send": out.get("can_send", False),
            "in_deadzone": out.get("in_deadzone", False),
            "lock_alpha": out.get("lock_filter_alpha", 1.0),
        })
        return out

    tracker.engine.compute = recording_compute
    perceiver = create_perceiver("replay", sim_cfg["perception"], env.rng, None)

    while True:
        if hasattr(tracker, "advance_to"):
            tracker.advance_to(env.t)
        env.update_red_dot()
        if hasattr(tracker, "set_reference"):
            tracker.set_reference(*env.red_dot())
        boxes = perceiver.perceive(env, 0)
        if hasattr(tracker, "set_live_latency"):
            tracker.set_live_latency(getattr(env, "replay_latency", None))
        tracker.update(boxes, env.dt)
        running = env.step(0.0, 0.0, advance_to=tracker.advance_to)
        if not running:
            break

    rec.summary = env.summary()
    return rec


# ── per-episode metrics ──────────────────────────────────────────────────

def episode_metrics(rec, cap_counts=280.3026):
    s = rec.summary
    obs = rec.obs
    m = {
        "first_inner60": s.get("first_inner60_time"),
        "dwell_inner60": s.get("post_entry_inner60_dwell", 0.0),
        "dwell_reddot": s.get("dwell_reddot", 0.0),
        "true_timeout": bool(s.get("true_timeout")),
        "entry_censored": bool(s.get("entry_censored")),
        "exhausted": bool(s.get("replay_exhausted")),
        "observed_s": s.get("replay_observed_seconds", 0.0),
        "n_obs": len(obs),
        "n_ticks": len(rec.ticks),
    }
    errs, lags_all, lags_strafe, speeds, strafe_vx = [], [], [], [], []
    quiet_cmds = []
    reasons = {}
    horizons = []
    sent_rows = 0
    cap_sat = 0
    dir_wrong = 0
    dir_wrong_total = 0
    for row in obs:
        reason = str(row["reason"])
        cmd = row["cmd"]
        locked = (row["target"] is not None
                  and str(row["lock_reason"] or "").startswith("locked"))
        if row["target"] is not None:
            reasons[reason] = reasons.get(reason, 0) + 1
        if cmd != (0.0, 0.0):
            sent_rows += 1
            if math.hypot(*cmd) >= cap_counts - 0.5:
                cap_sat += 1
        if row["horizon_ms"] > 0:
            horizons.append(row["horizon_ms"])
        if not locked:
            continue
        ex, ey = row["obs_err"]
        errs.append(math.hypot(ex, ey))
        vx, vy = row["vel_raw"]
        sp = math.hypot(vx, vy)
        lead = row["lead"]
        if (abs(vx) >= STRAFE_VX and abs(vx) >= 1.5 * abs(vy)
                and lead[0] != 0.0):
            dir_wrong_total += 1
            if lead[0] * vx < 0.0:
                dir_wrong += 1
        if sp < 100.0 and cmd != (0.0, 0.0):
            quiet_cmds.append(math.hypot(*cmd))
        if sp > 1.0:
            lag = (ex * vx + ey * vy) / sp
            if abs(vx) >= STRAFE_VX and abs(vx) >= 1.5 * abs(vy):
                lags_strafe.append(lag)
                strafe_vx.append(vx)
            else:
                lags_all.append(lag)
            speeds.append(sp)
    m["err_p50"] = percentile(errs, .5)
    m["err_p75"] = percentile(errs, .75)
    m["err_p90"] = percentile(errs, .9)
    m["xlag_p75"] = percentile(lags_strafe, .75)
    m["xlag_lead_p75"] = (-percentile([-v for v in lags_strafe], .75)
                          if lags_strafe else None)
    m["xdir_wrong_rate"] = dir_wrong / dir_wrong_total if dir_wrong_total else 0.0
    m["xdir_wrong_total"] = dir_wrong_total
    m["speed_p90"] = percentile(speeds, .9)
    m["strafe_samples"] = len(lags_strafe)
    m["reason_counts"] = reasons
    m["horizon_p50"] = percentile(horizons, .5)
    m["cap_saturation_rate"] = cap_sat / sent_rows if sent_rows else 0.0
    # tick-stream stability
    big_flips = flip_pairs = 0
    micro_sq = micro_pairs = 0
    peak = step_sum = 0.0
    last_move = last_any = None
    for tick in rec.ticks:
        c = tick["counts"]
        mag = math.hypot(*c)
        if mag <= 0:
            continue
        step_sum += mag
        peak = max(peak, mag)
        if last_any is not None and mag < 20 and math.hypot(*last_any) < 20:
            micro_pairs += 1
            micro_sq += (c[0] - last_any[0]) ** 2 + (c[1] - last_any[1]) ** 2
        last_any = c
        if mag >= 20:
            if last_move is not None:
                flip_pairs += 1
                if c[0] * last_move[0] + c[1] * last_move[1] < 0:
                    big_flips += 1
            last_move = c
    m["mean_step"] = step_sum / len(rec.ticks) if rec.ticks else 0.0
    m["peak_step"] = peak
    m["flip_rate"] = big_flips / flip_pairs if flip_pairs else 0.0
    m["micro_jitter_rms"] = math.sqrt(micro_sq / micro_pairs) if micro_pairs else 0.0
    # command jitter on quasi-static targets + sudden-jump rate
    m["quiet_cmd_p75"] = percentile(quiet_cmds, .75)
    m["quiet_cmd_p95"] = percentile(quiet_cmds, .95)
    jumps = 0
    prev_cmd = None
    for tick in rec.ticks:
        c = tick["counts"]
        if math.hypot(*c) <= 0:
            continue
        if prev_cmd is not None:
            d = math.hypot(c[0] - prev_cmd[0], c[1] - prev_cmd[1])
            if d > 40.0:
                jumps += 1
        prev_cmd = c
    m["jump_count"] = jumps
    return m


# ── set-level evaluation (the search scoring function) ───────────────────

def evaluate_set(sim_cfg, main_cfg, episodes, phases=None, collect_dets=False):
    """Run every episode; return pooled metrics + speed-bucket breakdown.

    episodes: list of replay.ReplayEpisode (same order as the split manifest).
    phases: optional per-episode 15ms output-grid phase (seconds).
    """
    cap_counts = float((main_cfg.get("aim_control") or {}).get(
        "unit_max_counts", 280.3026))
    recs = []
    for i, ep in enumerate(episodes):
        phase = phases[i] if phases else None
        recs.append(run_episode(sim_cfg, main_cfg, ep, ep_id=i,
                                output_phase_s=phase, collect_dets=collect_dets))
    per_ep = [episode_metrics(r, cap_counts) for r in recs]

    # pooled per-observation samples
    errs, strafe_lags, near_lags = [], [], []
    strafe_errs, quiet_errs = [], []
    reasons = {}
    horizon_vals = []
    jitter_vals, flip_rates = [], []
    inner = []
    dwell_num = dwell_den = 0.0
    timeout = 0
    for rec, m in zip(recs, per_ep):
        errs.extend(_locked_errors(rec))
        strafe, near, quiet, sp = _bucketed_samples(rec)
        strafe_lags.extend(v[0] for v in strafe)
        strafe_errs.extend(v[1] for v in strafe)
        near_lags.extend(v[0] for v in near)
        quiet_errs.extend(quiet)
        for k, v in m["reason_counts"].items():
            reasons[k] = reasons.get(k, 0) + v
        horizon_vals.extend(row["horizon_ms"] for row in rec.obs
                            if row["horizon_ms"] > 0)
        jitter_vals.append(m["micro_jitter_rms"])
        flip_rates.append(m["flip_rate"])
        if m["first_inner60"] is not None:
            inner.append(m["first_inner60"])
        dwell_num += rec.summary.get("post_entry_inner60_seconds", 0.0)
        dwell_den += rec.summary.get("post_entry_observed_seconds", 0.0)
        timeout += bool(m["true_timeout"])
    n = len(recs)
    sent = sum(1 for r in recs for row in r.obs if row["cmd"] != (0.0, 0.0))
    cap_sat = sum(1 for r in recs for row in r.obs
                  if row["cmd"] != (0.0, 0.0)
                  and math.hypot(*row["cmd"]) >= cap_counts - 0.5)
    quiet_cmds = []
    jump_total = 0
    flip_rates = []
    for r, m in zip(recs, per_ep):
        for row in r.obs:
            locked = (row["target"] is not None
                      and str(row["lock_reason"] or "").startswith("locked"))
            if locked and row["cmd"] != (0.0, 0.0):
                vx, vy = row["vel_raw"]
                if math.hypot(vx, vy) < 100.0:
                    quiet_cmds.append(math.hypot(*row["cmd"]))
        jump_total += m["jump_count"]
    pred_active = sum(v for k, v in reasons.items() if k.startswith("arrival_"))
    pred_unready = (reasons.get("velocity_unready", 0)
                    + reasons.get("lock_warmup", 0))
    pred_blocked = sum(v for k, v in reasons.items() if k.startswith(PRED_BLOCK_REASONS))
    total_locked_obs = sum(1 for r in recs for row in r.obs
                           if row["target"] is not None
                           and str(row["lock_reason"] or "").startswith("locked"))
    return {
        "episodes": n,
        # entry metrics
        "first_inner60_median": statistics.median(inner) if inner else None,
        "first_inner60_p75": percentile(inner, .75) if inner else None,
        "first_inner60_p90": percentile(inner, .9) if inner else None,
        "first_inner60_success": len(inner),
        "no_entry_rate": 1.0 - len(inner) / n if n else 1.0,
        "dwell_inner60": dwell_num / dwell_den if dwell_den else 0.0,
        "timeout_rate": timeout / n if n else 0.0,
        # moving-error metrics
        "err_p50": percentile(errs, .5), "err_p75": percentile(errs, .75),
        "err_p90": percentile(errs, .9),
        "strafe_xlag_p50": percentile(strafe_lags, .5),
        "strafe_xlag_p75": percentile(strafe_lags, .75),
        "strafe_xlag_p90": percentile(strafe_lags, .9),
        "strafe_err_p75": percentile(strafe_errs, .75),
        "strafe_dir_wrong_rate": (
            sum(m.get("xdir_wrong_rate", 0.0) * m.get("xdir_wrong_total", 0)
                for m in per_ep)
            / max(1, sum(m.get("xdir_wrong_total", 0) for m in per_ep))),
        "near_xlag_p75": percentile(near_lags, .75),
        "quiet_err_p75": percentile(quiet_errs, .75),
        "strafe_samples": len(strafe_lags),
        "total_locked_obs": total_locked_obs,
        # prediction state ratios
        "pred_active_rate": pred_active / max(1, total_locked_obs),
        "pred_unready_rate": pred_unready / max(1, total_locked_obs),
        "pred_blocked_rate": pred_blocked / max(1, total_locked_obs),
        "pred_missing_rate": (reasons.get("predict_missing", 0)
                              + sum(v for k, v in reasons.items()
                                    if k.startswith("missing_decay"))
                              ) / max(1, total_locked_obs),
        "horizon_p50": percentile(horizon_vals, .5),
        "cap_saturation_rate": cap_sat / sent if sent else 0.0,
        # stability
        "micro_jitter_rms_median": statistics.median(jitter_vals) if jitter_vals else 0.0,
        "flip_rate_median": statistics.median(flip_rates) if flip_rates else 0.0,
        "quiet_cmd_p75": percentile(quiet_cmds, .75),
        "quiet_cmd_p95": percentile(quiet_cmds, .95),
        "jump_count_total": jump_total,
        "reason_counts": reasons,
        "_per_ep": per_ep,
        "_recs": recs,
    }


def _locked_errors(rec):
    out = []
    for row in rec.obs:
        if (row["target"] is not None
                and str(row["lock_reason"] or "").startswith("locked")):
            out.append(math.hypot(*row["obs_err"]))
    return out


def _bucketed_samples(rec):
    """Return (strafe[(lag,err)], near[(lag,err)], quiet_errs, speeds)."""
    strafe, near, quiet, speeds = [], [], [], []
    for row in rec.obs:
        locked = (row["target"] is not None
                  and str(row["lock_reason"] or "").startswith("locked"))
        if not locked:
            continue
        ex, ey = row["obs_err"]
        err = math.hypot(ex, ey)
        vx, vy = row["vel_raw"]
        sp = math.hypot(vx, vy)
        if sp > 1.0:
            lag = (ex * vx + ey * vy) / sp
            if abs(vx) >= STRAFE_VX and abs(vx) >= 1.5 * abs(vy):
                strafe.append((lag, err))
            else:
                near.append((lag, err))
            speeds.append(sp)
        else:
            quiet.append(err)
    return strafe, near, quiet, speeds
