# -*- coding: utf-8 -*-
"""Single-variable replay probes against the latest structured live log."""
import copy
import argparse
import json
import math
import os
import statistics

from config import load_config
from logger import SilentLogger
import replay
import run_sim


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LOG = os.path.join(ROOT, "logs", "aim_20260905_131152.log")


def patch(cfg, values):
    result = copy.deepcopy(cfg)
    for dotted, value in values.items():
        node = result
        parts = dotted.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return result


def percentile(values, q):
    if not values:
        return 0.0
    values = sorted(values)
    return values[round(q * (len(values) - 1))]


def evaluate(sim_cfg, main_cfg):
    episodes = replay.load_episodes(sim_cfg["replay"])
    moving_errors = []
    moving_lag = []
    moving_speeds = []
    moving_lag_ms = []
    moving_rows = []
    active = moving = 0
    inner_times = []
    post_inner = post_observed = above = observed = 0.0
    timeouts = 0
    for ep_id in range(len(episodes)):
        last_out = [None]

        def hook(_dx, _dy, out):
            nonlocal active, moving
            if out is last_out[0]:
                return
            last_out[0] = out
            vx, vy = out.get("velocity_raw") or (0.0, 0.0)
            speed = math.hypot(vx, vy)
            if speed < 100.0:
                return
            ex, ey = out.get("observed_error") or (0.0, 0.0)
            moving += 1
            error = math.hypot(ex, ey)
            lag = max(0.0, (ex * vx + ey * vy) / speed)
            lag_ms = lag / speed * 1000.0
            moving_errors.append(error)
            moving_lag.append(lag)
            moving_speeds.append(speed)
            moving_lag_ms.append(lag_ms)
            moving_rows.append((speed, error, lag, lag_ms))
            active += out.get("prediction_reason") == "arrival_active"

        summary = run_sim.run_episode(
            sim_cfg, "auto", ep_id, renderer=None, logger=SilentLogger(),
            quiet=True, main_cfg_path=main_cfg, send_hook=hook)
        if summary.get("first_inner60_time") is not None:
            inner_times.append(summary["first_inner60_time"])
        timeouts += bool(summary.get("true_timeout"))
        post_inner += summary.get("post_entry_inner60_seconds", 0.0)
        post_observed += summary.get("post_entry_observed_seconds", 0.0)
        above += summary.get("above_box_seconds", 0.0)
        observed += summary.get("replay_observed_seconds", 0.0)

    speed_cuts = [percentile(moving_speeds, q) for q in (0.25, 0.50, 0.75)]
    speed_bands = {}
    edges = [0.0] + speed_cuts + [float("inf")]
    for index in range(4):
        rows = [row for row in moving_rows
                if edges[index] <= row[0] < edges[index + 1]]
        speed_bands[f"q{index + 1}"] = {
            "samples": len(rows),
            "speed_p50_px_s": round(percentile([r[0] for r in rows], 0.50), 1),
            "error_p75_px": round(percentile([r[1] for r in rows], 0.75), 3),
            "lag_p75_px": round(percentile([r[2] for r in rows], 0.75), 3),
            "lag_time_p75_ms": round(percentile([r[3] for r in rows], 0.75), 2),
        }
    return {
        "episodes": len(episodes),
        "moving_samples": moving,
        "moving_error_p50_px": round(percentile(moving_errors, 0.50), 3),
        "moving_error_p75_px": round(percentile(moving_errors, 0.75), 3),
        "moving_positive_lag_p75_px": round(percentile(moving_lag, 0.75), 3),
        "moving_speed_p50_px_s": round(percentile(moving_speeds, 0.50), 1),
        "moving_speed_p75_px_s": round(percentile(moving_speeds, 0.75), 1),
        "moving_speed_p90_px_s": round(percentile(moving_speeds, 0.90), 1),
        "moving_lag_time_p75_ms": round(percentile(moving_lag_ms, 0.75), 2),
        "speed_quartiles": speed_bands,
        "prediction_active_rate": round(active / moving, 4) if moving else 0.0,
        "first_inner60_median": round(statistics.median(inner_times), 4)
        if inner_times else None,
        "target_dwell_rate": round(post_inner / post_observed, 4)
        if post_observed else 0.0,
        "above_box_ratio": round(above / observed, 4) if observed else 0.0,
        "true_timeouts": timeouts,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log", default=LOG,
        help="Structured live aim log to replay.",
    )
    parser.add_argument(
        "--source-chest-ratio", type=float, default=0.65,
        help="chest_ratio used when the selected log was recorded.",
    )
    parser.add_argument(
        "--alphas", type=float, nargs="+",
        help="Run only these fixed lock-box alpha values, e.g. 0.2 0.3 0.4.",
    )
    parser.add_argument(
        "--check-current",
        action="store_true",
        help="Check the current production config against the live-log replay gate.",
    )
    args = parser.parse_args()
    sim_cfg = load_config(os.path.join(HERE, "config.json"))
    sim_cfg["replay"].update({
        "logs": [os.path.abspath(args.log)],
        "holdout_logs": [os.path.abspath(args.log)],
        "source_chest_ratio": args.source_chest_ratio,
        "tracking_only": True,
    })
    main_cfg = json.load(open(os.path.join(ROOT, "config.json"), encoding="utf-8"))
    if args.check_current:
        result = evaluate(sim_cfg, main_cfg)
        print("current", json.dumps(result, ensure_ascii=False, sort_keys=True))
        failures = []
        if result["moving_error_p75_px"] > 16.0:
            failures.append("moving_error_p75_px > 16.0")
        if result["moving_positive_lag_p75_px"] > 9.0:
            failures.append("moving_positive_lag_p75_px > 9.0")
        if result["true_timeouts"] > 0:
            failures.append("true_timeouts > 0")
        if failures:
            raise SystemExit("FAIL: " + ", ".join(failures))
        print("PASS: current config satisfies the live tracking replay gate")
        return
    variants = {
        "baseline": {},
        "filter_adaptive": {"aim_control.lock_box_filter_mode": "adaptive"},
        "filter_fixed_065": {
            "aim_control.lock_box_filter_mode": "fixed",
            "aim_control.lock_box_smoothing_alpha": 0.65,
        },
        "box_alpha_1": {"aim_control.lock_box_smoothing_alpha": 1.0},
        "box_alpha_02": {"aim_control.lock_box_smoothing_alpha": 0.2},
        "box_alpha_01": {"aim_control.lock_box_smoothing_alpha": 0.1},
        "box_alpha_015": {"aim_control.lock_box_smoothing_alpha": 0.15},
        "box_alpha_025": {"aim_control.lock_box_smoothing_alpha": 0.25},
        "box_alpha_03": {"aim_control.lock_box_smoothing_alpha": 0.3},
        "box_alpha_06": {"aim_control.lock_box_smoothing_alpha": 0.6},
        "box_alpha_08": {"aim_control.lock_box_smoothing_alpha": 0.8},
        "prediction_lock_2": {"aim_control.prediction_lock_frames": 2},
        "prediction_tau_008": {"aim_control.prediction_vel_tau": 0.08},
        "prediction_tau_012": {"aim_control.prediction_vel_tau": 0.12},
        "prediction_tau_020": {"aim_control.prediction_vel_tau": 0.20},
        "prediction_min_70": {"aim_control.prediction_min_ms": 70.0},
        "prediction_min_90": {"aim_control.prediction_min_ms": 90.0},
        "move_steps_2": {"unit.move_steps": 2},
        "move_steps_1": {"unit.move_steps": 1},
        "gain_softness_x_8": {"unit.gain_softness_px_x": 8.0},
        "gain_softness_x_2": {"unit.gain_softness_px_x": 2.0},
        "prediction_current": {
            "aim_control.prediction_mode": "current",
            "aim_control.unit_prediction_enabled": False,
        },
        "current_steps_1": {
            "aim_control.prediction_mode": "current",
            "aim_control.unit_prediction_enabled": False,
            "unit.move_steps": 1,
        },
        "current_steps_1_gain_x8": {
            "aim_control.prediction_mode": "current",
            "aim_control.unit_prediction_enabled": False,
            "unit.move_steps": 1,
            "unit.gain_softness_px_x": 8.0,
        },
        "current_steps_1_gain_x2": {
            "aim_control.prediction_mode": "current",
            "aim_control.unit_prediction_enabled": False,
            "unit.move_steps": 1,
            "unit.gain_softness_px_x": 2.0,
        },
        "box_02_gain_x2": {
            "aim_control.lock_box_smoothing_alpha": 0.2,
            "unit.gain_softness_px_x": 2.0,
        },
        "box_02_steps_1": {
            "aim_control.lock_box_smoothing_alpha": 0.2,
            "unit.move_steps": 1,
        },
        "box_02_gain_x2_steps_1": {
            "aim_control.lock_box_smoothing_alpha": 0.2,
            "unit.gain_softness_px_x": 2.0,
            "unit.move_steps": 1,
        },
        "box_02_gain_x2_steps_1_min70": {
            "aim_control.lock_box_smoothing_alpha": 0.2,
            "unit.gain_softness_px_x": 2.0,
            "unit.move_steps": 1,
            "aim_control.prediction_min_ms": 70.0,
        },
        "chest_ratio_035": {"chest_ratio": 0.35},
    }
    if args.alphas:
        invalid = [value for value in args.alphas if not 0.05 <= value <= 1.0]
        if invalid:
            parser.error("--alphas values must be between 0.05 and 1.0")
        variants = {
            f"box_alpha_{value:g}": {
                "aim_control.lock_box_filter_mode": "fixed",
                "aim_control.lock_box_smoothing_alpha": value,
            }
            for value in args.alphas
        }
    for name, values in variants.items():
        if "aim_control.lock_box_smoothing_alpha" in values:
            values.setdefault("aim_control.lock_box_filter_mode", "fixed")
        print(name, json.dumps(evaluate(sim_cfg, patch(main_cfg, values)),
                               ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
