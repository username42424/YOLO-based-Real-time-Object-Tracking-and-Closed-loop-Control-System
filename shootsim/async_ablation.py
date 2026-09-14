# -*- coding: utf-8 -*-
"""Deterministic closed-loop A/B/C/D timing experiment.

The plant, capture worker, single inference worker, output worker, and display
feedback are simulated separately.  A/B are fixed/rate-hold references and
C/D are remaining-error controls respectively; D's velocity comes from prior
observations, never from the plant's true velocity.
"""
import argparse
import json
import math
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from async_control import ObservationSnapshot, RemainingErrorController
from motion_arbiter import MotionLedgerSnapshot
from aim_engine import MainEngine


def target_state(t):
    if t < 1.0:
        return 80.0, 0.0
    if t < 2.0:
        return 80.0 + 120.0 * (t - 1.0), 120.0
    if t < 3.0:
        return 200.0, 0.0
    if t < 4.0:
        return 200.0 - 180.0 * (t - 3.0), -180.0
    return 20.0, 0.0


def _percentile(values, q):
    if not values:
        return None
    values = sorted(values)
    return values[min(len(values) - 1, round(q * (len(values) - 1)))]


def run(kind, *, seconds=5.0, observation_period=0.010,
        processing_s=0.009, feedback_s=0.020, output_period=0.010,
        prediction=False, noise_px=0.0, plant_px_per_count=0.44,
        controller_px_per_count=None, tau_s=0.040):
    # Plant calibration is deliberately independent from the controller's
    # configured calibration in config.json.  This lets the experiment expose
    # calibration error instead of silently making the plant an oracle.
    plant_px_per_count = float(plant_px_per_count)
    t = 0.0
    next_capture = 0.0
    obs_busy_until = 0.0
    pending_observation = None
    next_output = output_period
    applied = []  # (feedback_due, counts)
    successful_aim = []  # (commit_t, counts)
    camera_at = lambda at: sum(dx for due, dx in applied if due <= at + 1e-12) * plant_px_per_count
    controller = None
    engine = None
    if kind in ("A", "B", "C", "D"):
        with open(os.path.join(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__))), "config.json"), encoding="utf-8") as handle:
            engine_cfg = json.load(handle)
        engine_cfg["aim_mode"] = "unit"
        engine_cfg["target_classes"] = [0]
        engine_cfg.setdefault("aim_control", {})["prediction_mode"] = (
            "arrival" if kind in ("A", "B", "D") else "current")
        engine = MainEngine(engine_cfg)
        engine.set_prediction_mode("arrival" if kind in ("A", "B", "D") else "current")
    if kind in ("C", "D"):
        unit_cfg = engine_cfg.get("unit", {})
        controller_px_per_count = (float(unit_cfg.get("px_per_count", 0.44))
                                   if controller_px_per_count is None
                                   else float(controller_px_per_count))
        controller = RemainingErrorController(
            # Keep controller calibration separate from plant calibration.
            px_per_count=(controller_px_per_count,
                          float(unit_cfg.get("px_per_count_y", 0.51))),
            # tau is a time constant, not a frame-count multiplier.
            tau_s=float(tau_s),
            max_speed_counts_s=(4000.0, 4000.0), deadzone_px=1.0,
            stop_deadzone_px=0.5, feedback_delay_s=feedback_s,
            max_observation_age_s=0.080)
    fixed_chunks = []
    rate = 0.0
    last_target_measurement = None
    last_known_ledger_counts = 0.0
    estimated_velocity = 0.0
    estimate_confidence = 0.0
    output_times, output_counts = [], []
    obs_times, obs_capture_times = [], []
    errors = []
    display_errors = []
    output_dt = output_period
    last_output_t = 0.0
    current_t = 0.0
    while current_t < seconds - 1e-12:
        current_t = round(current_t + 0.001, 6)
        t = current_t

        # One capture/inference worker: an observation period cannot create
        # impossible throughput when processing takes longer than the period.
        if (pending_observation is None and t + 1e-12 >= next_capture
                and t + 1e-12 >= obs_busy_until):
            capture_t = t
            world_x, _ = target_state(capture_t)
            measured_target = world_x - camera_at(capture_t) + noise_px
            pending_observation = (capture_t + processing_s, capture_t,
                                   measured_target)
            obs_busy_until = capture_t + processing_s
            next_capture += observation_period

        if pending_observation is not None and t + 1e-12 >= pending_observation[0]:
            _, capture_t, measured_target = pending_observation
            pending_observation = None
            image_error = measured_target
            known_ledger_counts = sum(
                dx for committed, dx in successful_aim
                if committed <= capture_t + 1e-12)
            if engine is not None:
                if last_target_measurement is not None:
                    # This is the production-visible ledger, not hidden plant
                    # camera motion.  The plant applies the same input later
                    # through ``applied`` and only uses it for scoring.
                    camera_counts = round(known_ledger_counts -
                                          last_known_ledger_counts)
                    if camera_counts:
                        engine.notify_net(camera_counts, 0)
                dets = [{"cls": 0, "conf": 0.95,
                         "bbox": [160.0 + measured_target - 40.0, 20.0,
                                  160.0 + measured_target + 40.0, 300.0]}]
                engine_dt = (capture_t - last_target_measurement[0]
                             if last_target_measurement is not None else 1.0 / 120.0)
                engine_out = engine.compute(
                    dets, (160.0, 104.0), max(1e-4, engine_dt),
                    processing_latency_s=processing_s)
                # Match production's frame lifecycle; compute() alone does
                # not advance the MainEngine lock/velocity bookkeeping.
                engine.end_frame()
                filtered_error = tuple(engine_out.get("observed_error", (image_error, 0.0)))
                control_error = (float(engine_out.get("raw_dx", filtered_error[0])),
                                 float(engine_out.get("raw_dy", filtered_error[1])))
            else:
                filtered_error = (image_error, 0.0)
                control_error = filtered_error
            if last_target_measurement is not None:
                prev_t, prev_measurement, prev_known_counts = last_target_measurement
                dt = capture_t - prev_t
                scale_x = (float(controller_px_per_count)
                           if controller_px_per_count is not None else
                           float(engine_cfg.get("unit", {}).get(
                               "px_per_count", 0.44)))
                camera_delta = (known_ledger_counts - prev_known_counts) * scale_x
                if dt > 1e-6:
                    estimate = (measured_target - prev_measurement + camera_delta) / dt
                    if abs(estimate) <= 1000.0:
                        alpha = 1.0 - math.exp(-dt / 0.060)
                        estimated_velocity += (estimate - estimated_velocity) * alpha
                        estimate_confidence = min(1.0, estimate_confidence + 0.5)
                    else:
                        estimated_velocity = 0.0
                        estimate_confidence = 0.0
            last_known_ledger_counts = known_ledger_counts
            last_target_measurement = (capture_t, measured_target,
                                       known_ledger_counts)
            event_copy = tuple((ot, (dx, 0), (0, 0)) for ot, dx in successful_aim
                               if ot <= capture_t + 1e-12)
            if engine is None and prediction and estimate_confidence >= 1.0:
                horizon = min(0.020, processing_s + feedback_s) * estimate_confidence
                control_error += estimated_velocity * horizon
            snap = ObservationSnapshot(
                session_id=1, target_id=1, observation_id=len(obs_times) + 1,
                source_frame_id=len(obs_times) + 1,
                image_time_s=capture_t, image_time_kind="sim_capture",
                capture_completed_s=capture_t, published_s=t,
                target_point_px=(160.0 + image_error, 104.0),
                crosshair_px=(160.0, 104.0), raw_error_px=(image_error, 0.0),
                filtered_error_px=filtered_error, box_size_px=(40.0, 100.0),
                confidence=0.9, state="real", last_real_observation_s=capture_t,
                motion_ledger=MotionLedgerSnapshot(capture_t, (0, 0), (0, 0), event_copy),
                target_velocity_px_s=(tuple(engine_out.get("velocity_filtered", (estimated_velocity, 0.0)))
                                     if engine is not None else (estimated_velocity, 0.0)),
                target_velocity_confidence=(tuple(
                    1.0 if bool(v) else 0.0 for v in getattr(
                        engine, "_unit_velocity_valid_axes", (False, False)))
                    if engine is not None else estimate_confidence),
                control_error_px=control_error,
                prediction_mode=(engine.prediction_mode if engine is not None
                                 else "arrival" if prediction else "current"),
                prediction_reason=(engine_out.get("prediction_reason") if engine is not None
                                   else "estimated_velocity" if prediction and estimate_confidence >= 1.0
                                   else "current_mode" if not prediction else "velocity_unready"),
                prediction_horizon_ms=(float(engine_out.get("prediction_horizon_ms", 0.0))
                                       if engine is not None else
                                       min(0.020, processing_s + feedback_s)
                                       * estimate_confidence * 1000.0 if prediction else 0.0),
            )
            obs_times.append(t)
            obs_capture_times.append(capture_t)
            if kind in ("C", "D"):
                controller.observe(snap, t)
            elif kind == "B":
                command = float(engine_out.get("dx", image_error / plant_px_per_count * 0.85))
                unit_cap = float(engine_cfg.get("aim_control", {}).get(
                    "unit_max_counts", 800.0))
                if abs(command) > unit_cap:
                    command *= unit_cap / abs(command)
                rate = command / 0.022
            else:
                command = float(engine_out.get(
                    "dx", image_error / plant_px_per_count * 0.85))
                unit_cap = float(engine_cfg.get("aim_control", {}).get(
                    "unit_max_counts", 800.0))
                if abs(command) > unit_cap:
                    command *= unit_cap / abs(command)
                # Production fixed mode publishes a two-step plan to the
                # shared arbiter.  Reproduce that committed motion here;
                # sending the whole plan in one tick is a different controller.
                fixed_chunks = [command * 0.5, command - command * 0.5]

        if t + 1e-12 >= next_output:
            if output_times:
                output_dt = t - last_output_t
            if kind in ("C", "D"):
                dx, _ = controller.next_motion(output_dt, t, valid=True)
            elif kind == "B":
                dx = rate * output_dt
                if not obs_times or t - obs_times[-1] > 0.080:
                    dx = 0.0
            else:
                dx = fixed_chunks.pop(0) if fixed_chunks else 0.0
            output_times.append(t)
            output_counts.append(dx)
            last_output_t = t
            if abs(dx) > 1e-12:
                successful_aim.append((t, dx))
                applied.append((t + feedback_s, dx))
                if kind in ("C", "D"):
                    controller.on_send_succeeded(dx, 0.0, t)
            next_output += output_period

        errors.append(abs(target_state(t)[0] - camera_at(t)))
        display_errors.append(abs(target_state(t)[0] - camera_at(t)))

    visible = errors[100:]
    static = [e for i, e in enumerate(errors) if 0.4 <= i * 0.001 < 0.9]
    post_stop = [e for i, e in enumerate(errors) if 3.0 <= i * 0.001 < 3.5]
    obs_intervals = [b - a for a, b in zip(obs_times, obs_times[1:])]
    out_intervals = [b - a for a, b in zip(output_times, output_times[1:])]
    return {
        "group": kind,
        "frame_schedule": "fixed" if kind == "A" else "asap",
        "control_strategy": {
            "A": "legacy_unit",
            "B": "rate_hold",
            "C": "remaining_error",
            "D": "remaining_error",
        }[kind],
        "prediction": bool(engine is not None and engine.prediction_mode == "arrival"),
        "effective_prediction_mode": (engine.prediction_mode if engine is not None
                                       else "arrival" if prediction else "current"),
        "observation_interval_ms": _percentile(obs_intervals, .5) * 1000 if obs_intervals else None,
        "observation_p95_ms": _percentile(obs_intervals, .95) * 1000 if obs_intervals else None,
        "output_p50_ms": _percentile(out_intervals, .5) * 1000 if out_intervals else None,
        "output_p95_ms": _percentile(out_intervals, .95) * 1000 if out_intervals else None,
        "processing_ms": processing_s * 1000, "feedback_ms": feedback_s * 1000,
        "median_error_px": statistics.median(visible),
        "p95_error_px": _percentile(visible, .95),
        "within_60_ratio": sum(e <= 60.0 for e in visible) / max(1, len(visible)),
        "settle_s": next((i * 0.001 for i, e in enumerate(errors, 1) if e <= 5.0), None),
        "static_jitter_px": statistics.pstdev(static) if len(static) > 1 else 0.0,
        "post_stop_max_px": max(post_stop) if post_stop else None,
        "max_error_px": max(errors), "total_abs_counts": sum(abs(x) for x in output_counts),
        "observations": len(obs_times), "outputs": len(output_times),
        "estimated_velocity_samples": sum(1 for _ in obs_times if prediction),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feedback-ms", type=float, default=20.0)
    ap.add_argument("--processing-ms", type=float, default=9.0)
    ap.add_argument("--observation-ms", type=float, default=10.0)
    ap.add_argument("--output-ms", type=float, default=10.0)
    ap.add_argument("--tau-ms", type=float, default=40.0,
                    help="remaining-error response time constant")
    ap.add_argument("--plant-px-per-count", type=float, default=0.44,
                    help="simulated physical calibration, independent of controller config")
    ap.add_argument("--controller-px-per-count", type=float, default=None,
                    help="remaining-error calibration; default reads config.json")
    ap.add_argument("--scan", action="store_true",
                    help="independently scan processing/feedback/observation/output timing")
    args = ap.parse_args()
    common = dict(feedback_s=args.feedback_ms / 1000.0,
                  processing_s=args.processing_ms / 1000.0,
                  observation_period=args.observation_ms / 1000.0,
                  output_period=args.output_ms / 1000.0,
                  plant_px_per_count=args.plant_px_per_count,
                  controller_px_per_count=args.controller_px_per_count,
                  tau_s=args.tau_ms / 1000.0)
    rows = [run(k, observation_period=(0.025 if k == "A" else common["observation_period"]),
                prediction=(k == "D"), **{x: y for x, y in common.items()
                                          if x != "observation_period"})
            for k in ("A", "B", "C", "D")]
    print("group obs_ms out_p50 out_p95 proc_ms feedback_ms median_px p95_px in60 jitter post_stop")
    for row in rows:
        print("{group} {observation_interval_ms} {output_p50_ms} {output_p95_ms} "
              "{processing_ms:.1f} {feedback_ms:.1f} {median_error_px:.2f} "
              "{p95_error_px:.2f} {within_60_ratio:.3f} {static_jitter_px:.3f} "
              "{post_stop_max_px}".format(**row))
    payload = {"assumptions": {
            "plant_feedback_delay_s": common["feedback_s"],
            "plant_px_per_count": common["plant_px_per_count"],
            "controller_px_per_count": ("config.json" if common["controller_px_per_count"] is None
                                         else common["controller_px_per_count"]),
            "processing_s": common["processing_s"],
            "observation_period_s": common["observation_period"],
            "output_period_s": common["output_period"],
            "D_velocity_source": "previous observations with known aim-motion ledger",
            "true_velocity_for_scoring_only": True},
            "results": rows}
    if args.scan:
        base = {k: v for k, v in common.items() if k != "observation_period"}
        payload["scans"] = {"processing_ms": [], "feedback_ms": [],
                             "observation_ms": [], "output_ms": [], "tau_ms": []}
        for value in (6.0, 12.0, 20.0):
            c = run("C", processing_s=value / 1000.0,
                    observation_period=common["observation_period"], **{
                        k: v for k, v in base.items() if k != "processing_s"})
            payload["scans"]["processing_ms"].append({"value": value,
                "observation_ms": c["observation_interval_ms"],
                "median_error_px": c["median_error_px"]})
        for value in (5.0, 20.0, 40.0):
            c = run("C", feedback_s=value / 1000.0,
                    observation_period=common["observation_period"], **{
                        k: v for k, v in base.items() if k != "feedback_s"})
            d = run("D", feedback_s=value / 1000.0,
                    observation_period=common["observation_period"], **{
                        k: v for k, v in base.items() if k != "feedback_s"})
            payload["scans"]["feedback_ms"].append({"value": value,
                "C_median_error_px": c["median_error_px"],
                "D_median_error_px": d["median_error_px"]})
        for value in (8.0, 15.0, 25.0):
            c = run("C", observation_period=value / 1000.0, **base)
            payload["scans"]["observation_ms"].append({"value": value,
                "measured_ms": c["observation_interval_ms"],
                "median_error_px": c["median_error_px"]})
        for value in (5.0, 10.0, 20.0):
            c = run("C", output_period=value / 1000.0,
                    observation_period=common["observation_period"], **{
                        k: v for k, v in base.items() if k != "output_period"})
            payload["scans"]["output_ms"].append({"value": value,
                "measured_ms": c["output_p50_ms"],
                "median_error_px": c["median_error_px"]})
        for value in (25.0, 40.0, 60.0):
            tau_base = {k: v for k, v in base.items() if k != "tau_s"}
            c = run("C", tau_s=value / 1000.0, **tau_base)
            d = run("D", tau_s=value / 1000.0, **tau_base)
            payload["scans"]["tau_ms"].append({"value": value,
                "C_median_error_px": c["median_error_px"],
                "C_p95_error_px": c["p95_error_px"],
                "D_median_error_px": d["median_error_px"],
                "D_p95_error_px": d["p95_error_px"]})
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results",
                       "async_ablation_results.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=1)
    print("saved", os.path.relpath(out, os.path.dirname(os.path.dirname(out))))


if __name__ == "__main__":
    main()
