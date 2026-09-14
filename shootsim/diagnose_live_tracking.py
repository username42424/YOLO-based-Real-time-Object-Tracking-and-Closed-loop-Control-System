# -*- coding: utf-8 -*-
"""Deterministic tracking-lag diagnostic for structured live aim logs."""
import argparse
import json
import math
import statistics


MARKER = "[瞄准观测] "


def percentile(values, q):
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[round(q * (len(ordered) - 1))]


def analyze(path):
    rows = []
    with open(path, encoding="utf-8", errors="replace") as handle:
        lines = list(handle)
        for line in lines:
            if MARKER not in line:
                continue
            try:
                rows.append(json.loads(line.split(MARKER, 1)[1]))
            except (ValueError, TypeError):
                continue

    locked = [row for row in rows
              if str(row.get("lock_reason", "")).startswith("locked")]
    moving = []
    for row in locked:
        vx, vy = row.get("velocity_raw") or (0.0, 0.0)
        speed = math.hypot(vx, vy)
        if speed < 100.0:
            continue
        ex, ey = row.get("observed_error") or (0.0, 0.0)
        lag = (ex * vx + ey * vy) / speed
        moving.append((row, math.hypot(ex, ey), lag, speed))

    errors = [item[1] for item in moving]
    positive_lag = [max(0.0, item[2]) for item in moving]
    lag_time_ms = [max(0.0, item[2]) / item[3] * 1000.0 for item in moving]
    speeds = [item[3] for item in moving]
    speed_cuts = [percentile(speeds, q) for q in (0.25, 0.50, 0.75)]
    speed_bands = {}
    edges = [0.0] + speed_cuts + [float("inf")]
    for index in range(4):
        band = [item for item in moving
                if edges[index] <= item[3] < edges[index + 1]]
        speed_bands[f"q{index + 1}"] = {
            "samples": len(band),
            "speed_p50_px_s": round(percentile([item[3] for item in band], 0.5), 1),
            "error_p75_px": round(percentile([item[1] for item in band], 0.75), 3),
            "lag_p75_px": round(percentile(
                [max(0.0, item[2]) for item in band], 0.75), 3),
            "lag_time_p75_ms": round(percentile(
                [max(0.0, item[2]) / item[3] * 1000.0 for item in band], 0.75), 2),
        }

    recoil_rows = [row for row in locked
                   if any(abs(float(value)) > 0.0
                          for value in (row.get("observed_recoil") or (0, 0)))]
    quiet_rows = [row for row in locked if row not in recoil_rows]

    def _row_error_metrics(items):
        radial, vertical, signed_y = [], [], []
        for row in items:
            ex, ey = row.get("observed_error") or (0.0, 0.0)
            radial.append(math.hypot(ex, ey))
            vertical.append(abs(ey))
            signed_y.append(float(ey))
        return {
            "samples": len(items),
            "error_p75_px": round(percentile(radial, 0.75), 3),
            "abs_y_error_p75_px": round(percentile(vertical, 0.75), 3),
            "signed_y_error_median_px": round(statistics.median(signed_y), 3)
            if signed_y else None,
        }
    active = [item for item in moving
              if str(item[0].get("prediction_reason", "")).startswith("arrival_")]
    unready = [item for item in moving
               if item[0].get("prediction_reason") in
               ("velocity_unready", "lock_warmup")]

    body_y = []
    body_widths = []
    body_heights = []
    body_inner60 = 0
    for row in locked:
        crosshair = row.get("crosshair")
        target = row.get("target")
        bodies = [det for det in (row.get("detections") or [])
                  if int(det.get("cls", -1)) == 0]
        if not crosshair or not target or not bodies:
            continue
        body = min(bodies, key=lambda det: abs(
            (float(det["bbox"][0]) + float(det["bbox"][2])) * 0.5
            - float(target[0])))
        x1, y1, x2, y2 = map(float, body["bbox"])
        if y2 > y1:
            body_y.append((float(crosshair[1]) - y1) / (y2 - y1))
            body_widths.append(x2 - x1)
            body_heights.append(y2 - y1)
            body_inner60 += bool(
                x1 <= float(crosshair[0]) <= x2
                and 0.2 <= body_y[-1] <= 0.8)

    reset_x = reset_y = reset_both = unready_transitions = 0
    previous = None
    for row in rows:
        if previous is not None and int(row.get("frame", 0)) <= int(previous.get("frame", 0)):
            previous = None
        if (previous is not None
                and previous.get("prediction_reason") == "arrival_active"
                and row.get("prediction_reason") == "velocity_unready"):
            unready_transitions += 1
            rvx, rvy = row.get("velocity_raw") or (0.0, 0.0)
            pvx, pvy = previous.get("velocity_filtered") or (0.0, 0.0)
            x_reverse = rvx * pvx < 0.0 and abs(rvx) > 80.0 and abs(pvx) > 80.0
            y_reverse = rvy * pvy < 0.0 and abs(rvy) > 80.0 and abs(pvy) > 80.0
            reset_x += x_reverse
            reset_y += y_reverse
            reset_both += x_reverse and y_reverse
        previous = row

    return {
        "observations": len(rows),
        "locked_observations": len(locked),
        "moving_observations": len(moving),
        "moving_error_p50_px": round(percentile(errors, 0.50), 3),
        "moving_error_p75_px": round(percentile(errors, 0.75), 3),
        "moving_error_p90_px": round(percentile(errors, 0.90), 3),
        "moving_positive_lag_p75_px": round(percentile(positive_lag, 0.75), 3),
        "moving_lag_time_p75_ms": round(percentile(lag_time_ms, 0.75), 2),
        "moving_speed_p50_px_s": round(percentile(speeds, 0.50), 1),
        "moving_speed_p75_px_s": round(percentile(speeds, 0.75), 1),
        "moving_speed_p90_px_s": round(percentile(speeds, 0.90), 1),
        "speed_quartiles": speed_bands,
        "moving_prediction_active_rate": round(len(active) / len(moving), 4)
        if moving else 0.0,
        "moving_prediction_unready_rate": round(len(unready) / len(moving), 4)
        if moving else 0.0,
        "active_to_unready_transitions": unready_transitions,
        "transitions_with_x_reverse": reset_x,
        "transitions_with_y_reverse": reset_y,
        "transitions_with_both_reverse": reset_both,
        "crosshair_body_y_median": round(statistics.median(body_y), 3)
        if body_y else None,
        "crosshair_body_y_p25": round(percentile(body_y, 0.25), 3)
        if body_y else None,
        "crosshair_body_y_p75": round(percentile(body_y, 0.75), 3)
        if body_y else None,
        "body_width_median_px": round(statistics.median(body_widths), 2)
        if body_widths else None,
        "body_height_median_px": round(statistics.median(body_heights), 2)
        if body_heights else None,
        "live_inner60_ratio": round(body_inner60 / len(body_y), 4)
        if body_y else 0.0,
        "recoil_active": _row_error_metrics(recoil_rows),
        "recoil_inactive": _row_error_metrics(quiet_rows),
        "sendinput_failures": sum("SendInput失败" in line for line in lines),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    metrics = analyze(args.log)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    if args.check:
        failures = []
        if metrics["moving_error_p75_px"] > 12.0:
            failures.append("moving_error_p75_px")
        if metrics["moving_prediction_active_rate"] < 0.60:
            failures.append("moving_prediction_active_rate")
        if failures:
            print("FAIL: " + ", ".join(failures))
            raise SystemExit(1)
        print("PASS")


if __name__ == "__main__":
    main()
