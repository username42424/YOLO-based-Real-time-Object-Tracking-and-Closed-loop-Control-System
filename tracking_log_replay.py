# -*- coding: utf-8 -*-
"""Deterministic same-observation replay for current-vs-arrival aim modes.

New logs contain one ``[瞄准观测]`` JSON object per YOLO capture.  This module
feeds those exact detections, crosshair locations, timings, and attributed mouse
motion through MainEngine twice.  It is an A/B decision replay; final gameplay
validation is still required because an alternate command would change later
screenshots in a real closed loop.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import statistics
from pathlib import Path

from aim_engine import MainEngine


MARKER = "[瞄准观测] "


def parse_observation_lines(lines):
    records = []
    for line in lines:
        pos = line.find(MARKER)
        if pos < 0:
            continue
        try:
            value = json.loads(line[pos + len(MARKER):])
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if isinstance(value, dict) and "frame" in value:
            records.append(value)
    return records


def parse_observation_log(path):
    return parse_observation_lines(
        Path(path).read_text(encoding="utf-8", errors="replace").splitlines())


def replay_mode(records, config, mode):
    cfg = copy.deepcopy(config)
    ac = cfg.setdefault("aim_control", {})
    ac["prediction_mode"] = mode
    ac["unit_prediction_enabled"] = mode == "arrival"
    engine = MainEngine(cfg)
    outputs = []
    previous_frame = None
    for row in records:
        frame = int(row["frame"])
        if previous_frame is not None and frame <= previous_frame:
            engine.reset()
        previous_frame = frame
        aim = row.get("observed_aim", (0, 0))
        recoil = row.get("observed_recoil", (0, 0))
        engine.notify_net(
            float(aim[0]) + float(recoil[0]),
            float(aim[1]) + float(recoil[1]),
        )
        out = engine.compute(
            row.get("detections", []),
            tuple(row.get("crosshair", (0.0, 0.0))),
            max(1e-4, float(row.get("dt", 0.06))),
            processing_latency_s=max(
                0.0, float(row.get("processing_latency_ms", 0.0)) / 1000.0),
            recoil_feedforward_active=bool(row.get("recoil_feedforward", False)),
        )
        engine.end_frame()
        outputs.append(out)
    return outputs


def _percentile(values, fraction):
    if not values:
        return float("nan")
    values = sorted(values)
    pos = (len(values) - 1) * fraction
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return values[lo]
    return values[lo] * (hi - pos) + values[hi] * (pos - lo)


def compare_modes(records, config):
    current = replay_mode(records, config, "current")
    arrival = replay_mode(records, config, "arrival")
    command_delta = [
        math.hypot(a["dx"] - c["dx"], a["dy"] - c["dy"])
        for c, a in zip(current, arrival)
    ]
    arrival_lead = [math.hypot(*out["lead"]) for out in arrival]
    changed = sum(value > 0.5 for value in command_delta)
    return {
        "observations": len(records),
        "changed_observations": changed,
        "changed_percent": 100.0 * changed / len(records) if records else 0.0,
        "command_delta_median": statistics.median(command_delta) if command_delta else 0.0,
        "command_delta_p90": _percentile(command_delta, 0.90),
        "arrival_lead_median_px": statistics.median(arrival_lead) if arrival_lead else 0.0,
        "arrival_lead_p90_px": _percentile(arrival_lead, 0.90),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Replay structured aim observations in current and arrival modes")
    parser.add_argument("log")
    parser.add_argument("--config", default="config.json")
    args = parser.parse_args(argv)
    records = parse_observation_log(args.log)
    if not records:
        parser.error(
            "log has no [瞄准观测] records; record a new run with the updated engine")
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    metrics = compare_modes(records, config)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print("提示：这是同观测决策A/B；最终闭环效果仍以分阶段实机日志为准。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
