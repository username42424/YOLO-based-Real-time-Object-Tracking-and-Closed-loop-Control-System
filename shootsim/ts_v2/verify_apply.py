# -*- coding: utf-8 -*-
"""Verify the applied config.json: keys, calibration, JSON validity, and a
MainEngine smoke test (construct + one compute)."""
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.dirname(HERE))

live = json.load(open(os.path.join(ROOT, "config.json"), encoding="utf-8"))
backup = json.load(open(os.path.join(ROOT, "config_backup_alpha020_20260906.json"),
                        encoding="utf-8"))

expect = {
    "aim_control.lock_box_smoothing_alpha": 0.18,
    "aim_control.prediction_min_ms": 30.0,
    "aim_control.prediction_max_ms": 105.0,
    "aim_control.prediction_box_ratio": 0.65,
    "aim_control.prediction_vel_tau": 0.06,
    "aim_control.reversal_damp": 0.45,
    "unit.far_gain": 1.05,
    "unit.near_gain": 0.55,
    "unit.gain_softness_px_x": 8.0,
}
ok = True
for dotted, want in expect.items():
    sec, key = dotted.split(".")
    got = live.get(sec, {}).get(key)
    status = "OK " if got == want else "FAIL"
    ok = ok and got == want
    print(f"{status} {dotted} = {got}")
print("changed-keys count:", sum(
    1 for s in ("aim_control", "unit") for k in live[s]
    if live[s][k] != backup[s][k]))
print("calibration intact:",
      live["unit"]["px_per_count"] == 0.44,
      live["unit"]["px_per_count_y"] == 0.51,
      live["aim_control"]["view_scale"] == 0.44,
      live["aim_control"]["view_scale_y"] == 0.51)
print("aim_mode/unit frame/steps:", live["aim_mode"], live["unit"]["frame_ms"],
      live["unit"]["move_steps"], live["unit"]["deadzone"],
      live["unit"]["stop_deadzone"], live["unit"]["target_filter_alpha"])

sys.path.insert(0, os.path.join(os.path.dirname(HERE)))
from trackers import _load_aim_main  # noqa: E402

mod = _load_aim_main()
engine = mod.MainEngine(live)
print("engine constructed:",
      "alpha=%.2f pred=%.0f/%.0fms tau=%.3f far=%.3f near=%.3f soft=%.1f" % (
          engine.lock_box_smoothing_alpha, engine.prediction_min_s * 1000,
          engine.prediction_max_s * 1000, engine.prediction_vel_tau,
          engine._unit_far_gain, engine._unit_near_gain,
          engine._unit_gain_softness[0]))
out = engine.compute(
    [{"cls": 0, "conf": 0.9, "bbox": [180.0, 120.0, 280.0, 240.0]}],
    (160.0, 160.0), 0.04, processing_latency_s=0.013)
print("smoke compute:", "target=", out["target"], "dx=%.1f dy=%.1f" % (out["dx"], out["dy"]),
      "can_send=", out["can_send"])
print("ALL OK" if ok else "MISMATCH FOUND")
