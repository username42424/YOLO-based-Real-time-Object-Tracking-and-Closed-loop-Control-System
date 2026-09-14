# -*- coding: utf-8 -*-
"""Data-driven live-log comparison.

No filename is a strategy declaration. Files are grouped by metadata found
in headers, runtime switch lines, and observation rows.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_cadence_sweep import group_metrics  # noqa: E402
from logparse import load_log, fill_world  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOGS = os.path.join(ROOT, "logs")


def strategy_label(meta):
    meta = meta or {}
    return "%s/%s/%s" % (
        meta.get("frame_schedule", "unknown"),
        meta.get("control_strategy", "unknown"),
        meta.get("effective_prediction_mode", "unknown"),
    )


def collect(names):
    groups, skipped = {}, []
    for name in names:
        # Accept a log basename, a workspace-relative path, or the absolute
        # paths supplied by the user.  The path form must never determine the
        # strategy label; only parsed metadata does that.
        path = (name if os.path.isabs(name) or os.path.exists(name)
                else os.path.join(LOGS, name))
        if not os.path.exists(path):
            skipped.append({"log": name, "reason": "not_found"})
            continue
        data = load_log(path)
        integrity = data.get("integrity", {})
        if integrity.get("polluted"):
            skipped.append({"log": name, "reason": "polluted_log",
                            "pollution_reasons": integrity.get(
                                "pollution_reasons", []),
                            "excluded_observations": integrity.get(
                                "excluded_observation_count", 0)})
            continue
        used = 0
        for episode in data["episodes"]:
            fill_world(episode)
            pts = episode["points"]
            if not (any(p["detected"] for p in pts)
                    and sum(1 for p in pts if p.get("raw_box") is not None) >= 3):
                continue
            label = strategy_label(episode.get("snap"))
            bucket = groups.setdefault(label, {"logs": [], "points": [],
                                                "parameters": [],
                                                "legacy_unverified_logs": []})
            bucket["points"].append(pts)
            bucket["parameters"].append({
                k: episode.get("snap", {}).get(k)
                for k in ("frame_schedule", "control_strategy",
                          "effective_prediction_mode", "frame_ms",
                          "reference_frame_ms", "output_period_ms",
                          "px_per_count", "tau_ms", "feedback_delay_ms",
                          "lock_box_smoothing_alpha")
            })
            if name not in bucket["logs"]:
                bucket["logs"].append(name)
            if integrity.get("legacy_unverified") and name not in bucket[
                    "legacy_unverified_logs"]:
                bucket["legacy_unverified_logs"].append(name)
            used += 1
        if not used:
            skipped.append({"log": name, "reason": "no_valid_episode"})
    results = {}
    for label, bucket in sorted(groups.items()):
        metrics = group_metrics(bucket["points"])
        metrics["logs"] = bucket["logs"]
        metrics["strategy"] = label
        metrics["parameters"] = bucket["parameters"]
        metrics["legacy_unverified_logs"] = bucket["legacy_unverified_logs"]
        for key in ("output_period_ms", "tau_ms", "feedback_delay_ms",
                    "px_per_count", "lock_box_smoothing_alpha"):
            values = [p.get(key) for p in bucket["parameters"]
                      if p.get(key) is not None]
            if values and all(v == values[0] for v in values):
                metrics[key] = values[0]
        results[label] = metrics
    return results, skipped


def _fmt(value):
    return "-" if value is None else "%.2f" % value


def main():
    names = sys.argv[1:]
    if not names:
        names = sorted(n for n in os.listdir(LOGS) if n.lower().endswith(".log"))
    results, skipped = collect(names)
    print("%-32s %4s %6s %9s %9s %9s %8s %8s %8s" % (
        "实际策略组", "eps", "obs", "raw75", "filtered75", "xlag75",
        "obsHz", "obs_ms", "lat95"))
    for label, m in results.items():
        print("%-32s %4d %6d %9s %9s %9s %8s %8s %8s" % (
            label, m["episodes"], m["observation_count"],
            _fmt(m.get("raw_err_p75")), _fmt(m.get("err_p75")),
            _fmt(m.get("strafe_xlag_p75")), _fmt(m.get("observation_rate_hz")),
            _fmt(m.get("dt_p50_ms")), _fmt(m.get("lat_p95_ms"))))
    if skipped:
        print("skipped:", json.dumps(skipped, ensure_ascii=False))
    payload = {"groups": results, "skipped": skipped,
               "grouping": "header/runtime/observation metadata; unknown retained",
               "pollution_policy": "concrete provenance failures excluded; legacy unverified retained and marked by parser"}
    for out in (os.path.join(ROOT, "日志V4", "cadence", "logic_compare.json"),
                os.path.join(ROOT, "shootsim", "results", "tracking_logic_v4",
                             "cadence", "logic_compare.json")):
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w", encoding="utf-8", newline="") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=1)
    print("saved 日志V4/cadence/logic_compare.json")


if __name__ == "__main__":
    main()
