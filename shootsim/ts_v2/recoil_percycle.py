# -*- coding: utf-8 -*-
"""Per-capture and per-second recoil delivery from observation rows."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
MARKER = "[瞄准观测] "

GROUPS = {
    "cadence30_robust": ["aim_20260906_135713.log", "aim_20260906_140550.log",
                         "aim_20260906_141313.log"],
    "cadence40_robust": ["aim_20260906_125713.log", "aim_20260906_130338.log",
                         "aim_20260906_131131.log"],
}


def main():
    out = {}
    for g, logs in GROUPS.items():
        counts, dts, cycles = [], [], []
        for name in logs:
            prev_t = None
            for line in open(os.path.join(ROOT, "logs", name),
                             encoding="utf-8", errors="replace"):
                if MARKER not in line:
                    continue
                try:
                    row = json.loads(line.split(MARKER, 1)[1])
                except Exception:
                    continue
                t = float(row["capture_t"])
                rc = float((row.get("observed_recoil") or [0, 0])[1])
                if prev_t is not None:
                    dt = t - prev_t
                    if dt > 0:
                        dts.append(dt)
                        counts.append(rc)
                        # counts per 30ms-equivalent cycle
                        cycles.append(rc * 0.030 / dt)
                prev_t = t
        out[g] = {
            "captures": len(counts),
            "recoil_per_capture_p50": sorted(counts)[len(counts) // 2],
            "recoil_per_second_p50": sorted(
                [c / d for c, d in zip(counts, dts)])[len(counts) // 2],
            "recoil_per_30ms_cycle_p50": sorted(cycles)[len(cycles) // 2],
        }
    for g, v in out.items():
        print("%-18s captures=%5d  per_capture_p50=%.1f  per_second_p50=%.0f  "
              "per_30ms_cycle_p50=%.2f" % (
                  g, v["captures"], v["recoil_per_capture_p50"],
                  v["recoil_per_second_p50"], v["recoil_per_30ms_cycle_p50"]))
    with open(os.path.join(ROOT, "日志V4", "recoil", "recoil_percycle.json"),
              "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
