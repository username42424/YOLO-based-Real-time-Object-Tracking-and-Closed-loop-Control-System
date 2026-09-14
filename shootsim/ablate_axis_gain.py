# -*- coding: utf-8 -*-
"""Training-only scan of independent continuous gain softness by axis."""
import json
import os

import replay
import sweep_staged as staged
from ablate_unit_control import FAST_BASE


def main():
    staged._init_worker()
    cfg = staged._G["cfg"]
    episodes = replay.load_episodes(cfg["replay"])
    holdout_names = {os.path.basename(path).lower()
                     for path in cfg["replay"].get("holdout_logs", [])}
    train_ids = [i for i, episode in enumerate(episodes)
                 if os.path.basename(episode.src).lower() not in holdout_names]
    rows = []
    for sx in (8.0, 16.0, 28.0):
        for sy in (16.0, 28.0, 45.0, 70.0):
            name = f"axis_x{sx:g}_y{sy:g}"
            extra = {
                "unit.continuous_gain": True,
                "unit.gain_softness_px_x": sx,
                "unit.gain_softness_px_y": sy,
            }
            params = {**FAST_BASE, **extra}
            result = staged.run_group((len(rows), "axis_gain", params, train_ids))
            rows.append({"name": name, "params": extra,
                         "metrics": result[3], "score": result[4]})
            print(name, result[3])
    rows.sort(key=lambda row: (
        not row["metrics"].get("accepted", False),
        row["metrics"]["flip_rate"],
        -row["metrics"]["target_dwell_rate"],
        row["metrics"]["first_inner60_median"],
    ))
    out = os.path.join(staged.HERE, "results", "axis_gain_ablation.json")
    with open(out, "w", encoding="utf-8") as handle:
        json.dump({"train_n": len(train_ids), "rows": rows}, handle,
                  ensure_ascii=False, indent=2)
    print("BEST", rows[0])
    print(out)


if __name__ == "__main__":
    main()
