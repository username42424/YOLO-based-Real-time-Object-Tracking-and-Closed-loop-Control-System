# -*- coding: utf-8 -*-
"""Training-only ablation for unit-mode gain and target filtering."""
import itertools
import json
import os

import replay
import sweep_staged as staged


FAST_BASE = {
    "unit.frame_ms": 55.1864,
    "unit.move_steps": 3,
    "unit.far_gain": 1.0476,
    "unit.near_gain": 0.4344,
    "unit.near_radius_px": 15.8581,
    "unit.deadzone": 3.0,
    "unit.stop_deadzone": 1.1139,
    "unit.target_filter_alpha": 0.4963,
    "aim_control.reversal_damp": 0.5,
    "aim_control.unit_max_counts": 280.3026,
    "aim_control.missing_decay": 0.25,
    "aim_control.missing_max_counts": 8.1572,
    "target_lock_distance": 133.5258,
    "target_lock_iou": 0.0553,
    "lock_ambiguity_margin": 0.3069,
    "aim_control.prediction_min_ms": 20.0,
    "aim_control.prediction_max_ms": 149.7102,
    "aim_control.prediction_box_ratio": 0.6146,
    "aim_control.prediction_cap_px": 28.7998,
    "aim_control.prediction_lock_frames": 3,
    "aim_control.prediction_vel_tau": 0.05,
    "unit.px_per_count": 0.3106,
    "unit.px_per_count_y": 0.23,
    "chest_ratio": 0.65,
}


def variants():
    yield "legacy", {}
    for softness in (8.0, 16.0, 28.0, 45.0):
        yield f"gain_s{softness:g}", {
            "unit.continuous_gain": True,
            "unit.gain_softness_px_x": softness,
            "unit.gain_softness_px_y": softness,
        }
    euro = list(itertools.product((1.0, 2.0, 4.0), (0.0, 0.03, 0.08)))
    for cutoff, beta in euro:
        yield f"euro_c{cutoff:g}_b{beta:g}", {
            "unit.target_filter_mode": "one_euro",
            "unit.target_filter_min_cutoff": cutoff,
            "unit.target_filter_beta": beta,
            "unit.target_filter_d_cutoff": 1.0,
        }
    for softness in (8.0, 16.0, 28.0, 45.0):
        for cutoff, beta in euro:
            yield f"both_s{softness:g}_c{cutoff:g}_b{beta:g}", {
                "unit.continuous_gain": True,
                "unit.gain_softness_px_x": softness,
                "unit.gain_softness_px_y": softness,
                "unit.target_filter_mode": "one_euro",
                "unit.target_filter_min_cutoff": cutoff,
                "unit.target_filter_beta": beta,
                "unit.target_filter_d_cutoff": 1.0,
            }


def main():
    staged._init_worker()
    cfg = staged._G["cfg"]
    episodes = replay.load_episodes(cfg["replay"])
    holdout_names = {os.path.basename(path).lower()
                     for path in cfg["replay"].get("holdout_logs", [])}
    train_ids = [i for i, episode in enumerate(episodes)
                 if os.path.basename(episode.src).lower() not in holdout_names]
    screen_ids = staged._G["search_eps"]
    rows = []
    for index, (name, extra) in enumerate(variants()):
        params = {**FAST_BASE, **extra}
        result = staged.run_group((index, "ablation", params, screen_ids))
        metrics = result[3]
        rows.append({"name": name, "params": extra, "screen": metrics,
                     "score": result[4]})
        print(name, metrics)

    accepted = [row for row in rows if row["screen"].get("accepted")]
    accepted.sort(key=lambda row: (
        row["screen"]["flip_rate"],
        -row["screen"]["target_dwell_rate"],
        row["screen"]["first_inner60_median"],
    ))
    finalists = accepted[:6]
    for index, row in enumerate(finalists):
        params = {**FAST_BASE, **row["params"]}
        result = staged.run_group((index, "full_train", params, train_ids))
        row["full_train"] = result[3]
        print("FULL", row["name"], result[3])

    out = os.path.join(staged.HERE, "results", "unit_control_ablation.json")
    with open(out, "w", encoding="utf-8") as handle:
        json.dump({"screen_n": len(screen_ids), "train_n": len(train_ids),
                   "rows": rows, "finalists": finalists},
                  handle, ensure_ascii=False, indent=2)
    print(out)


if __name__ == "__main__":
    main()
