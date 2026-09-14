# -*- coding: utf-8 -*-
"""One-shot holdout validation for the training-selected unit controller."""
import json
import os

import replay
import sweep_staged as staged
from ablate_unit_control import FAST_BASE


SELECTED = {
    "unit.continuous_gain": True,
    "unit.gain_softness_px_x": 28.0,
    "unit.gain_softness_px_y": 70.0,
    "unit.target_filter_mode": "legacy",
}


def main():
    staged._init_worker()
    cfg = staged._G["cfg"]
    episodes = replay.load_episodes(cfg["replay"])
    holdout_names = {os.path.basename(path).lower()
                     for path in cfg["replay"].get("holdout_logs", [])}
    holdout_ids = [i for i, episode in enumerate(episodes)
                   if os.path.basename(episode.src).lower() in holdout_names]
    if not holdout_ids:
        raise RuntimeError("No configured holdout episodes")
    legacy_params = {**FAST_BASE,
                     "unit.continuous_gain": False,
                     "unit.target_filter_mode": "legacy"}
    selected_params = {**FAST_BASE, **SELECTED}
    legacy = staged.run_group((0, "holdout_legacy", legacy_params, holdout_ids))
    selected = staged.run_group((1, "holdout_selected", selected_params, holdout_ids))
    payload = {
        "holdout_n": len(holdout_ids),
        "legacy_fast": {"params": legacy_params, "metrics": legacy[3],
                        "score": legacy[4]},
        "selected": {"params": selected_params, "metrics": selected[3],
                     "score": selected[4]},
    }
    out = os.path.join(staged.HERE, "results", "unit_candidate_holdout.json")
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(out)


if __name__ == "__main__":
    main()
