# -*- coding: utf-8 -*-
"""Emit recommended_configs.json with FULL parameter sets (complete
aim_control + unit sections) for the three live-test candidates."""
import copy
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from harness import load_production_cfg, patch_cfg  # noqa: E402
from search import OUT_DIR  # noqa: E402
from final2 import ROBUST, BALANCED  # noqa: E402
from stage_e import C40 as AGGRESSIVE_C40  # noqa: E402

NAMES = {
    "robust": ("稳健候选", "改动最小、抖动与驻留几乎不变，适合第一个实机测试"),
    "balanced": ("平衡候选", "首入/驻留/高速跟随综合最好，holdout 上 dwell 仅 -0.16pp"),
    "aggressive": ("激进候选", "高速跟随最快（train xlag -30%），holdout 上 dwell -1.49pp、"
                              "静止抖动 P95 +12%，需要接受回撤风险"),
}


def main():
    prod = load_production_cfg()
    out = {
        "seed": 20260906,
        "note": ("以下为完整参数段（aim_control 与 unit），可直接对照填入 GUI/config.json。"
                 "物理标定 unit.px_per_count=0.44 / px_per_count_y=0.51 / "
                 "aim_control.view_scale=0.44 / view_scale_y=0.51 保持不变；"
                 "所有候选均假设 anti_recoil=false、default_recoil.enabled=false。"),
        "control": {"label": "对照组（当前生产 alpha=0.2）",
                    "config": prod},
        "candidates": {},
    }
    for key, (label, desc) in NAMES.items():
        vals = {"robust": ROBUST, "balanced": BALANCED,
                "aggressive": AGGRESSIVE_C40}[key]
        full = patch_cfg(prod, vals)
        out["candidates"][key] = {
            "label": label,
            "description": desc,
            "config": full,
            "aim_control": full["aim_control"],
            "unit": full["unit"],
        }
    with open(os.path.join(OUT_DIR, "recommended_configs.json"), "w",
              encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    for key in out["candidates"]:
        print("==", key)
        print(json.dumps(out["candidates"][key]["aim_control"],
                         ensure_ascii=False, sort_keys=True))
        print(json.dumps(out["candidates"][key]["unit"],
                         ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
