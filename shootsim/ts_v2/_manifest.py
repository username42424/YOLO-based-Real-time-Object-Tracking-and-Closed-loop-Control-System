# -*- coding: utf-8 -*-
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from dataset3 import build_split_manifest, OUT_DIR

metas, assign, idx = build_split_manifest()
print("total episodes:", len(metas))
for k in ("train", "validation", "holdout_ext"):
    ids = idx[k]
    logs = {}
    for i in ids:
        logs[metas[i]["log"]] = logs.get(metas[i]["log"], 0) + 1
    streaks = [metas[i]["elig"]["max_detected_streak_s"] for i in ids]
    n08 = sum(1 for s in streaks if s >= 0.8)
    print(k, len(ids), "episodes", logs, "streak>=0.8s:", n08)
print("manifest ->", os.path.join(OUT_DIR, "dataset_split_v3.json"))
