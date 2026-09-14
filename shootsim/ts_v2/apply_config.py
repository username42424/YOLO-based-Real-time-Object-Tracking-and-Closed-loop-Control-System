# -*- coding: utf-8 -*-
"""Apply the ROBUST candidate to the live config.json (with verification)."""
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(os.path.dirname(HERE), "results", "tracking_sweep_20260906_v2")

with open(os.path.join(OUT, "recommended_configs.json"), encoding="utf-8") as f:
    rec = json.load(f)
robust = rec["candidates"]["robust"]["config"]
live_path = os.path.join(ROOT, "config.json")
backup_path = os.path.join(ROOT, "config_backup_alpha020_20260906.json")

with open(live_path, encoding="utf-8") as f:
    before = json.load(f)
with open(backup_path, encoding="utf-8") as f:
    backup = json.load(f)

# sanity: the backup must equal the pre-apply live config
assert backup == before, "backup/config mismatch - aborting"

changed = {}
for section in ("aim_control", "unit"):
    for k, v in robust[section].items():
        if before.get(section, {}).get(k) != v:
            changed[f"{section}.{k}"] = (before.get(section, {}).get(k), v)
# no other section may differ
for section, node in robust.items():
    if section in ("aim_control", "unit"):
        continue
    assert node == before.get(section), f"unexpected change in section {section}"

with open(live_path, "w", encoding="utf-8") as f:
    json.dump(robust, f, ensure_ascii=False, indent=2)
    f.write("\n")

print("applied ROBUST config ->", live_path)
print("backup:", backup_path)
print("changed keys (%d):" % len(changed))
for k, (a, b) in sorted(changed.items()):
    print(f"  {k}: {a} -> {b}")
calib = all(robust["unit"]["px_per_count"] == 0.44
            and robust["unit"]["px_per_count_y"] == 0.51
            and robust["aim_control"]["view_scale"] == 0.44
            and robust["aim_control"]["view_scale_y"] == 0.51)
print("calibration intact:", calib)
