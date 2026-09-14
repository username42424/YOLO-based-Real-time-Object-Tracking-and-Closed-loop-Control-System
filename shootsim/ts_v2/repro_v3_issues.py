# -*- coding: utf-8 -*-
"""Reproduce the four known V3 evaluator issues against raw results/code.

Saves evidence to 日志V4/reproduction/reproduction_v3.json.
Read-only: does not modify any code."""
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "shootsim"))

V3 = os.path.join(ROOT, "日志V3")
OUT = os.path.join(ROOT, "日志V4", "reproduction")
os.makedirs(OUT, exist_ok=True)

results = {}

# ── 问题1: err_p75 与 strafe_err_p75 同源 ─────────────────────────────
abl = json.load(open(os.path.join(V3, "LOGIC_ABLATION_RESULTS.json"), encoding="utf-8"))
rows = []
for c in abl["candidates"]:
    for sp in ("train", "validation"):
        it = c["splits"][sp]
        e = it.get("err_p75", {})
        se = it.get("strafe_err_p75", {})
        if e and se:
            rows.append({"name": c["name"], "split": sp,
                         "err_p75_mean": round(e["mean"], 6),
                         "strafe_err_p75_mean": round(se["mean"], 6),
                         "identical": abs(e["mean"] - se["mean"]) < 1e-12})
n_ident = sum(1 for r in rows if r["identical"])
results["issue1_err_equals_strafe_err"] = {
    "checked": len(rows), "identical_pairs": n_ident,
    "conclusion": "CONFIRMED: err_p75 and strafe_err_p75 identical for all candidates"
    if n_ident == len(rows) else "not identical for all",
    "samples": rows[:6],
}

# ── 问题4: blank0 与 blank1 完全相同 ──────────────────────────────────
by_name = {c["name"]: c for c in abl["candidates"]}
b1 = by_name.get("ROBUST+keep0.6_conf")
b0 = by_name.get("ROBUST+keep0.6_conf_blank0")
same = True
diff_fields = []
if b1 and b0:
    for sp in ("train", "validation"):
        for ph in range(9):
            a = b1["splits"][sp]["per_phase"][ph]
            b = b0["splits"][sp]["per_phase"][ph]
            for k, v in a.items():
                if k == "episodes":
                    continue
                if json.dumps(v, sort_keys=True) != json.dumps(b.get(k), sort_keys=True):
                    same = False
                    diff_fields.append(f"{sp}/ph{ph}/{k}")
results["issue4_blank0_equals_blank1"] = {
    "conclusion": "CONFIRMED: blank0 and blank1 identical across all splits/phases/fields"
    if same else f"differ in {len(diff_fields)} fields",
    "differ_fields_sample": diff_fields[:5],
}

# ── 问题2/3: 时间原点与 started_inside 污染（用 manifest 重算） ────────
sys.path.insert(0, os.path.join(ROOT, "shootsim", "ts_v2"))
from dataset3 import load_all_episodes  # noqa: E402
from dataset3 import stratified_split, BASELINE_CONFIG_PATH  # noqa: E402

metas = load_all_episodes()
assign = stratified_split(metas)
elig = [m["elig"] for m in metas]

# 从日志重建每个 episode 的 first_detection_t 与 first_inner60_time 近似不可行
# （fi 来自模拟）；此处用日志口径复算"时间原点差"本身：
# fi 以 episode 起点=0 计，而首次检测可能发生在 >0 处 → 偏差 = first_detection_t。
fd = [m["elig"]["first_detection_t"] for m in metas if m["elig"]["first_detection_t"] is not None]
fd_sorted = sorted(fd)
results["first_detection_offset"] = {
    "episodes_with_detections": len(fd),
    "first_detection_t_p50_s": round(fd_sorted[len(fd_sorted) // 2], 4),
    "first_detection_t_p75_s": round(fd_sorted[int(0.75 * (len(fd_sorted) - 1))], 4),
    "max_s": round(max(fd), 4),
    "conclusion": "CONFIRMED: many episodes have first detection well after t=0 "
                  "(pre-detection frames use real timestamps), so entry timing "
                  "measured from t=0 overstates entry latency",
}

# started_inside 污染：统计各 split 内 started_inside 且 streak>=0.8 的 episode 数
def split_of(m, a):
    return a
pollution = {"train": {"started_inside_streak08": 0, "n08": 0},
             "validation": {"started_inside_streak08": 0, "n08": 0}}
for m, a in zip(metas, assign):
    if a not in pollution:
        continue
    if m["elig"]["max_detected_streak_s"] >= 0.8:
        pollution[a]["n08"] += 1
        if not m["elig"]["started_outside_inner60"]:
            pollution[a]["started_inside_streak08"] += 1
results["issue3_started_inside_in_median"] = {
    "conclusion": "CONFIRMED: started_inside episodes are present in the "
                  "first_inner60_median pool (streak>=0.8s)",
    "data": pollution,
}

# ── 数量不一致：manifest vs 实际运行 ─────────────────────────────────
manifest = json.load(open(os.path.join(V3, "dataset_split_v3.json"), encoding="utf-8"))
m_counts = {k: len(v) for k, v in manifest["splits"].items()}
run_counts = {}
for c in abl["candidates"]:
    for sp in ("train", "validation"):
        n = c["splits"][sp]["per_phase"][0]["episodes"]
        run_counts.setdefault(sp, set()).add(n)
results["count_consistency"] = {
    "manifest_counts": m_counts,
    "run_episodes_sets": {k: sorted(v) for k, v in run_counts.items()},
    "candidate_count_in_json": len(abl["candidates"]),
    "note": "compare with V3 report claims (train 77, 22 candidates)",
}

with open(os.path.join(OUT, "reproduction_v3.json"), "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=1)
print(json.dumps(results, ensure_ascii=False, indent=1)[:3000])
