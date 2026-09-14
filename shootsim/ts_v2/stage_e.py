# -*- coding: utf-8 -*-
"""Stage E: conservative refinement ladder around the A+B+C winner.

Adds a stricter dwell constraint (no material inner60 dwell loss on train AND
val) and probes the deadzone/softness boundary question.  Ranking on
train+val only; holdout is evaluated once at the end (see final2).
"""
import itertools
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from search import OUT_DIR, prepare_dataset, improvement_score, hard_gates  # noqa: E402
from harness import load_production_cfg, patch_cfg, evaluate_set  # noqa: E402

C40 = {
    "aim_control.lock_box_filter_mode": "fixed",
    "aim_control.lock_box_smoothing_alpha": 0.1509,
    "aim_control.prediction_box_ratio": 0.6966,
    "aim_control.prediction_cap_px": 50.87,
    "aim_control.prediction_lock_frames": 3,
    "aim_control.prediction_max_ms": 127.51,
    "aim_control.prediction_min_ms": 44.56,
    "aim_control.prediction_vel_tau": 0.0759,
    "aim_control.reversal_damp": 0.355,
    "aim_control.unit_max_counts": 254.9,
    "unit.deadzone": 4.98,
    "unit.far_gain": 1.089,
    "unit.gain_softness_px_x": 17.79,
    "unit.move_steps": 1,
    "unit.near_gain": 0.6715,
    "unit.near_radius_px": 16.62,
    "unit.stop_deadzone": 0.63,
    "unit.target_filter_alpha": 0.4722,
}


def main():
    metas, episodes, phases, idx = prepare_dataset()
    with open(os.path.join(OUT_DIR, "baseline_metrics.json"), encoding="utf-8") as f:
        base_train = json.load(f)["splits"]["train"]
    base_val = json.load(open(os.path.join(OUT_DIR, "baseline_metrics.json"),
                              encoding="utf-8"))["splits"]["val"]
    prod = load_production_cfg()
    from harness import build_sim_cfg
    sim_cfg = build_sim_cfg(["aim_20260906_003353.log", "aim_20260906_001842.log",
                             "aim_20260906_000134.log"])

    rows = []
    jsonl = os.path.join(OUT_DIR, "stage_e_results.jsonl")
    ladder = []
    for cap in (28.8, 36.0, 43.0, 50.87):
        for mmax in (105.0, 127.51):
            for soft in (2.0, 8.0, 17.79):
                for dz in (3.0, 5.0, 6.5):
                    for msteps in (1, 2):
                        v = dict(C40)
                        v["aim_control.prediction_cap_px"] = cap
                        v["aim_control.prediction_max_ms"] = mmax
                        v["unit.gain_softness_px_x"] = soft
                        v["unit.deadzone"] = dz
                        v["unit.move_steps"] = msteps
                        ladder.append(v)
    # targeted boundary probe: deadzone beyond 5
    for dz in (7.0, 8.0):
        v = dict(C40)
        v["unit.deadzone"] = dz
        v["unit.stop_deadzone"] = 0.9
        ladder.append(v)
    print("ladder size:", len(ladder))
    for i, vals in enumerate(ladder):
        cfg = patch_cfg(prod, vals)
        res_all = {}
        ok = True
        for sp in ("train", "val"):
            ep_ids = idx[sp]
            res = evaluate_set(sim_cfg, cfg, [episodes[j] for j in ep_ids],
                               [phases[j] for j in ep_ids])
            res.pop("_recs", None)
            res.pop("_per_ep", None)
            res_all[sp] = res
            b = base_train if sp == "train" else base_val
            if res["dwell_inner60"] < b["dwell_inner60"] - 0.005:
                ok = False
        score = improvement_score(res_all["train"], base_train)
        fails = hard_gates(res_all["train"], base_train)
        row = {"name": f"E_{i}", "values": vals, "splits": res_all,
               "score": score, "fails": fails, "gate_pass": ok and not fails,
               "dwell_ok": ok}
        rows.append(row)
        with open(jsonl, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        t = res_all["train"]
        print("E_%03d score=%7.3f pass=%s dwell_ok=%s xlag=%.2f err=%.2f "
              "dwell=%.3f quiet=%.1f dz=%.1f soft=%.1f cap=%.1f steps=%d" % (
                  i, score, row["gate_pass"], ok, t["strafe_xlag_p75"],
                  t["err_p75"], t["dwell_inner60"], t["quiet_cmd_p95"],
                  vals["unit.deadzone"], vals["unit.gain_softness_px_x"],
                  vals["aim_control.prediction_cap_px"], vals["unit.move_steps"]))
    rows.sort(key=lambda r: -r["score"])
    print("\nTOP 8 (train+val dwell-constrained):")
    n = 0
    for r in rows:
        if r["gate_pass"]:
            print(r["name"], round(r["score"], 3))
            n += 1
            if n >= 8:
                break


if __name__ == "__main__":
    main()
