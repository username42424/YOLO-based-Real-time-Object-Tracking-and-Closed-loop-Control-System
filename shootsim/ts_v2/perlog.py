# -*- coding: utf-8 -*-
"""Per-log robustness check for the three finalists + baseline."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from search import OUT_DIR, prepare_dataset  # noqa: E402
from harness import (load_production_cfg, patch_cfg, evaluate_set,
                     build_sim_cfg)  # noqa: E402
from final2 import ROBUST, BALANCED, AGGRESSIVE  # noqa: E402


def main():
    metas, episodes, phases, idx = prepare_dataset()
    # the live config.json may already carry an applied candidate; the true
    # production baseline is the pre-apply backup
    backup_path = os.path.join(os.path.dirname(os.path.dirname(HERE)),
                               "config_backup_alpha020_20260906.json")
    if os.path.exists(backup_path):
        with open(backup_path, encoding="utf-8") as f:
            prod = json.load(f)
    else:
        prod = load_production_cfg()
    sim_cfg = build_sim_cfg(["aim_20260906_003353.log", "aim_20260906_001842.log",
                             "aim_20260906_000134.log"])
    by_log = {}
    for i, m in enumerate(metas):
        by_log.setdefault(m["log"], []).append(i)
    named = {"production_baseline": {}, "robust": ROBUST,
             "balanced": BALANCED, "aggressive": AGGRESSIVE}
    out = {}
    for log, ids in sorted(by_log.items()):
        eps = [episodes[i] for i in ids]
        ph = [phases[i] for i in ids]
        out[log] = {}
        for name, vals in named.items():
            cfg = patch_cfg(prod, vals)
            res = evaluate_set(sim_cfg, cfg, eps, ph)
            res.pop("_recs", None)
            res.pop("_per_ep", None)
            out[log][name] = {k: res[k] for k in (
                "episodes", "strafe_xlag_p75", "err_p75", "strafe_err_p75",
                "first_inner60_median", "dwell_inner60", "quiet_cmd_p95",
                "no_entry_rate", "strafe_samples")}
        b = out[log]["production_baseline"]
        print("==", log, "eps=%d strafe=%d" % (b["episodes"], b["strafe_samples"]))
        for name in named:
            r = out[log][name]
            print("  %-20s xlag=%6.2f err=%6.2f serr=%6.2f dwell=%.3f "
                  "quiet=%6.1f noent=%.3f" % (
                      name, r["strafe_xlag_p75"], r["err_p75"],
                      r["strafe_err_p75"], r["dwell_inner60"],
                      r["quiet_cmd_p95"], r["no_entry_rate"]))
    with open(os.path.join(OUT_DIR, "per_log_robustness.json"), "w",
              encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
