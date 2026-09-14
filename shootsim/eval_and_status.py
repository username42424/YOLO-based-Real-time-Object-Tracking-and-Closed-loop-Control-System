# -*- coding: utf-8 -*-
"""跑 N 回合评估根目录 main 配置, 解析新评估标准并写 eval_status.json。

用法(从 shootsim 目录):
    python eval_and_status.py [回合数]      # 缺省=全部日志回合

评估标准: 首入60%区域、进入失败率、红点驻留、偏上率和真实超时率。
阈值取 shootsim/config.json 的 eval 段；
任一阈值为 null 视为"未定标", done 恒为 false, 等用户给定目标后自动生效。
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main():
    n = sys.argv[1] if len(sys.argv) > 1 else "--holdout"
    main_cfg = os.path.join(ROOT, "config.json")
    py = sys.executable
    cmd = [py, "eval_main.py", main_cfg, n]
    print(">>> " + " ".join(cmd))
    child_env = dict(os.environ)
    child_env["PYTHONIOENCODING"] = "utf-8"
    r = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", env=child_env)
    out = (r.stdout or "") + (r.stderr or "")
    print(out)

    marker = "EVAL_JSON="
    payload = next((line[len(marker):] for line in out.splitlines()
                    if line.startswith(marker)), None)
    if payload is None:
        print("[status] 未找到 EVAL_JSON,未写 eval_status.json")
        sys.exit(1)
    metrics = json.loads(payload)

    with open(os.path.join(HERE, "config.json"), encoding="utf-8") as f:
        ev = (json.load(f).get("eval") or {})
    t_inner = ev.get("first_inner60_max", ev.get("first_hit_max"))
    t_entry = ev.get("inner60_entry_failure_max")
    t_dw = ev.get("dwell_min")
    t_above = ev.get("above_box_max")
    t_to = ev.get("timeout_max")
    thresholds_set = all(v is not None for v in (t_inner, t_entry, t_dw, t_above, t_to))
    done = bool(
        thresholds_set
        and metrics["first_inner60_median"] is not None
        and metrics["first_inner60_median"] <= t_inner
        and metrics["inner60_entry_failure_rate"] <= t_entry
        and metrics["target_dwell_rate"] >= t_dw
        and metrics["above_box_ratio"] <= t_above
        and metrics["timeout_rate"] <= t_to
    )
    status = {
        **metrics,
        "thresholds": {"first_inner60_max": t_inner,
                       "inner60_entry_failure_max": t_entry,
                       "dwell_min": t_dw, "above_box_max": t_above,
                       "timeout_max": t_to},
        "done": done,
    }
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", "eval_status.json"), "w", encoding="utf-8") as f:
        json.dump(status, f, ensure_ascii=False, indent=2)
    note = "" if thresholds_set else "  (eval 阈值未定标, done 恒 false)"
    print(f"[status] {json.dumps(status, ensure_ascii=False)}  -> done={done}{note}")
    sys.exit(0)


if __name__ == "__main__":
    main()
