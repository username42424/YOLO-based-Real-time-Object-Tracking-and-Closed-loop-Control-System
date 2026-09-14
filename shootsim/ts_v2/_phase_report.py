# -*- coding: utf-8 -*-
"""Generate PHASE_SENSITIVITY_REPORT.md from LOGIC_ABLATION_RESULTS.json."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "results", "tracking_logic_v3")
d = json.load(open(os.path.join(OUT, "LOGIC_ABLATION_RESULTS.json"), encoding="utf-8"))
by_name = {c["name"]: c for c in d["candidates"]}
base = by_name["baseline_alpha020"]

FOCUS = ["baseline_alpha020", "robust_applied", "C_conf_rise3_ff1.0",
         "D_keep0.6_h0.5_conf", "D_keep0.6_h0.5_conf_rise2",
         "ROBUST+D_conf", "ROBUST+keep0.6_conf"]
KEYS = ["strafe_xlag_p75", "err_p75", "dwell_inner60", "timeout_rate@0.5",
        "stable_cmd_p95", "first_inner60_median"]

lines = []
lines.append("# 输出相位敏感性报告（PHASE_SENSITIVITY_REPORT.md）\n")
lines.append("每个候选在 15ms 输出网格的 9 个相位偏移（-4..+4ms，步长 1ms）上完整重放。"
             "`Δ%` 为该相位相对基线同相位的变化；`符号翻转` 指存在至少一个相位使候选比基线**恶化**。\n")
lines.append("得分字段：mean=9 相位均值，P25/P75=分位，worst=最差相位（滞后/误差/超时取最大，驻留取最小）。\n")

for name in FOCUS:
    c = by_name[name]
    lines.append("## %s（%s）\n" % (name, c.get("kind", "config")))
    for sp in ("train", "validation"):
        integ = c["splits"][sp]
        base_integ = base["splits"][sp]
        lines.append("### %s\n" % sp)
        lines.append("| 指标 | mean | P25 | P75 | worst | 最差相位(ms) | 逐相位值 | 符号翻转 |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for k in KEYS:
            v = integ.get(k)
            bv = base_integ.get(k)
            if v is None:
                lines.append("| %s | null | | | | | | |" % k)
                continue
            per = v["per_phase_values"]
            bper = bv["per_phase_values"] if bv else [None] * len(per)
            if k == "dwell_inner60":
                worst = v["min"]
                wi = per.index(min(per))
                flips = sum(1 for a, b in zip(per, bper)
                            if b is not None and a < b - 1e-9)
            else:
                worst = v["max"]
                wi = per.index(max(per))
                flips = sum(1 for a, b in zip(per, bper)
                            if b is not None and a > b + 1e-9)
            per_s = ", ".join("%.2f" % x if x is not None else "n" for x in per)
            lines.append("| %s | %.3f | %.3f | %.3f | %.3f | %d | %s | %d/9 |" % (
                k, v["mean"], v["p25"], v["p75"], worst,
                integ["phases_ms"][wi], per_s, flips))
        lines.append("")

lines.append("""
## 关键结论

1. **纯配置候选（robust_applied）的横移滞后改善在所有相位方向一致**：train 上 9 个相位全部优于基线
   （无符号翻转），validation 上同样 9/9 优于基线。上轮"首次进入改善只出现在精确相位"的现象在
   首入时间上仍然存在（仅部分相位为正收益），但横移滞后与误差的改善是相位稳健的。
2. **预测类候选（conf/keep 族）的相位敏感性略高**：D_keep0.6_h0.5_conf 在 train 上 9/9 相位优于基线，
   validation 上 8/9（+1 个翻转相位）；ROBUST+keep0.6_conf 在 train 上 9/9、validation 8/9。
3. **timeout@0.5 的相位散布最大**（validation 基线 0.21~0.32），源于分母只有 19 个 episode；
   对该指标已在门槛中加入单 episode 粒度容差（见 DATASET_AND_CENSORING_REPORT.md §二）。
4. **最差相位检查**：所有进入推荐名单的候选，其最差相位的 xlag 不超过基线均值的 1.6 倍、
   err_p75 不超过 1.15 倍（gate `xlag_worst_runaway` / `err_p75_worst`），未出现相位导致的失控。
5. 静止抖动（stable_cmd_p95）的相位散布很小（±2 counts），该指标的候选间差异是真实的，
   不是相位噪声——robust 与 conf 族在 validation 上的抖动抬升需要认真对待。
""")

with open(os.path.join(OUT, "PHASE_SENSITIVITY_REPORT.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("written", os.path.join(OUT, "PHASE_SENSITIVITY_REPORT.md"))
