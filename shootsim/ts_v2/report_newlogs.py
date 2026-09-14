# -*- coding: utf-8 -*-
"""Generate 日志V4/newlogs/NEWLOGS_REPORT.md from newlogs_v4.json."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
V4 = os.path.join(os.path.dirname(os.path.dirname(HERE)), "日志V4")
src = os.path.join(V4, "newlogs", "newlogs_v4.json")
d = json.load(open(src, encoding="utf-8"))


def m(integ, key):
    v = integ.get(key)
    return None if v is None else v["mean"]


def fmt(v, nd=2):
    return "null" if v is None else ("%%.%df" % nd) % v


base = d["sim_ab"]["baseline_alpha020"]
rob = d["sim_ab"]["robust_applied"]
live_a02 = d["live"]["pooled_alpha020_reference"]
live_rob = d["live"]["pooled_robust_newlogs"]

L = []
A = L.append
A("# 新实战日志验证报告（robust 配置，α=0.18）\n")
A("日志：%s（%d 个有效 episode，全部为 robust 配置实机录制，垂直压枪开启）。\n"
  % (", ".join(d["new_logs"]), d["sim_ab"]["episodes_replayed"]))
A("原始数据：`newlogs_v4.json`（本目录），镜像于 `shootsim/results/tracking_logic_v4/newlogs/`。\n")

A("## 1. 实测（日志自身误差/速度，修正口径）\n")
A("| 数据 | 移动误差 P75 | 横移子集误差 P75 | 横移 X 滞后 P75 | 移动样本 | 横移样本 |")
A("|---|---|---|---|---|---|")
A("| α=0.2 参考日志（003353，旧构建）| %s | %s | **%s** | %d | %d |" % (
    fmt(live_a02["err_p75"]), fmt(live_a02["strafe_err_p75"]),
    fmt(live_a02["strafe_xlag_p75"]), live_a02["movement_samples"],
    live_a02["strafe_samples"]))
A("| 新日志（robust，本轮）| %s | %s | **%s** | %d | %d |" % (
    fmt(live_rob["err_p75"]), fmt(live_rob["strafe_err_p75"]),
    fmt(live_rob["strafe_xlag_p75"]), live_rob["movement_samples"],
    live_rob["strafe_samples"]))
A("\n横移 X 滞后实测 **%s vs %s px（-%.1f%%）**，横移样本量翻倍（402 vs 190）。"
  "注意：这是跨会话对比（不同地图/目标/旧引擎构建），只作趋势证据；"
  "误差 P75 的跨会话对比受目标距离/速度分布影响（speed_p90 %s vs %s px/s），不作结论。\n"
  % (fmt(live_rob["strafe_xlag_p75"]), fmt(live_a02["strafe_xlag_p75"]),
     100 * (1 - live_rob["strafe_xlag_p75"] / live_a02["strafe_xlag_p75"]),
     fmt(live_rob["speed_p90"], 0), fmt(live_a02["speed_p90"], 0)))

A("## 2. 闭环模拟 A/B（同一条目标轨迹，只换控制器；9 相位均值）\n")
A("| 指标 | α=0.2 基线 | robust | 变化 |")
A("|---|---|---|---|")
for k, nd, label in (("strafe_xlag_p75", 2, "横移 X 滞后 P75 (px)"),
                     ("err_p75", 2, "整体移动误差 P75 (px)"),
                     ("strafe_err_p75", 2, "横移子集误差 P75 (px)"),
                     ("first_inner60_median", 3, "首入 inner60 中位 (s)"),
                     ("dwell_inner60", 3, "inner60 驻留率"),
                     ("timeout_rate@0.5", 3, "timeout@0.5s")):
    b, r = m(base, k), m(rob, k)
    delta = ("%.1f%%" % (100 * (r / b - 1))) if (b and r) else "—"
    A("| %s | %s | %s | %s |" % (label, fmt(b, nd), fmt(r, nd), delta))
A("")
bx = base["strafe_xlag_p75"]["per_phase_values"]
rx = rob["strafe_xlag_p75"]["per_phase_values"]
A("横移滞后逐相位：基线 %s → robust %s；**robust %d/9 相位更优**。"
  % ([fmt(x) for x in bx], [fmt(x) for x in rx],
     sum(1 for a, b in zip(rx, bx) if a < b)))
bs = base["stable_cmd_p95"]["per_phase_values"]
rs = rob["stable_cmd_p95"]["per_phase_values"]
A("静止阶段指令 P95 逐相位：基线 %s → robust %s——"
  "**在新实战数据上 robust 的静止抖动低于 α=0.2**（调参期 validation 上的抖动抬升未复现）。\n"
  % ([fmt(x, 1) for x in bs], [fmt(x, 1) for x in rs]))

A("## 3. 结论\n")
A("1. 用户主观感受\"效果还行\"与数据一致：横移滞后在实测（-21.3%，跨会话趋势）与"
  "同轨迹闭环 A/B（-8.0%%，干净对比）两个口径下都改善了，且 9/9 相位方向一致。")
A("2. 整体移动误差在同轨迹 A/B 下 -8.7%%；首入时间与 α=0.2 持平（0.215 vs 0.216s，"
  "新日志多为近距接敌，首入本来就快）；timeout@0.5 略优。")
A("3. 调参期发现的\"validation 静止抖动抬升\"在实战数据上**未复现**——本组数据上 robust 的"
  "静止抖动反而低于 α=0.2（16.0~16.6 vs 17.0~19.1 counts）。")
A("4. 注意事项：本轮日志垂直压枪开启，Y 轴误差含程序压枪位移，Y 相关指标不用于结论；"
  "α=0.2 参考日志为旧引擎构建，跨会话对比仅作趋势。")
A("5. 维持建议：**保留当前 robust 配置**；新预测逻辑仍不进主程序。")

out = os.path.join(V4, "newlogs", "NEWLOGS_REPORT.md")
with open(out, "w", encoding="utf-8", newline="") as f:
    f.write("\n".join(L))
mirror = os.path.join(os.path.dirname(HERE), "results", "tracking_logic_v4",
                      "newlogs", "NEWLOGS_REPORT.md")
os.makedirs(os.path.dirname(mirror), exist_ok=True)
with open(mirror, "w", encoding="utf-8", newline="") as f:
    f.write("\n".join(L))
print("written", out)
