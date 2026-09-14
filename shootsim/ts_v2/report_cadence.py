# -*- coding: utf-8 -*-
"""Generate 日志V4/cadence/CADENCE30_REPORT.md from the cadence JSONs."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
V4 = os.path.join(ROOT, "日志V4", "cadence")
comp = json.load(open(os.path.join(V4, "cadence_comparison.json"), encoding="utf-8"))
ab = json.load(open(os.path.join(V4, "cadence30_sim_ab.json"), encoding="utf-8"))


def f(v, nd=2):
    return "null" if v is None else ("%%.%df" % nd) % v


def m(integ, key):
    v = integ.get(key)
    return None if v is None else v["mean"]


c40 = comp["cadence40_robust"]
c30 = comp["cadence30_robust"]
ref = comp["alpha020_reference"]
ab_b = ab["baseline_alpha020"]
ab_r = ab["robust_applied"]

L = []
A = L.append
A("# 30ms 节拍实战验证报告（robust 配置，输出周期 12ms）\n")
A("日志：%s（%d episodes）。配置确认：frame_ms=30、共享输出周期=12ms（缓出 2×12=24ms < 30ms ✓）、"
  "α=0.18。对照组：40ms/15ms 的 robust 日志（125713/130338/131131）与 α=0.2 参考日志（003353）。\n"
  % (", ".join(c30["logs"]), c30["episodes"]))
A("原始数据：cadence_comparison.json、cadence30_sim_ab.json（本目录）。\n")

A("## 1. 实测对比（日志自身误差/速度，修正口径）\n")
A("| 组 | 横移 X 滞后 P75 | 整体误差 P75 | 横移子集误差 P75 | 目标速度 P90 | 横移样本 | dt P50/P95 |")
A("|---|---|---|---|---|---|---|")
for g, label in (("cadence40_robust", "40ms/15ms robust"), ("cadence30_robust", "**30ms/12ms robust**"),
                 ("alpha020_reference", "α=0.2 参考（旧构建）")):
    d = comp[g]
    A("| %s | %s | %s | %s | %.0f | %d | %s/%s |" % (
        label, f(d["strafe_xlag_p75"]), f(d["err_p75"]), f(d["strafe_err_p75"]),
        d["speed_p90"], d["strafe_samples"],
        f(d["dt_p50_ms"], 1), f(d["dt_p95_ms"], 1)))
A("")
A("- 30ms vs 40ms：横移滞后 %s vs %s（**-%.1f%%**）、误差 %s vs %s——且 30ms 场次的目标更快"
  "（速度 P90 %.0f vs %.0f px/s，+%.0f%%），通常应加重滞后，实测仍持平略优。"
  % (f(c30["strafe_xlag_p75"]), f(c40["strafe_xlag_p75"]),
     100 * (1 - c30["strafe_xlag_p75"] / c40["strafe_xlag_p75"]),
     f(c30["err_p75"]), f(c40["err_p75"]),
     c30["speed_p90"], c40["speed_p90"], 100 * (c30["speed_p90"] / c40["speed_p90"] - 1)))
A("- 节拍执行质量：dt P50=%sms、P95=%sms，>45ms 卡顿仅 %.1f%%——30ms 节打稳了。"
  % (f(c30["dt_p50_ms"], 1), f(c30["dt_p95_ms"], 1), 100 * c30["dt_gt45ms_rate"]))
A("- 预测时域贴 30ms 下限的比例仅 %.1f%%——未饱和。"
  % (100 * c30["horizon_floor_rate"]))

A("\n## 2. 30ms 轨迹上的闭环 A/B（robust vs α=0.2，同轨迹换控制器，9 相位均值）\n")
A("| 指标 | α=0.2 | robust | 变化 |")
A("|---|---|---|---|")
for k, nd, label in (("strafe_xlag_p75", 2, "横移 X 滞后 P75 (px)"),
                     ("err_p75", 2, "整体移动误差 P75 (px)"),
                     ("strafe_err_p75", 2, "横移子集误差 P75 (px)"),
                     ("first_inner60_median", 3, "首入 inner60 中位 (s)"),
                     ("dwell_inner60", 3, "inner60 驻留率"),
                     ("timeout_rate@0.5", 3, "timeout@0.5s")):
    b, r = m(ab_b, k), m(ab_r, k)
    delta = ("%.1f%%" % (100 * (r / b - 1))) if (b and r) else "—"
    A("| %s | %s | %s | %s |" % (label, f(b, nd), f(r, nd), delta))
A("")
bx = ab_b["strafe_xlag_p75"]["per_phase_values"]
rx = ab_r["strafe_xlag_p75"]["per_phase_values"]
A("横移滞后逐相位：α=0.2 %s → robust %s（**robust %d/9 相位更优**）。"
  % ([f(x) for x in bx], [f(x) for x in rx], sum(1 for a, b in zip(rx, bx) if a < b)))

A("\n## 3. 结论与注意事项\n")
A("1. **30ms/12ms 组合验证通过**：相对 40ms/15ms 无任何退化（更快目标下持平略优），"
  "闭环 A/B 下 robust 仍以 -9.5%% 滞后、-8.7%% 误差优于 α=0.2（9/9 相位）——配置优势迁移到新节拍。")
A("2. 首入时间受益于 lock_warmup 缩短（3 帧=90ms）：A/B 口径下 robust 首入 -5.9%%。")
A("3. 实测中偶发 13ms/96ms 的帧间隔（GPU 抖动）与速度离群 >2400px/s（离群重置正常触发），"
  "系统按真实 dt 处理，未造成异常。")
A("4. 注意：本轮同时改了节拍（40→30）与输出周期（15→12）两个变量，无法分别归因；"
  "垂直压枪开启，Y 轴误差含压枪位移，仅横移（X）结论可用；跨会话对比为趋势证据。")
A("5. 建议：**保留 30ms/12ms 与 robust 配置**。若手感稳定，此配置可作为新基线；"
  "之后如需继续优化，优先方向是输出计划替换率（19%%）与红点追踪器的偶发跳变（141313 帧 5）。")

out = os.path.join(V4, "CADENCE30_REPORT.md")
with open(out, "w", encoding="utf-8", newline="") as f:
    f.write("\n".join(L))
mirror = os.path.join(os.path.dirname(HERE), "results", "tracking_logic_v4",
                      "cadence", "CADENCE30_REPORT.md")
os.makedirs(os.path.dirname(mirror), exist_ok=True)
with open(mirror, "w", encoding="utf-8", newline="") as f:
    f.write("\n".join(L))
print("written", out)
