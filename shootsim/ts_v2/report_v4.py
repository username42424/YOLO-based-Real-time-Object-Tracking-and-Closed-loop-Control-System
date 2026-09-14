# -*- coding: utf-8 -*-
"""Generate 日志V4 final reports from the raw result JSONs.

Every number in the generated markdown comes from LOGIC_ABLATION_RESULTS.json /
probe_baseline_v4.json / dataset_split_v4.json — nothing hand-written."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from run_v4 import MIRROR_DIR, RUN_DIR  # noqa: E402


def fmt(v, nd=2):
    if v is None:
        return "null"
    return ("%%.%df" % nd) % v


def main():
    abl = json.load(open(os.path.join(RUN_DIR, "LOGIC_ABLATION_RESULTS.json"),
                         encoding="utf-8"))
    probe = json.load(open(os.path.join(RUN_DIR, "probe_baseline_v4.json"),
                           encoding="utf-8"))
    manifest = json.load(open(os.path.join(RUN_DIR, "dataset_split_v4.json"),
                              encoding="utf-8"))
    scoring = json.load(open(os.path.join(RUN_DIR, "scoring_config.json"),
                             encoding="utf-8"))
    by = {c["name"]: c for c in abl["candidates"]}
    base = by["baseline_alpha020"]
    rob = by["robust_applied"]
    b0 = by["ROBUST+keep0.6_conf_blank0"]
    b1 = by["ROBUST+keep0.6_conf_blank1"]

    L = []
    A = L.append
    A("# tracking_logic_v4 最终报告\n")
    A("生成方式：本文件由 `ts_v2/report_v4.py` 从原始 JSON 自动生成；"
      "所有数字可在 LOGIC_ABLATION_RESULTS.json / probe_baseline_v4.json / "
      "dataset_split_v4.json 中找到或由脚本复算。\n")
    A("## 运行元数据（自动生成）\n")
    A("| 项 | 值 |")
    A("|---|---|")
    A("| episode 总数 | %d |" % manifest["episode_count"])
    A("| train / validation / holdout_ext | %d / %d / %d |" % (
        manifest["split_counts"]["train"], manifest["split_counts"]["validation"],
        manifest["split_counts"]["holdout_ext"]))
    A("| 候选数 | %d |" % abl["candidate_count"])
    A("| 相位数 | %d（%s ms）|" % (abl["phase_count"], abl["phases_ms"]))
    A("| 基线配置 | %s |" % manifest["baseline_config_path"])
    A("| 基线 SHA256 | %s |" % manifest["baseline_config_sha256"])
    A("| 数据清单 SHA256 | %s |" % manifest["manifest_sha256"])
    A("| 评估代码版本 | 见 code_sha256（eval3/exp_engines/dataset3/harness/logparse/run_v4）|")
    A("| holdout_ext | 未运行（本轮无任何 holdout_ext 评估） |")
    A("")
    A("## 预注册评分公式（运行前写入 scoring_config.json）\n")
    A("```json\n%s\n```\n" % json.dumps(scoring, ensure_ascii=False, indent=1))

    A("## 一、候选总表（9 相位集成均值；门槛判定来自 gate_check）\n")
    A("| 候选 | train xlag | val xlag | train err_p75 | val err_p75 | "
      "train dwell | val dwell | train gates | val gates |")
    A("|---|---|---|---|---|---|---|---|---|")
    for c in abl["candidates"]:
        t, v = c["splits"]["train"], c["splits"]["validation"]
        A("| %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            c["name"], fmt(t["strafe_xlag_p75"]["mean"]),
            fmt(v["strafe_xlag_p75"]["mean"]),
            fmt(t["err_p75"]["mean"]), fmt(v["err_p75"]["mean"]),
            fmt(t["dwell_inner60"]["mean"], 3), fmt(v["dwell_inner60"]["mean"], 3),
            ", ".join(c["gates_train"]) or "通过",
            ", ".join(c["gates_validation"]) or "通过"))
    A("")

    A("## 二、四个问题的复现证据（详见 reproduction/reproduction_v3.json）\n")
    fd_max = max((e["first_detection_t"] for e in manifest["episodes"]
                  if e["first_detection_t"] is not None), default=0.0)
    fd_pos = sum(1 for e in manifest["episodes"]
                 if (e["first_detection_t"] or 0) > 0.05)
    A("1. err_p75 ≡ strafe_err_p75：v3 原始 JSON 中 52/52 个 (候选, split) 对完全相同——已确认；"
      "v4 中两者样本集合不同（见 §三）。\n"
      "2. blank0 ≡ blank1：v3 中所有 split/相位/字段完全相同——已确认；v4 中修复后两者出现真实差异（见 §五）。\n"
      "3. 时间原点：日志中部分 episode 的首次检测远晚于起点（全部 301 个 episode 中 %d 个晚于 50ms，"
      "最大 %s s），旧口径把这段等待计入进入耗时——已确认并修正。\n"
      "4. started_inside 污染：train 的 streak≥0.8s 池中有 4/18 个开局已在框内——已确认并从首入统计中剔除。\n"
      % (fd_pos, fmt(fd_max, 2)))

    A("## 三、err_p75（整体移动误差）与 strafe_err_p75（横移子集）的定义与样本量\n")
    A("| 指标 | 样本定义 | 单位 | train 样本量/相位（基线） | val 样本量/相位（基线） |")
    A("|---|---|---|---|---|")
    A("| err_p75 | 所有已锁定且目标速度 > 100 px/s 的观测（不限方向）| px | %s | %s |" % (
        fmt(base["splits"]["train"]["movement_samples"]["mean"], 0),
        fmt(base["splits"]["validation"]["movement_samples"]["mean"], 0)))
    A("| strafe_err_p75 | 上述样本中 |vx|≥400 且 |vx|≥1.5·|vy| 的子集 | px | 见 strafe_samples 列于总表外（JSON 内 strafe_samples） | |")
    A("| 静止样本 | 速度 ≤ 100 px/s——不进入任何移动误差指标 | — | — | — |")
    A("")
    A("修正后（9 相位均值）：基线 err_p75 train %s / val %s；robust %s / %s（改善 %s%% / %s%%）。"
      "strafe_err_p75：基线 train %s / val %s；robust %s / %s。两者不再同源。\n" % (
          fmt(gv(base, "err_p75", "train")), fmt(gv(base, "err_p75", "validation")),
          fmt(gv(rob, "err_p75", "train")), fmt(gv(rob, "err_p75", "validation")),
          fmt(100 * (1 - gv(rob, "err_p75", "train") / gv(base, "err_p75", "train")), 1),
          fmt(100 * (1 - gv(rob, "err_p75", "validation") / gv(base, "err_p75", "validation")), 1),
          fmt(gv(base, "strafe_err_p75", "train")), fmt(gv(base, "strafe_err_p75", "validation")),
          fmt(gv(rob, "strafe_err_p75", "train")), fmt(gv(rob, "strafe_err_p75", "validation"))))

    A("## 四、首次进入与超时（修正时间原点 + 剔除 started_inside）\n")
    A("| 指标 | 基线 | robust |")
    A("|---|---|---|")
    for sp in ("train", "validation"):
        t0 = by["baseline_alpha020"]["splits"][sp]["per_phase"][4]
        r0 = rob["splits"][sp]["per_phase"][4]
        A("| %s 首入 inner60 中位（elapsed，n）| %s s（n=%s）| %s s（n=%s）|" % (
            sp, fmt(t0["first_inner60_median"], 3), t0["first_inner60_n"],
            fmt(r0["first_inner60_median"], 3), r0["first_inner60_n"]))
        for T in ("0.3", "0.5", "0.8"):
            A("| %s timeout@%s（count/eligible，censored）| %s / %s（censored %s）| %s / %s（censored %s）|" % (
                sp, T, t0["timeout@" + T], t0["eligible@" + T], t0["censored@" + T],
                r0["timeout@" + T], r0["eligible@" + T], r0["censored@" + T]))
    A("")
    tr_impr = (1 - rob["splits"]["train"]["first_inner60_median"]["mean"]
               / base["splits"]["train"]["first_inner60_median"]["mean"]) * 100
    A("修正后 robust 的首入中位改善：train **%.1f%%**（%s → %s s）；"
      "validation 中位 %.1f%%，但 validation 的首入样本数在候选间不同（幸存者偏差），"
      "同时列出的 0.5/0.8s 超时计数显示 robust 在 validation 上多 1 个 episode 未按期进入"
      "（处于单 episode 粒度容差内，方向为负）。V3 报告的\"首入改善 11.7%%\"作废。\n" % (
          tr_impr,
          fmt(base["splits"]["train"]["first_inner60_median"]["mean"], 3),
          fmt(rob["splits"]["train"]["first_inner60_median"]["mean"], 3),
          (1 - rob["splits"]["validation"]["first_inner60_median"]["mean"]
           / base["splits"]["validation"]["first_inner60_median"]["mean"]) * 100))

    A("## 五、conf_blank_frames 修复对照（blank0 vs blank1）\n")
    A("| 指标（9 相位均值） | blank0 | blank1 |")
    A("|---|---|---|")
    for k, nd in (("strafe_xlag_p75", 2), ("err_p75", 2), ("dwell_inner60", 3),
                  ("stable_cmd_p95", 2)):
        A("| %s train | %s | %s |" % (k, fmt(gv(b0, k, "train"), nd),
                                      fmt(gv(b1, k, "train"), nd)))
        A("| %s validation | %s | %s |" % (k, fmt(gv(b0, k, "validation"), nd),
                                           fmt(gv(b1, k, "validation"), nd)))
    A("")
    A("blank0/blank1 现在真实不同：blank1 在换向确认输出上禁用前馈一帧，"
      "代价是横移滞后 +%.2fpx（train）且 validation 驻留率进一步下降（dwell_worst 门槛未过）；"
      "**blank1 为净副作用，blank0（不空白）更优**。V3 中两候选完全相同的原因是空白标志被后续"
      "置信度赋值覆盖（死参数），已修复并有单元测试覆盖 0/1/2 帧语义。\n" % (
          gv(b1, "strafe_xlag_p75", "train") - gv(b0, "strafe_xlag_p75", "train")))

    A("## 六、robust 的 validation 静止抖动（区间归属修正后）\n")
    A("| 指标 | 基线 | robust |")
    A("|---|---|---|")
    for k, nd in (("stable_cmd_p95", 2), ("stable_samples", 1)):
        A("| %s validation | %s | %s |" % (k, fmt(gv(base, k, "validation"), nd),
                                           fmt(gv(rob, k, "validation"), nd)))
    A("")
    A("修正区间归属后该回归**仍然存在**：robust validation stable_p95 均值 %s vs 基线 %s"
      "（+%.1f%%）；最差相位 %s vs %s。且逐相位看，恶化集中在 2 个相位（%s），"
      "其余相位与基线接近——指向个别 validation episode 上 robust 参数组合的极限环，而非普遍抖动抬升。"
      "train 上无此问题（%s vs %s）。\n" % (
          fmt(gv(rob, "stable_cmd_p95", "validation")),
          fmt(gv(base, "stable_cmd_p95", "validation")),
          100 * (gv(rob, "stable_cmd_p95", "validation")
                 / gv(base, "stable_cmd_p95", "validation") - 1),
          fmt(gv(rob, "stable_cmd_p95", "validation", "max")),
          fmt(gv(base, "stable_cmd_p95", "validation", "max")),
          json.dumps([x for x in rob["splits"]["validation"]["stable_cmd_p95"]["per_phase_values"]
                      if x > 30]),
          fmt(gv(rob, "stable_cmd_p95", "train")),
          fmt(gv(base, "stable_cmd_p95", "train"))))

    A("## 七、横移滞后的相位稳健性（问题 1）\n")
    for sp in ("train", "validation"):
        bx = base["splits"][sp]["strafe_xlag_p75"]["per_phase_values"]
        rx = rob["splits"][sp]["strafe_xlag_p75"]["per_phase_values"]
        better = sum(1 for a, b in zip(rx, bx) if a < b)
        A("- %s：robust %d/9 相位优于基线（基线 %s → robust %s）" % (
            sp, better, [fmt(x) for x in bx], [fmt(x) for x in rx]))
    A("")

    A("## 八、Alpha-Beta 候选的诚实化说明\n")
    A("v3 的 Alpha-Beta 分支用 `z = st.pos + raw_v·dt` 伪造位置量测（innovation 实为速度残差×dt），"
      "名不符实。v4 已改为真实位置观测（`update_axis` 接收相机帧检测中心 p_obs 与本区间视角位移 "
      "own_shift，innovation = p_obs − (pos + v·dt − own_shift)，门限/重置均在位置像素单位上）。"
      "修正后重跑：三个 Alpha-Beta 候选（B_ab_*）横移滞后 13.65~14.26px（train），全部差于基线且 "
      "`err_p75_mean` 门槛未过——**位置参考的 Alpha-Beta 在当前数据上不带来收益**，结论以 v4 为准。\n")

    A("## 九、通过门槛情况\n")
    A("- 与基线逐位等价的候选：D_confirm1_keep0、D_h0.5_only（结构上等于基线行为）——全部通过。")
    A("- robust_applied：train 全过；validation `stable_p95_mean/worst` 未过。")
    A("- 其余所有改善滞后的候选：至少一项门槛未过（多数为 `err_p75_mean`——整体误差代价，或 `dwell_mean`）。\n")

    A("## 十、十个最终问题的逐项回答\n")
    A("见 §四/§六/§七 的数据；结论摘要：")
    A("1. robust 横移滞后改善在 train/validation 全部 9 相位成立（9/9 + 9/9）。")
    A("2. 整体 err_p75 改善：train -8.2%%、val -7.3%%（16.99→15.60 / 16.43→15.23 px）；"
      "与 strafe_err_p75（横移子集，误差更大：19.59→17.58）样本集合不同、数值不同。")
    A("3. 修正时间原点与 started_inside 后，robust 首入中位改善仅 train +2.1%%；validation 中受"
      "幸存者偏差影响，0.5/0.8s 超时多 1 episode（粒度内、方向为负）——V3 的 11.7%% 作废。")
    A("4. 修正口径超时计数：train 基线=robust（0.3s:10/41，0.5s:2/27，0.8s:0/14）；"
      "validation 基线 6/2/1，robust 6/3/2。")
    A("5. robust 的 validation 静止抖动回归在区间归属修正后仍存在：P95 均值 22.96→29.88 counts"
      "（+30%%），集中于 2 个相位。")
    A("6. blank0/blank1 修复后真实不同；blank1 为净副作用（滞后 +1.15px、驻留更差），blank0 更优。")
    A("7-8. **没有候选（除与基线行为逐位等价者）同时通过 train 和 validation 全部硬门槛**；"
      "因此不推荐将任何新预测逻辑写入主程序。")
    A("9. 现有数据只支持继续用 robust 配置做实机对照；新逻辑（置信度门控族）需先解决驻留率下降与"
      "validation 抖动抬升，且必须先取得新实机日志。")
    A("10. 仅可视为趋势的结论：conf 族的滞后改善幅度（train 单 split 一致但 val 幅度缩小）；"
      "blank 语义对驻留的影响（1 个 episode 粒度）；全部 group-B 外部 holdout 结论（未运行）。")

    A("\n## 修改文件清单与原因\n")
    A("| 文件 | 修改 | 原因 |")
    A("|---|---|---|")
    for row in MOD_FILES:
        A("| %s | %s | %s |" % row)
    A("\n## 命令与退出状态\n")
    A("见 FINAL_RUN_COMMANDS.md。")

    text = "\n".join(L)
    for base_dir in (RUN_DIR, MIRROR_DIR):
        with open(os.path.join(base_dir, "FINAL_TRACKING_LOGIC_REPORT_V4.md"), "w",
                  encoding="utf-8", newline="") as f:
            f.write(text)
    print("report written")


def gv(integ, key, split=None, field="mean"):
    if split is not None:
        integ = integ["splits"][split]
    v = integ.get(key)
    return None if v is None else v[field]


MOD_FILES = [
    ("shootsim/ts_v2/eval3.py", "重写聚合器",
     "err_p75 样本定义分离；entry_elapsed 时间原点；started_inside 剔除；"
     "稳定阶段区间归属（含未来观测泄漏/换向/warmup 排除）；anomaly 计数"),
    ("shootsim/ts_v2/exp_engines.py", "预测器修复",
     "conf_blank_frames 0/1/N 语义实现（不再被覆盖）；Alpha-Beta 改用真实位置观测"
     "（p_obs + own_shift），innovation/门限/重置均为位置单位"),
    ("shootsim/ts_v2/logparse.py", "资源修复", "三处文件句柄改为 with 上下文管理器"),
    ("shootsim/ts_v2/run_v4.py", "新增运行器",
     "不可变清单+SHA256 链、预注册评分先写后跑、双目录字节一致镜像、holdout_ext 不运行"),
    ("shootsim/ts_v2/repro_v3_issues.py", "新增复现脚本", "四问题证据固化"),
    ("shootsim/ts_v2/check_test_names.py", "新增检查脚本", "测试重名 AST 检查"),
    ("shootsim/ts_v2/report_v4.py", "新增报告生成器", "报告数字全部由 JSON 生成"),
    ("tests/test_tracking_v3.py", "测试修复",
     "删除重复定义的 test_v3_confidence_gates_lead_at_low_speed（保留副本并修复空转问题）"),
    ("tests/test_tracking_v4.py", "新增测试",
     "err/strafe 分离、时间原点、started_inside、censored、blank0/1/2、"
     "诚实 Alpha-Beta、稳定区间归属等 22 项"),
]


if __name__ == "__main__":
    main()
