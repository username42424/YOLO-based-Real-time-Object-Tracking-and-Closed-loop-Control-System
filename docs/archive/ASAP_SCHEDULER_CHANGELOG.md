# ASAP_SCHEDULER_CHANGELOG — 可切换识别调度（fixed / asap）改造记录

日期：2026-09-07
范围：调度、时间归一化、瞄准计划输出语义、GUI/日志/模拟器/测试。
未改动：目标选择策略、跟踪增益与预测参数数值、chest_ratio/head_ratio、模型文件、
检测阈值、日志/模型/备份/模拟结果文件、DXcam 主后端、单写入 SendInput 架构。

---

## 1. 修改的文件

| 文件 | 修改内容 |
| --- | --- |
| `main.py` | ① 新增模块级助手 `_frame_schedule_of` / `_reference_frame_seconds` / `_unit_frame_wait_seconds`；② `run_engine` 读取 `unit.frame_schedule`、`unit.reference_frame_ms`，派生 `asap_mode`、`reference_frame_s`、统一 TTL `_aim_fresh_ttl_s`；③ `aim_frame` 固定节拍等待改为经 `_unit_frame_wait_seconds` 判定（asap 恒 0，不推进 `_seg_next_t`），并记录实际人工等待；④ unit 发布路径按调度分流：fixed=`publish_aim(chunks)`，asap=`publish_aim_rate(dx/ref_s, dy/ref_s)`；⑤ 输出线程按真实 tick dt 积分速率、TTL 归零并计数 stale clear，新增周期性 `[输出器]` 诊断行；⑥ `[YOLO性能]` 周期日志扩展调度字段；⑦ 默认垂直后坐力 tick 强度换算在 asap 下使用 reference_frame_ms；⑧ 红点 tracker 实例化传入调度参数；⑨ 启动日志输出调度语义；⑩ `DEFAULT["unit"]` 增加 `frame_schedule`/`reference_frame_ms` |
| `motion_arbiter.py` | 新增速率语义 API：`publish_aim_rate(rate_x, rate_y)`、`next_aim_rate_motion(dt_s)`、`aim_rate()`、`has_pending_aim()`；`publish_aim` 与 `publish_aim_rate` 互斥（各自清除另一语义的待发量）；`clear_aim`/`reset` 同时清零两套语义。chunk/量化/账本原逻辑未动 |
| `aim_engine.py` | 时间归一化（**仅 asap 生效，fixed 逐位保持原实现**）：`frame_schedule`/`reference_dt` 配置解析；`_frame_scale`/`_norm_alpha`/`_norm_retention`/`_norm_disp`/`_frames_gate` 五个换算助手；对 target EMA、`_own_vel`、lock_box 平滑 alpha、unit target filter（legacy）alpha 与 6/40px 阈值、`ema_max_step`、`vel_deadzone`、`vel_lw` 地板、`lock_prediction_max_step` 做按 dt 的换算；`prediction_lock_frames`、`lock_grace`、`switch_grace`、`_no_target≥2帧`、`_VEL_CONFIRM=3`、速度轴确认≥2帧、`missing_decay` 指数改为帧计数/累计时间双轨门控 |
| `crosshair_tracker.py` | 新增 `frame_schedule`/`reference_frame_ms` 参数；asap 下候选确认（2帧→44ms 连续稳定）、丢失保持（2帧→44ms）、受限回归（4帧→88ms，末步精确吸附）、EMA/fallback alpha 均按真实时间运行；fixed 路径逐位保持原实现；状态字符串在 asap 下显示 ms |
| `gui.py` | "固定节拍"区新增"识别调度"下拉框（跟随推理速度 / 固定周期）；加载/保存 `unit.frame_schedule`；保存时始终写 `frame_ms` 并同步 `reference_frame_ms=frame_ms`（asap 下 frame_ms 保存但不用于等待）；`u_steps` 下新增说明"缓出分配仅固定周期模式生效"；`DEFAULT["unit"]` 同步新字段 |
| `config.json` | 用户当前配置：`unit.frame_schedule="asap"`，`frame_ms=22.0`（保留），新增 `reference_frame_ms=22.0` |
| `shootsim/trackers.py` | `MainEngineTracker`（main_real）读取主配置调度：fixed 保持 frame_ms 闸门与 chunk 输出（旧行为不变）；asap 跳过 frame_ms 闸门、按 `publish_aim_rate` 发布、输出 tick 按真实 dt 积分；新增 `asap_e2e_ms`（数值或序列）作为可配置端到端观测间隔；replay 模式（日志真实时间戳）不变 |
| `tests/test_frame_schedule.py` | 新增 24 项测试（见 §7） |
| `tests/fixtures/fixed_baseline.json`、`tests/fixtures/make_fixed_baseline.py` | 修改前 fixed 模式的确定性逐帧输出基线（引擎 160 帧 + 红点 80 帧 + arbiter 序列，全精度浮点），以及再生成脚本 |

## 2. fixed / asap 的准确语义

- **fixed（固定周期）**：与修改前完全一致。unit 模式下每 `frame_ms` 开始一次
  "截图→预处理→推理→选点→发布"；左键沿可经 `fire_wake` 提前唤醒当次等待；
  发布 = `publish_aim(dx, dy, steps=move_steps, profile)` 离散 chunks，10ms 输出
  线程每 tick 消费一块。所有"每帧参数"按帧原样生效。
- **asap（跟随推理速度）**：`_unit_frame_wait_seconds("asap", …) == 0`，即上一帧
  完成后**立即**开始下一次截图；`frame_ms` 不参与等待；无第二推理线程、无并发
  推理（`_INFER_LOCK` 与单 consumer 线程结构未动）；热键仍以 `hk.poll(timeout=0)`
  非阻塞处理；松键/停止/退出后不会多执行一帧（consumer 在进入下一帧前处理
  事件并检查 `active`/`stop_event`，capture 后亦有停止检查）。
  observation dt 仍由实际 capture 时间戳差计算，真实 processing latency 仍传给
  到达预测器；capture/replay 模块、DXcam、模型、阈值、目标选择全部未动。
- `reference_frame_ms`：控制参数的**标称整定周期**（当前 22ms），只用于把
  "每帧参数"换算到真实时间与速率换算，**绝不是** asap 的等待时间。

## 3. 时间归一化公式（仅 asap）

记 `k = actual_dt / reference_dt`（`reference_dt = reference_frame_ms/1000`）：

- 修正系数（correction 形态，`x += (new - x) * alpha`）：
  `alpha_dt = 1 - (1 - alpha_ref) ** k`
- 保留权重（retention 形态，`x = x*a + new*(1-a)`，如 target EMA / own_vel）：
  `a_dt = a_ref ** k`（修正权重 = `1 - a_dt`；两权重同归一，和恒为 1）
- 每帧位移阈值（`ema_max_step`、`vel_deadzone(+frac)`、`vel_lw_floor(+frac)`、
  unit_tp 6/40px、`lock_prediction_max_step`）：`threshold_dt = threshold_ref * k`
- 帧数阈值 → 时间（`reference_frame_ms=22ms` 兼容时间：2帧≈44ms、3帧≈66ms、
  4帧≈88ms）：`t = frames * reference_dt`，应用于
  `prediction_lock_frames(3→66ms)`、`lock_grace/switch_grace(2→44ms)`、
  无目标 snap（2→44ms）、速度确认（`_VEL_CONFIRM` 3→66ms）、
  unit 速度轴确认（2→44ms）、红点 confirm（2→44ms）/hold（2→44ms）/return（4→88ms）
- 丢失衰减指数：fixed 用丢帧数；asap 用 `missing_time / reference_dt`
- 已按真实时间工作的项**未重复换算**：`prediction_vel_tau`（exp 秒级 tau）、
  `vel_lw_t`（exp 秒级 tau）、one_euro 滤波（dt 公式）、到达预测时域（秒+实测
  processing latency）、`unit_velocity`（px/s，换向阈值 80/2400 px/s）。
- 兼容时间表（reference_frame_ms=22）：2帧≈44ms；3帧≈66ms；4帧≈88ms。

## 4. 动态控制速率输出（仅 asap）

- 引擎输出 `out.dx/dy` 仍表示 `reference_frame_ms` 下的一帧控制量（counts）。
- 发布时换算：`publish_aim_rate(dx / reference_frame_s, dy / reference_frame_s)`
  （counts/second）。
- 10ms 输出线程每个 tick 用**本次真实 tick dt** 积分：
  `aim_dx, aim_dy = arbiter.next_aim_rate_motion(tick_dt)`（= 速率 × tick_dt）。
- 新观测只**整体替换**最新速率；没有"尚未发完的尾段"，旧计划不会补发
  （旧计划可能已过时）；换向后下一 tick 立即输出新方向。
- 目标无效 / 松键 / 锁定失效：发布路径与输出线程调用 `clear_aim()` 立即清零速率。
- TTL：控制状态超过 `_aim_fresh_ttl_s = max(0.20, reference_frame_s × 2.5)` 未更新
  自动归零，并计入 `stale_rate_clears`（仅"仍在瞄准且目标曾有效"的超时才计数，
  松键不计）。
- 整数余量：沿用 `quantize_components` 累计期望差分（累计发送恒等于累计期望的
  四舍五入）；瞄准与后坐力仍各自独立小数账本；合成后仍走 `mix_motion_components`
  的 `unit_max_counts` 限幅和唯一 `_send_relative` 写入点。
- fixed 模式继续使用 `move_steps`/chunks 完整基线兼容；asap 下 `move_steps`
  保留在配置中但不参与速率输出（GUI 有明确说明）。

## 5. 后坐力兼容性

- 后坐力仍完全由固定周期输出线程驱动（`recoil.output_period_ms=10`），与 YOLO
  帧率无关；手动曲线按 `perf_counter_ns` 真实时间轴回放，未动。
- 默认垂直后坐力：tick 强度 = `strength × period / 参考帧周期`。fixed 用
  `unit.frame_ms`、asap 用 `reference_frame_ms`（当前两者同为 22ms），因此每秒
  累计强度 `strength / 0.022` 两种调度完全一致（测试 §7-7 验证 ≤1 count）。
- YOLO 更新频率变化不影响后坐力：两控制器只依赖输出线程的时钟。
- 瞄准与后坐力仍统一限幅、一次 SendInput 发出；前馈抑制
  （`suppress_prediction_during_replay`）与入框开火门控行为未动；
  标定单位与 `recoil_profiles.json` 未动。

## 6. 模拟器同步（shootsim）

- `main_real` 策略读取主配置 `unit.frame_schedule`：
  - fixed：`frame_ms` 闸门 + chunk 输出，历史行为与 V2/V3/V4 结果文件不变；
  - asap：无 frame_ms 闸门；观测节奏 = env 步长（replay=日志真实时间戳），
    或 `tracker.asap_e2e_ms` 指定的端到端数值/序列（如 `[13, 15.5, 17]`，按观测
    循环）——未把 asap 简化成单一固定 13/15ms；
  - 输出：`publish_aim_rate` + 输出网格真实 tick dt 积分；发布延迟仿真
    （`_chunks_ready_t`）两种调度一致保留。
- 引擎时间归一化由同一份主配置自动生效（模拟与实机共用 MainEngine）。
- `run_sim.py`、日志格式、历史结果文件均未改动（冒烟运行通过）。

## 7. 测试命令与结果

```
python -m unittest discover -s tests -q
```

结果：**Ran 192 tests — OK**（原 167 项全部保留、断言未放宽；新增 25 项，
含实机回归后追加的接口完整性守卫）。

新增测试 → 任务要求映射（`tests/test_frame_schedule.py`）：

1. fixed 与修改前逐帧一致：修改前生成的全精度基线夹具回放——引擎 160 帧
   `dx/dy/can_send/in_deadzone/prediction_reason/observed_error` 全部精确相等；
   红点 tracker 80 帧 `x/y/status` 精确相等；arbiter chunk/量化序列一致。
2. asap 无 frame_ms 人工等待：`_unit_frame_wait_seconds("asap",…)==0`（任意
   节拍/frame_ms），fixed 语义不变；`aim_frame` 经该助手门控 `fire_wake.wait`，
   asap 不推进 `_seg_next_t`（源级断言）。
3. 14/18/22ms 同一 1 秒轨迹：时间归一化后 EMA 终值 153.26/153.18/153.32
   （极差 0.14px），`observed_error` 极差 0.21px；红点 tracker 在 14ms 周期下
   候选确认≈44ms、hold≈44ms、return≈88ms 末步精确吸附。
4. 恒定误差 1 秒墙钟累计：fixed 1874.2 counts/s vs asap（三周期均值）1874.2
   counts/s（每帧 dx 三周期完全一致 41.23），误差 ≈0%（<5%）。
5. 换向：新速率发布即刻整体替换，`next_aim_rate_motion` 下一 tick 即输出新方向；
   速率与 chunk 语义互斥。
6. 丢失/TTL：`clear_aim` 后积分恒 0；TTL 超时归零并计数。
7. 默认垂直后坐力 1 秒累计：fixed/asap tick 强度逐位相等，秒累计相等（<1 count），
   且 = `strength/参考帧周期`。
8. 手动后坐力时间轴：同一输出 tick 序列输出逐 tick 一致；60×10ms 与 30×20ms
   两种 tick 划分同一墙钟总回放量一致（YOLO FPS 不出现在控制器任何输入中）。
9. 单一 SendInput 写入者：源级守卫——`_send_relative(` 直接调用全 main.py 仅
   输出线程一处；输出线程内含 mix→quantize→send 合成链；unit 分支只经
   arbiter 发布。
10. 小数账本：3.7/-1.3 counts/s 非整速率 100 秒积分后 `sent == round(rate×T)`
    精确成立（无系统丢失/漂移）；任意 tick 序列积分和 = 速率×时长。
11. 缺字段兼容：`unit.frame_schedule`/`reference_frame_ms` 缺失 → fixed 启动，
    `load_config` 注入默认，逐帧输出与基线一致。
12. GUI：下拉框与"仅固定周期生效"说明存在；加载读 `unit.frame_schedule`；
    保存写 `frame_schedule` 且 `reference_frame_ms=frame_ms`；用户 config 为 asap。
13. 模拟器：fixed/asap/可变 e2e 序列三种配置冒烟通过，fixed 行为不变。

## 8. 实机前验收对照（本轮可得数据）

| 指标 | fixed 22ms | asap | 来源 |
| --- | --- | --- | --- |
| 截图 P50/P95 | 运行时（日志 `[YOLO性能]`） | 同左（代码路径未动） | 需实机日志 |
| 推理 P50/P95 | 运行时 | 同左 | 需实机日志（离线参考值见下） |
| 端到端 P50/P95 | 运行时 | 预期 ≈截图间隔（无等待） | 需实机日志 |
| observation interval P50/P95 | ≈frame_ms(22)+超出量 | ≈端到端耗时 | 需实机日志 |
| 实际 YOLO FPS | ≈1000/22 ≈ 45（上限） | 理论 ≈60–75（按 13–17ms e2e），以实测为准 | 需实机日志 |
| artificial wait | ≈frame_ms−e2e（≥0） | ≈0（设计+测试断言） | 需实机确认 |
| 输出 tick P50/P95 | ≈10ms（1ms 定时器） | 同左（线程未动，仅积分语义变化） | 需实机日志 |
| stale clear 次数 | — | 运行时计数 `[输出器]` | 需实机日志 |
| 换向后旧方向输出次数 | — | 速率语义下为 0（chunk 尾段不存在）；测试 §7-5 | 已由测试证明 |
| 1 秒累计后坐力差异 | 0 | 0（测试 §7-7，<1 count） | 已由测试证明 |
| 测试 | 191/191 OK | 191/191 OK | 本机实测 |

离线参考（本机、`onnxruntime` **仅 CPU**、yolodeltav1 640×640、n=120）：
推理 P50=20.8ms / P95=21.5ms。该环境无 DML，不代表实机 DML+fp16 路径，
**不能**据此推断 asap 实际 FPS；实机 FPS 必须以 `[YOLO性能]` 的 `识别FPS` 为准。

## 9.5 实机首跑回归与修复记录

- **问题**：实机日志 `logs/aim_20260907_120855.log` 显示每帧
  `AttributeError: 'CrosshairTracker' object has no attribute 'preferred_position'`，
  瞄准管线完全失效（预热与所有识别帧均抛异常）。根因：本轮重写
  `crosshair_tracker.py` 时遗漏了原类的 `preferred_position` 属性
  （`main.py` 红点检测的时间先验依赖它），而既有测试与基线夹具都只调用
  `update()`，未能拦截。
- **修复**：补回该只读属性（`ever` 时返回 `(x, y)`，否则 `None`，语义与
  修改前一致）。
- **防再发**：新增测试 `test_tracker_public_surface_unchanged`——按 `main.py`
  实际使用的 `crosshair_tracker.*` 成员清单逐项断言存在。
- **排查结论（非缺陷）**：日志中"RMB 松开后仍打印 `净移=(+0,+4)`"经与修改前
  日志（`logs/22ms _1.log`）对比并以 `输出t` 时间戳核对，全部后坐力输出 tick
  均发生在 RMB up 之前，只是鼠标日志线程的打印滞后；修改前后行为一致。
- **顺带修正**：`[输出器]` 周期日志改为线程启动 1 秒后才开始输出，消除首行
  `tick_dt≈0.1ms` 的无意义样本。
- 修复后：`python -m unittest discover -s tests -q` → **Ran 192 tests — OK**。

## 9. 尚需实机验证的风险

1. **实际 FPS 与端到端**：asap 的 FPS 由实机 e2e 决定；若 e2e 劣化（DML 抖动、
   截图重试回退 PIL），帧率会自动下降——属预期行为，但需日志确认无病态尖峰。
2. **快速横移手感**：本轮未声称解决快速横移问题；asap 下观测更密、预测时域
   仍由 processing latency 推导，实机手感（增益相位、红点状态机时间制）需人工
   验证，必要时只调 `reference_frame_ms` 等标称周期参数而非回退架构。
3. **stale clear 计数**：若实机日志中持续增长，说明 e2e 偶发超过
   `max(200ms, reference×2.5)`，需要检查截图/推理尖峰而非放宽 TTL。
4. **时间制红点状态机**：asap 下 confirm/hold/return 按 44/44/88ms 运行，极端
   帧率波动下的观感（状态字符串频率）会与 fixed 不同，需实机确认无回归。
5. **GUI 覆盖**：Tk 界面的实际操作（切固定周期→保存→重启引擎）未做自动化
   UI 测试，仅源级与配置级验证。

## 10. 一键恢复 fixed 模式

- **GUI**：瞄准页 → "固定节拍" → "识别调度" 选 **固定周期** → 保存 → 重启引擎。
- **config.json**：将 `"unit"."frame_schedule"` 改为 `"fixed"`（或删除该键，缺省
  即 fixed），保持 `"frame_ms": 22.0` 与 `"reference_frame_ms": 22.0` 即可完整
  复现改造前的固定节拍行为（已由基线夹具逐帧比对证明）。

## 11. 结论分级

- **已由自动测试证明**：fixed 模式与修改前逐位一致（含缺字段回退）；asap 无
  frame_ms 人工等待；时间归一化在 14/18/22ms 下滤波结果一致（≤0.3px 级）；
  恒定误差墙钟累计 fixed/asap 一致；换向无旧尾段；丢失/TTL 立即清零；后坐力
  秒累计一致、手动曲线时间轴不变；小数账本无系统丢失；单写入点结构守卫；
  模拟器双调度可运行。
- **仅由离线模拟支持**：asap 在可变 e2e 序列下的端到端追踪表现
  （`shootsim` main_real，真实日志时间戳/可配置 e2e）。
- **必须由新实机日志确认**：实际 YOLO FPS 与各 P50/P95 指标、artificial wait≈0、
  stale clear 次数、换向/横移实机手感、GUI 实操切换。
