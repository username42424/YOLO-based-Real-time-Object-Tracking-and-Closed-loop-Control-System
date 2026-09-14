# ASAP 调度修复报告

日期：2026-09-07

## 结论

已完成本次范围内的调度、控制时序、时间归一化、模拟器一致性和回归测试修复。没有修改 YOLO 模型、推理后端、瞄准参数、灵敏度、后坐力强度、日志、基线夹具或第三方依赖。

附件实战日志已作为诊断背景阅读。该日志没有提供可用于证明目标跟随改善的合格目标跟踪对照段，因此本文不宣称实机跟随效果、首次进入 60% 时间或 60% 停留率已经改善。

## 修改文件与精确函数

- `asap_timing.py`
  - `clamp`
  - `bounded_rate_integration_dt`
  - `ObservationFreshness.reset/observe/recent_dt/ewma_dt/ttl_s`
  - `merge_config_defaults`
- `main.py`
  - `DEFAULT` 中 `unit.reference_frame_ms=22.0`
  - `load_config`
  - `run_engine`
  - `run_engine` 内 `_clear_current_aim`
  - `run_engine` 内 `_output_worker_loop`
  - `run_engine` 内 `_apply_recoil_state`
  - `run_engine` 内 `_reset_fps_meter`
- `aim_engine.py`
  - `MainEngine._frames_gate`
  - `MainEngine.compute` 中长窗口速度估计门控
- `crosshair_tracker.py`
  - `CrosshairTracker.preferred_position`
  - `CrosshairTracker.update`
- `gui.py`
  - `DEFAULT`
  - `load_config`
  - `AimGUI._ui_to_config`
- `shootsim/trackers.py`
  - `MainEngineTracker.__init__`
  - `set_output_phase`
  - `reset`
  - `_submit_unit_plan`
  - `_activate_pending_plan`
  - `_drain_unit_output`
  - `update`
- `tests/test_frame_schedule.py`
  - 新增 ASAP TTL、输出卡顿、初始帧门限、配置兼容、长窗口速度、仿真延迟计划和闭环反馈测试；同时关闭测试文件句柄。

## 问题原因与修改后的语义

### 1. 旧速率最多持续 200ms

原因：ASAP TTL 使用固定的 `max(0.20, reference_frame_s * 2.5)`，推理/捕获暂时停顿时，输出器继续积分上一帧速率。

现在由 `ObservationFreshness` 记录近期观测间隔的稳健中位数，并使用：

```text
TTL = clamp(recent_observation_dt * 2.5, 40ms, 60ms)
```

超过 TTL 后立即清空瞄准计划和速率；恢复观测时只发布新计划，不补发失效期间的位移。fixed 模式继续使用原有固定节拍语义。

### 2. 输出线程卡顿补发大位移

原因：真实输出线程间隔直接作为 `rate * actual_dt`，卡顿 100ms/500ms 会一次性折算错过的时间。

现在 `bounded_rate_integration_dt` 将速率积分使用的 dt 限制为 `output_period * 1.5`，超出的 dt 直接丢弃并累计诊断：

- `tick_stall_count`
- `discarded_dt_ms`
- `max_actual_tick_dt_ms`

实际 tick 间隔仍单独记录用于诊断；后坐力仍走原固定输出节拍和积分路径，没有用瞄准修正改变其总强度。

### 3. ASAP 锁定、预测和红点确认多等一帧

原因：首次观测已经将计数设为 1，但时间门限仍从 `frames * reference_dt` 开始计算。

现在只有明确标记 `initial_counted=True` 的预测锁门限使用 `(frames - 1) * reference_dt`；丢失宽限、换锁宽限、速度样本确认等从 0 开始的门限未改变。红点 `confirm_frames=2` 也在第二次稳定观测激活。fixed 仍按原帧数判断。

### 4. 长窗口速度异常阈值未归一化

原因：普通速度估计按实际 dt 缩放阈值，长窗口仍用未缩放的 `ema_max_step + 1`，导致 ASAP 短周期下部分异常跳变被长窗口接受。

现在长窗口使用 `self._norm_disp(self.ema_max_step + 1.0)`。异常样本被拒绝时不更新 `_vel_lw`，也不改变 `_vel_lw_valid`；因此不会用异常跳变污染长窗口状态。正常样本仍保持已有的时间归一化滤波语义。

### 5. 模拟器推理完成延迟

原因：旧模拟器在观测时立即写入新速率，却又在 `_chunks_ready_t` 前跳过输出 tick，结果同时出现“新计划提前生效”和“推理期间旧计划停止”两种错误。

现在模拟器明确维护：

- `active_plan`：当前已发布并正在执行的计划
- `pending_plan`：推理完成但尚未到发布时间的计划
- `pending_available_at`：计划可用时间

输出 tick 到达 `pending_available_at` 前继续执行 active plan；到达的第一个 tick 原子替换为 pending plan，或执行延迟的 clear。新的完成结果覆盖尚未发布的 pending 结果，符合该串行模拟器的“最新已完成结果”顺序。TTL 和输出 dt 限制复用 `asap_timing.py` 的实现。

### 6. GUI 解耦两个周期

`frame_ms` 仅表示 fixed 模式人工识别周期；`reference_frame_ms` 是控制器标称整定周期，用于速率换算、滤波、预测门限和默认后坐力时间归一化。GUI 当前不单独暴露后者，保存时保留已加载值，不再无条件覆盖。修改 fixed 的 `frame_ms` 或切换 ASAP 不会暗中改动 `reference_frame_ms`。

### 7. 旧配置回退

`merge_config_defaults` 在深合并前检查源配置字段：缺少 `frame_schedule` 时回退 `fixed`；缺少 `reference_frame_ms` 时优先使用源配置实际的 `frame_ms`。因此旧配置只有 `frame_ms=22` 时，加载结果为 `fixed` 和 `reference_frame_ms=22`，不会被默认值 100 覆盖。main、GUI 和直接传给 `run_engine` 的配置走同一兼容逻辑。

### 8. 输出诊断

无 active plan 时 `control_age` 显示 `N/A(inactive)`；没有历史发布时发布间隔显示 `N/A(no-history)`。日志额外显示最近历史发布距今、当前 plan 状态及 `stale_clear`/`no_target`/`deadzone`/`stopped`/`never_published` 等 clear 原因，并记录近期观测周期和输出线程卡顿指标。诊断字段只反映状态，不改变输出决策。

另外，实战日志暴露的 `CrosshairTracker.preferred_position` 缺失接口也已补齐；这是主循环读取该属性时的必要兼容修复，避免红点路径在启用时崩溃。

## 测试覆盖

新增或加强的测试覆盖：

1. 14/18/22ms 时间归一化，以及 fixed/asap 在 `actual_dt=reference_dt` 时的锁定、预测和红点激活时刻一致。
2. 近期观测 TTL 的 40–60ms 边界。
3. 10ms 正常输出和 100ms/500ms 卡顿的 dt 限制与丢弃时间。
4. 推理延迟期间 active 速率继续运行，延迟到点后切换新速率；无目标 clear 同样延迟。
5. 长窗口归一化异常跳变拒绝和不同周期的等价物理速度。
6. GUI 周期解耦和旧配置回退。
7. 输出诊断 inactive/history/stale 字段的源级守卫。
8. 实际闭环仿真：鼠标输出更新模拟相机，下一帧检测框由该相机状态生成；比较首次进入 60% 误差区时间、60% 区域停留率、超时率、峰值误差以及换向后的旧方向输出。
9. 测试中的文本/JSON 文件全部使用上下文管理器；另以 `ResourceWarning` 升级为错误运行。

## 完整结果

执行：

```text
.venv\Scripts\python.exe -m unittest discover -s tests -q
```

结果：`Ran 203 tests in 0.393s — OK`。

另以 `.venv\Scripts\python.exe -W error::ResourceWarning -m unittest discover -s tests -q` 严格模式运行，结果同样为 `Ran 203 tests — OK`。

同时通过了相关模块的 `py_compile` 检查。

## fixed 兼容性

fixed 基线回放、旧配置 fixed 回退和现有 fixed 相关测试均通过；`tests/fixtures/fixed_baseline.json` 未修改、未重新生成。当前证据是仓库内固定夹具的逐字段自动回放，不包含原始版本源码或独立实机重放，因此不将其扩大解释为所有机器环境下的逐位兼容证明。

## 模拟器与 main.py 的已知差异

- main.py 使用真实线程调度、捕获时间、YOLO 推理完成时间和 Windows 输入注入；模拟器使用可控的 `advance_to` 时钟和 `send_fn`。
- main 的 active/pending 关系由识别线程与输出线程的共享状态实现；模拟器在单线程回放中显式建模同一发布顺序。
- 模拟器的相机响应、检测框生成和端到端延迟是确定性模型，不等价于真实游戏渲染、输入采样和网络/引擎延迟。
- 两者共用 TTL 和速率积分 dt 限制；模拟器 fixed chunk 消费和真实 main 的 Windows 输入时序仍不是同一个操作系统调度实例。

## 自动测试结论与仍需实机验证的结论

自动测试证明了：时间门限的计数语义、TTL 上限、卡顿 dt 丢弃、active/pending 发布顺序、GUI/旧配置映射、长窗口异常拒绝和闭环仿真中的反馈链路。

仍需实机日志验证：真实捕获/推理停顿分布、实际输出线程卡顿频率、TTL 清零是否覆盖所有设备状态、真实红点检测稳定性、真实目标运动下的首次进入 60% 时间、60% 停留率、峰值误差、超时率和换向尾段。当前附件没有可用于这些改善声明的目标跟踪对照段。

## 下一次实机采集建议

由用户在不改变模型和瞄准参数的前提下采集一段可对照日志，至少保留：

1. 连续无目标段，包含一次人为或自然的捕获/推理停顿。
2. 静止目标段和单方向移动目标段，各持续足够时间。
3. 一次明确的目标运动换向，并保留 `[输出器]`、`[鼠标]`、`[YOLO性能]` 的时间戳。
4. 记录 `control_age`、`last_publish_history_age`、`plan`、`stale_clear`、`tick_stall`、`discarded_dt` 和 `max_actual_tick_dt`。
5. 同一配置下分别保留 fixed 和 ASAP 的可比短段，避免把不同灵敏度、模型或后坐力状态混入对照。

本报告不代替用户运行游戏，也不把仿真指标当作实机效果证明。
