# 异步视觉与鼠标控制改造日志

日期：2026-09-07  
范围：视觉观测、瞄准输出、运动账本、配置入口、自动测试与实战日志核对

## 1. 编辑方法

1. 先使用 `rg` 检查现有 `fixed/asap`、`publish_aim_rate()`、输出线程和运动仲裁器，复用已有实现。
2. 使用补丁方式逐文件修改，未重写整个文件，未建立第二套输出线程或第二套仲裁器。
3. 视觉、输出和账本先分别增加单元测试，再运行回归测试。
4. 自动测试使用模拟检测、可注入时钟和模拟 `send_fn`，不向真实桌面发送鼠标。
5. 实战日志只读分析，区分观测间隔、输出 tick、控制年龄和输入到画面反馈延迟；没有把 YOLO 推理时间当作闭环延迟。

## 2. 主要修改

### `async_control.py`

- 新增不可变 `ObservationSnapshot`。
- 新增 `LatestObservation` mailbox，只保留最新完整观测。
- 新增 `RemainingErrorController`。
- 以观测误差为基础，扣除已成功提交且预计尚未生效的运动量。
- 同一观测重复输出时，剩余预算持续减少，不重复满额纠正。
- 新会话、新目标、失检、歧义、过期时清空控制状态。
- 第一阶段关闭目标运动预测，仅保留比例纠错、死区、速度限制、加速度限制和小数累计。

### `motion_arbiter.py`

- 复用原有唯一输入入口。
- 新增 `MotionLedgerSnapshot` 和运动事件账本。
- 区分计划量、成功提交量、失败发送量以及按捕获时间可见的运动量。
- 发送失败不消耗成功运动预算。
- `next_aim_rate_motion(dt)` 使用真实 tick 时间积分，并限制异常长 dt，避免卡顿后突发补发。
- 瞄准和压枪仍通过同一个仲裁器合并，并保留分量账本。

### `main.py`

- 增加 `unit.control_strategy` 选择：
  - `rate_hold`：现有 ASAP 最新速率保持策略。
  - `remaining_error`：新的异步剩余误差策略。
- 视觉循环只负责截图、红点定位、YOLO、目标关联、滤波和发布观测。
- 输出线程独立读取最新快照，每个周期重新计算剩余误差并发送增量。
- 增加 `session_id`、`target_id`、`observation_id` 的内部状态检查，迟到观测不得重新激活旧会话。
- 输出周期继续使用 10ms，并按实际 `dt` 计算；错过的调度时间直接丢弃，不连续补发。
- 根据近期观测间隔计算有限 TTL，过期后清空旧瞄准计划和速率。
- 增加 `plan`、`residual`、`pending_pre`、`stale_control`、`tick_stall`、`discarded_dt` 等诊断日志。
- 保留 fixed 路径和旧 `rate_hold` 路径作为对照与回退。
- 修复启动异常：在读取 `unit_control_strategy` 前先初始化 `aim_mode`，避免：

  ```text
  UnboundLocalError: cannot access local variable 'aim_mode'
  ```

### `gui.py`

- 增加异步控制策略配置入口。
- 保留 `fixed/asap` 调度选择。
- 保存配置时保留已有 `reference_frame_ms`，不让 GUI 隐式覆盖用户配置。
- 没有修改无关的热键、模型和压枪参数。

### `asap_timing.py`

- 统一实际时间积分、dt 限幅、观测新鲜度和 TTL 计算。
- fixed 模式保持原固定节拍语义。
- ASAP 模式不再使用 `reference_frame_ms` 作为隐含等待时间。

### `aim_engine.py`、`mouse_control.py`

- `MainEngine` 的可变跟踪状态仍由视觉线程独占。
- 输出线程不调用同一个 `MainEngine.compute()`。
- `mouse_control.py` 保留既有发送职责，没有新增第二个写入者。

## 3. 测试与验证

- 异步剩余误差测试覆盖：同一观测重复输出、无新图停止、输入反馈延迟、发送失败、小数累计、会话切换和输出卡顿。
- 启动回归测试覆盖 `aim_mode` 初始化顺序。
- fixed/asap 公共能力和原有瞄准回归测试继续执行。
- 模拟闭环独立扫描推理耗时、观测周期、输入反馈延迟、输出周期、延迟抖动和标定误差。
- 早期核心测试记录：异步控制 8/8 通过，瞄准回归 124/124 通过。
- 仓库修复报告记录的完整套件为 203/203 通过；若当前 `config.json` 被用户改为 fixed，旧的“默认必须 asap”配置测试需要按当前用户配置单独处理。

## 4. 配置使用

```json
{
  "unit": {
    "frame_schedule": "asap",
    "frame_ms": 26.0,
    "reference_frame_ms": 22.0,
    "control_strategy": "remaining_error"
  }
}
```

- 新策略：`frame_schedule=asap` + `control_strategy=remaining_error`。
- 旧 ASAP 对照：`frame_schedule=asap` + `control_strategy=rate_hold`。
- 固定对照：`frame_schedule=fixed`，`frame_ms` 使用 25 或 26。

## 5. 实战日志结论

- 改后连续观测间隔约为 P50 13.6ms、P95 18.7ms，实际低于 25ms。
- 输出 tick 仍约为 10ms，控制快照年龄略有下降。
- 同一观测下的输出增量会递减，说明剩余误差预算正在工作。
- 现有实战日志没有游戏命中、击杀、标准化超时和真实 60% 停留结果，不能从日志推算命中率或击杀率。
- 实战日志仍发现松键后可能出现迟到的非零瞄准输出；这属于后续必须继续收紧的取消/唯一发送入口问题。
- 日志中的 `发布间隔=0.0` 或 `N/A` 仍不能作为有效观测发布延迟，需要继续补充发布时刻、SendInput 结果和输入到画面反馈时间戳。

## 6. 未做的修改

- 未修改 YOLO 模型、推理后端、灵敏度、瞄准比例和压枪曲线。
- 未删除 fixed 或旧 ASAP 速率策略。
- 未覆盖用户现有 `config.json`。
- 未把仿真结果冒充实战命中率、击杀率或过冲结论。

## 7. 本轮修复（2026-09-08）

### 评估链路

- `shootsim/ts_v2/logparse.py` 新增从日志头、运行时行和观测 JSON 解析
  `frame_schedule`、`control_strategy`、`effective_prediction_mode`、输出周期、
  标定、滤波和响应参数；缺失字段保持 `unknown`。
- `compare_logic.py` 不再使用文件名策略映射，默认扫描日志或接受显式文件参数，
  并按连续策略段分组。`report_logic_compare.py` 删除固定百分比和固定胜负结论，
  所有差异从 JSON 指标计算。
- 速度重建改为共同时间窗口、连续目标、跳变/切换/失检过滤；仅已知瞄准输入用于
  自身运动补偿，后坐力污染样本标低可信。报告同时列原始误差与滤波误差，FPS改为
  观测率，并增加有效观测数、有效时长、锁定时间占比、连续锁定时长和丢锁次数。

### 取消与发布/发送

- `AimControlEpoch` 将按键状态、会话和取消代次作为权威 token。采集开始保存 token，
  推理完成和发布前再次检查；取消确认与最终发送检查使用同一锁，取消后不能进入旧
  会话的新 `SendInput`。这不撤回操作系统已接收的输入。
- 日志新增结构化 `[发送]`（真实 begin/end/result、单调 `event_id`、实际 commit 时间、
  aim/recoil/net 分量和残差诊断）；取消拒绝、发送失败和正常零净分量保持不同语义。
  观测发布与控制元数据在同一状态锁内交换。
- 新观测 JSON 增加 `session_id`、`target_id`、`observation_id`、图像时间含义、发布时刻、
  `raw_error`、`filtered_error`、`control_error`、速度及逐轴可信度。

### 运动账本与控制约束

- `MotionArbiter` 引入 `MotionEvent.event_id`；目标/会话切换通过
  `reset_control_scope()` 清计划而保留已提交物理历史，快照去重按事件编号，不再按
  时间+分量猜同一事件。
- `RemainingErrorController` 不再在换目标时丢掉在途历史，跨 scope 仍扣除捕获前和
  推理期间已成功提交的瞄准输入；拒绝倒序观测。加速度限制在 counts/s 速度域执行，
  死区采用明确的停止迟滞。
- 新策略的预测开关不再被强制为 `current`；读取实际 `prediction_mode`，并将预测后的
  `control_error` 与评价用滤波误差分开。GUI 增加 tau、速度上限、加速度、反馈延迟和
  最大观测年龄入口，仍不自动修改用户当前配置。

## 8. 本轮验证结果

- 自动测试：`python -m unittest discover -s tests -p "test_*.py"`，240 项通过；新增
  对照组实际生效预测模式、处理超载观测降速和启动配置初始化顺序测试。
  另外对主模块和评估脚本执行 `py_compile` 通过。
- 指定八份历史日志重新分组：
  `asap/rate_hold/arrival`（2份）、`asap/remaining_error/current`（2份）、
  `fixed/unknown/arrival`（4份）。固定日志缺少控制策略字段，未猜成旧 arrival 控制器。
- 按共同窗口重建速度的实战观察：remaining/current 横移滞后 P75=13.91px，
  rate_hold/arrival=26.43px，fixed/unknown/arrival=13.95px；对应滤波误差 P75 为
  26.02、28.90、16.84px。以上是记录差异，未消除场景、标定和目标切换混杂，不能作为
  因果或普遍优于结论。
- 仿真 `shootsim/async_ablation.py` 使用独立的处理、观测、输出和画面反馈时间轴，
  并复用 `MainEngine` 与 `RemainingErrorController`。D 的速度来自历史观测估计而非真值；
  被控对象标定与控制器标定分别配置。当前基线（处理9ms、反馈20ms、输出10ms、
  被控对象0.44px/count）结果为：A fixed/arrival 中位误差36.62px、P95 82.52px、
  60px内占比0.684；B asap/rate_hold/current 中位5.14px、P95 20.18px；
  C asap/remaining_error/current 中位6.38px、P95 35.36px；D
  asap/remaining_error/arrival 中位5.55px、P95 30.09px。C/D 静止段抖动
  分别0.110/0.107px，急停后最大误差35.66/30.40px。A 的结果提示固定节拍在该
  延迟与标定条件下的响应滞后，不能外推成所有场景结论；B/C/D 的差异也仍是仿真观察，
  不是命中率因果证据。
- `--scan` 独立扫描处理耗时、反馈延迟、观测周期和输出周期；处理6/12/20ms时实测
  观测间隔为10/13/21ms，反馈5/20/40ms时C中位误差为4.52/6.38/9.22px、D为
  4.42/5.55/8.82px，观测周期8/15/25ms时实测为10/15/25ms，输出周期5/10/20ms
  时实测为5/10/20ms；tau=25/40/60ms时C中位误差为5.72/6.38/8.03px、D为
  4.93/5.55/7.52px。说明处理超过周期时没有伪造吞吐且变量没有绑成一个“帧时间”；
  tau=25ms是当前仿真候选点，但尚未据此改写用户配置，也不能仅凭仿真替代实战调参。
  该仿真仍是工程闭环证据，不是实际游戏命中率。

## 9. 尚待新实战验证

- 现有历史日志没有命中、击杀、标准化超时和红点60%停留事件，因此这些指标仍缺失。
- `new_frame_only=False` 后端未提供可确认的源帧序号时，日志明确使用
  `capture_completed` 作为获取完成时刻，不把它冒充图像生成时刻；需在支持源帧时间戳/序号
  的采集后端上继续验证重复帧和输入到画面反馈延迟。
- 预测开关、tau/反馈延迟和固定/ASAP输出周期需要在相同场景、相同标定和开火状态下重新
  录制配对场次，才能回答命中率、击杀率和过冲是否实际改善。

## 10. 2026-09-08 启动配置摘要初始化修复

### 问题

- GUI 启动引擎时，在日志文件 `logs/aim_20260908_003343.log` 对应的启动阶段报错：
  `UnboundLocalError: cannot access local variable '_recoil_cfg' where it is not associated with a value`。
- 触发位置是 `main.py` 的启动配置摘要。摘要先读取
  `_recoil_cfg.get('output_period_ms', 30.0)`，而 `_recoil_cfg` 原本在后续后坐力控制器
  初始化处才赋值。
- 这是局部变量初始化顺序错误，与异步控制策略、鼠标发送和用户当前配置内容无关。

### 修改

- 将 `_recoil_cfg = config.get("recoil", {}) or {}` 提前到启动配置摘要之前，统一供摘要、
  后坐力控制器和输出线程使用。
- 未修改用户当前 `config.json`，未改变 fixed、rate_hold、remaining_error 或预测模式的
  选择逻辑。
- 增加启动顺序回归测试，检查 `_recoil_cfg` 的赋值必须出现在配置摘要读取之前。

### 验证

- 先运行针对该故障的测试，确认修复前稳定失败：赋值位置索引 `7922`，摘要读取位置索引
  `6109`，断言失败。
- 应用最小修复后重新运行同一测试，结果：`37 tests ... OK`。
- 全量回归：`python -m unittest discover -s tests -p "test_*.py"`，结果 `240 tests ... OK`。
- 主程序、GUI、异步控制、运动仲裁器及评估脚本执行 `py_compile`，全部通过。
- 测试使用模拟/静态验证，不向真实桌面发送鼠标输入。

### 结论

- 本次启动崩溃已修复，重新启动 GUI 时配置摘要可以正常读取 `output_period_ms`。
- 本次属于启动初始化缺陷修复，不代表新的实战命中率、击杀率或超时率已经得到验证。

## 11. 本轮日志生命周期与验证链路修复（2026-09-08）

### 已修复

- 新增 `runtime_logging.py` 的运行级 `RunLogSink`。每次引擎运行使用独立
  `run_id`、单调 `event_seq` 和独立文件/标准输出队列；事件同时保留产生、入队、
  写入时间，队列年龄按单调时钟计算。文件、GUI、stdout 不再共享可替换的全局文件
  handler。模型加载器也改为接收本次运行的 emit，避免后台检测器把旧消息写入新运行。
- GUI 改为有界 `GuiLogPump`，Tk 主线程每 50ms 批量刷新（每批最多80条、约4ms），
  文本保留上限2000行；停止时清空GUI详情队列，不回放旧详情。引擎线程不调用
  `root.after`，只投递普通线程安全队列。慢stdout/慢文件接收端不会让生产者等待；
  文件过载记录丢弃、缺口、高水位和收尾截断，stdout/GUI丢弃单独统计。
- GUI保存本次运行线程句柄、配置、stop_event、poller和run_id；旧运行完成回调不能
  改变新运行状态。停止超时显示“旧运行仍在收尾”，未确认线程结束前拒绝重叠启动。
  引擎停止观察线程会立即使发送授权失效，但不承诺撤回操作系统已接收的输入。
- `MotionArbiter` 事件加入运行账本命名空间；目标/会话切换仍清除控制计划，但保留
  成功物理输入历史。量化、最终发送授权、成功/失败提交现在由同一事务锁保护，
  发送成功时间取实际发送边界，不用tick开始时间冒充。
- `RemainingErrorController` 在首次采用快照、换目标和新会话时也合并快照账本中的
  在途事件，并按 namespace+event_id 去重；发送失败不消耗成功预算，零净分量包不
  作为实际瞄准修正重复扣减。停止迟滞改为小阈值停止、大阈值重新启动。
- 新会话重置滤波/观测时间基准和发布间隔窗口；remaining-error 实际发布会记录
  `publish_intervals`。失检诊断由帧数后缀改为稳定类别 `missing_decay` 加独立的
  `missing_duration_ms`。
- 评估解析器支持新的JSON日志封套，按事件序号重建观测顺序；检测到多run、身份
  混写、启动后追加、时间或会话倒退的文件标为污染并从性能汇总排除，报告保留排除
  数量。当前指定 `aim_20260908_124724.log` 的248条观测已排除。
- `shootsim/async_ablation.py` 的A/B/C/D已复用 `MainEngine`、`end_frame()` 和
  `RemainingErrorController`；自运动补偿只使用提交账本和控制器标定，隐藏真实相机
  位移只用于评分。四组返回显式 `frame_schedule/control_strategy/prediction` 标签，
  处理、观测、输出、反馈时间轴分开。

### 反例与行为验证

- 快照误差40px、0.5px/count、快照账本内捕获前30counts、反馈估计20ms且控制器本地
  历史为空：剩余预算为50counts，不是80counts。
- 停止阈值1px、重启阈值4px：运动中2px继续输出；已停止后2px保持停止；达到4px才
  重新启动。新观测不会无条件清除停止状态。
- 取消与发送交错：发送中的取消等待发送边界完成，之后旧token发送被拒绝；新会话
  token可以发送。旧快照不能临时取得新会话token。
- 正常文件通道500条事件全部写入且`file_dropped=0`；慢stdout生产者提交500条耗时
  约4.6ms，文件仍写入500条；慢文件接收端队列上限8、产生500条时记录492条文件
  缺口且高水位不超过8；GUI队列上限64、产生5000条时保留64条并记录4936条丢弃。
- 运行级A/B隔离测试确认A关闭后消息不会出现在B文件；停止后GUI队列深度为0。
- 瞄准取消不会撤销独立压枪授权；只有压枪自身取消或整引擎停止才会使压枪token失效。

### 当前自动验证结果

- `python -m unittest discover -s tests -p "test_*.py"`：251 tests，全部通过。
- `py_compile`：`main.py`、`gui.py`、`aim_engine.py`、`async_control.py`、
  `motion_arbiter.py`、`runtime_logging.py`、异步仿真和评估脚本全部通过。
- 代码路径的发送、账本、GUI和解析行为均使用模拟发送/临时事件验证，不向真实桌面
  发送鼠标。

### 本轮仿真对照（处理9ms、反馈20ms、输出10ms、被控对象0.44px/count）

| 组 | 实际策略 | 观测间隔(ms) | 输出P50/P95(ms) | 误差中位/P95(px) | 60px内 | 静止抖动(px) | 急停后最大误差(px) |
|---|---|---:|---:|---:|---:|---:|---:|
| A | fixed / legacy_unit / arrival | 25.0 | 10.0/10.0 | 2.96/20.36 | 1.000 | 0.000 | 22.53 |
| B | asap / rate_hold / arrival | 10.0 | 10.0/10.0 | 4.86/20.18 | 1.000 | 0.381 | 21.18 |
| C | asap / remaining_error / current | 10.0 | 10.0/10.0 | 5.68/35.53 | 1.000 | 0.312 | 36.49 |
| D | asap / remaining_error / arrival | 10.0 | 10.0/10.0 | 5.65/30.22 | 1.000 | 0.193 | 30.94 |

这些是闭环仿真结果，不是实际游戏命中率。A的控制器标签为legacy_unit，未把手写
比例控制冒充生产旧控制器；D的预测来自历史观测估计和账本补偿，不读取真值速度。

### 日志压力实测（本轮独立注入）

- 正常通道：产生/写入500/500，文件丢弃0，队列年龄P95约9.83ms。
- 慢stdout（2ms/条）：产生500，文件写入500，stdout收尾时有95条低优先级显示记录
  被计入stdout截断；控制生产提交耗时约4.55ms。
- 慢文件（2ms/条、队列8）：产生500，写入8，文件丢弃492，队列高水位8，缺口492；
  这是明确的过载结果，不假称文件完整。
- GUI批量提交5000条：队列深度64、丢弃4936，提交耗时约5.88ms。GUI批处理函数
  有单批条数/时间上限；尚需在真实Tk窗口中测量批处理耗时分布。

### 历史实战重新分组

按实际日志头/运行时/观测字段，20260907四份记录拆为：

- `asap/rate_hold/arrival`：2份，3531条有效观测，观测率59.34Hz，间隔P50/P95
  15.25/24.00ms，原始误差P75 27.79px，滤波误差P75 28.90px，横移滞后P75 26.43px。
- `asap/remaining_error/current`：2份，6542条有效观测，观测率66.89Hz，间隔P50/P95
  13.86/19.40ms，原始误差P75 23.24px，滤波误差P75 26.02px，横移滞后P75 13.91px。

两组历史文件均没有run_id，属于legacy-unverified，只能描述观察差异，不能证明
生命周期隔离或因果优劣。命中率、击杀率、超时率、红点60%停留率在实战日志中没有
事件来源，保持缺失；不从这些记录猜测。

### 尚需新实战验证

- 新格式日志的真实连续启动/停止、旧运行零污染、GUI回调数量和停止收尾耗时；
- 采集后端源帧序号/生成时间，以及输入到画面的实际反馈延迟（`capture_completed`只
  是抓取完成时刻）；
- fixed、rate_hold、remaining_error和预测在相同场景/目标尺度/开火状态/标定下的
  命中率、击杀率、超时率、红点60%停留、过冲和急停额外位移；
- 不根据受污染日志或本轮仿真单独改写用户当前tau、前导或`config.json`。

## 2026-09-08 本轮生产发送与日志刷屏修复（复核结果）

本节是对上一节历史声明的重新核实，不把历史的“251 tests passed”当作生产发送
链路证据。本轮没有修改用户当前 `config.json`，没有向真实桌面发送鼠标。

### 根因与修改

- P0 发送失效：`run_engine` 内层事务函数的形参是 `send_token`，却引用了不可见的
  `_send_token`。新增模块级生产入口 `main.execute_motion_transaction`，输出线程只
  传入本次快照/控制状态的 token，并在入口内调用
  `send_epoch.run_send(send_token, send_fn, ...)`；没有新增同名全局变量。内层函数
  现在只是注入本次运行依赖的薄包装，fixed、rate_hold路径未改变。
- P0 事务污染：量化、最终授权、发送结果和账本提交位于同一事务锁。量化/授权/发送
  前异常或失败调用`mark_send_failed`恢复pending和累计目标；成功进入物理提交后，
  controller反馈或日志异常只记error，不回滚、不重发。
- P1 空转刷屏：`MotionArbiter.mark_send_succeeded(record_event=False)`允许保留小数
  累计或分量账务而不生成物理事件。纯零包不调用fake/真实发送入口、不生成send_id；
  aim/recoil整数分量抵消只增加`component_cancel`计数，不冒充净视角移动。输出摘要
  现在按秒合并`idle_tick/fractional_wait/component_cancel/send_attempt/send_success/
  send_failed`。
- P1 日志路由：`RunLogSink`默认文件保留完整事件，stdout/GUI只接收生命周期、配置、
  error/warning、取消和低频summary；observation/send高频详情仅文件通道。主动过滤、
  GUI提交异常、stdout/GUI丢弃和文件缺口分开计数。observation正文只放在`fields`，
  `message`改为短摘要，`shootsim/ts_v2/logparse.py`已兼容该格式。
- P1 输出故障：输出器对确定性异常（包括NameError/UnboundLocalError）清理瞄准状态、
  发出带`event_type=error`和traceback的事件并停止，不再每100ms刷“准备恢复”。
  OSError/TimeoutError最多重试3次；日志故障和输入提交故障分别统计。
- P2 通道和GUI统计：文件、stdout分别记录队列年龄P95和消费耗时P95；GUI pump记录
  入队到显示年龄、完整Tk刷新耗时、批字符上限和丢弃数。GUI使用批量insert，单批最多
  80条/12000字符，保留2000行，停止时清空详情队列且拒收旧run_id。
- 关闭语义：关闭前写入`[日志统计] phase=pre_close`快照，关闭后只向stdout输出
  `[日志最终] phase=closed close_reason=...`，不重新打开全局日志通道；因此文件中的
  `close_reason=open`只表示关闭前快照，不再被误读为最终状态。
- GUI终止回调的错误消息现在与旧run_id一起在清空详情后写入、再解除活动身份；因此
  错误可见，同时不会让旧完成回调重新灌入新运行。

### 针对本次故障的行为证据

- 修复前提供的 `logs/aim_20260908_134327_2bc134e6.log`：单一run_id、114条
  observation，但1646条send全部是`result=no_packet`，且每条都有递增物理事件号；
  收尾统计仍显示`close_reason=open`。这是修复前的刷屏证据，不用于证明修复后的实战
  表现。
- 修复后生产事务行为测试使用实际`main.execute_motion_transaction`、真实
  `MotionArbiter`/`AimControlEpoch`和注入fake send：有效token非零包调用一次，账本和
  controller反馈各一次；过期token不调用send；量化异常、发送异常和false结果均无残留
  pending且下一次有效包不补发旧包；成功提交后的诊断异常不会回滚或重发。
- 连续1000次纯空转的fake-send测试：实际send为0、successful physical event为0、
  逐tick emit为0；小数等待和净零分量抵消分别计数且保留累计语义。
- 日志行为测试：高频observation/send完整留在文件，stdout/GUI过滤计数增加；短消息
  文件仍可由现有`parse_rows`从`fields`恢复；GUI测试记录显示年龄和完整刷新耗时。

### 当前自动验证

- `python -m unittest discover -s tests -p 'test_frame_schedule.py'`：44 tests，全部通过。
- `python -m unittest discover -s tests -p 'test_runtime_lifecycle.py'`：8 tests，全部通过，
  包含慢stdout、慢文件、慢GUI/有界队列、A/B run隔离、路由过滤、解析兼容和GUI显示统计。
- `python -m unittest discover -s tests -p 'test_*.py'`：260 tests，全部通过。
- `python -m py_compile main.py motion_arbiter.py runtime_logging.py gui.py
  shootsim/ts_v2/logparse.py`：通过。

### 证据边界与待验证项

以上是内存fake-send、生产模块集成和临时文件日志测试；没有真实鼠标、真实游戏或无人
值守桌面验证。尚未真实验证游戏结束后Tk窗口显示、真实连续启停期间的系统级句柄回收、
真实SendInput到下一张画面的反馈延迟，以及命中率/击杀率/超时率/红点60%停留率。
下一次实战应使用新的run_id文件，检查非零send比例、空转摘要、最终close记录和跨运行
污染计数后，再与旧实战分组比较；本轮不据此调整tau、前导、反馈延迟、增益或输出周期。

## 2026-09-08 输出周期下限调整

按用户要求，将共享鼠标输出周期下限从10ms调整为5ms：`main.py`的实际周期限幅、
启动摘要和`gui.py`滑块范围已统一。保留实际`dt`积分、小数累计和每秒速率语义，
未修改用户当前`config.json`。新增下限回归测试；定向45项、全量265项测试均通过，
`main.py`和`gui.py`编译通过。5ms下的真实系统调度、游戏画面反馈、静止抖动和过冲
仍需实战测量，不能仅因输出调用更密集就视为闭环延迟改善。

## 2026-09-08 热键绑定稳定性修复

### 根因

- GUI 轮询器原先用物理 VK 作为唯一边沿状态键。若瞄准和压枪误设为同一个物理键，
  前一个动作会先更新 `_prev[vk]`，后一个动作看不到同一次按键边沿，表现为动作
  “被改掉”或偶尔不响应。
- GUI 轮询器缺少退出键时的回退值是 `p`，而 GUI、主引擎默认配置是反引号 `` ` ``；
  配置字段缺失时因此出现显示/实际监听不一致。
- GUI 启动和保存会把控件中的完整热键映射写入配置。此前没有保存前的冲突校验，
  重复绑定会被静默接受。

### 修改

- 新增 `hotkey_config.py`，集中定义五个动作的默认值，并统一规范化、检查未知键和
  重复物理键；重复绑定现在在轮询器创建前拒绝，不再静默吞边沿。
- GUI 的保存与启动均在写配置前校验热键；校验失败只提示并保留当前文件，不启动新
  运行，不修改用户当前 `config.json`。
- GUI 与 CLI 均使用同一组缺省值；退出键缺失时统一回退到反引号。
- 轮询器边沿状态改为按动作保存，作为重复绑定保护之外的防回归措施；真实按键发送
  链路未改变。

### 行为验证

- 修复前红灯：`test_duplicate_hotkey_bindings_fail_before_polling` 未抛冲突异常，
  `test_missing_quit_binding_uses_the_gui_default` 得到 VK 80（`p`）而非反引号 VK 192。
- 修复后：`python -m unittest discover -s tests -p 'test_aim_regressions.py'`，126项
  通过；全量 `python -m unittest discover -s tests -p 'test_*.py'`，269项通过；
  `python -m py_compile hotkey_config.py gui.py main.py` 通过。
- 测试使用模拟配置和模拟 `GetAsyncKeyState`，未发送真实鼠标输入，未覆盖真实桌面上
  多个 GUI 进程同时写配置的系统级竞争。

## 2026-09-08 头框/身框候选优先级可选

- 在 `aim_engine.MainEngine` 增加 `target_priority` 配置，取值为 `body` 或 `head`，
  默认仍为 `body`，因此未改变原有实战默认行为。
- GUI“工作台 → 目标类别”新增“同时检测头/身框时：优先身框/优先头框”选项；
  保存到 `config.json` 后，启动日志会记录实际生效的 `target_priority`。
- 仅改变首次新锁定候选的类别排序：头框优先时先选 `cls=1`，身框优先时保持原规则；
  同类内距离优先和身框 `cls=0` 优先规则均保留。已有锁定仍按原有头/身类别
  连续匹配、切换宽限、滤波和目标状态机处理，不会因为改选项强制抢锁。
- 行为测试覆盖：默认身框仍选身框、显式头框优先选择头框；全量测试
  `python -m unittest discover -s tests -p 'test_*.py'` 为270项通过，相关模块编译通过。
- 本次未修改用户当前 `config.json`，未进行真实鼠标或真实游戏验证；目标点命中效果仍需
  在实际使用中分别测量。

## 2026-09-10 引擎启停提示音

- GUI 增加 `play_engine_status_tone()`：启动使用异步 `SystemAsterisk`，关闭使用异步
  `SystemExclamation`，音频不可用时返回失败但不影响引擎、GUI 或鼠标输出。
- 启动提示音绑定在 `run_engine` 完成模型、采集和输出器初始化后的 `ready_callback`；
  关闭提示音绑定在 GUI 确认运行线程已退出后的停止状态，因此不会把“点击停止”误报成
  “引擎已经关闭”。同一运行只播放一次关闭音。
- 新增模拟 `winsound` 回归测试，验证启动/关闭别名和 `SND_ASYNC` 标志；全量测试
  `python -m unittest discover -s tests -p 'test_*.py'`：271项通过，`gui.py`、`main.py`
  编译通过。未发送真实鼠标，真实声卡/系统声音主题仍需桌面验证。

## 2026-09-10 启停状态语音播报

- 将启停提示从单纯的“叮”升级为 Windows SAPI 异步语音：启动播报“引擎已开启”，
  停止播报“引擎已关闭”。原 `winsound` 异步提示音仅作为 PowerShell/SAPI 不可用时
  的回退，不引入 `pyttsx3` 或其他新依赖。
- 仍沿用上一节的生命周期边界：启动在模型、采集和输出器就绪后播报，关闭在运行线程
  真正退出并由 GUI 确认后播报；新播报会终止尚未完成的旧语音进程，避免快速启停时重叠。
- 新增语音命令行为测试，验证两个中文状态文本；专项测试12项、全量测试272项通过。
  测试未调用真实语音设备或真实鼠标，系统是否安装中文 SAPI 声音仍需用户桌面验证。
