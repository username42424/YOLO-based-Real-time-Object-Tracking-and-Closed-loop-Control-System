# YOLO-based Real-time Object Tracking and Closed-loop Control System

基于 YOLO 的实时目标跟踪与闭环控制实验平台

> [!WARNING]
> **这是实验性项目，不是成熟产品。** 仓库中存在多套并行演进的控制实现、历史兼容路径和未经完整验证的假设。请在理解源码并完成独立测试后使用。

> [!IMPORTANT]
> **本项目绝大部分代码在 AI 工具的帮助下编写和迭代。** AI 参与了设计、实现、重构、调试与文档整理；代码尚未经过完整人工审计，可能包含重复实现、风格不一致、逻辑缺陷或错误假设。

## English summary

An experimental Windows-based platform for YOLO object detection, target tracking, closed-loop mouse control, replay analysis, and offline simulation. Most of the code was written and iterated with substantial AI assistance. The repository is a research prototype, not production-ready software, and should be reviewed and tested independently before use.

The project uses FPS-style visual scenes as a validation environment, but it is not designed for or affiliated with any specific game.

## 项目简介

该项目把屏幕采集、ONNX 推理、目标选择、状态跟踪、控制量计算、Windows 鼠标输入和运行诊断组织为一条实时闭环。它主要用于研究视觉检测结果如何转化为连续控制输出，以及延迟、目标抖动、预测、滤波和输出调度对闭环行为的影响。

仓库同时包含主程序和 ShootSim 离线模拟器。主程序负责真实屏幕采集与 Windows 输入；ShootSim 用于在不依赖真实桌面输入的情况下复用部分跟踪逻辑、回放日志并比较控制策略。

## 项目状态

- 当前处于实验和快速迭代阶段，没有稳定版本承诺。
- 多套鼠标控制路径同时存在，部分参数在不同路径中的含义并不完全一致。
- 源码仍包含未在当前界面开放的历史兼容分支。
- 当前公开仓库没有完整的自动化测试集。
- 仿真结果只能用于工程比较，不能代表真实环境中的最终表现。
- 使用前需要自行检查代码、模型、输入方向、灵敏度、分辨率和安全边界。
- 绝大部分代码由 AI 辅助完成，人工复核覆盖并不完整。

## AI 辅助开发声明

本项目不是由传统人工开发流程完整实现的软件。绝大部分代码在 AI 工具帮助下生成、修改或重构，AI 也参与了问题定位、控制策略讨论、实验脚本和技术文档的整理。

这意味着仓库可能同时存在：

- 重复或相互重叠的实现；
- 不同阶段遗留的命名和配置语义；
- 只经过局部验证、没有经过完整系统验证的逻辑；
- 注释、实验报告与当前实现不同步的情况；
- 看似合理但尚未由人工充分证明的技术结论。

阅读、复用或贡献代码时，请把 AI 输出视为需要审查的草稿，而不是天然正确的实现。

## 核心功能

- YOLOv5、YOLOv11 和 YOLO26 ONNX 推理适配；
- 屏幕区域采集、图像预处理和检测结果后处理；
- 目标选择、优先级、连续锁定、切换宽限和丢失恢复；
- 目标框滤波、速度估计、预测和准星参考点跟踪；
- Windows `SendInput` 相对鼠标移动；
- 固定节拍与跟随推理速度的调度实验；
- 后坐力轨迹采集、持久化和回放；
- 自动触发与反应时间实验模块；
- GUI 配置入口、CLI 入口和运行诊断；
- ShootSim 离线仿真、日志回放和参数比较。

## 鼠标控制路径

当前代码中存在三套用户可选择的主要控制路径：

| 路径 | 配置 | 当前定位 |
|---|---|---|
| 平滑旧架构 | `aim_mode = smooth` | 使用平滑、增益、死区和目标预测计算逐帧控制量 |
| 速率保持 | `aim_mode = unit` + `control_strategy = rate_hold` | 将最新控制量转换为离散计划或控制速率，由独立输出节拍发送 |
| 异步剩余误差 | `aim_mode = unit` + `control_strategy = remaining_error` | 依据最新观测和剩余误差，在独立输出线程中持续生成控制量 |

此外，`aim_engine.py` 中仍保留一个未在当前 GUI 开放的旧 `direct` 分支。它属于历史兼容实现，不计入上面的三套用户可选路径。多套实现并存正是该项目仍被标记为实验版的重要原因之一。

## 工作流程

```text
屏幕采集
   ↓
YOLO ONNX 推理
   ↓
检测结果后处理与目标选择
   ↓
锁定状态、滤波与运动预测
   ↓
控制误差和鼠标控制量计算
   ↓
输出仲裁与 Windows SendInput
   ↓
画面反馈、下一帧观测与运行诊断
```

瞄准、后坐力和其他运动分量会先经过统一仲裁，再由单一物理输出路径发送。不同控制策略的观测节拍和输出节拍可以相互独立。

## 系统要求

- Windows；核心输入与热键模块依赖 Windows API。
- Python 3.10 或更高版本；当前本地开发环境主要使用 Python 3.12。
- OpenCV、NumPy、ONNX Runtime、dxcam 和 Pillow。
- 运行 ShootSim 可视化模式时需要 pygame。
- 需要自行提供与当前检测接口兼容的 ONNX 模型。

`requirements.txt` 默认使用 CPU 版 `onnxruntime`。DirectML 或 CUDA 环境需要根据硬件和 ONNX Runtime 官方说明替换推理包，避免同时安装互相冲突的运行时。

## 快速入口

```powershell
# GUI 配置入口（README 不提供界面展示）
python gui.py

# CLI 入口
python main.py

# ShootSim 可视化入口
python shootsim/run_sim.py
```

这些命令只表示程序入口，并不保证当前机器已经具备模型、依赖、硬件后端或适配参数。

## 常用配置

| 配置项 | 作用 |
|---|---|
| `model` | ONNX 模型路径 |
| `conf` / `iou` | 检测置信度与 NMS 阈值 |
| `capture_size` | 屏幕中心采集区域尺寸 |
| `target_priority` | 新锁定候选的优先策略 |
| `aim_mode` | 选择平滑旧架构或 unit 控制路径 |
| `unit.control_strategy` | 在 unit 模式下选择速率保持或异步剩余误差 |
| `unit.frame_schedule` | 固定节拍或跟随推理速度 |
| `crosshair_mode` | 屏幕中心、红点或回退策略 |
| `precision` | 推理精度与预处理模式 |
| `mouse_log_enabled` | 是否记录详细运行诊断 |

高级控制参数与不同策略紧密耦合。修改前应先阅读 `aim_engine.py`、`main.py`、`async_control.py` 和 `motion_arbiter.py` 的当前实现。

## ShootSim

[ShootSim](shootsim/README.md) 是仓库内的离线闭环模拟与分析工具，包含目标运动、感知噪声、跟踪器、控制器、射击判定、日志回放和参数比较。

根 README 不引用具体命中率、击杀时间或所谓“最优参数”。这些结果依赖模型、日志、配置、随机种子和评估口径，不能作为整个项目的性能承诺。

## 项目结构

| 路径 | 主要职责 |
|---|---|
| `main.py` | 主运行编排、屏幕采集、检测器适配和输出线程 |
| `gui.py` | 配置入口与运行控制 |
| `aim_engine.py` | 目标选择、锁定状态、预测和控制量计算 |
| `mouse_control.py` | Windows 相对鼠标输入与发送前处理 |
| `async_control.py` | 最新观测邮箱与异步剩余误差控制 |
| `motion_arbiter.py` | 瞄准、后坐力等运动分量的仲裁与输出计划 |
| `crosshair_tracker.py` | 准星或红点参考位置跟踪 |
| `recoil_controller.py` | 后坐力轨迹采集与回放 |
| `runtime_logging.py` | 运行日志与 GUI 日志转发 |
| `shootsim/` | 离线模拟、回放、评估和参数实验 |
| `docs/archive/` | 历史开发记录和阶段性技术报告 |

## 已知限制

- 真实输入路径依赖 Windows，不是跨平台实现。
- 仓库不附带训练数据或 ONNX 模型。
- 模型输出、截图范围和控制方向需要与实际环境匹配。
- 多套控制路径并存，尚未形成统一、稳定的控制接口。
- 历史分支和兼容参数增加了理解与维护成本。
- 当前测试、实机验证和长期稳定性验证均不完整。
- AI 辅助代码可能包含尚未发现的逻辑错误。
- 离线仿真不能替代真实系统验证。

## 使用边界

本项目以视觉跟踪、闭环控制和离线仿真研究为目的。请只在离线、私人、测试或明确获得授权的环境中使用，并遵守软件平台规则、服务条款和当地法律。项目与任何具体游戏、发行商或平台均无关联。

作者和贡献者不保证项目适合任何特定用途，也不对未经授权的使用负责。

## 贡献与审查

欢迎通过 Issue 或 Pull Request 讨论可复现的问题、控制策略、测试和文档改进。提交修改时建议：

- 明确指出影响的是哪条控制路径；
- 给出可复现条件和验证方法；
- 区分仿真结论、日志回放结论和真实环境观察；
- 对 AI 生成或 AI 修改的代码进行人工复核；
- 不把未经验证的 AI 结论写成确定事实。

由于本项目绝大部分代码由 AI 辅助开发，后续贡献尤其需要重视人工审查、测试覆盖和实现去重。

## 许可证

本仓库当前尚未确定开源许可证。在第三方代码来源和许可兼容性完成核对之前，请不要假定本项目采用 MIT、GPL 或其他开放许可证。

## 历史开发记录

以下阶段性记录已归档，内容可能描述旧实现，不代表当前稳定接口：

- [ASAP 调度改造记录](docs/archive/ASAP_SCHEDULER_CHANGELOG.md)
- [ASAP 调度修复报告](docs/archive/ASAP_SCHEDULER_FIX_REPORT.md)
- [异步控制修改记录](docs/archive/ASYNC_CONTROL_EDIT_LOG.md)
