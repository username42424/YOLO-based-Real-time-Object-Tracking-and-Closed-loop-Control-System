# ShootSim — FPS 射击追踪模拟与优化系统

模拟 FPS 射击场景的完整闭环：目标框随机移动 + 鼠标视角控制（准星固定、场景移动）
+ 自动射击 + hit-scan 伤害判定 + 全程日志 + 可视化 + 参数搜索优化。
最终目标：通过调整追踪/射击策略与参数，使目标框被击杀的时间最短。

## 快速开始

```
cd shootsim
python run_sim.py                     # 可视化运行（auto 模式：追踪器自动瞄准+射击）
python run_sim.py --mode human        # 人手模式：物理鼠标=视角，自动射击
python optimize.py --compare          # 预设对比实验（策略/瞄准点/增益/延迟…）
python optimize.py                    # 随机参数搜索（200组 × 8种子）
python sweep_staged.py --smoke --workers 1  # 修正版回放寻优冒烟测试
python sweep_staged.py                # 修正版分阶段大量寻优
```

- 可视化窗口内：**准星固定在屏幕中心**；移动鼠标时场景（网格+目标）相对移动。
- `ESC` 退出；回合结束（击杀或超时）自动打印汇总。
- 日志输出在 `results/`（每回合一个 JSONL 文件，逐发子弹记录）。

## 架构（模块化）

| 文件 | 职责 |
|---|---|
| `config.py` | 参数默认值 + JSON 读写 + 搜索空间 `SEARCH_SPACE` |
| `env.py` | 核心环境：世界/相机坐标模型、目标移动、射击判定、伤害/血量（纯逻辑，可 headless） |
| `movers.py` | 目标移动策略：`random4`(默认上下左右随机) / `random8` / `sine` / `jitter` / `teleport` / `chase` |
| `trackers.py` | 追踪策略：`p` / `pid` / `ema` / `predict` / `main` / `main_real` / `other_test` / `obs_auto_aim` |
| `shooters.py` | 射击策略：`always`(连续) / `lock_delay`(锁定延迟) / `oracle_on_aim`(真实命中框触发，仅演示) |
| `perception.py` | 感知：`gt`(精确) / `gt_noise`(高斯噪声+丢帧) / `yolo`(复用根目录 main.py 的视觉识别) |
| `logger.py` | JSONL 日志：逐发子弹（区域/伤害/累计/血量）+ 回合汇总（击杀时间） |
| `renderer.py` | pygame 渲染：网格场景/目标框三区域/准星/血条/命中特效/弹道/HUD |
| `run_sim.py` | 可视化运行入口（auto / human 模式） |
| `optimize.py` | 参数搜索（随机采样 × 多种子，中位数排序）+ 预设对比实验 |

## 坐标与射击模型

- **视角模型**：世界平面 w×h；相机位置=屏幕中心对应的世界坐标；
  目标屏幕位置 = 目标世界坐标 − 相机坐标 + 屏幕中心。
  鼠标右移 → 相机右移 → 目标在屏幕上左移。准星固定中心。
- **伤害**：目标框垂直三分区——上部头部 40 / 中部 20 / 下部 10；初始血量 150。
- **射速**：20 发/秒（每 50ms 一发）。hit-scan：准星点在框内即命中，按所在区域计伤害。
- **感知延迟**：`tracker.obs_delay_frames`（默认 2 帧）模拟"截图+推理+相机响应"回路延迟，
  追踪器读到的是延迟前的位置——这是真实系统调参的关键因素。

## 关键参数（config.json）

`main_real` 直接复用根目录 `MainEngine`，并按真实程序的识别周期、独立鼠标输出周期、
分步计划替换和整数累计运行。回放优先读取当前日志中的
`[瞄准观测]` JSON；其中 `observed_aim + observed_recoil` 已代表截图实际观察到的位移，
不会再叠加虚构后坐力或额外两帧相机延迟。鼠标分步输出会在两次日志观测之间按20ms
节拍生效，而不是等到下一帧集中发送。`tracker.processing_latency_ms` 应使用实测日志中位数。

回放会保留每帧全部检测框、日志记录的红点位置和最后一个目标位置；日志结束即结束该
回合，不再跳回第一帧或补成3秒“超时”。跨回合驻留率按目标实际可见秒数加权。
`sweep_staged.py` 的新结果写入 `results/staged_v8_tracking/`，不会续跑或混用修复前的旧结果。
实战日志 `aim_20260904_220626.log` 与 `aim_20260905_131152.log` 被整份留作独立验证集，
不会把同一文件切片后同时放进搜索集和验证集。不同日志录制时使用的瞄点比例由
`replay.source_chest_ratio_by_log` 分别还原。
低于日志可分辨截图周期的 `unit.frame_ms` 候选会被明确拒绝，避免把日志无法验证的
40–45ms 数值当成精确最优。
`chest_ratio` 固定使用生产配置值，不参与搜索；“准星位于框上方”仅保留为诊断数据，
不参与评分或验收。最终参数必须同时通过首入60%区域、进入失败率、驻留率和真实超时率门槛；没有
候选全部通过时，报告会明确拒绝产生冠军，也不会建议写回生产配置。

| 参数 | 默认 | 说明 |
|---|---|---|
| `target.speed` | 120 | 目标移动速度 px/s |
| `target.mover` | random4 | 移动策略 |
| `tracker.strategy` | p | 追踪策略 p/pid/ema/predict/main/main_real/other_test/obs_auto_aim |
| `tracker.gain` | 0.35 | 比例增益（移动=误差×gain） |
| `tracker.cap` | 60 | 单帧位移上限 px |
| `tracker.deadzone` | 6 | 误差死区 px |
| `tracker.aim_ratio` | 0.166 | 瞄准点高度：0=框顶，0.166≈头部中心，0.42≈胸口，0.5=框中心 |
| `tracker.lead_frames` | 0 | predict 策略的速度外推帧数 |
| `tracker.obs_delay_frames` | 0 | 回放日志已经包含实际闭环延迟，不重复添加延迟帧 |
| `shooter.policy` | always | 连续射击 / 瞄准才射 |
| `perception.mode` | gt_noise | 感知模式 gt / gt_noise / yolo |
| `episode.max_seconds` | 30 | 回合超时 |
| `view.sensitivity` | 1.0 | 鼠标 px → 视角 px |

`obs_auto_aim` 是对公开仓库 `160037lk/obs-auto-aim` 本地版瞄准逻辑的确定性移植，包含距离+置信度选目标、距离相关增益和自适应平滑。可直接使用：

```powershell
.\.venv\Scripts\python.exe shootsim\run_sim.py --headless `
  --config shootsim\config_obs_auto_aim.json --tracker obs_auto_aim `
  --episodes 100 --seed 2001
```

三方 100 次结果见 [`OBS_AUTO_AIM_COMPARISON.md`](OBS_AUTO_AIM_COMPARISON.md)。

## 调优结果（2026-08-13，实测数据）

理论下限：头部 40 伤害 × 4 发 = 160 ≥ 150，20 发/秒 → **理论最短击杀 = 0.20s**。

**预设对比实验结论**（每组 8 回合）：

1. **追踪策略**：P 控制 0.281s 最优；PID(积分) 反而差（1.68s，饱和拖累）；EMA/预测略差。
2. **瞄准点**：头部中心(0.166) 0.281s；瞄准胸口(0.42)/中心(0.5) 约 0.47-0.48s——**打头比打身体快 40%**（40×4 发 vs 20×8 发）。
3. **增益**：gain 0.2-0.4 最优（0.26-0.28s）；gain≥0.5 开始过冲变差，0.7 恶化到 0.61s。
4. **射击策略**：`oracle_on_aim`/兼容别名 `on_aim` 会读取真实命中框，只能用于演示，命中率不参与算法比较；算法评估应使用 `lock_delay` 或 `always`。
5. **目标速度**：60-260 px/s 对击杀时间影响很小（追踪能力足够）。
6. **感知延迟**：1 帧 0.250s ≈ 2 帧 0.281s；3 帧 0.456s；4 帧 1.169s——**延迟是最敏感的瓶颈**，
   对应真实系统"端到端延迟越低，击杀越快"。

**随机搜索结论**（300 组 × 8 种子，按中位数排序防过拟合）：
- 前 10 名全部瞄准头部区（aim_ratio 0.10-0.33），增益集中在 0.2-0.48 —— 与对比实验一致。
- 最优示例（`results/best_config.json`）：策略 predict(gain 0.48/cap 100/aim 0.159/延迟1帧)，
  平均击杀 **0.213s**，命中率 95%，0 超时。
- 教训：少种子 + 固定种子池会选出"幸运扫射"配置（少数种子上靠振荡偶尔爆头取胜，
  换种子现原形：27s 击杀/1.5% 命中率）。已改为每配置独立随机种子 ×8 + 中位数排序。

**可直接落地的"最优策略"**：瞄准头部中心（aim_ratio≈0.16）+ P 控制（gain 0.3-0.4，
cap 60-100）+ 死区 6 + 感知延迟尽量 ≤2 帧 + 连续或瞄准后射击皆可。

## 目标图片与 yolo 感知模式

- `target.image` 填图片路径即可替换目标外观（空=程序生成占位图）。
- `perception.mode = "yolo"`：可视化运行时截取窗口中心 640×640，调用根目录 main.py
  的 YOLO 检测器识别目标框（模型路径 `perception.yolo_model`）。
  注意：模型是游戏画面训练出的，占位图可能检测不到——此时用 `gt_noise` 模式
  （高斯噪声+丢帧）等效模拟真实检测的不确定性。

## 输出文件

| 文件 | 内容 |
|---|---|
| `results/episode_*.jsonl` | 每回合日志：episode_start / 逐发 shot（区域、伤害、血量）/ episode_end（击杀时间汇总） |
| `results/search.csv` | 参数搜索全表（每组合的参数 + 平均/中位/标准差/超时数/命中率/逐回合击杀时间） |
| `results/best_config.json` | 搜索最优参数，可合并进 config.json 使用 |
| `results/staged_v8_tracking/` | 修正版分阶段回放寻优结果；旧版本结果仅供历史追溯 |
