# obs-auto-aim 瞄准逻辑对比

## 接入内容

新增策略 `obs_auto_aim`，移植自公开仓库的本地 `main_aim.py`：

- 目标评分：`距离 × 0.6 + (1 - 置信度) × 100`
- 已锁定时加入上一目标偏移距离，降低目标切换
- FOV 过滤
- 距离相关灵敏度
- 非线性距离加速，指数默认 `1.8`
- 自适应平滑，近距离更平滑、远距离更快

没有移植云端 `video_bridge.py` 中的随机反应延迟、随机抖动、故意偏移、随机过冲和硬件鼠标输出，因为这些会把控制器比较变成随机性/硬件比较。

## 运行命令

```powershell
.\.venv\Scripts\python.exe shootsim\run_sim.py `
  --headless `
  --config shootsim\config_obs_auto_aim.json `
  --tracker obs_auto_aim `
  --episodes 100 `
  --seed 2001
```

## 100 次结果

条件：100 个种子 `2001–2100`、GTNoise、噪声标准差 2px、漏检率 5%、同一组 7 个目标素材、`always` 射击、关闭人手跟枪。

| 策略 | 命中率 | 击杀/100 | 首次进入内圈 P50 | 红点内停留 | P50 误差 |
|---|---:|---:|---:|---:|---:|
| `main`（你的主逻辑） | 23.79% | 88 | 0.23s | 56.40% | 55.59px |
| `other_test` | 12.77% | 59 | 0.04s | 16.33% | 98.54px |
| `obs_auto_aim_head` | 11.95% | 61 | 0.03s | 17.06% | 125.21px |
| `obs_auto_aim_body` | 22.48% | 82 | 0.03s | 57.30% | 59.48px |

## 结论

1. 对方逻辑不是天然优于你的逻辑；它的头部模式只处理 `cls=1`，遇到只有身体框的目标时不会回退，因此结果较差。
2. 切到身体模式后，`obs_auto_aim_body` 已接近你的 `main`：击杀 82 对 88，P50 误差 59.48px 对 55.59px。
3. 当前 `other_test` 的主要问题仍然是瞄准点/控制参数组合，而不是检测可见率：它的检测可见比例 97.23%，但红点内停留只有 16.33%。
4. `obs_auto_aim` 的“距离加速 + 自适应平滑”值得保留，但必须解决头/身类别回退和输出比例标定后再与 PID 进行最终判断。

这轮是同种子、同配置族的闭环比较，但控制器移动会改变后续画面，所以不是逐帧冻结的完全相同检测序列。若要做最终归因，下一步应录制一次检测框序列，再让三个控制器读取同一序列。

来源：

- [上游 main_aim.py](https://github.com/160037lk/obs-auto-aim/blob/master/main_aim.py)
- [上游 video_bridge.py](https://github.com/160037lk/obs-auto-aim/blob/master/video_bridge.py)
