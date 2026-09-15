# -*- coding: utf-8 -*-
"""参数配置：默认值 + JSON 读写 + 搜索空间定义。

所有关键参数均可通过 config.json 覆盖；optimize.py 的参数搜索基于 SEARCH_SPACE。
"""

import json
import os

DEFAULT = {
    "world": {
        "width": 4000,          # 世界宽度 px（场景平面）
        "height": 4000,         # 世界高度 px
        "spawn_margin": 200,    # 目标出生时距屏幕边缘的最小边距 px
    },
    "screen": {"width": 640, "height": 640},
    "target": {
        "image": "",            # 目标图片路径；空 = 程序生成占位图
        "images": [],           # 目标图片列表（单目标：随机选一张）
        "img_cls_types": [],    # 每张图 cls 类型: "both"(身+头) / "single"(仅单类)，与 images 对齐
        "box_w": 80.0,          # 目标框宽 px（基础值；保持宽高比时按图片比例派生）
        "box_h": 160.0,         # 目标框高 px（基础值，仅作 scale 统计基准）
        "size_h_min": 50.0,     # 目标框高随机下限 px（远目标）
        "img_max_h": [],        # 每张图可检出的最大框高 px（贴脸大目标），与 images 对齐
        "fixed_h": 0.0,         # 0=随机；>0 时强制目标框高（演示/调试用）
        "hp": 150,              # 初始血量
        "mover": "jumpstrafe",  # 移动策略: random4/random8/sine/jitter/teleport/chase/fixeddir/strafe/jumpstrafe
        "speed": 240.0,         # 移动速度上限 px/s（fixeddir 随机 0~speed）
        "dir_change_min": 0.4,  # 方向变化间隔下限 s
        "dir_change_max": 1.2,  # 方向变化间隔上限 s
        "bounce": True,         # 边界反弹
        "spawn_in_crop": True,  # 出生在中心截图区内（yolo 感知用，保证出生即可见）
        "spawn_radius_min": 100.0,  # 环形出生内半径（让目标偏向截图区边缘）
        "spawn_radius_max": 260.0,  # 环形出生外半径
        "disappear_visible_min": 40,  # 随机消失：显身帧数下限（~50帧）
        "disappear_visible_max": 70,  # 显身帧数上限
        "disappear_hidden_min": 8,    # 消失帧数下限（~10帧）
        "disappear_hidden_max": 14,   # 消失帧数上限
    },
    "weapon": {
        "fire_rate_hz": 20.0,   # 射速 20 发/秒（每 50ms 一发）
        "damage": {"head": 40, "mid": 20},  # 命中区域伤害（已取消脚部 low 区）
        "red_dot_offset_px": 15.0,   # 红点(子弹落点)相对屏幕中心的向上偏移 px
        "red_dot_jitter_px": 5.0,    # 开枪时红点抖动半径 px
        "red_dot_recoil_max_px": 30.0,     # 连发后坐力：红点额外上飘上限 px（总偏移≈15+30=45）
        "red_dot_recoil_per_shot_px": 6.0, # 每发子弹红点上飘量 px
        "red_dot_recoil_recover_px": 1.0,  # 停火后每帧回落量 px
    },
    "view": {
        "sensitivity": 1.0,     # 视角灵敏度：1 鼠标 px = 多少世界 px（镜头移动量）
        "fps": 60,              # 逻辑帧率
    },
    "occlusion": {
        "enabled": True,        # 屏幕遮挡：右键 ADS 后枪身盖住屏幕中下部九分之一
        "duration_ms": 350,     # 按下右键到完全遮挡所需时间 ms
        "hold_fire": True,      # 黑框完全出现前不开枪（贴近实战）
    },
    "perception": {
        "mode": "gt_noise",     # gt(精确) / gt_noise(带噪声) / yolo(复用main.py视觉)
        "noise_std_px": 4.0,    # gt_noise: 框中心高斯噪声标准差 px
        "miss_rate": 0.02,      # gt_noise: 单帧丢失概率
        "yolo_model": "../yolodeltav1.onnx",
        "yolo_conf": 0.35,
        "yolo_classes": [0, 1],
        "capture_size": 640,    # yolo 模式：屏幕中心截图区域
    },
    "tracker": {
        "strategy": "p",        # 追踪策略: p / pid / ema / predict
        "gain": 0.35,           # 比例增益：移动 = 误差 × gain
        "cap": 60.0,            # 单帧最大位移 px
        "deadzone": 6.0,        # 误差死区 px（两轴均小于则不动）
        "ema_alpha": 0.26,      # ema 策略：瞄准点 EMA 系数
        "ki": 0.0,              # pid 策略：积分系数
        "kd": 0.0,              # pid 策略：微分系数
        "lead_frames": 0,       # predict 策略：速度外推帧数
        "vel_ema": 0.6,         # predict 策略：速度 EMA 系数
        "aim_ratio": 0.166,     # 瞄准点高度比：0=框顶 0.166≈头部中心 0.42≈胸口 0.5=框中心 1=框底
        "obs_delay_frames": 2,  # 感知延迟（模拟截图+推理+相机响应，约2帧）
        "control_interval_frames": 1,  # 追踪控制周期（帧）
    },
    "shooter": {
        "policy": "always",     # always / lock_delay / oracle_on_aim(on_aim兼容别名)
        "auto_fire": True,      # 自动射击开关
        "reaction_ms": 120.0,    # lock_delay: 稳定锁定后反应时间
        "lost_grace_ms": 50.0,   # lock_delay: 丢失/WAITING容忍时间
    },
    "human_track": {            # 人手跟枪：bot 失明时保持目标在截图区内（不居中，实战目标偏离准星100~260px）
        "enabled": False,       # 默认关（gt 模式不启用）
        "gain": 0.35,           # 拉回强度（目标越出安全区越远拉得越狠）
        "jitter": 0.2,          # 人手抖动（gain 的随机比例）
        "margin": 60.0,         # 截图区边缘安全边距 px：目标进入该区域才拉回
    },
    "episode": {
        "max_seconds": 30.0,    # 回合超时时间（超时=击杀失败）
        "seed": 0,              # 0=随机种子
    },
    "logging": {
        "dir": "results",       # 日志目录
        "per_frame": False,     # 是否逐帧记录
        "frame_interval": 10,   # 逐帧记录间隔（帧）
    },
    "optimize": {
        "iterations": 200,      # 随机搜索轮数
        "seeds": 4,             # 每组参数重复回合数
        "output": "results/search.csv",
        "best": "results/best_config.json",
    },
}

# 参数搜索空间（optimize.py 随机搜索从这里采样）
SEARCH_SPACE = [
    ("tracker.strategy",        "choice",   ["p", "pid", "ema", "predict"]),
    ("tracker.gain",            "uniform",  [0.15, 0.55]),
    ("tracker.cap",             "uniform",  [30.0, 110.0]),
    ("tracker.deadzone",        "uniform",  [0.0, 12.0]),
    ("tracker.aim_ratio",       "uniform",  [0.08, 0.60]),
    ("tracker.lead_frames",     "choice",   [0, 1, 2, 3]),
    ("tracker.ema_alpha",       "uniform",  [0.10, 0.60]),
    ("tracker.ki",              "uniform",  [0.0, 0.4]),
    ("tracker.kd",              "uniform",  [0.0, 0.3]),
    ("tracker.obs_delay_frames", "choice",  [1, 2, 3]),
    ("shooter.policy",          "choice",   ["always", "lock_delay"]),
    ("target.speed",            "uniform",  [60.0, 260.0]),
    ("target.mover",            "choice",   ["random4", "random8", "sine", "jitter"]),
]


def deep_merge(base, override):
    """递归合并 override 到 base（返回全新深拷贝 dict）。

    注意：必须深拷贝——浅拷贝会让未覆盖的嵌套 dict 与 base 共享，
    set_path 修改配置时会原地污染全局 DEFAULT。
    """
    import copy
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path):
    """读取 config.json 并合并默认值。"""
    cfg = DEFAULT
    if path and os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                cfg = deep_merge(DEFAULT, json.load(f))
        except Exception as e:
            print(f"[config] 读取失败，使用默认值: {e}")
    return cfg


def save_config(cfg, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)


def get(cfg, dotted, default=None):
    """按 'a.b.c' 路径读取配置。"""
    cur = cfg
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def set_path(cfg, dotted, value):
    parts = dotted.split(".")
    cur = cfg
    for part in parts[:-1]:
        cur = cur.setdefault(part, {})
    cur[parts[-1]] = value
