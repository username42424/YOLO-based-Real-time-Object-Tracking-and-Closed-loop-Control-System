# -*- coding: utf-8 -*-
"""用局部画面位移自动标定纵向 screen-px / mouse-count。"""

import argparse
import ctypes
import json
import os
import statistics
import time

import cv2
import numpy as np

from mouse_control import _send_relative, set_right_hold


HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(HERE, "config.json")
VK_ESCAPE = 0x1B


def key_down(vk):
    return bool(ctypes.windll.user32.GetAsyncKeyState(vk) & 0x8000)


def grab_gray():
    from PIL import ImageGrab

    image = ImageGrab.grab(all_screens=False).convert("L")
    return np.asarray(image, dtype=np.uint8)


def estimate_vertical_shift(before, after, roi_width=0.80, roi_height=0.55):
    """用正反向光流估计固定场景在屏幕上的局部平移。"""
    if before.shape != after.shape or before.ndim != 2:
        raise ValueError("before/after 必须是同尺寸灰度图")

    height, width = before.shape
    mask = np.zeros_like(before, dtype=np.uint8)
    roi_w = max(32, int(width * float(roi_width)))
    roi_h = max(32, int(height * float(roi_height)))
    x1 = max(0, (width - roi_w) // 2)
    x2 = min(width, x1 + roi_w)
    y1 = max(0, int(height * 0.10))
    y2 = min(height, y1 + roi_h)
    mask[y1:y2, x1:x2] = 255

    points0 = cv2.goodFeaturesToTrack(
        before,
        maxCorners=600,
        qualityLevel=0.01,
        minDistance=8,
        blockSize=7,
        mask=mask,
    )
    if points0 is None or len(points0) < 30:
        raise RuntimeError("画面纹理不足：请对准墙角、门框或带纹理的固定场景")

    lk = {
        "winSize": (31, 31),
        "maxLevel": 4,
        "criteria": (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            40,
            0.01,
        ),
    }
    points1, status1, _ = cv2.calcOpticalFlowPyrLK(
        before, after, points0, None, **lk
    )
    if points1 is None:
        raise RuntimeError("无法跟踪画面特征")
    points0_back, status2, _ = cv2.calcOpticalFlowPyrLK(
        after, before, points1, None, **lk
    )
    if points0_back is None:
        raise RuntimeError("无法完成反向画面校验")

    p0 = points0.reshape(-1, 2)
    p1 = points1.reshape(-1, 2)
    p0_back = points0_back.reshape(-1, 2)
    valid = (
        (status1.reshape(-1) > 0)
        & (status2.reshape(-1) > 0)
        & (np.linalg.norm(p0_back - p0, axis=1) <= 1.5)
    )
    displacement = p1[valid] - p0[valid]
    if len(displacement) < 24:
        raise RuntimeError("有效跟踪点不足：请换一个纹理更明显的固定场景")

    dx = displacement[:, 0]
    dy = displacement[:, 1]
    median_y = float(np.median(dy))
    deviations = np.abs(dy - median_y)
    mad = float(np.median(deviations))
    tolerance = max(2.0, 4.0 * 1.4826 * mad)
    inliers = deviations <= tolerance
    if int(inliers.sum()) < 20:
        raise RuntimeError("场景运动不一致：标定时请保持人物和鼠标不动")

    return {
        "dx": float(np.median(dx[inliers])),
        "dy": float(np.median(dy[inliers])),
        "tracked": int(inliers.sum()),
        "mad_y": mad,
    }


def robust_scale(samples):
    """剔除不一致的测量轮次后返回中位标定值。"""
    values = [float(value) for value in samples if float(value) > 0.0]
    if len(values) < 3:
        raise ValueError("至少需要3个有效样本")
    median = statistics.median(values)
    deviations = [abs(value - median) for value in values]
    mad = statistics.median(deviations)
    tolerance = max(0.01, median * 0.12, 3.0 * 1.4826 * mad)
    kept = [value for value in values if abs(value - median) <= tolerance]
    if len(kept) < 3:
        raise ValueError("样本一致性不足")
    return statistics.median(kept), kept


def apply_scale_config(config, value):
    """把同一实测尺度写到两条会被现有控制路径读取的配置中。"""
    value = round(float(value), 4)
    config.setdefault("aim_control", {})["view_scale_y"] = value
    config.setdefault("unit", {})["px_per_count_y"] = value
    return value


def write_scale(config, value):
    value = apply_scale_config(config, value)
    with open(CONFIG_PATH, "w", encoding="utf-8") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return value


def main():
    parser = argparse.ArgumentParser(
        description="用自动小幅正反脉冲标定纵向 screen-px/mouse-count"
    )
    parser.add_argument(
        "--counts", type=int, default=120, help="每次脉冲的鼠标 count，默认120"
    )
    parser.add_argument(
        "--trials", type=int, default=8, help="测量次数，方向自动交替，默认8"
    )
    parser.add_argument(
        "--settle", type=float, default=0.16, help="发送后等待画面稳定的秒数"
    )
    parser.add_argument(
        "--dir", choices=("down", "up"), default="down", help="第一次脉冲方向"
    )
    parser.add_argument("--ads", action="store_true", help="标定期间自动按住鼠标右键")
    parser.add_argument(
        "--write",
        action="store_true",
        help="同时写入 view_scale_y 和 px_per_count_y",
    )
    args = parser.parse_args()

    counts = max(20, abs(int(args.counts)))
    trials = max(4, int(args.trials))
    settle = max(0.05, float(args.settle))
    direction = 1 if args.dir == "down" else -1
    with open(CONFIG_PATH, encoding="utf-8") as handle:
        config = json.load(handle)
    current_view = float(config.get("aim_control", {}).get("view_scale_y", 0.0))
    current_unit = float(config.get("unit", {}).get("px_per_count_y", 0.0))

    print("=== 纵向局部自动标定 ===")
    print(
        f"当前 view_scale_y={current_view:.4f}, "
        f"px_per_count_y={current_unit:.4f}"
    )
    print(f"每次 {counts} counts，{trials} 次，正反方向交替，等待 {settle:.2f}s")
    print("请站着不动，把视角放在上下极限之间，并对准有纹理的墙面/门框。")
    print("不需要点击鼠标；脚本会自动测量并等量回到原视角。按 ESC 中止。")
    print("3秒后开始……")
    for remaining in range(3, 0, -1):
        print(f"  {remaining}...")
        time.sleep(1.0)

    samples = []
    rejected = 0
    if args.ads and not set_right_hold(True):
        print("[失败] 无法按下右键，请确认程序权限。")
        return
    if args.ads:
        # 等待开镜动画和FOV变化结束，否则第一轮会把缩放动画误当成鼠标位移。
        time.sleep(0.65)
    try:
        for index in range(trials):
            if key_down(VK_ESCAPE):
                print("[中止] ESC")
                break

            pulse = direction * counts
            motion = None
            sent = False
            time.sleep(0.05)
            before = grab_gray()
            try:
                if not _send_relative(0, pulse):
                    raise RuntimeError("SendInput 失败")
                sent = True
                time.sleep(settle)
                after = grab_gray()
                motion = estimate_vertical_shift(before, after)
            finally:
                if sent:
                    if not _send_relative(0, -pulse):
                        raise RuntimeError("回位 SendInput 失败")
                    time.sleep(settle)

            shift = abs(motion["dy"])
            horizontal = abs(motion["dx"])
            scale = shift / counts
            reason = None
            if shift < 4.0:
                reason = "位移过小，可能碰到俯仰极限或游戏未聚焦"
            elif shift > min(before.shape[0] * 0.30, 180.0):
                reason = "位移过大，请降低 --counts"
            elif horizontal > max(4.0, shift * 0.35):
                reason = "横向污染过大，请勿移动鼠标"

            if reason:
                rejected += 1
                print(
                    f"  [{index + 1}/{trials}] 剔除：{reason} "
                    f"dx={motion['dx']:+.1f} dy={motion['dy']:+.1f}"
                )
            else:
                samples.append(scale)
                print(
                    f"  [{index + 1}/{trials}] dy={motion['dy']:+.2f}px "
                    f"dx={motion['dx']:+.2f}px 点={motion['tracked']} "
                    f"=> {scale:.4f} px/count"
                )
            direction = -direction
    except Exception as exc:
        print(f"[失败] {exc}")
    finally:
        if args.ads:
            set_right_hold(False)

    print(f"\n有效样本={len(samples)}，剔除={rejected}")
    try:
        scale, kept = robust_scale(samples)
    except ValueError as exc:
        print(f"无法生成结果：{exc}。请在视角中段、更明显的纹理前重试。")
        return

    spread = ((max(kept) - min(kept)) / scale) if scale else 0.0
    print("================ 结果 ================")
    print("保留样本：" + ", ".join(f"{value:.4f}" for value in kept))
    print(f"纵向实测 = {scale:.4f} screen-px / mouse-count")
    print(f"样本极差 = {spread:.1%}")
    if spread > 0.18:
        print("警告：样本离散偏大，建议换纹理更明显的场景重跑，不要写入。")

    if args.write:
        if spread > 0.25:
            print("离散超过25%，为避免写入错误标定，本次拒绝 --write。")
            return
        written = write_scale(config, scale)
        print(
            f"已写入 config.json: aim_control.view_scale_y={written:.4f}, "
            f"unit.px_per_count_y={written:.4f}"
        )
    else:
        print("未写入配置；确认结果稳定后加 --write。")


if __name__ == "__main__":
    main()
