"""
cell_pipeline_v2.py
===================
细胞变形分析流水线 v2

用法：
    python cell_pipeline_v2.py --video path/to/video.avi [--roi x1 y1 x2 y2] [--fps 2000]
    python cell_pipeline_v2.py --video "G:/work/260416_RTDC/20260415-mcf10A-2500fps-shuttertime394us/avi/100atm-1600atm-1 - frames 26058-26414.avi"  --fps 2500

    输出：
    results/{video_name}_per_frame.csv   — 每帧参数
    results/{video_name}_per_cell.csv    — 每细胞汇总（SVM输入）
    results/{video_name}_debug.mp4       — 可视化视频（可选）--debug

运行前先执行 validate_new_video.py 确认参数。
"""

import cv2
import numpy as np
import pandas as pd
import argparse
import os
import json
from pathlib import Path
from scipy.optimize import curve_fit, linear_sum_assignment

MAX_SQUEEZE_ZONES = 9  # 最大挤压区数量，不足的列填 NaN
SQUEEZE_ZONE_WIDTH_PX = 55  # 挤压区矩形固定宽度（原始帧像素），保证各区等宽可比


# ─────────────────────────────────────────────
# 配置（根据 validate_new_video.py 的输出调整）
# ─────────────────────────────────────────────
class Config:
    # 视频
    fps: float = 2500.0          # 实际帧率，视频传来后确认
    px_to_um: float = 1.25        # 像素→微米，需根据芯片通道宽度标定

    # 背景建模
    bg_n_frames: int = 100       # 采样帧数

    # 检测
    diff_threshold = 20        # None=自动取5；或填整数手动指定（低对比度视频建议5-8）
    min_area: int = 200          # px²，从validate脚本获取
    max_area: int = 5000         # px²
    min_circularity: float = 0.20
    max_aspect_ratio: float = 5.0

    # 细胞分类
    cluster_area_threshold: int = 8000   # px²，超过此值视为细胞团
    min_cell_area: int = 200             # px²，低于此值视为碎片

    # 追踪
    max_distance: int = 50       # px，初始最大匹配距离
    max_gap: int = 20             # 允许丢失的最大帧数
    min_track_length: int = 15   # 最短有效轨迹帧数

    # 通道ROI（None=全帧；或 (x1,y1,x2,y2)）
    channel_roi = None

    # 有效统计的x坐标范围（None=全帧宽度）
    # 只有细胞质心在 x_stat_min <= cx <= x_stat_max 范围内才计入统计
    x_stat_min: int = None
    x_stat_max: int = None

    # 输出
    output_dir: str = "results"
    save_debug_video: bool = False
    profile_long_step_um: float = 2.0


# ─────────────────────────────────────────────
# 背景建模
# ─────────────────────────────────────────────
def build_background(cap, n_frames=100):
    """顺序读取并均匀采样n_frames帧取中位数，建静态背景。

    注意：不用 cap.set(POS_FRAMES, idx) 随机跳帧——对于大体积 AVI
    （高速相机录制，单文件可达数 GB~数十 GB），随机 seek 会触发
    ffmpeg 30s 读取超时而卡死。改为从头顺序读、按固定间隔采样。
    """
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # 采样间隔：在前 total 帧里均匀取约 n_frames 个样本
    step = max(1, total // max(1, n_frames))
    frames = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            if len(frames) >= n_frames:
                break
        idx += 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    bg = np.median(frames, axis=0).astype(np.uint8)
    print(f"  背景建模完成，采样 {len(frames)} 帧（顺序读，间隔 {step} 帧）")
    return bg


# ─────────────────────────────────────────────
# 分水岭分割：分离粘连区域
# ─────────────────────────────────────────────
def _watershed_split(fg_mask: np.ndarray) -> np.ndarray:
    """
    对二值前景图做分水岭分割，分离粘连的细胞。
    原理：距离变换找各区域的局部极大值（细胞中心），以这些点为种子向外生长，
    遇到两个种子的边界时停止，从而切开粘连区域。
    """
    dist = cv2.distanceTransform(fg_mask, cv2.DIST_L2, 5)

    # 局部极大值作为前景种子：取距离图中 > 0.4 * 全局最大值的区域
    # 0.4 是经验值：太高会漏掉小细胞，太低会把一个细胞切成多块
    _, sure_fg = cv2.threshold(dist, 0.4 * dist.max(), 255, cv2.THRESH_BINARY)
    sure_fg = sure_fg.astype(np.uint8)

    # 确定背景（膨胀后取反）
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    sure_bg = cv2.dilate(fg_mask, kernel, iterations=2)

    # 未知区域（边界）
    unknown = cv2.subtract(sure_bg, sure_fg)

    # 连通组件标记种子
    _, markers = cv2.connectedComponents(sure_fg)
    markers += 1                    # 背景从1开始，0留给未知区域
    markers[unknown == 255] = 0

    # 分水岭需要3通道图像
    img_3ch = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)
    cv2.watershed(img_3ch, markers)

    # markers == -1 是分水岭边界，其余正整数是各区域标签
    result = np.zeros_like(fg_mask)
    result[markers > 1] = 255       # 标签1是背景，>1是细胞区域
    return result


# ─────────────────────────────────────────────
# 单帧检测
# ─────────────────────────────────────────────
def detect_cells_single_frame(frame_gray, background, config: Config):
    """
    返回检测列表，每项为 dict：
      cx, cy, area, perimeter, circularity, diameter_px,
      major_axis, minor_axis, bbox, contour, cell_type
    """
    # 应用ROI
    if config.channel_roi:
        x1, y1, x2, y2 = config.channel_roi
        roi_gray = frame_gray[y1:y2, x1:x2]
        roi_bg = background[y1:y2, x1:x2]
        offset = (x1, y1)
    else:
        roi_gray = frame_gray
        roi_bg = background
        offset = (0, 0)

    # 帧差
    diff = cv2.absdiff(roi_gray, roi_bg)

    # 阈值：手动指定优先；否则固定 10（5 太低会把光晕圈进去，10 更贴近细胞边界）
    thr = config.diff_threshold if config.diff_threshold is not None else 10
    _, fg_mask = cv2.threshold(diff, thr, 255, cv2.THRESH_BINARY)

    # 闭运算（核缩小到 9x9，减少桥接相邻细胞的概率）；开运算去孤立噪点
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    kernel_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel_close)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN,  kernel_open)

    # 凸包填充：填补亮环内部缺口
    cnts_raw, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    fg_mask = np.zeros_like(fg_mask)
    for c in cnts_raw:
        hull = cv2.convexHull(c)
        cv2.drawContours(fg_mask, [hull], -1, 255, cv2.FILLED)

    # 分水岭分割：分离凸包后仍粘连的区域
    fg_mask = _watershed_split(fg_mask)

    # 轮廓提取（CHAIN_APPROX_NONE保留所有点，用于精确椭圆拟合）
    contours, _ = cv2.findContours(
        fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )

    detections = []
    frame_h, frame_w = frame_gray.shape

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 10:
            continue

        perimeter = cv2.arcLength(cnt, True)
        if perimeter < 1:
            continue

        circularity = 4 * np.pi * area / (perimeter ** 2)
        M = cv2.moments(cnt)
        if M["m00"] < 1e-6:
            continue

        cx = M["m10"] / M["m00"] + offset[0]
        cy = M["m01"] / M["m00"] + offset[1]
        x, y, w, h = cv2.boundingRect(cnt)
        x += offset[0]
        y += offset[1]

        # 椭圆拟合（需要>=5个点）
        if len(cnt) >= 5:
            ellipse = cv2.fitEllipse(cnt)
            (_, _), (ax1, ax2), _ = ellipse
            major_axis = max(ax1, ax2)
            minor_axis = min(ax1, ax2)
        else:
            major_axis = float(max(w, h))
            minor_axis = float(min(w, h))

        diameter_px = 2 * np.sqrt(area / np.pi)

        # 轮廓坐标加上offset
        cnt_global = cnt.copy()
        cnt_global[:, :, 0] += offset[0]
        cnt_global[:, :, 1] += offset[1]

        det = {
            "cx": cx,
            "cy": cy,
            "area": area,
            "perimeter": perimeter,
            "circularity": circularity,
            "diameter_px": diameter_px,
            "major_axis": major_axis,
            "minor_axis": minor_axis,
            "bbox": (x, y, w, h),
            "contour": cnt_global,
        }
        det["cell_type"] = classify_detection(det, frame_w, frame_h, config)
        detections.append(det)

    return detections


# ─────────────────────────────────────────────
# 检测分类
# ─────────────────────────────────────────────
def classify_detection(det, frame_w, frame_h, config: Config):
    """返回 'cell' / 'cluster' / 'debris' / 'artifact'"""
    area = det["area"]
    circ = det["circularity"]
    x, y, w, h = det["bbox"]
    aspect = det["major_axis"] / (det["minor_axis"] + 1e-6)

    # 触碰边界 → 通道壁伪影
    if x <= 2 or y <= 2 or x + w >= frame_w - 2 or y + h >= frame_h - 2:
        return "artifact"

    # x坐标统计范围过滤（细胞质心不在有效区间内 → 不统计）
    cx = det["cx"]
    if config.x_stat_min is not None and cx < config.x_stat_min:
        return "out_of_roi"
    if config.x_stat_max is not None and cx > config.x_stat_max:
        return "out_of_roi"

    # 面积过大 → 细胞团
    if area > config.cluster_area_threshold:
        return "cluster"

    # 面积过小或圆度过低 → 碎片
    if area < config.min_cell_area or circ < config.min_circularity:
        return "debris"

    # 过于细长 → 碎片或伪影
    if aspect > config.max_aspect_ratio:
        return "debris"

    return "cell"


# ─────────────────────────────────────────────
# 每帧参数提取
# ─────────────────────────────────────────────
def extract_per_frame_params(det, frame_idx, fps):
    """从单帧检测结果提取参数，返回 dict。"""
    major = det["major_axis"]
    minor = det["minor_axis"]
    D = major / (minor + 1e-6)
    aspect_ratio = D  # same formula: major/minor
    diameter_px = det["diameter_px"]

    return {
        "frame": frame_idx,
        "time_s": round(frame_idx / fps, 6),
        "cell_id": None,          # 由追踪器填入
        "circularity": round(det["circularity"], 4),
        "aspect_ratio": round(aspect_ratio, 4),
        "deformation_D": round(D, 4),
        "diameter_px": round(diameter_px, 2),
        "area_px": round(det["area"], 1),
        "centroid_x_px": round(det["cx"], 1),
        "centroid_y_px": round(det["cy"], 1),
        "major_axis_px": round(major, 2),
        "minor_axis_px": round(minor, 2),
        "cell_type": det["cell_type"],
    }


# ─────────────────────────────────────────────
# 细胞汇总（变形曲线 + 黏弹性拟合）
# ─────────────────────────────────────────────
def _fit_tau(t_arr, D_arr):
    """对恢复段拟合指数衰减，返回 (tau_s, D_residual) 或 (nan, nan)。"""
    if len(t_arr) < 5:
        return np.nan, np.nan
    def exp_decay(t, D0, tau, D_inf):
        return D_inf + (D0 - D_inf) * np.exp(-t / (tau + 1e-9))
    try:
        p0 = [D_arr[0], 0.005, D_arr[-1]]
        bounds = ([0, 1e-5, 0], [1.0, 1.0, 1.0])
        popt, _ = curve_fit(exp_decay, t_arr, D_arr, p0=p0, bounds=bounds, maxfev=3000)
        return float(popt[1]), float(popt[2])
    except Exception:
        return np.nan, np.nan


def extract_cell_summary(track_df, cell_id, px_to_um, squeeze_zones=None,
                         channel_widths_um=None):
    """
    输入：单个细胞的所有帧数据（DataFrame）
    squeeze_zones:      list of (x_start, x_end)，收缩口 x 像素范围列表
    channel_widths_um:  list of float，每个收缩口的通道最窄宽度（微米），与 squeeze_zones 一一对应
    输出：一行汇总 dict（per_cell.csv）
    """
    D_curve = track_df["deformation_D"].values
    circ    = track_df["circularity"].values
    t       = track_df["time_s"].values
    cx      = track_df["centroid_x_px"].values
    fps_dt  = float(t[1] - t[0]) if len(t) > 1 else 0.0

    # ── 挤压前基线帧（进入第一个挤压区之前）────────────────────
    if squeeze_zones:
        first_sq_x = squeeze_zones[0][0]
        pre_mask = cx < first_sq_x
        ref_df = track_df[pre_mask] if pre_mask.sum() >= 3 else pd.DataFrame()
        pre_free_time_ms = round(float(pre_mask.sum() * fps_dt * 1000), 3) if pre_mask.sum() >= 1 else np.nan
    else:
        ref_df = track_df
        pre_free_time_ms = round(float(len(track_df) * fps_dt * 1000), 3) if len(track_df) >= 1 else np.nan

    if len(ref_df) >= 1:
        diameter_um      = float(ref_df["diameter_px"].median()) * px_to_um
        area_um2         = float(ref_df["area_px"].median()) * (px_to_um ** 2)
        circ_baseline    = float(ref_df["circularity"].mean())
    else:
        # 细胞进入视野时已在挤压区内
        diameter_um   = np.nan
        area_um2      = np.nan
        circ_baseline = np.nan

    # ── 全局 D_max ────────────────────────────────────────────
    D_max_global = float(D_curve.max())

    # ── 逐挤压区统计 ──────────────────────────────────────────
    n_squeezes_passed = 0
    row = {
        "cell_id":            cell_id,
        "n_frames":           len(track_df),
        "diameter_um":        round(diameter_um, 3) if not np.isnan(diameter_um) else np.nan,
        "area_um2":           round(area_um2, 3)    if not np.isnan(area_um2)    else np.nan,
        "circ_baseline":      round(circ_baseline, 4) if not np.isnan(circ_baseline) else np.nan,
        "D_max_global":       round(D_max_global, 4),
        "n_squeezes_passed":  0,
        "pre_free_time_ms":   pre_free_time_ms,
        "label":              "",
    }

    # 恢复目标圆度阈值（初始圆度 × 0.95）
    circ_recovery_target = circ_baseline * 0.95 if not np.isnan(circ_baseline) else np.nan

    # 自由区边界（与 extract_cell_profile 口径统一：本区结束→下一区开始，
    # 最后一个区用前面自由区宽度平均值外推）
    zones_sorted = sorted(squeeze_zones or [], key=lambda z: z[0])
    free_bounds = _free_region_bounds(zones_sorted)

    for i in range(MAX_SQUEEZE_ZONES):
        prefix = f"sq{i+1}"
        if zones_sorted and i < len(zones_sorted):
            x_lo, x_hi = zones_sorted[i]
            sq_mask = (cx >= x_lo) & (cx <= x_hi)
            sq_idx  = np.where(sq_mask)[0]

            if len(sq_idx) >= 2:
                n_squeezes_passed += 1
                D_sq   = D_curve[sq_mask]
                t_sq   = t[sq_mask]
                circ_sq = circ[sq_mask]
                local_max_idx = int(np.argmax(D_sq))

                sq_D_max     = round(float(D_sq.max()), 4)
                sq_circ_min  = round(float(circ_sq.min()), 4)
                sq_t_max_ms  = round(float((t_sq[local_max_idx] - t_sq[0]) * 1000), 3)
                sq_time_ms   = round(float(len(sq_idx) * fps_dt * 1000), 3)

                # 自由区：本区结束 → 下一区开始（最后一区用平均宽度外推）
                free_x_lo, free_x_hi = free_bounds[i]
                free_mask = (cx > free_x_lo) & (cx < free_x_hi)
                free_idx = np.where(free_mask)[0]

                # 区段长度（um）：挤压区 x 长度、自由区 x 长度
                squeeze_len_um = round(float((x_hi - x_lo) * px_to_um), 3)
                free_len_um    = round(float((free_x_hi - free_x_lo) * px_to_um), 3)

                if len(free_idx) >= 1:
                    free_time_ms = round(float(len(free_idx) * fps_dt * 1000), 3)
                    # 恢复时间：自由区内圆度首次 >= circ_recovery_target 的时间
                    if not np.isnan(circ_recovery_target):
                        circ_free = circ[free_idx]
                        t_free    = t[free_idx]
                        t_dmax    = t_sq[local_max_idx]
                        recovered = np.where(
                            (t_free >= t_dmax) & (circ_free >= circ_recovery_target)
                        )[0]
                        if len(recovered) > 0:
                            recovery_time_ms = round(
                                float((t_free[recovered[0]] - t_dmax) * 1000), 3
                            )
                        else:
                            recovery_time_ms = np.nan
                    else:
                        recovery_time_ms = np.nan
                else:
                    free_time_ms     = np.nan
                    recovery_time_ms = np.nan

                row[f"{prefix}_channel_width_um"] = (
                    round(channel_widths_um[i], 3)
                    if channel_widths_um and i < len(channel_widths_um)
                       and not np.isnan(channel_widths_um[i])
                    else np.nan
                )
                row[f"{prefix}_D_max"]           = sq_D_max
                row[f"{prefix}_circ_min"]        = sq_circ_min
                row[f"{prefix}_t_max_ms"]        = sq_t_max_ms
                row[f"{prefix}_squeeze_time_ms"] = sq_time_ms
                row[f"{prefix}_squeeze_len_um"]  = squeeze_len_um
                row[f"{prefix}_free_time_ms"]    = free_time_ms
                row[f"{prefix}_free_len_um"]     = free_len_um
                row[f"{prefix}_recovery_time_ms"] = recovery_time_ms
            else:
                # 细胞进入视野时已在该挤压区内，或该区帧数不足
                row[f"{prefix}_channel_width_um"]  = (
                    round(channel_widths_um[i], 3)
                    if channel_widths_um and i < len(channel_widths_um)
                       and not np.isnan(channel_widths_um[i])
                    else np.nan
                )
                row[f"{prefix}_D_max"]            = np.nan
                row[f"{prefix}_circ_min"]         = np.nan
                row[f"{prefix}_t_max_ms"]         = np.nan
                row[f"{prefix}_squeeze_time_ms"]  = np.nan
                row[f"{prefix}_squeeze_len_um"]   = round(float((x_hi - x_lo) * px_to_um), 3)
                row[f"{prefix}_free_time_ms"]     = np.nan
                row[f"{prefix}_free_len_um"]      = round(float((free_bounds[i][1] - free_bounds[i][0]) * px_to_um), 3)
                row[f"{prefix}_recovery_time_ms"] = np.nan
        else:
            row[f"{prefix}_channel_width_um"]  = np.nan
            row[f"{prefix}_D_max"]             = np.nan
            row[f"{prefix}_circ_min"]          = np.nan
            row[f"{prefix}_t_max_ms"]          = np.nan
            row[f"{prefix}_squeeze_time_ms"]   = np.nan
            row[f"{prefix}_squeeze_len_um"]    = np.nan
            row[f"{prefix}_free_time_ms"]      = np.nan
            row[f"{prefix}_free_len_um"]       = np.nan
            row[f"{prefix}_recovery_time_ms"]  = np.nan

    row["n_squeezes_passed"] = n_squeezes_passed
    return row


# ─────────────────────────────────────────────
# 变形轮廓表（空间采样）
# ─────────────────────────────────────────────
def _spatial_sample(cx, values, x_lo, x_hi, n_pts):
    """
    在 x 坐标范围 [x_lo, x_hi] 内均匀划分 n_pts 个采样位置，
    对每个采样位置取最近帧的值（若该范围内无数据则全 NaN）。
    返回长度为 n_pts 的列表。
    """
    if x_lo >= x_hi or len(cx) == 0:
        return [np.nan] * n_pts
    sample_xs = np.linspace(x_lo, x_hi, n_pts)
    result = []
    mask = (cx >= x_lo) & (cx <= x_hi)
    cx_seg = cx[mask]
    val_seg = values[mask]
    if len(cx_seg) == 0:
        return [np.nan] * n_pts
    for sx in sample_xs:
        nearest = int(np.argmin(np.abs(cx_seg - sx)))
        result.append(float(val_seg[nearest]))
    return result


def _free_region_bounds(squeeze_zones):
    """给定挤压区列表，返回每个挤压区对应自由区的 (x_lo, x_hi)。

    自由区定义：第 i 个挤压区结束(x_hi) → 第 i+1 个挤压区开始(x_lo)。
    最后一个挤压区没有"下一个"，用前面各自由区宽度的平均值外推：
      x_hi + mean(前面自由区宽度)。
    返回 list，与排序后的 zones 一一对应。
    """
    zones = sorted(squeeze_zones or [], key=lambda z: z[0])
    n = len(zones)
    if n == 0:
        return []
    free_widths = [zones[i + 1][0] - zones[i][1] for i in range(n - 1)]
    avg_free = float(np.mean(free_widths)) if free_widths else 0.0
    bounds = []
    for i in range(n):
        x_hi = float(zones[i][1])
        x_next = float(zones[i + 1][0]) if i + 1 < n else x_hi + avg_free
        bounds.append((x_hi, x_next))
    return bounds


def extract_cell_profile(track_df, cell_id, squeeze_zones=None):
    """
    生成空间采样变形轮廓（per_cell_profile.csv 的一行）。

    采样规则（按 x 坐标均匀划分，消除流速差异）：
      - 挤压前自由区（轨迹起点 → 第一挤压区左边界）：5 个点
      - 每个挤压区（x_lo ~ x_hi）：9 个点（奇数，中间点对应最窄处）
      - 每个挤压区后的恢复区（x_hi ~ x_hi + 挤压区宽度）：9 个点
    每个采样点提取 D 值和圆度，列名格式：
      pre_D_1..5, pre_C_1..5
      sq1_D_1..9, sq1_C_1..9, sq1_rec_D_1..9, sq1_rec_C_1..9
      ...
    若某区域内无数据则填 NaN。
    """
    D_curve = track_df["deformation_D"].values
    circ    = track_df["circularity"].values
    cx      = track_df["centroid_x_px"].values

    row = {"cell_id": cell_id}

    # ── 挤压前自由区（5点）────────────────────────────────────
    if squeeze_zones:
        pre_x_hi = squeeze_zones[0][0]
        pre_x_lo = float(cx.min())
    else:
        pre_x_lo = float(cx.min())
        pre_x_hi = float(cx.max())

    pre_D = _spatial_sample(cx, D_curve, pre_x_lo, pre_x_hi, 5)
    pre_C = _spatial_sample(cx, circ,    pre_x_lo, pre_x_hi, 5)
    for k in range(5):
        row[f"pre_D_{k+1}"] = round(pre_D[k], 4) if not np.isnan(pre_D[k]) else np.nan
        row[f"pre_C_{k+1}"] = round(pre_C[k], 4) if not np.isnan(pre_C[k]) else np.nan

    # ── 逐挤压区（9点）+ 恢复区（9点）────────────────────────
    # 恢复区(rec)= 本挤压区结束 → 下一个挤压区开始（最后一个用平均自由区宽度外推）
    zones_sorted = sorted(squeeze_zones or [], key=lambda z: z[0])
    free_bounds = _free_region_bounds(zones_sorted)
    for i in range(MAX_SQUEEZE_ZONES):
        prefix = f"sq{i+1}"
        if zones_sorted and i < len(zones_sorted):
            x_lo, x_hi = zones_sorted[i]

            # 挤压区 9 点
            sq_D = _spatial_sample(cx, D_curve, x_lo, x_hi, 9)
            sq_C = _spatial_sample(cx, circ,    x_lo, x_hi, 9)
            for k in range(9):
                row[f"{prefix}_D_{k+1}"] = round(sq_D[k], 4) if not np.isnan(sq_D[k]) else np.nan
                row[f"{prefix}_C_{k+1}"] = round(sq_C[k], 4) if not np.isnan(sq_C[k]) else np.nan

            # 恢复区（自由区）：本区结束 → 下一区开始，均匀 9 点
            rec_x_lo, rec_x_hi = free_bounds[i]
            rec_D = _spatial_sample(cx, D_curve, rec_x_lo, rec_x_hi, 9)
            rec_C = _spatial_sample(cx, circ,    rec_x_lo, rec_x_hi, 9)
            for k in range(9):
                row[f"{prefix}_rec_D_{k+1}"] = round(rec_D[k], 4) if not np.isnan(rec_D[k]) else np.nan
                row[f"{prefix}_rec_C_{k+1}"] = round(rec_C[k], 4) if not np.isnan(rec_C[k]) else np.nan
        else:
            for k in range(9):
                row[f"{prefix}_D_{k+1}"]     = np.nan
                row[f"{prefix}_C_{k+1}"]     = np.nan
                row[f"{prefix}_rec_D_{k+1}"] = np.nan
                row[f"{prefix}_rec_C_{k+1}"] = np.nan

    d_cols = [c for c in row if c != "cell_id" and "_D_" in c]
    c_cols = [c for c in row if c != "cell_id" and "_C_" in c]
    other_cols = [c for c in row if c != "cell_id" and c not in d_cols and c not in c_cols]

    return {
        "cell_id": row["cell_id"],
        **{c: row[c] for c in d_cols},
        **{c: row[c] for c in c_cols},
        **{c: row[c] for c in other_cols},
    }


def _spatial_sample_long_region(track_df, cell_id, region_type, region_name,
                                x_lo, x_hi, px_to_um, step_um,
                                squeeze_id=np.nan, free_id=np.nan):
    if x_lo >= x_hi or step_um <= 0 or px_to_um <= 0:
        return []

    cx = track_df["centroid_x_px"].values
    mask = (cx >= x_lo) & (cx <= x_hi)
    if mask.sum() == 0:
        return []

    step_px = step_um / px_to_um
    sample_xs = list(np.arange(x_lo, x_hi + 1e-9, step_px))
    if not sample_xs or sample_xs[-1] < x_hi:
        sample_xs.append(float(x_hi))

    seg_df = track_df.loc[mask].copy()
    seg_cx = seg_df["centroid_x_px"].values
    rows = []

    for point_idx, sample_x in enumerate(sample_xs, start=1):
        nearest = int(np.argmin(np.abs(seg_cx - sample_x)))
        nearest_row = seg_df.iloc[nearest]
        rows.append({
            "cell_id": cell_id,
            "region_type": region_type,
            "region_name": region_name,
            "squeeze_id": squeeze_id,
            "free_id": free_id,
            "point_idx": point_idx,
            "x_region_start_px": round(float(x_lo), 3),
            "x_region_end_px": round(float(x_hi), 3),
            "x_sample_px": round(float(sample_x), 3),
            "x_rel_um": round(float((sample_x - x_lo) * px_to_um), 3),
            "x_abs_um": round(float(sample_x * px_to_um), 3),
            "nearest_frame": int(nearest_row["frame"]),
            "nearest_x_px": round(float(nearest_row["centroid_x_px"]), 3),
            "nearest_time_s": round(float(nearest_row["time_s"]), 6),
            "deformation_D": round(float(nearest_row["deformation_D"]), 4),
            "circularity": round(float(nearest_row["circularity"]), 4),
        })

    return rows


def extract_cell_profile_long(track_df, cell_id, px_to_um, squeeze_zones=None,
                              step_um=2.0):
    """
    生成 long-format 空间采样表。

    每行对应一个空间采样点，并标注该点属于挤压前自由区、某个挤压区、
    或某个挤压区后的自由/恢复区。
    """
    cx = track_df["centroid_x_px"].values
    if len(cx) == 0:
        return []

    rows = []
    zones = sorted(squeeze_zones or [], key=lambda z: z[0])

    if not zones:
        return _spatial_sample_long_region(
            track_df, cell_id, "pre_free", "pre_free",
            float(cx.min()), float(cx.max()), px_to_um, step_um,
            squeeze_id=np.nan, free_id=0,
        )

    pre_rows = _spatial_sample_long_region(
        track_df, cell_id, "pre_free", "pre_free",
        float(cx.min()), float(zones[0][0]), px_to_um, step_um,
        squeeze_id=np.nan, free_id=0,
    )
    rows.extend(pre_rows)

    # 自由区口径与 extract_cell_profile 统一：本区结束→下一区开始，
    # 最后一个区用前面自由区宽度平均值外推（见 _free_region_bounds）
    free_bounds = _free_region_bounds(zones)
    for i, (x_lo, x_hi) in enumerate(zones):
        squeeze_id = i + 1
        rows.extend(_spatial_sample_long_region(
            track_df, cell_id, "squeeze", f"sq{squeeze_id}",
            float(x_lo), float(x_hi), px_to_um, step_um,
            squeeze_id=squeeze_id, free_id=np.nan,
        ))

        free_x_lo, free_x_hi = free_bounds[i]

        rows.extend(_spatial_sample_long_region(
            track_df, cell_id, "free", f"free{squeeze_id}",
            float(free_x_lo), float(free_x_hi), px_to_um, step_um,
            squeeze_id=squeeze_id, free_id=squeeze_id,
        ))

    return rows


# ─────────────────────────────────────────────
# 追踪器
# ─────────────────────────────────────────────
class Track:
    def __init__(self, track_id, det, frame_idx):
        self.track_id = track_id
        self.frames = [frame_idx]
        self.positions = [(det["cx"], det["cy"])]
        self.areas = [det["area"]]
        self.miss_count = 0
        self.is_active = True
        self.frame_data = []  # 每帧参数列表

    def predict_position(self):
        """用最近3帧的速度预测下一帧位置。"""
        if len(self.positions) < 2:
            return self.positions[-1]
        n = min(3, len(self.positions))
        vx = (self.positions[-1][0] - self.positions[-n][0]) / (n - 1 + 1e-6)
        vy = (self.positions[-1][1] - self.positions[-n][1]) / (n - 1 + 1e-6)
        return (self.positions[-1][0] + vx, self.positions[-1][1] + vy)

    def adaptive_distance_threshold(self, config: Config):
        """根据近期位移自适应调整匹配距离阈值。"""
        if len(self.positions) < 2:
            return config.max_distance
        n = min(4, len(self.positions))
        disps = [
            np.linalg.norm(
                np.array(self.positions[-i]) - np.array(self.positions[-i - 1])
            )
            for i in range(1, n)
        ]
        mean_disp = np.mean(disps)
        return float(np.clip(3 * mean_disp, 15, config.max_distance))


class CellTracker:
    def __init__(self, config: Config):
        self.config = config
        self.tracks: dict[int, Track] = {}
        self.completed_tracks: list[Track] = []
        self.next_id = 0

    def update(self, detections, frame_idx):
        """
        用匈牙利算法将当前帧检测结果与已有轨迹匹配。
        只处理 cell_type == 'cell' 的检测。
        """
        cell_dets = [d for d in detections if d["cell_type"] == "cell"]
        active_ids = [tid for tid, t in self.tracks.items() if t.is_active]

        if not active_ids:
            # 没有活跃轨迹，全部新建
            for det in cell_dets:
                self._new_track(det, frame_idx)
            return

        if not cell_dets:
            # 没有检测，所有活跃轨迹miss_count+1
            for tid in active_ids:
                self._increment_miss(tid)
            return

        # 构建代价矩阵
        n_tracks = len(active_ids)
        n_dets = len(cell_dets)
        cost_matrix = np.full((n_tracks, n_dets), 1e6)

        for i, tid in enumerate(active_ids):
            track = self.tracks[tid]
            pred_pos = track.predict_position()
            dist_thresh = track.adaptive_distance_threshold(self.config)

            for j, det in enumerate(cell_dets):
                dist = np.linalg.norm(
                    np.array([det["cx"], det["cy"]]) - np.array(pred_pos)
                )
                if dist > dist_thresh:
                    continue

                # 单向流动约束：细胞不应大幅倒退
                dx = det["cx"] - track.positions[-1][0]
                dy = det["cy"] - track.positions[-1][1]
                # 如果主要流动方向是X轴，则dx不应小于-10
                # 如果主要流动方向是Y轴，则dy不应小于-10
                # 这里用保守约束：任意方向倒退超过15px则拒绝
                if dx < -15 and dy < -15:
                    continue

                # 面积相似度（防止跨细胞匹配）
                area_ratio = det["area"] / (track.areas[-1] + 1e-6)
                if area_ratio > 3.0 or area_ratio < 0.33:
                    continue

                cost = 0.7 * dist + 0.3 * abs(1 - area_ratio) * dist_thresh
                cost_matrix[i, j] = cost

        # 匈牙利算法
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matched_tracks = set()
        matched_dets = set()

        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] >= 1e6:
                continue
            tid = active_ids[r]
            det = cell_dets[c]
            track = self.tracks[tid]

            track.positions.append((det["cx"], det["cy"]))
            track.areas.append(det["area"])
            track.frames.append(frame_idx)
            track.miss_count = 0

            params = extract_per_frame_params(det, frame_idx, self.config.fps)
            params["cell_id"] = tid
            track.frame_data.append(params)

            matched_tracks.add(tid)
            matched_dets.add(c)

        # 未匹配轨迹
        for i, tid in enumerate(active_ids):
            if tid not in matched_tracks:
                self._increment_miss(tid)

        # 未匹配检测 → 新轨迹
        for j, det in enumerate(cell_dets):
            if j not in matched_dets:
                self._new_track(det, frame_idx)

    def _new_track(self, det, frame_idx):
        tid = self.next_id
        self.next_id += 1
        track = Track(tid, det, frame_idx)
        params = extract_per_frame_params(det, frame_idx, self.config.fps)
        params["cell_id"] = tid
        track.frame_data.append(params)
        self.tracks[tid] = track

    def _increment_miss(self, tid):
        track = self.tracks[tid]
        track.miss_count += 1
        if track.miss_count > self.config.max_gap:
            track.is_active = False
            self.completed_tracks.append(track)
            del self.tracks[tid]

    def finalize(self):
        """流水线结束时，将所有剩余活跃轨迹标记为完成。"""
        for tid, track in list(self.tracks.items()):
            track.is_active = False
            self.completed_tracks.append(track)
        self.tracks.clear()

    def get_valid_tracks(self):
        """返回满足最短帧数要求的轨迹。"""
        return [
            t for t in self.completed_tracks
            if len(t.frame_data) >= self.config.min_track_length
        ]


# ─────────────────────────────────────────────
# 可视化（可选）
# ─────────────────────────────────────────────
def draw_frame(frame_bgr, detections, active_tracks):
    """在帧上绘制检测结果和轨迹。"""
    vis = frame_bgr.copy()
    color_map = {
        "cell": (0, 255, 0),
        "cluster": (0, 165, 255),
        "debris": (128, 128, 128),
        "artifact": (0, 0, 255),
    }

    for det in detections:
        color = color_map.get(det["cell_type"], (255, 255, 255))
        x, y, w, h = det["bbox"]
        cv2.rectangle(vis, (x, y), (x + w, y + h), color, 1)

        # 轮廓（青色）
        if det.get("contour") is not None:
            cv2.drawContours(vis, [det["contour"]], -1, (255, 255, 0), 1)

        if det["cell_type"] == "cell":
            major = det["major_axis"]
            minor = det["minor_axis"]
            D = major / (minor + 1e-6)

            # 拟合椭圆（红色）
            cnt = det.get("contour")
            if cnt is not None and len(cnt) >= 5:
                try:
                    ellipse = cv2.fitEllipse(cnt)
                    cv2.ellipse(vis, ellipse, (0, 0, 255), 1)
                except Exception:
                    pass

            # 等效圆（蓝色）：diameter = 2*sqrt(area/pi)
            r_eq = int(round(np.sqrt(det["area"] / np.pi)))
            cx_i, cy_i = int(det["cx"]), int(det["cy"])
            cv2.circle(vis, (cx_i, cy_i), r_eq, (255, 80, 0), 1)

            # 圆度和D值标注
            circ = det["circularity"]
            cv2.putText(
                vis,
                f"D={D:.2f} C={circ:.2f}",
                (x, y - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                color,
                1,
            )

    # 绘制轨迹尾迹
    for tid, track in active_tracks.items():
        if len(track.positions) > 1:
            pts = np.array(track.positions[-20:], dtype=np.int32)
            for k in range(1, len(pts)):
                cv2.line(vis, tuple(pts[k - 1]), tuple(pts[k]), (255, 200, 0), 1)
        if track.positions:
            cx, cy = int(track.positions[-1][0]), int(track.positions[-1][1])
            cv2.putText(
                vis,
                str(tid),
                (cx + 5, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (255, 200, 0),
                1,
            )

    return vis


# ─────────────────────────────────────────────
# 交互框选收缩口
# ─────────────────────────────────────────────
def _draw_baseline(disp, title, color=(0, 255, 255)):
    """画一条全局基准线：左键点两点（沿通道一侧壁）。
    返回 [(x1,y1),(x2,y2)]（显示坐标），Esc 取消返回 None。
    """
    state = {"pts": [], "done": False}

    def _render():
        img = disp.copy()
        for p in state["pts"]:
            cv2.circle(img, p, 2, color, -1)
        if len(state["pts"]) == 2:
            cv2.line(img, state["pts"][0], state["pts"][1], color, 1)
        return img

    def on_mouse(event, x, y, flags, param):
        if state["done"]:
            return
        if event == cv2.EVENT_LBUTTONDOWN and len(state["pts"]) < 2:
            state["pts"].append((x, y))
            cv2.imshow(title, _render())
            if len(state["pts"]) == 2:
                state["done"] = True

    cv2.namedWindow(title)
    cv2.setMouseCallback(title, on_mouse)
    cv2.imshow(title, disp.copy())
    while not state["done"]:
        key = cv2.waitKey(20) & 0xFF
        if key == 27:
            cv2.destroyWindow(title)
            return None
    cv2.destroyWindow(title)
    return list(state["pts"])


def _measure_with_fixed_baseline(disp, title, base_pts, color=(0, 255, 255)):
    """给定固定基准线 base_pts，用户移动鼠标实时预览平行线，
    左键点击确认对侧壁位置；返回平行线到基准线的垂直距离（像素）。
    Esc 跳过返回 None。
    """
    p0, p1 = base_pts
    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]
    length = max(np.sqrt(dx*dx + dy*dy), 1e-6)
    nx, ny = -dy / length, dx / length  # 基准线法向量

    state = {"parallel_pt": None, "done": False}

    def _render(par_pt):
        img = disp.copy()
        cv2.line(img, p0, p1, color, 1)  # 始终画出基准线
        if par_pt is not None:
            h, w = img.shape[:2]
            t_vals = []
            if abs(dx) > 1e-6:
                t_vals += [(-par_pt[0]) / dx, (w - par_pt[0]) / dx]
            if abs(dy) > 1e-6:
                t_vals += [(-par_pt[1]) / dy, (h - par_pt[1]) / dy]
            t_vals = sorted(t_vals)
            if len(t_vals) >= 2:
                q1 = (int(par_pt[0] + t_vals[0]*dx),  int(par_pt[1] + t_vals[0]*dy))
                q2 = (int(par_pt[0] + t_vals[-1]*dx), int(par_pt[1] + t_vals[-1]*dy))
                cv2.line(img, q1, q2, (0, 200, 255), 1)
            px = par_pt[0] - p0[0]
            py = par_pt[1] - p0[1]
            dist = abs(px * nx + py * ny)
            cv2.putText(img, f"dist={dist:.1f}px", (par_pt[0]+6, par_pt[1]-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
        return img

    def on_mouse(event, x, y, flags, param):
        if state["done"]:
            return
        if event == cv2.EVENT_MOUSEMOVE:
            cv2.imshow(title, _render((x, y)))
        elif event == cv2.EVENT_LBUTTONDOWN:
            state["parallel_pt"] = (x, y)
            state["done"] = True

    cv2.namedWindow(title)
    cv2.setMouseCallback(title, on_mouse)
    cv2.imshow(title, _render(None))
    while not state["done"]:
        key = cv2.waitKey(20) & 0xFF
        if key == 27:
            cv2.destroyWindow(title)
            return None
    cv2.destroyWindow(title)

    par_pt = state["parallel_pt"]
    px = par_pt[0] - p0[0]
    py = par_pt[1] - p0[1]
    return float(abs(px * nx + py * ny))


def _pick_point_x(disp, title, color=(0, 255, 0)):
    """让用户左键点一个点，返回其显示坐标 x（int）。Esc 取消返回 None。"""
    state = {"x": None, "done": False}

    def on_mouse(event, x, y, flags, param):
        if state["done"]:
            return
        if event == cv2.EVENT_MOUSEMOVE:
            img = disp.copy()
            cv2.line(img, (x, 0), (x, img.shape[0]), color, 1)
            cv2.imshow(title, img)
        elif event == cv2.EVENT_LBUTTONDOWN:
            state["x"] = x
            state["done"] = True

    cv2.namedWindow(title)
    cv2.setMouseCallback(title, on_mouse)
    cv2.imshow(title, disp.copy())
    while not state["done"]:
        key = cv2.waitKey(20) & 0xFF
        if key == 27:
            cv2.destroyWindow(title)
            return None
    cv2.destroyWindow(title)
    return int(state["x"])


def _measure_parallel_width(disp, title, color=(0, 255, 255)):
    """
    平行线法测量通道最窄宽度：
      Step 1：左键点两点画基准线（沿通道一侧壁）
      Step 2：移动鼠标，实时显示平行线；左键点击确认平行线位置
      返回两平行线之间的垂直距离（像素），Esc 跳过返回 None。

    操作说明：
      - 先点两点定基准线（通道一侧壁）
      - 移动鼠标到对侧壁，实时看平行线预览
      - 左键点击确认平行线位置
      - Esc 随时跳过
    """
    state = {"pts": [], "parallel_pt": None, "done": False}
    img_show = [disp.copy()]

    def _draw_current(base_pts, par_pt):
        img = disp.copy()
        if len(base_pts) >= 1:
            cv2.circle(img, base_pts[0], 2, color, -1)
        if len(base_pts) >= 2:
            cv2.line(img, base_pts[0], base_pts[1], color, 1)
            # 基准线方向向量
            dx = base_pts[1][0] - base_pts[0][0]
            dy = base_pts[1][1] - base_pts[0][1]
            length = max(np.sqrt(dx*dx + dy*dy), 1e-6)
            # 法向量（垂直于基准线）
            nx, ny = -dy / length, dx / length
            if par_pt is not None:
                # 平行线偏移量 = par_pt 到基准线的有符号距离
                px, py = par_pt[0] - base_pts[0][0], par_pt[1] - base_pts[0][1]
                dist = px * nx + py * ny
                # 平行线过 par_pt，方向与基准线相同
                # 延长到图像边界显示
                h, w = img.shape[:2]
                t_vals = []
                if abs(dx) > 1e-6:
                    t_vals += [(-par_pt[0]) / dx, (w - par_pt[0]) / dx]
                if abs(dy) > 1e-6:
                    t_vals += [(-par_pt[1]) / dy, (h - par_pt[1]) / dy]
                t_vals = sorted(t_vals)
                if len(t_vals) >= 2:
                    p1 = (int(par_pt[0] + t_vals[0]*dx), int(par_pt[1] + t_vals[0]*dy))
                    p2 = (int(par_pt[0] + t_vals[-1]*dx), int(par_pt[1] + t_vals[-1]*dy))
                    cv2.line(img, p1, p2, (0, 200, 255), 1)
                # 显示距离
                cv2.putText(img, f"dist={abs(dist):.1f}px",
                            (par_pt[0]+6, par_pt[1]-6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
        return img

    def on_mouse(event, x, y, flags, param):
        if state["done"]:
            return
        if event == cv2.EVENT_MOUSEMOVE and len(state["pts"]) == 2:
            state["parallel_pt"] = (x, y)
            img_show[0] = _draw_current(state["pts"], state["parallel_pt"])
            cv2.imshow(title, img_show[0])
        elif event == cv2.EVENT_LBUTTONDOWN:
            if len(state["pts"]) < 2:
                state["pts"].append((x, y))
                img_show[0] = _draw_current(state["pts"], None)
                cv2.imshow(title, img_show[0])
            elif len(state["pts"]) == 2:
                state["parallel_pt"] = (x, y)
                state["done"] = True

    cv2.namedWindow(title)
    cv2.setMouseCallback(title, on_mouse)
    cv2.imshow(title, img_show[0])

    while not state["done"]:
        key = cv2.waitKey(20) & 0xFF
        if key == 27:
            cv2.destroyWindow(title)
            return None

    cv2.destroyWindow(title)

    base_pts = state["pts"]
    par_pt = state["parallel_pt"]
    dx = base_pts[1][0] - base_pts[0][0]
    dy = base_pts[1][1] - base_pts[0][1]
    length = max(np.sqrt(dx*dx + dy*dy), 1e-6)
    nx, ny = -dy / length, dx / length
    px = par_pt[0] - base_pts[0][0]
    py = par_pt[1] - base_pts[0][1]
    dist = abs(px * nx + py * ny)
    return float(dist)


def select_squeeze_zones(video_path: str, px_to_um: float = 1.0,
                         calib_frame: int = 0) -> tuple:
    """
    弹出视频指定帧（默认第一帧），让用户依次框选每个收缩口，然后对每个收缩口画线段量最窄宽度。
    检测到同名 JSON 文件则直接读取，跳过框选。

    calib_frame: 用于标定的帧序号（0=第一帧）。若该帧细胞太多导致画面杂乱，
                 可指定一个较干净的帧。注意：这里始终只读取「一帧」原始图像，
                 不做任何平均/叠加，画面里的内容就是该帧的真实内容。

    返回 (zones, channel_widths_um)
      zones:              list of (x_start, x_end)，每个区固定宽 SQUEEZE_ZONE_WIDTH_PX、等间距
      channel_widths_um:  list of float（微米），与 zones 一一对应（全部为同一标定值）
    """
    json_path = Path(video_path).with_suffix(".zones.json")

    # ── 读缓存 ────────────────────────────────────────────────
    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        zones = [tuple(z) for z in data["zones"]]
        widths = data.get("channel_widths_um", [np.nan] * len(zones))
        print(f"[zones] 读取缓存: {json_path}")
        print(f"  挤压区: {zones}")
        print(f"  通道宽度(um): {widths}")
        return zones, widths

    # ── 读指定帧（单帧原始图像，无平均/叠加）────────────────────
    cap = cv2.VideoCapture(video_path)
    frame = None
    if calib_frame <= 0:
        ret, frame = cap.read()
    else:
        # 顺序读到目标帧（大体积 AVI 随机 seek 会触发 ffmpeg 超时，故顺序读）
        ret = False
        for _ in range(calib_frame + 1):
            ret, frame = cap.read()
            if not ret:
                break
    cap.release()
    if frame is None or not ret:
        print(f"[WARN] 无法读取视频第 {calib_frame} 帧，跳过框选")
        return [], []
    print(f"[zones] 用第 {calib_frame} 帧标定（单帧原始图像）")

    h, w = frame.shape[:2]
    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        screen_w = root.winfo_screenwidth()
        screen_h = root.winfo_screenheight()
        root.destroy()
    except Exception:
        screen_w, screen_h = 1920, 1080

    scale = min(screen_w * 0.95 / w, screen_h * 0.85 / h)
    disp_orig = cv2.resize(frame, (max(1, int(w * scale)), max(1, int(h * scale))))

    print("\n标定流程：画一次基准线 → 逐个挤压区（点中心 + 点对侧壁量宽度）")
    print("  Step 0：沿通道一侧壁点两点，画一条基准线（所有挤压区共用）")
    print(f"  逐个挤压区：左键点【中心】，自动生成宽 {SQUEEZE_ZONE_WIDTH_PX}px 的矩形")
    print("            然后移动到对侧壁，左键点击，量该挤压区的通道宽度")
    print("  Esc：结束标定（在点中心环节按 Esc 即结束）")

    # ── Step 0：画基准线（所有挤压区共用） ────────────────────
    disp_base = disp_orig.copy()
    cv2.putText(disp_base, "Step 0: 沿通道一侧壁点两点，画基准线",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    baseline = _draw_baseline(disp_base, "Step0 画基准线 — 沿一侧壁点两点（Esc取消）")
    if baseline is None:
        print("[WARN] 未画基准线，无法测量宽度，且无法标定，跳过该视频。")
        return [], []

    # ── 逐个标定挤压区：点中心生成固定宽矩形 + 点对侧壁量宽度 ──
    half = SQUEEZE_ZONE_WIDTH_PX / 2.0
    zones = []
    widths_um = []
    while True:
        n = len(zones)
        disp_c = disp_orig.copy()
        cv2.line(disp_c, baseline[0], baseline[1], (0, 255, 255), 1)
        for idx, (xs, xe) in enumerate(zones):
            xs_d, xe_d = int(xs * scale), int(xe * scale)
            cv2.rectangle(disp_c, (xs_d, 0), (xe_d, disp_c.shape[0]), (0, 255, 0), 2)
            cv2.putText(disp_c, f"#{idx+1}", (xs_d + 2, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        cv2.putText(disp_c, f"点第 {n+1} 个挤压区中心（Esc结束）",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cx_d = _pick_point_x(disp_c, f"点第 {n+1} 个挤压区中心（Esc结束标定）")
        if cx_d is None:
            break  # 结束标定

        cx = cx_d / scale
        x_start = int(round(cx - half))
        x_end   = int(round(cx + half))
        zones.append((x_start, x_end))
        print(f"  挤压区 {n+1}: 中心 {cx:.0f}px, x = {x_start} ~ {x_end} px")

        # 量该挤压区的通道宽度（基于固定基准线，点对侧壁）
        disp_w = disp_orig.copy()
        cv2.line(disp_w, baseline[0], baseline[1], (0, 255, 255), 1)
        xs_d, xe_d = int(x_start * scale), int(x_end * scale)
        cv2.rectangle(disp_w, (xs_d, 0), (xe_d, disp_w.shape[0]), (0, 255, 0), 2)
        cv2.putText(disp_w, f"挤压区 {n+1}：移到对侧壁，左键点击量宽度（Esc跳过）",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        length_px = _measure_with_fixed_baseline(
            disp_w, f"量挤压区 {n+1} 通道宽度 — 点击对侧壁（Esc跳过）", baseline)
        if length_px is not None:
            width_um = length_px / scale * px_to_um
            widths_um.append(width_um)
            print(f"    通道宽度: {length_px/scale:.1f} px = {width_um:.2f} um")
        else:
            widths_um.append(np.nan)
            print(f"    通道宽度: 跳过")

    if not zones:
        print("[WARN] 未标定任何挤压区。")
        return [], []

    # 按 x 排序，宽度跟随
    order = sorted(range(len(zones)), key=lambda k: zones[k][0])
    zones = [zones[k] for k in order]
    widths_um = [widths_um[k] for k in order]

    # ── 预览确认 ──────────────────────────────────────────────
    disp_prev = disp_orig.copy()
    cv2.line(disp_prev, baseline[0], baseline[1], (0, 255, 255), 1)
    for idx, (xs, xe) in enumerate(zones):
        xs_d, xe_d = int(xs * scale), int(xe * scale)
        cv2.rectangle(disp_prev, (xs_d, 0), (xe_d, disp_prev.shape[0]), (0, 255, 0), 2)
        cv2.putText(disp_prev, f"#{idx+1}", (xs_d + 2, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
    cv2.putText(disp_prev, "Enter确认 / Esc重来",
                (10, disp_prev.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
    cv2.imshow("预览挤压区 — Enter确认 / Esc重来", disp_prev)
    key = cv2.waitKey(0) & 0xFF
    cv2.destroyAllWindows()
    if key == 27:
        print("  取消，重新标定该视频。")
        return select_squeeze_zones(video_path, px_to_um, calib_frame)

    print(f"\n共标定 {len(zones)} 个挤压区（等宽 {SQUEEZE_ZONE_WIDTH_PX}px）: {zones}")
    print(f"各区通道宽度(um): {[round(w, 2) if not np.isnan(w) else None for w in widths_um]}")

    # ── 存 JSON ───────────────────────────────────────────────
    data = {
        "zones": [list(z) for z in zones],
        "channel_widths_um": [round(w, 3) if not np.isnan(w) else None for w in widths_um],
        "zone_width_px": SQUEEZE_ZONE_WIDTH_PX,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"已保存: {json_path}\n")

    return zones, widths_um


# ─────────────────────────────────────────────
# 主流水线
# ─────────────────────────────────────────────
def run_pipeline(video_path: str, config: Config, squeeze_zones: list = None,
                 channel_widths_um: list = None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"无法打开视频: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_video = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 如果config.fps未设置，使用视频元数据
    if config.fps <= 0:
        config.fps = fps_video
    print(f"视频: {total_frames} 帧, {config.fps:.0f} fps, {width}x{height}")

    # 建背景
    print("建立背景模型...")
    background = build_background(cap, config.bg_n_frames)

    # 输出目录
    os.makedirs(config.output_dir, exist_ok=True)
    video_stem = Path(video_path).stem

    # 调试视频
    debug_writer = None
    if config.save_debug_video:
        
        # 改后
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        debug_path = os.path.join(config.output_dir, f"{video_stem}_debug.avi")
        
        debug_writer = cv2.VideoWriter(debug_path, fourcc, min(config.fps, 30), (width, height))

    tracker = CellTracker(config)

    print("处理帧...")
    for frame_idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detections = detect_cells_single_frame(frame_gray, background, config)
        tracker.update(detections, frame_idx)

        if debug_writer:
            vis = draw_frame(frame, detections, tracker.tracks)
            debug_writer.write(vis)

        if frame_idx % 500 == 0:
            n_active = len(tracker.tracks)
            n_done = len(tracker.completed_tracks)
            print(f"  帧 {frame_idx}/{total_frames}，活跃轨迹: {n_active}，已完成: {n_done}")

    cap.release()
    if debug_writer:
        debug_writer.release()

    tracker.finalize()
    valid_tracks = tracker.get_valid_tracks()
    print(f"\n有效轨迹数: {len(valid_tracks)}（≥{config.min_track_length}帧）")

    # 汇总数据
    all_frame_rows = []
    all_cell_rows = []
    all_profile_rows = []
    all_profile_long_rows = []

    for track in valid_tracks:
        if not track.frame_data:
            continue
        track_df = pd.DataFrame(track.frame_data)
        all_frame_rows.append(track_df)

        summary = extract_cell_summary(track_df, track.track_id, config.px_to_um,
                                       squeeze_zones, channel_widths_um)
        all_cell_rows.append(summary)

        profile = extract_cell_profile(track_df, track.track_id, squeeze_zones)
        all_profile_rows.append(profile)

        profile_long = extract_cell_profile_long(
            track_df, track.track_id, config.px_to_um, squeeze_zones,
            config.profile_long_step_um,
        )
        all_profile_long_rows.extend(profile_long)

    # 保存CSV
    if all_frame_rows:
        per_frame_df = pd.concat(all_frame_rows, ignore_index=True)
        per_frame_path = os.path.join(config.output_dir, f"{video_stem}_per_frame.csv")
        per_frame_df.to_csv(per_frame_path, index=False)
        print(f"per_frame CSV: {per_frame_path}（{len(per_frame_df)} 行）")

    if all_cell_rows:
        per_cell_df = pd.DataFrame(all_cell_rows)
        per_cell_path = os.path.join(config.output_dir, f"{video_stem}_per_cell.csv")
        per_cell_df.to_csv(per_cell_path, index=False)
        print(f"per_cell CSV:  {per_cell_path}（{len(per_cell_df)} 行）")
        print(f"\n特征统计：")
        cols_stat = [c for c in ["diameter_um", "D_max_global", "circ_baseline",
                                  "n_squeezes_passed", "pre_free_time_ms"] if c in per_cell_df.columns]
        print(per_cell_df[cols_stat].describe().round(4))

    if all_profile_rows:
        per_profile_df = pd.DataFrame(all_profile_rows)
        per_profile_path = os.path.join(config.output_dir, f"{video_stem}_per_cell_profile.csv")
        per_profile_df.to_csv(per_profile_path, index=False)
        print(f"per_cell_profile CSV: {per_profile_path}（{len(per_profile_df)} 行，{len(per_profile_df.columns)} 列）")

        if all_cell_rows:
            per_cell_profile_merged_df = pd.merge(
                per_cell_df,
                per_profile_df,
                on="cell_id",
                how="left",
                suffixes=("", "_profile"),
            )
            per_cell_profile_merged_path = os.path.join(
                config.output_dir,
                f"{video_stem}_per_cell_with_profile.csv",
            )
            per_cell_profile_merged_df.to_csv(per_cell_profile_merged_path, index=False)
            print(
                f"per_cell_with_profile CSV: {per_cell_profile_merged_path}"
                f"（{len(per_cell_profile_merged_df)} 行，{len(per_cell_profile_merged_df.columns)} 列）"
            )

    if all_profile_long_rows:
        per_profile_long_df = pd.DataFrame(all_profile_long_rows)
        per_profile_long_path = os.path.join(config.output_dir, f"{video_stem}_per_cell_profile_long.csv")
        per_profile_long_df.to_csv(per_profile_long_path, index=False)
        print(f"per_cell_profile_long CSV: {per_profile_long_path}（{len(per_profile_long_df)} 行）")

    return per_cell_df if all_cell_rows else pd.DataFrame()


# ─────────────────────────────────────────────
# 入口
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="细胞变形分析流水线 v2")
    parser.add_argument("--video", required=True, help="AVI视频路径")
    parser.add_argument("--fps", type=float, default=0, help="帧率（0=从视频读取）")
    parser.add_argument("--px_to_um", type=float, default=0.5, help="像素→微米换算")
    parser.add_argument(
        "--roi", nargs=4, type=int, metavar=("X1", "Y1", "X2", "Y2"),
        default=None, help="通道ROI区域"
    )
    parser.add_argument("--debug", action="store_true", help="保存调试视频")
    parser.add_argument("--min_area", type=int, default=50, help="最小细胞面积(px²)")
    parser.add_argument("--max_area", type=int, default=8000, help="最大细胞面积(px²)")
    parser.add_argument("--threshold", type=int, default=None, help="固定diff阈值（不填=Otsu）")
    parser.add_argument("--x_min", type=int, default=None, help="统计有效区间左边界(px)")
    parser.add_argument("--x_max", type=int, default=None, help="统计有效区间右边界(px)")
    parser.add_argument("--profile_long_step_um", type=float, default=2.0,
                        help="per_cell_profile_long.csv 的空间采样间隔(um)")
    parser.add_argument("--no_select", action="store_true", help="跳过收缩口框选（不统计挤压时间）")
    parser.add_argument("--calib_frame", type=int, default=0,
                        help="用于标定的帧序号（0=第一帧）。若第一帧细胞太多画面杂乱，可指定较干净的一帧")
    args = parser.parse_args()

    config = Config()
    config.fps = args.fps
    config.px_to_um = args.px_to_um
    config.channel_roi = tuple(args.roi) if args.roi else None
    config.save_debug_video = args.debug
    config.min_area = args.min_area
    config.max_area = args.max_area
    config.min_cell_area = args.min_area
    config.diff_threshold = args.threshold
    config.x_stat_min = args.x_min
    config.x_stat_max = args.x_max
    config.profile_long_step_um = args.profile_long_step_um

    squeeze_zones = []
    channel_widths_um = []
    if not args.no_select:
        squeeze_zones, channel_widths_um = select_squeeze_zones(
            args.video, config.px_to_um, args.calib_frame)

    run_pipeline(args.video, config, squeeze_zones, channel_widths_um)


if __name__ == "__main__":
    main()
