# Cell Mechanical Phenotype Extraction Pipeline (OpenCV)

基于 **OpenCV + SciPy + Pandas** 的高速显微视频细胞力学表型提取流水线。  
该程序面向微流控/收缩通道中的单细胞高速成像视频，可自动完成：

- 静态背景建模
- 前景细胞检测
- 粘连细胞分水岭分割
- 细胞/细胞团/碎片/伪影分类
- 多目标细胞追踪
- 单帧形态学参数提取
- 收缩区与恢复区力学表型计算
- 细胞空间变形曲线采样
- 结果 CSV 导出
- 可选调试视频可视化

适用于基于高速显微视频的细胞变形、恢复、圆度变化、通过时间等力学表型分析。

---

## 1. Pipeline Overview

整体处理流程如下：

```text
Input AVI video
      │
      ▼
Static background modeling
      │
      ▼
Frame difference
      │
      ▼
Threshold segmentation
      │
      ▼
Morphological closing / opening
      │
      ▼
Convex hull filling
      │
      ▼
Watershed separation
      │
      ▼
Contour detection
      │
      ▼
Cell morphology extraction
      │
      ▼
Cell / cluster / debris / artifact classification
      │
      ▼
Hungarian multi-object tracking
      │
      ▼
Per-frame phenotype extraction
      │
      ├──────────────► per_frame.csv
      │
      ▼
Per-cell trajectory aggregation
      │
      ├──────────────► per_cell.csv
      ├──────────────► per_cell_profile.csv
      ├──────────────► per_cell_with_profile.csv
      └──────────────► per_cell_profile_long.csv
```

---

## 2. Main Features

### 2.1 Static background modeling

程序从视频中顺序读取并均匀采样若干帧，计算灰度中位数作为静态背景。

这种实现避免了在超大 AVI 文件中频繁随机 seek，从而降低高速相机大体积视频读取卡顿或 ffmpeg 超时的风险。

核心参数：

```python
bg_n_frames = 100
```

---

### 2.2 Cell foreground segmentation

单帧检测流程：

1. 灰度化
2. 与静态背景做绝对帧差
3. 二值阈值分割
4. 形态学闭运算
5. 形态学开运算
6. 凸包填充
7. 分水岭分割
8. 轮廓提取

默认形态学结构元：

```python
close kernel = 9 × 9 ellipse
open kernel  = 3 × 3 ellipse
```

---

### 2.3 Watershed separation

对于粘连区域，程序利用距离变换构造前景种子，再执行 OpenCV watershed 分水岭算法。

主要目的：

- 分离相互接触的细胞
- 减少多个细胞被识别为单一大轮廓的问题
- 提高后续面积、长短轴和变形度计算的可靠性

---

### 2.4 Cell classification

检测到的目标会被分为：

| 类型 | 含义 |
|---|---|
| `cell` | 有效单细胞 |
| `cluster` | 面积过大的细胞团 |
| `debris` | 小碎片、低圆度目标或过度细长目标 |
| `artifact` | 接触图像边界的伪影 |
| `out_of_roi` | 位于有效统计 x 区间之外的目标 |

主要判据包括：

- 面积
- 圆度
- 长宽比
- 是否接触图像边缘
- 是否位于指定统计区间

---

## 3. Extracted Morphological Features

每一帧中，对有效细胞提取以下参数：

| 参数 | 含义 |
|---|---|
| `frame` | 帧序号 |
| `time_s` | 时间，单位 s |
| `cell_id` | 细胞追踪编号 |
| `circularity` | 圆度 |
| `aspect_ratio` | 长短轴比 |
| `deformation_D` | 变形指标 |
| `diameter_px` | 等效直径，pixel |
| `area_px` | 轮廓面积，pixel² |
| `centroid_x_px` | 质心 x 坐标 |
| `centroid_y_px` | 质心 y 坐标 |
| `major_axis_px` | 椭圆拟合长轴 |
| `minor_axis_px` | 椭圆拟合短轴 |
| `cell_type` | 目标类别 |

### 3.1 Circularity

圆度定义为：

```text
Circularity = 4πA / P²
```

其中：

- `A` 为细胞轮廓面积
- `P` 为细胞轮廓周长

越接近 1，表示轮廓越接近圆形。

---

### 3.2 Equivalent diameter

等效圆直径：

```text
diameter = 2 × sqrt(A / π)
```

程序进一步使用 `px_to_um` 将像素尺寸换算为实际微米尺寸。

---

### 3.3 Deformation index

当前代码中的变形指标定义为：

```text
D = major_axis / minor_axis
```

即椭圆拟合长轴与短轴之比。

当：

```text
D ≈ 1
```

时，细胞接近圆形；

`D` 越大，表示细胞轮廓越细长、形变越明显。

> 注意：本代码中的 `deformation_D` 是长短轴比，并不是 RT-DC 文献中常见的 `(1-circularity)` 或其他 deformation 定义。进行论文写作时应明确给出本文采用的定义。

---

## 4. Cell Tracking

程序采用多目标追踪策略对不同帧中的同一细胞进行关联。

核心步骤：

1. 根据最近若干帧轨迹预测下一帧位置
2. 根据近期速度自适应调整最大匹配距离
3. 构建轨迹-检测目标之间的代价矩阵
4. 使用 Hungarian algorithm 进行全局最优匹配
5. 使用面积变化约束降低跨细胞误匹配
6. 允许短时间检测丢失

默认参数：

```python
max_distance = 50
max_gap = 20
min_track_length = 15
```

匹配代价：

```text
cost =
0.7 × spatial distance
+
0.3 × area similarity penalty
```

仅 `cell_type == "cell"` 的目标参与正式追踪。

---

## 5. Squeeze Zone Calibration

程序支持交互式标定多个收缩区。

运行时若没有已有的：

```text
*.zones.json
```

程序会弹出 OpenCV 标定窗口。

### 标定流程

#### Step 0

沿通道一侧壁点击两个点，建立全局基准线。

#### Step 1

依次点击各收缩区中心位置。

每一个收缩区的 x 方向宽度固定为：

```python
SQUEEZE_ZONE_WIDTH_PX = 55
```

即：

```text
center_x ± 27.5 px
```

生成对应的收缩区域。

#### Step 2

将鼠标移动至通道另一侧壁并点击。

程序计算该点到基准线的垂直距离，以此估计该收缩区的最窄通道宽度。

#### Step 3

继续标定下一收缩区。

在“选择收缩区中心”步骤按：

```text
Esc
```

结束标定。

#### Step 4

预览所有收缩区。

- `Enter`：确认
- `Esc`：重新标定

---

## 6. Calibration Cache

标定完成后，程序会在视频同目录生成：

```text
<video_name>.zones.json
```

其中保存：

```json
{
  "zones": [
    [x_start_1, x_end_1],
    [x_start_2, x_end_2]
  ],
  "channel_widths_um": [
    10.5,
    9.8
  ],
  "zone_width_px": 55
}
```

下一次分析同一视频时，如果该 JSON 已存在，程序会直接读取，不再重复弹出标定窗口。

如果需要重新标定，可删除对应 `.zones.json` 文件后重新运行。

---

## 7. Mechanical Phenotypes Extracted per Cell

程序不仅提取单帧形态，还会基于完整轨迹计算单细胞级力学表型。

主要包括：

### Baseline phenotype

| 参数 | 含义 |
|---|---|
| `diameter_um` | 进入第一个收缩区之前的中位等效直径 |
| `area_um2` | 进入第一个收缩区之前的中位面积 |
| `circ_baseline` | 进入第一个收缩区之前的平均圆度 |
| `pre_free_time_ms` | 进入首个收缩区之前的自由运动时间 |
| `D_max_global` | 整条轨迹中的最大变形指标 |

如果细胞进入视野时已经处于收缩区中，则部分基线参数会被记为 `NaN`。

---

### Phenotypes for each squeeze zone

最多支持：

```python
MAX_SQUEEZE_ZONES = 9
```

每个收缩区输出：

```text
sq1_channel_width_um
sq1_D_max
sq1_circ_min
sq1_t_max_ms
sq1_squeeze_time_ms
sq1_squeeze_len_um
sq1_free_time_ms
sq1_free_len_um
sq1_recovery_time_ms
```

并依次扩展至：

```text
sq2_...
sq3_...
...
sq9_...
```

---

### `sqN_D_max`

细胞在第 N 个收缩区内的最大长短轴比。

表示该区域内观测到的最大形态变形程度。

---

### `sqN_circ_min`

细胞在第 N 个收缩区内的最小圆度。

---

### `sqN_t_max_ms`

从进入当前收缩区开始，到达到该区最大变形 `D_max` 所需要的时间。

---

### `sqN_squeeze_time_ms`

细胞在当前收缩区中出现的帧数 × 单帧时间。

可近似表征通过该收缩区域的停留时间。

---

### `sqN_squeeze_len_um`

收缩区在 x 方向的长度：

```text
(x_end - x_start) × px_to_um
```

---

### `sqN_free_time_ms`

细胞离开当前收缩区后，到下一收缩区之前的自由区域停留时间。

---

### `sqN_free_len_um`

当前收缩区末端至下一收缩区起点之间的空间长度。

对于最后一个收缩区，程序使用前面自由区宽度的平均值向后外推。

---

### `sqN_recovery_time_ms`

程序定义恢复目标为：

```text
recovery circularity threshold
=
baseline circularity × 0.95
```

细胞达到该圆度阈值时：

```text
recovery_time
=
恢复时刻 - 当前收缩区最大变形时刻
```

如果细胞在自由区内没有恢复到目标圆度，则记录为：

```text
NaN
```

---

## 8. Spatial Mechanical Phenotype Profile

为了避免不同细胞运动速度不同导致时间轴不可直接比较，程序还提供基于 **x 空间坐标** 的标准化采样。

### 8.1 Fixed-point profile

输出文件：

```text
*_per_cell_profile.csv
```

采样规则：

#### 收缩前自由区

```text
5 points
```

输出：

```text
pre_D_1 ... pre_D_5
pre_C_1 ... pre_C_5
```

#### 每个收缩区

每区均匀采样：

```text
9 points
```

输出：

```text
sq1_D_1 ... sq1_D_9
sq1_C_1 ... sq1_C_9
```

#### 每个收缩区后的恢复区

同样采样：

```text
9 points
```

输出：

```text
sq1_rec_D_1 ... sq1_rec_D_9
sq1_rec_C_1 ... sq1_rec_C_9
```

这种表示形式适合：

- 机器学习输入
- 细胞间标准化比较
- 热图绘制
- PCA / UMAP
- 聚类
- SVM / Random Forest / XGBoost 等分类模型

---

### 8.2 Long-format spatial profile

输出：

```text
*_per_cell_profile_long.csv
```

该表每一行代表一个空间采样点。

主要字段：

| 字段 | 含义 |
|---|---|
| `cell_id` | 细胞 ID |
| `region_type` | `pre_free` / `squeeze` / `free` |
| `region_name` | 区域名称 |
| `squeeze_id` | 收缩区编号 |
| `free_id` | 自由区编号 |
| `point_idx` | 区域内采样点编号 |
| `x_region_start_px` | 区域起始 x |
| `x_region_end_px` | 区域结束 x |
| `x_sample_px` | 理论采样 x |
| `x_rel_um` | 相对区域起点的空间位置 |
| `x_abs_um` | 图像中的绝对空间位置 |
| `nearest_frame` | 最邻近采样位置对应帧 |
| `nearest_x_px` | 实际最近质心位置 |
| `nearest_time_s` | 对应时间 |
| `deformation_D` | 变形指标 |
| `circularity` | 圆度 |

默认空间采样间隔：

```python
profile_long_step_um = 2.0
```

可通过命令行修改。

---

## 9. Installation

建议使用 Python 3.9 或更高版本。

安装依赖：

```bash
pip install opencv-python numpy pandas scipy
```

如果需要 GUI 标定窗口，请确保当前 Python/OpenCV 环境支持桌面图形界面。

主要依赖：

```text
opencv-python
numpy
pandas
scipy
```

代码还使用 Python 标准库：

```text
argparse
os
json
pathlib
tkinter
```

---

## 10. Usage

基本运行：

```bash
python 1cell_pipeline_v4.py --video path/to/video.avi
```

指定视频帧率：

```bash
python 1cell_pipeline_v4.py \
  --video path/to/video.avi \
  --fps 2500
```

指定像素-微米换算关系：

```bash
python 1cell_pipeline_v4.py \
  --video path/to/video.avi \
  --fps 2500 \
  --px_to_um 0.5
```

保存调试视频：

```bash
python 1cell_pipeline_v4.py \
  --video path/to/video.avi \
  --fps 2500 \
  --px_to_um 0.5 \
  --debug
```

指定 ROI：

```bash
python 1cell_pipeline_v4.py \
  --video path/to/video.avi \
  --roi 100 50 1800 700
```

其中：

```text
--roi X1 Y1 X2 Y2
```

表示仅在：

```text
x = X1 ~ X2
y = Y1 ~ Y2
```

区域进行细胞检测。

---

## 11. Command-line Arguments

| 参数 | 类型 | 默认值 | 含义 |
|---|---:|---:|---|
| `--video` | str | required | AVI 视频路径 |
| `--fps` | float | `0` | 视频帧率；0 表示从视频元数据读取 |
| `--px_to_um` | float | `0.5` | pixel → μm 换算比例 |
| `--roi` | 4 × int | None | 检测 ROI：X1 Y1 X2 Y2 |
| `--debug` | flag | False | 保存调试视频 |
| `--min_area` | int | `50` | 最小有效细胞面积 |
| `--max_area` | int | `8000` | 最大面积参数 |
| `--threshold` | int | None | 帧差二值化阈值 |
| `--x_min` | int | None | 有效统计 x 左边界 |
| `--x_max` | int | None | 有效统计 x 右边界 |
| `--profile_long_step_um` | float | `2.0` | long-format 空间采样步长 |
| `--no_select` | flag | False | 不进行收缩区交互标定 |
| `--calib_frame` | int | `0` | 用于标定通道结构的帧序号 |

---

## 12. Recommended Run Command

对于高速微流控视频，建议明确指定：

- 实际帧率
- 像素尺寸
- 检测阈值
- 面积范围

例如：

```bash
python 1cell_pipeline_v4.py \
  --video "example.avi" \
  --fps 2500 \
  --px_to_um 1.25 \
  --min_area 200 \
  --max_area 5000 \
  --threshold 20 \
  --debug
```

---

## 13. Output Files

默认输出目录：

```text
results/
```

主要文件如下。

### 13.1 Per-frame data

```text
<video_name>_per_frame.csv
```

每一行代表某个被追踪细胞在某一帧中的形态参数。

适合：

- 查看单细胞时间序列
- 绘制 deformation-time 曲线
- 绘制 circularity-time 曲线
- 检查追踪质量

---

### 13.2 Per-cell summary

```text
<video_name>_per_cell.csv
```

每一行代表一个细胞。

包含：

- 基线尺寸
- 基线圆度
- 全局最大变形
- 通过收缩区数量
- 各收缩区最大变形
- 各收缩区最小圆度
- 最大变形时间
- 收缩区通过时间
- 自由区停留时间
- 恢复时间
- 通道宽度

适合直接作为统计分析或机器学习输入。

---

### 13.3 Per-cell fixed spatial profile

```text
<video_name>_per_cell_profile.csv
```

每个细胞一行，将各区域标准化为空间采样点。

---

### 13.4 Combined phenotype table

```text
<video_name>_per_cell_with_profile.csv
```

为：

```text
per_cell.csv
+
per_cell_profile.csv
```

按 `cell_id` 合并后的完整单细胞特征表。

适合直接用于后续机器学习。

---

### 13.5 Long-format spatial profile

```text
<video_name>_per_cell_profile_long.csv
```

每行一个空间采样点，适合：

- 绘制平均变形曲线
- 分组统计
- seaborn / ggplot 风格绘图
- mixed-effects model
- 空间动力学分析

---

### 13.6 Debug video

启用：

```bash
--debug
```

后生成：

```text
<video_name>_debug.avi
```

其中可视化：

- 检测框
- 细胞轮廓
- 拟合椭圆
- 等效圆
- deformation D
- circularity
- cell ID
- 近期运动轨迹

建议在正式批量分析前先检查 debug 视频，以确认：

- 分割边界是否合理
- 是否存在细胞漏检
- 是否存在多个细胞被连成一个目标
- 轨迹 ID 是否频繁跳变

---

## 14. Important Configuration Parameters

代码内部 `Config` 中包含以下重要参数。

### Video

```python
fps = 2500.0
px_to_um = 1.25
```

---

### Background

```python
bg_n_frames = 100
```

---

### Detection

```python
diff_threshold = 20
min_area = 200
max_area = 5000
min_circularity = 0.20
max_aspect_ratio = 5.0
```

---

### Classification

```python
cluster_area_threshold = 8000
min_cell_area = 200
```

---

### Tracking

```python
max_distance = 50
max_gap = 20
min_track_length = 15
```

---

### Spatial profile

```python
profile_long_step_um = 2.0
```

---

## 15. Parameter Tuning Suggestions

### Too much background noise

提高：

```text
--threshold
```

例如：

```text
10 → 15 → 20 → 25
```

---

### Cell boundary is underestimated

可适当降低：

```text
--threshold
```

但阈值过低可能把光晕、背景纹理或通道结构识别为细胞。

---

### Small cells are removed

降低：

```text
--min_area
```

---

### Debris is treated as cells

提高：

```text
--min_area
```

或提高代码中的：

```python
min_circularity
```

---

### Tracking breaks frequently

可适当提高：

```python
max_distance
max_gap
```

但设置过高也会增加不同细胞之间发生 ID switch 的风险。

---

### Different cells are incorrectly merged

重点检查：

- morphology close kernel 是否过大
- watershed 是否正确分离
- `cluster_area_threshold`
- `max_distance`
- 视频中细胞浓度是否过高

---

## 16. Quality Control

建议每一批实验数据至少执行以下 QC。

### 1. Check segmentation

开启：

```bash
--debug
```

检查细胞轮廓是否贴合真实边界。

### 2. Check tracking

确认同一个细胞在整个运动过程中保持相同 `cell_id`。

### 3. Check calibration

确认：

```text
px_to_um
fps
squeeze zone position
channel width
```

均与实验条件一致。

### 4. Check baseline phenotype

重点查看：

```text
diameter_um
circ_baseline
D_max_global
```

是否存在明显异常值。

### 5. Check squeeze-zone completeness

查看：

```text
n_squeezes_passed
```

确认分析中的细胞是否完整通过预期数量的收缩区。

---

## 17. Notes and Limitations

### 17.1 Assumption of approximately one-directional motion

当前追踪策略包含一定的单向运动约束，因此更适用于微流控通道中总体沿固定方向运动的细胞。

---

### 17.2 Deformation metric is morphology-based

当前 `deformation_D` 来源于二维轮廓椭圆拟合：

```text
major_axis / minor_axis
```

它反映的是图像平面中的表观形态变化，而不是直接测得的：

- Young's modulus
- shear modulus
- cortical tension
- viscosity

如果需要将图像表型转换为真实力学参数，需要额外建立：

- 流体力学模型
- 有限元模型
- 标定实验
- 或数据驱动回归模型

---

### 17.3 Recovery time is threshold-based

当前恢复时间以：

```text
circularity ≥ 0.95 × baseline circularity
```

作为判定条件。

因此该指标属于算法定义的图像表型，而不是直接的材料本构时间常数。

---

### 17.4 `_fit_tau()` is currently not part of final exported phenotype

代码中保留了指数衰减拟合函数 `_fit_tau()`，可用于拟合恢复过程：

```text
D(t) = D_inf + (D0 - D_inf) exp(-t / tau)
```

但当前主流程的 `per_cell.csv` 并未实际调用该函数输出 `tau`。

如果后续需要获得黏弹性恢复时间常数，可在 `extract_cell_summary()` 中进一步接入该拟合模块。

---

### 17.5 Maximum number of squeeze zones

当前程序固定最多预留：

```python
MAX_SQUEEZE_ZONES = 9
```

超过 9 个收缩区时，需要修改代码中的该常量及相关输出逻辑。

---

## 18. Example Project Structure

```text
project/
├── 1cell_pipeline_v4.py
├── videos/
│   └── example.avi
├── example.zones.json
└── results/
    ├── example_per_frame.csv
    ├── example_per_cell.csv
    ├── example_per_cell_profile.csv
    ├── example_per_cell_with_profile.csv
    ├── example_per_cell_profile_long.csv
    └── example_debug.avi
```

---

## 19. Suggested Downstream Analysis

当前输出可直接用于：

- 单细胞力学异质性分析
- 不同细胞系对比
- 药物处理组与对照组比较
- 不同压力或流速条件比较
- deformation trajectory clustering
- PCA
- UMAP
- hierarchical clustering
- SVM
- Random Forest
- XGBoost
- time/space-resolved phenotype analysis

尤其推荐将：

```text
*_per_cell_with_profile.csv
```

作为单细胞机器学习特征矩阵的基础数据。

---

## 20. Citation / Method Description Template

如果该程序用于论文，可以将图像分析方法概括为：

> High-speed microscopy videos were analyzed using a custom OpenCV-based image-processing pipeline. A static background image was constructed from uniformly sampled video frames using median intensity projection. Foreground cells were segmented by background subtraction followed by thresholding, morphological operations, convex-hull filling, and watershed-based separation of contacting objects. Cell contours were fitted with ellipses to quantify morphology, including projected area, circularity, equivalent diameter, major and minor axes, and the deformation index defined as the ratio of the major to minor axis. Individual cells were tracked across frames using motion prediction and Hungarian assignment. Cell-level mechanical phenotypes were subsequently calculated from trajectories before, within, and after predefined microfluidic constriction regions.

请根据正式论文的实验装置、像素标定方法和算法版本进一步核对后再使用。

---

## 21. Quick Start

最简命令：

```bash
python 1cell_pipeline_v4.py \
  --video "example.avi" \
  --fps 2500 \
  --px_to_um 1.25
```

建议首次运行增加：

```bash
--debug
```

确认分割和追踪效果后，再进行批量分析。

---

## 22. License

本项目目前未在代码中声明开源许可证。

如果计划公开至 GitHub，建议根据实际使用需求选择：

- MIT License
- BSD-3-Clause
- GPL-3.0

在未明确许可证前，不建议默认声明为某一种开源许可。
