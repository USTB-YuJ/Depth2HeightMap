# depth_elevation

一个轻量级、与框架无关的 Python 库，将深度图转换为本地高程图（鸟瞰地形网格），专为腿式机器人运动控制设计。

核心计算全部基于 **纯 NumPy**，不依赖任何特定仿真器。  
同时提供 Isaac Lab 集成适配层，可一行接入 Isaac Lab 训练/播放环境，并支持仿真视口内实时可视化。

---

## 处理流程

```
深度图 (H×W)
      │  ┌─────────────────┐
      │  │ CameraIntrinsics│  (fx, fy, cx, cy, width, height, max_depth)
      └──▶  unproject_depth │
           └────────────────┘
                 │ pts_world  (N, 3)  世界系点云
      ┌──────────┘
      │  机器人根部 pos + quat
      ▼
  world_to_yaw_frame          （x 前、y 左、z 上，原点为机器人根部）
      │
      │ pts_yaw  (N, 3)
      ▼
  build_elevation_map         （BEV 网格，每格取 max z，使用 np.maximum.at 向量化）
      │
      │ ElevationMap
      │   .grid       (n_x, n_y)  float32  – 相对机器人根部的 max z（米）
      │   .x_centers  (n_x,)
      │   .y_centers  (n_y,)
      │   .occupied   (n_x, n_y)  bool
      ▼
  elevation_map_to_world_centers   →  世界系球心位置，供可视化使用
```

**深度约定：** `distance_to_image_plane` —— 射线命中点在相机前向轴上的投影距离。  
这是 Isaac Lab `GroupedRayCasterCamera` / `RayCasterCamera` 输出的深度格式。

---

## 安装

### 从源码安装（开发推荐，可编辑模式）

```bash
git clone https://github.com/<your-org>/depth_elevation.git
cd depth_elevation
python setup.py develop --user
```

或者使用 pip（需要 setuptools ≥ 61 以解析构建依赖）：

```bash
pip install -e .
```

**运行时依赖：** 仅 `numpy >= 1.20`。  
PyTorch 和 Isaac Lab **不是**核心包的依赖——只有运行 Isaac Lab 集成示例时才需要。

---

## 快速开始

### 纯 NumPy 使用示例

```python
import numpy as np
from depth_elevation import (
    CameraIntrinsics,
    ElevationConfig,
    unproject_depth,
    world_to_yaw_frame,
    build_elevation_map,
)

# --- 1. 构造相机内参（64×36 针孔，约 90°×65° FOV）---
intr = CameraIntrinsics(
    fx=31.56, fy=31.56,
    cx=32.0, cy=18.0,
    width=64, height=36,
    max_depth=2.5,
)

# --- 2. 深度图（distance_to_image_plane，单位：米）---
depth_hw = np.random.uniform(0.3, 2.0, (36, 64)).astype(np.float32)

# --- 3. 相机在世界系中的位姿 ---
cam_pos_world  = np.array([0.09, 0.0, 1.05])          # (3,)
cam_quat_world = np.array([0.887, 0.0, 0.462, 0.0])   # (4,) w, x, y, z

# --- 4. 反投影：深度图 → 世界系点云 ---
pts_world, valid = unproject_depth(depth_hw, intr, cam_pos_world, cam_quat_world)
# pts_world: (H*W, 3)，valid: (H*W,) bool

# --- 5. 转换到机器人 yaw 系 ---
root_pos  = np.array([0.0, 0.0, 0.0])
root_quat = np.array([1.0, 0.0, 0.0, 0.0])  # 单位四元数（无旋转）
pts_yaw = world_to_yaw_frame(pts_world[valid], root_pos, root_quat)

# --- 6. 构建 BEV 高程图 ---
cfg = ElevationConfig(
    x_min=0.0, x_max=1.5,   # 机器人前方 1.5 m
    y_min=-0.6, y_max=0.6,  # 左右各 0.6 m
    resolution=0.05,         # 30×24 网格，共 720 格
)
emap = build_elevation_map(pts_yaw, cfg)

print(emap.grid.shape)     # (30, 24)
print(emap.occupied.sum()) # 有点覆盖的格子数量
```

---

## API 文档

### `depth_elevation.CameraIntrinsics`

| 字段 | 类型 | 说明 |
|------|------|------|
| `fx`, `fy` | `float` | 像素单位焦距 |
| `cx`, `cy` | `float` | 主点像素坐标 |
| `width`, `height` | `int` | 图像分辨率（像素） |
| `max_depth` | `float` | 有效深度上限（米）。深度 ≥ `max_depth × 0.999` 的像素将被丢弃。默认 `3.0`。 |

### `depth_elevation.ElevationConfig`

所有距离单位均为**米**，坐标系为**机器人 yaw 系**（x 前、y 左）。

| 字段 | 默认值 | 说明 |
|------|--------|------|
| `x_min` | `0.0` | 地图前方起始距离 |
| `x_max` | `1.5` | 地图前方最大距离 |
| `y_min` | `-0.6` | 右侧边界（负值） |
| `y_max` | `0.6` | 左侧边界（正值） |
| `resolution` | `0.05` | 网格分辨率（m/格），默认生成 30×24 网格 |
| `min_depth` | `1e-4` | 深度有效下限（米） |
| `fill_value` | `0.0` | 无点覆盖格子的填充值（米，相对机器人根部） |

`n_x = round((x_max - x_min) / resolution)`，`n_y` 同理。

### `depth_elevation.ElevationMap`

| 字段 | 形状 | 说明 |
|------|------|------|
| `grid` | `(n_x, n_y)` float32 | 每格 max z（yaw 系，相对机器人根部，米） |
| `x_centers` | `(n_x,)` float32 | 格中心 x 坐标（yaw 系） |
| `y_centers` | `(n_y,)` float32 | 格中心 y 坐标（yaw 系） |
| `occupied` | `(n_x, n_y)` bool | `True` 表示该格至少有一个点覆盖 |
| `config` | `ElevationConfig` | 生成该高程图所用的配置 |

### `depth_elevation.unproject_depth`

```python
pts_world, valid = unproject_depth(depth_hw, intrinsics, cam_pos_world, cam_quat_world)
```

**深度约定（image_plane）的数学推导：**

```
d = (R^T · (t · dir_w))_x       （Isaac Lab 定义）
→  dir_cam = R^T · dir_w
→  t = d / dir_cam.x             （需要 |dir_cam.x| > 1e-4）
→  P_cam = t · dir_cam
→  P_world = pos_world + R · P_cam
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `depth_hw` | `(H, W)` ndarray | 深度图，米制 |
| `intrinsics` | `CameraIntrinsics` | 相机内参 |
| `cam_pos_world` | `(3,)` ndarray | 相机在世界系中的位置 |
| `cam_quat_world` | `(4,)` ndarray | 相机在世界系中的姿态（w, x, y, z） |

返回值：`pts_world` 形状 `(H*W, 3)` float64，`valid` 形状 `(H*W,)` bool。  
无效像素在 `pts_world` 中填 `NaN`。

### `depth_elevation.world_to_yaw_frame` / `yaw_frame_to_world`

```python
pts_yaw  = world_to_yaw_frame(pts_world, root_pos, root_quat)
pts_back = yaw_frame_to_world(pts_yaw,   root_pos, root_quat)
```

在世界系与机器人 yaw 对齐局部系之间相互转换。  
`root_quat` 中只使用 yaw 分量，roll 和 pitch 被忽略（保留相对高度）。

### `depth_elevation.build_elevation_map`

```python
emap = build_elevation_map(points_yaw, config)
```

使用 `numpy.maximum.at` 向量化聚合，每格取 max z，NaN 点自动过滤。

### `depth_elevation.elevation_map_to_world_centers`

```python
world_centers, z_vals = elevation_map_to_world_centers(emap, root_pos, root_quat)
```

将有点覆盖的格中心从 yaw 系转回世界系，用于在仿真视口中放置可视化小球。  
返回 `(K, 3)` float32 和 `(K,)` float32，`K` 为有点覆盖的格子数。

---

## Isaac Lab 集成

### 从 `GroupedRayCasterCamera` 提取内参

```python
from depth_elevation import CameraIntrinsics

def cam_intrinsics_from_isaaclab(cam, env_id: int) -> CameraIntrinsics:
    """读取单环境内参，优先使用域随机化后的逐环境值。"""
    max_d = float(cam.cfg.max_distance)
    if hasattr(cam._data, "intrinsic_matrices"):
        K = cam._data.intrinsic_matrices[env_id].cpu().numpy()
        fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    else:
        pc = cam.cfg.pattern_cfg
        h_ap = pc.horizontal_aperture
        v_ap = pc.vertical_aperture or h_ap * pc.height / pc.width
        fx = pc.width  * pc.focal_length / h_ap
        fy = pc.height * pc.focal_length / v_ap
        cx, cy = pc.width / 2, pc.height / 2
    return CameraIntrinsics(fx=fx, fy=fy, cx=cx, cy=cy,
                            width=cam.cfg.pattern_cfg.width,
                            height=cam.cfg.pattern_cfg.height,
                            max_depth=max_d)
```

从 `cam._data.intrinsic_matrices` 读取时，可正确反映域随机化后的逐环境焦距。

### 仿真视口可视化（`VisualizationMarkers`）

```python
from depth_elevation.examples.example_isaaclab import ElevationVizIsaacLab

viz = ElevationVizIsaacLab()   # 使用默认 ElevationConfig

# 在 play 循环中：
while simulation_app.is_running():
    actions = policy(obs)
    obs, _, _, extras = env.step(actions)
    viz.update(env, env_id=0)  # 在仿真视口中绘制彩色高程小球
```

`ElevationVizIsaacLab.update` 内部完成全部流程：  
深度图 → 反投影 → yaw 系 → 高程图 → `VisualizationMarkers.visualize`。

颜色从**蓝（最低）→ 青 → 绿 → 黄 → 橙 → 红（最高）**共 8 档，按当前帧 z 范围自动归一化。

### play.py 中的对应关系

`legged_lab/scripts/play.py` 中原有的私有函数 `_depth_image_plane_unproject_to_world` 和 `_camera_points_world_to_yaw_frame` 已被本包替换。  
`play.py` 中保留了两个轻量适配器函数（`_depth_image_plane_unproject_to_world_np`、`_cam_intrinsics_from_isaaclab`），用于将 Isaac Lab 的 tensor 接口桥接到本包的 NumPy API。  

---

## 坐标系约定

| 坐标系 | 原点 | x 轴 | y 轴 | z 轴 |
|--------|------|-------|-------|-------|
| 世界系 | 固定点 | 前方 / 东 | 左 / 北 | 上 |
| 相机系（Isaac Lab） | 相机光心 | 前方（光轴） | 左 | 上 |
| 机器人 yaw 系 | 机器人根部关节 | 前方（机头方向） | 左 | 上 |

四元数格式全程使用 **(w, x, y, z)**，与 Isaac Lab `math_utils` 保持一致。

---

## 扩展说明

若需支持其他深度约定（如标准 `distance_to_camera` = 欧氏距离），可在 `camera.py` 中新增反投影函数：

```python
def unproject_euclidean(depth_hw, intrinsics, cam_pos_world, cam_quat_world):
    """适用于输出欧氏（射线长度）深度的传感器。"""
    dirs_cam = _build_ray_directions(intrinsics)   # (N, 3) 单位向量
    pts_cam  = depth_hw.reshape(-1, 1) * dirs_cam  # 沿单位射线缩放
    ...
```

---

## 许可证

BSD-3-Clause
