"""数据类定义：CameraIntrinsics、ElevationConfig、ElevationMap。"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class CameraIntrinsics:
    """针孔相机内参。

    约定与 Isaac Lab ``PinholeCameraPatternCfg`` 一致：
    - ``fx``, ``fy``：像素单位焦距
    - ``cx``, ``cy``：主点（像素坐标）
    - 深度格式为 ``distance_to_image_plane``（沿相机前向轴的投影距离）
    """

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    max_depth: float = 3.0
    """有效深度上限（米）。超过此值的像素视为无效（未命中）。"""


@dataclass
class ElevationConfig:
    """高程网格配置（机器人 yaw 系，单位 m）。

    坐标约定：x 前（正向前）、y 左、z 上，原点为机器人根部。
    """

    x_min: float = 0.0
    """前方起始距离（m）。"""
    x_max: float = 1.5
    """前方最大距离（m）。"""
    y_min: float = -0.6
    """左右最小值（负为右侧，m）。"""
    y_max: float = 0.6
    """左右最大值（正为左侧，m）。"""
    resolution: float = 0.05
    """网格分辨率（m/格）。默认 30×24 = 720 格。"""
    min_depth: float = 1e-4
    """深度有效下限（m）。低于此值的像素丢弃。"""
    fill_value: float = 0.0
    """无点覆盖格子的填充值（m，相对机器人根部高度）。"""

    @property
    def n_x(self) -> int:
        return int(round((self.x_max - self.x_min) / self.resolution))

    @property
    def n_y(self) -> int:
        return int(round((self.y_max - self.y_min) / self.resolution))


@dataclass
class ElevationMap:
    """高程网格结果。

    ``grid[i, j]`` 为第 i 行（x 方向）、第 j 列（y 方向）格子的最大 z 值，
    单位为米，相对机器人根部（yaw 系 z 轴）。空格子填 ``fill_value``。
    """

    grid: np.ndarray
    """形状 ``(n_x, n_y)`` float32，每格 max z（yaw 系，相对根部）。"""
    x_centers: np.ndarray
    """形状 ``(n_x,)``，每行格中心的 x 坐标（m）。"""
    y_centers: np.ndarray
    """形状 ``(n_y,)``，每列格中心的 y 坐标（m）。"""
    occupied: np.ndarray
    """形状 ``(n_x, n_y)`` bool，True 表示该格至少有一个点覆盖。"""
    config: ElevationConfig = field(repr=False)
    """生成该高程图所用的配置。"""
