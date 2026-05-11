"""点云 → BEV 高程网格。"""

from __future__ import annotations

import numpy as np

from .types import ElevationConfig, ElevationMap


def build_elevation_map(
    points_yaw: np.ndarray,
    config: ElevationConfig,
) -> ElevationMap:
    """将 yaw 系点云聚合为 BEV 高程网格。

    每个网格格子取落入该格所有点中 **z 最大值**（相对机器人根部高度），
    无点覆盖的格子填 ``config.fill_value``。

    Args:
        points_yaw: ``(N, 3)`` yaw 系坐标点。坐标约定：x 前、y 左、z 上，
                    原点为机器人根部。可以包含 NaN（会被自动过滤）。
        config:     :class:`ElevationConfig` 网格参数。

    Returns:
        :class:`ElevationMap`，包含 ``grid``、``x_centers``、``y_centers``、
        ``occupied``、``config``。

    Example::

        from depth_elevation.types import ElevationConfig
        from depth_elevation.elevation import build_elevation_map
        import numpy as np

        pts = np.random.randn(500, 3)
        cfg = ElevationConfig()
        emap = build_elevation_map(pts, cfg)
        print(emap.grid.shape)   # (30, 24)
    """
    points_yaw = np.asarray(points_yaw, dtype=np.float64)

    n_x, n_y = config.n_x, config.n_y
    grid = np.full((n_x, n_y), fill_value=-np.inf, dtype=np.float32)

    if points_yaw.shape[0] > 0:
        # 过滤 NaN
        finite_mask = np.all(np.isfinite(points_yaw), axis=-1)
        pts = points_yaw[finite_mask]

        if pts.shape[0] > 0:
            px, py, pz = pts[:, 0], pts[:, 1], pts[:, 2]

            ix = np.floor((px - config.x_min) / config.resolution).astype(np.int32)
            iy = np.floor((py - config.y_min) / config.resolution).astype(np.int32)
            in_roi = (ix >= 0) & (ix < n_x) & (iy >= 0) & (iy < n_y)

            ix_v, iy_v, pz_v = ix[in_roi], iy[in_roi], pz[in_roi].astype(np.float32)
            np.maximum.at(grid, (ix_v, iy_v), pz_v)

    occupied = grid > -np.inf
    grid[~occupied] = config.fill_value

    x_centers = config.x_min + (np.arange(n_x) + 0.5) * config.resolution
    y_centers = config.y_min + (np.arange(n_y) + 0.5) * config.resolution

    return ElevationMap(
        grid=grid,
        x_centers=x_centers.astype(np.float32),
        y_centers=y_centers.astype(np.float32),
        occupied=occupied,
        config=config,
    )


def elevation_map_to_world_centers(
    emap: ElevationMap,
    root_pos: np.ndarray,
    root_quat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """将高程图中有点覆盖格子的中心坐标从 yaw 系转回世界系。

    用于在仿真视口里绘制彩色小球（``VisualizationMarkers``）。

    Args:
        emap:       :class:`ElevationMap` 高程图。
        root_pos:   ``(3,)`` 机器人根部世界系位置。
        root_quat:  ``(4,)`` 机器人根部世界系姿态 (w,x,y,z)。

    Returns:
        - ``world_centers``: ``(K, 3)`` 有效格子在世界系中的位置（高度取对应格子的 max z）。
        - ``z_vals``:        ``(K,)`` 每个格子的 z 值（yaw 系，用于颜色映射）。
    """
    from .transform import yaw_frame_to_world

    occ_ix, occ_iy = np.where(emap.occupied)
    if occ_ix.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros(0, dtype=np.float32)

    xc = emap.x_centers[occ_ix]
    yc = emap.y_centers[occ_iy]
    zc = emap.grid[occ_ix, occ_iy].astype(np.float64)
    pts_yaw = np.stack([xc, yc, zc], axis=1)

    world_centers = yaw_frame_to_world(pts_yaw, root_pos, root_quat).astype(np.float32)
    return world_centers, emap.grid[occ_ix, occ_iy]
