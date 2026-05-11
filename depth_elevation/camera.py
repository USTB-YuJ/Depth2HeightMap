"""深度图反投影（image_plane 约定，与 Isaac Lab 一致）。

Isaac Lab ``distance_to_image_plane`` 定义：
    d = (R^T (t * dir_w))_x
其中 R 为相机世界姿态旋转矩阵，dir_w 为世界系下射线方向（单位向量），t 为沿射线的真实距离。

在相机系下：
    dir_cam = R^T * dir_w
    d       = t * dir_cam.x   →  t = d / dir_cam.x
    P_cam   = t * dir_cam
    P_world = pos_w + R * P_cam
"""

from __future__ import annotations

import numpy as np

from .types import CameraIntrinsics
from .transform import quat_apply, quat_inv


def _build_ray_directions(intrinsics: CameraIntrinsics) -> np.ndarray:
    """生成图像平面所有像素在相机系下的归一化射线方向。

    遵循与 Isaac Lab ``PinholeCameraPatternCfg.func`` 相同的坐标约定：
    相机系 x 轴指向前方（光轴），y/z 轴分别对应行/列反方向。

    具体映射（Isaac Lab 的针孔模型，u 列索引从左到右，v 行索引从上到下）：

        dir_cam = normalize([ 1,
                             -(u - cx) / fx,
                             -(v - cy) / fy ])

    此处负号来自 Isaac Lab 将图像 y/z 翻转（与 ROS REP-103 相机系一致）。

    Returns:
        ``(H*W, 3)`` 相机系归一化射线方向，顺序与展平后的深度图像素对应。
    """
    h, w = intrinsics.height, intrinsics.width
    # 像素中心坐标（u=列, v=行）
    u = np.arange(w, dtype=np.float64)  # (W,)
    v = np.arange(h, dtype=np.float64)  # (H,)
    uu, vv = np.meshgrid(u, v)          # (H, W)

    dx = np.ones((h, w), dtype=np.float64)
    dy = -(uu - intrinsics.cx) / intrinsics.fx
    dz = -(vv - intrinsics.cy) / intrinsics.fy

    dirs = np.stack([dx, dy, dz], axis=-1)   # (H, W, 3)
    norms = np.linalg.norm(dirs, axis=-1, keepdims=True)
    dirs /= np.where(norms > 1e-10, norms, 1.0)
    return dirs.reshape(-1, 3)               # (H*W, 3)


def unproject_depth(
    depth_hw: np.ndarray,
    intrinsics: CameraIntrinsics,
    cam_pos_world: np.ndarray,
    cam_quat_world: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """将 ``distance_to_image_plane`` 深度图反投影为世界系 3D 点云。

    Args:
        depth_hw:       ``(H, W)`` 米制深度，类型 float32 / float64。
                        值为 Isaac Lab ``distance_to_image_plane``（前向轴投影距离）。
        intrinsics:     :class:`CameraIntrinsics` 相机内参。
        cam_pos_world:  ``(3,)`` 相机原点在世界系中的位置。
        cam_quat_world: ``(4,)`` 相机在世界系中的姿态，格式 (w,x,y,z)。

    Returns:
        - ``points_world``: ``(H*W, 3)`` float64，世界系 3D 点（包括无效点，用 NaN 填充）。
        - ``valid_mask``:   ``(H*W,)`` bool，True 表示该像素有效。

    Notes:
        从 Isaac Lab 的 ``cam._data.pos_w[env_id]``、
        ``cam._data.quat_w_world[env_id]`` 取外参；
        从 ``cam._data.intrinsic_matrices[env_id]`` 或
        ``_pinhole_base_intrinsics_from_pattern_cfg(cam.cfg.pattern_cfg)`` 取内参。
    """
    depth_hw = np.asarray(depth_hw, dtype=np.float64)
    cam_pos_world = np.asarray(cam_pos_world, dtype=np.float64)
    cam_quat_world = np.asarray(cam_quat_world, dtype=np.float64)

    h, w = intrinsics.height, intrinsics.width
    if depth_hw.shape != (h, w):
        raise ValueError(
            f"depth_hw shape {depth_hw.shape} does not match "
            f"intrinsics ({h}, {w})."
        )

    d_flat = depth_hw.reshape(-1)                       # (N,)
    dir_cam = _build_ray_directions(intrinsics)          # (N, 3)
    n = d_flat.shape[0]

    # 将相机系射线方向转到世界系
    q_n = np.broadcast_to(cam_quat_world[np.newaxis], (n, 4))
    dir_w = quat_apply(q_n, dir_cam)                    # (N, 3)

    # 转回相机系（验证 t 计算等价于直接用 dir_cam）
    # Isaac Lab 约定：d = dir_cam.x * t，dir_cam 已是归一化相机系方向
    den = dir_cam[:, 0]                                  # (N,)
    dir_ok = np.abs(den) > 1e-4
    depth_ok = (
        np.isfinite(d_flat)
        & (d_flat > intrinsics.min_depth if hasattr(intrinsics, "min_depth") else d_flat > 1e-4)
        & (d_flat < intrinsics.max_depth * 0.999)
    )
    valid = dir_ok & depth_ok

    t = np.where(valid, d_flat / den, np.nan)           # (N,)
    P_cam = t[:, np.newaxis] * dir_cam                  # (N, 3)

    # 世界系：pos_w + R * P_cam
    P_world = cam_pos_world[np.newaxis] + quat_apply(q_n, P_cam)  # (N, 3)

    # 结果中的 NaN 传播：对无效像素置 NaN
    P_world[~valid] = np.nan

    valid = valid & np.all(np.isfinite(P_world), axis=-1)
    return P_world, valid
