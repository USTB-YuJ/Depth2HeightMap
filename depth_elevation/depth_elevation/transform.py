"""坐标系变换工具（纯 NumPy）。

约定：四元数格式为 ``(w, x, y, z)``，与 Isaac Lab ``math_utils`` 一致。
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# 四元数基础运算
# ---------------------------------------------------------------------------

def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """四元数乘法，形状 ``(..., 4)`` × ``(..., 4)`` → ``(..., 4)``。"""
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    return np.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        axis=-1,
    )


def quat_inv(q: np.ndarray) -> np.ndarray:
    """单位四元数的逆（共轭），形状 ``(..., 4)``。"""
    inv = q.copy()
    inv[..., 1:] = -inv[..., 1:]
    return inv


def quat_apply(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """将旋转四元数 q 作用于向量 v。

    Args:
        q: ``(..., 4)`` 单位四元数 (w,x,y,z)。
        v: ``(..., 3)`` 向量。

    Returns:
        ``(..., 3)`` 旋转后的向量。
    """
    # 扩展 v 为纯四元数
    v_quat = np.concatenate([np.zeros_like(v[..., :1]), v], axis=-1)
    return quat_mul(quat_mul(q, v_quat), quat_inv(q))[..., 1:]


def quat_apply_inverse(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """将旋转四元数 q 的逆作用于向量 v（即在 q 所定义的坐标系中表达 v）。"""
    return quat_apply(quat_inv(q), v)


# ---------------------------------------------------------------------------
# yaw frame 工具
# ---------------------------------------------------------------------------

def quat_yaw_only(q_wxyz: np.ndarray) -> np.ndarray:
    """从完整旋转四元数中提取仅包含 yaw 分量的四元数。

    等价于 Isaac Lab ``math_utils.yaw_quat``。

    Args:
        q_wxyz: ``(4,)`` 或 ``(N, 4)`` 单位四元数 (w,x,y,z)。

    Returns:
        同形状的 yaw-only 四元数（已归一化）。
    """
    q = np.asarray(q_wxyz, dtype=np.float64)
    scalar = q.ndim == 1
    if scalar:
        q = q[np.newaxis]  # (1, 4)

    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    # 仅保留绕 z 轴旋转：从 euler yaw 角重建
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    half = yaw / 2.0
    yaw_q = np.stack([np.cos(half), np.zeros_like(half), np.zeros_like(half), np.sin(half)], axis=-1)

    if scalar:
        return yaw_q[0]
    return yaw_q


def world_to_yaw_frame(
    points_world: np.ndarray,
    root_pos: np.ndarray,
    root_quat: np.ndarray,
) -> np.ndarray:
    """世界坐标点 → 机器人 yaw 系（x 前、y 左、z 上）。

    yaw 系原点为机器人根部，只旋转掉 yaw 角，保留相对高度。

    Args:
        points_world: ``(N, 3)`` 世界系坐标点。
        root_pos:     ``(3,)`` 机器人根部世界系位置。
        root_quat:    ``(4,)`` 机器人根部世界系姿态 (w,x,y,z)。

    Returns:
        ``(N, 3)`` yaw 系坐标点。
    """
    points_world = np.asarray(points_world, dtype=np.float64)
    root_pos = np.asarray(root_pos, dtype=np.float64)
    root_quat = np.asarray(root_quat, dtype=np.float64)

    rel = points_world - root_pos[np.newaxis]          # (N, 3)
    yaw_q = quat_yaw_only(root_quat)                   # (4,)
    yaw_q_inv = quat_inv(yaw_q[np.newaxis])            # (1, 4)
    yaw_q_inv_n = np.broadcast_to(yaw_q_inv, (rel.shape[0], 4))
    return quat_apply(yaw_q_inv_n, rel)                # (N, 3)


def yaw_frame_to_world(
    points_yaw: np.ndarray,
    root_pos: np.ndarray,
    root_quat: np.ndarray,
) -> np.ndarray:
    """yaw 系坐标点 → 世界系（world_to_yaw_frame 的逆变换）。

    Args:
        points_yaw: ``(N, 3)`` yaw 系坐标点。
        root_pos:   ``(3,)`` 机器人根部世界系位置。
        root_quat:  ``(4,)`` 机器人根部世界系姿态 (w,x,y,z)。

    Returns:
        ``(N, 3)`` 世界系坐标点。
    """
    points_yaw = np.asarray(points_yaw, dtype=np.float64)
    root_pos = np.asarray(root_pos, dtype=np.float64)
    root_quat = np.asarray(root_quat, dtype=np.float64)

    yaw_q = quat_yaw_only(root_quat)                   # (4,)
    yaw_q_n = np.broadcast_to(yaw_q[np.newaxis], (points_yaw.shape[0], 4))
    return root_pos[np.newaxis] + quat_apply(yaw_q_n, points_yaw)
