"""Isaac Lab 适配示例：depth_elevation 包在 Dex camera 环境中的完整调用流程。

展示如何：
1. 从 Isaac Lab GroupedRayCasterCamera 实例提取内参与外参
2. 调用 depth_elevation 包完成 深度图 → 点云 → 高程图
3. 用 VisualizationMarkers 在仿真视口中绘制高程彩球

该示例与 legged_lab/scripts/play.py 中的 _update_elevation_viz_3d 实现同源，
可作为迁移到其他机器人/环境时的参考模板。

用法
----
在 play.py 或任意 Isaac Lab 脚本中导入并调用::

    from depth_elevation.examples.example_isaaclab import ElevationVizIsaacLab
    viz = ElevationVizIsaacLab()
    # 在每步循环中：
    viz.update(env, env_id=0)
"""

from __future__ import annotations

import numpy as np
import torch

from depth_elevation import (
    CameraIntrinsics,
    ElevationConfig,
    unproject_depth,
    world_to_yaw_frame,
    build_elevation_map,
    elevation_map_to_world_centers,
)

# ── 8 色彩条（蓝=低 → 红=高）────────────────────────────────────────────────
_ELEV_VIZ_COLORS = [
    (0.0,  0.0,  0.9),
    (0.0,  0.45, 1.0),
    (0.0,  0.85, 0.95),
    (0.0,  0.85, 0.3),
    (0.85, 0.9,  0.0),
    (1.0,  0.65, 0.0),
    (1.0,  0.2,  0.0),
    (0.85, 0.0,  0.0),
]


def cam_intrinsics_from_isaaclab(cam, env_id: int) -> CameraIntrinsics:
    """从 Isaac Lab 相机对象提取单环境内参。

    优先读取 ``cam._data.intrinsic_matrices[env_id]``（支持域随机化后的逐环境值）。
    若不可用，则从 ``cam.cfg.pattern_cfg`` 静态计算。

    Args:
        cam:    ``GroupedRayCasterCamera`` 实例（``env.camera``）。
        env_id: 环境索引。

    Returns:
        :class:`~depth_elevation.CameraIntrinsics`。
    """
    max_d = float(getattr(cam.cfg, "max_distance", 3.0))

    if hasattr(cam, "_data") and hasattr(cam._data, "intrinsic_matrices"):
        K = cam._data.intrinsic_matrices[env_id].cpu().numpy()
        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])
    else:
        pc = cam.cfg.pattern_cfg
        h_ap = pc.horizontal_aperture
        v_ap = pc.vertical_aperture if pc.vertical_aperture is not None else h_ap * pc.height / pc.width
        fx = pc.width * pc.focal_length / h_ap
        fy = pc.height * pc.focal_length / v_ap
        cx, cy = pc.width / 2.0, pc.height / 2.0

    h = int(cam.cfg.pattern_cfg.height)
    w = int(cam.cfg.pattern_cfg.width)
    return CameraIntrinsics(fx=fx, fy=fy, cx=cx, cy=cy, width=w, height=h, max_depth=max_d)


class ElevationVizIsaacLab:
    """在 Isaac Lab 仿真视口中实时绘制深度高程图彩球。

    Parameters
    ----------
    config:
        高程网格配置，默认前方 1.5 m × 左右 0.6 m，分辨率 0.05 m。
    prim_path:
        USD 场景中 VisualizationMarkers 的挂载路径。
    """

    def __init__(
        self,
        config: ElevationConfig | None = None,
        prim_path: str = "/Visuals/ElevationMapSpheres",
    ):
        self.config = config or ElevationConfig()
        self.prim_path = prim_path
        self._viz = None   # VisualizationMarkers，懒加载

    def _get_viz(self):
        if self._viz is not None:
            return self._viz
        try:
            from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
            import isaaclab.sim as sim_utils

            radius = self.config.resolution / 2.0
            markers = {
                f"h{i}": sim_utils.SphereCfg(
                    radius=radius,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c),
                )
                for i, c in enumerate(_ELEV_VIZ_COLORS)
            }
            cfg = VisualizationMarkersCfg(prim_path=self.prim_path, markers=markers)
            self._viz = VisualizationMarkers(cfg)
        except Exception as exc:
            print(f"[ElevationVizIsaacLab] VisualizationMarkers init failed: {exc}")
        return self._viz

    def update(self, env, env_id: int = 0) -> None:
        """用当前帧深度图刷新视口高程彩球。

        Args:
            env:    Isaac Lab 环境实例（需有 ``env.camera`` 和 ``env.robot``）。
            env_id: 环境索引（默认 0）。
        """
        viz = self._get_viz()
        if viz is None:
            return

        if not getattr(env.cfg.scene.camera, "enable_depth_camera", False):
            return
        if not hasattr(env, "camera"):
            return
        cam = env.camera
        out = cam.data.output
        if "distance_to_image_plane" not in out:
            return

        env_id = int(np.clip(env_id, 0, env.num_envs - 1))

        # ── 步骤 1：提取内外参 ─────────────────────────────────────────────
        intr = cam_intrinsics_from_isaaclab(cam, env_id)
        cam_pos_np: np.ndarray = cam._data.pos_w[env_id].cpu().numpy()           # (3,)
        cam_quat_np: np.ndarray = cam._data.quat_w_world[env_id].cpu().numpy()   # (4,) w,x,y,z
        root_pos_np: np.ndarray = env.robot.data.root_pos_w[env_id].cpu().numpy()
        root_quat_np: np.ndarray = env.robot.data.root_quat_w[env_id].cpu().numpy()

        # ── 步骤 2：深度图 → 世界系点云 ───────────────────────────────────
        raw_depth = out["distance_to_image_plane"][env_id, :, :, 0].detach()
        depth_np: np.ndarray = raw_depth.cpu().numpy()   # (H, W)

        pts_world, valid = unproject_depth(depth_np, intr, cam_pos_np, cam_quat_np)
        # 额外过滤深度噪声（distance_to_image_plane 在未命中时置为 max_distance）
        near_max = depth_np.reshape(-1) > intr.max_depth * 0.999
        valid = valid & ~near_max

        if not valid.any():
            viz.set_visibility(False)
            return

        # ── 步骤 3：世界系 → 机器人 yaw 系 ───────────────────────────────
        pts_yaw = world_to_yaw_frame(pts_world[valid], root_pos_np, root_quat_np)

        # ── 步骤 4：点云 → 高程网格 ───────────────────────────────────────
        emap = build_elevation_map(pts_yaw, self.config)

        # ── 步骤 5：高程格中心 → 世界系（用于 VisualizationMarkers 位置）──
        world_centers, z_vals = elevation_map_to_world_centers(emap, root_pos_np, root_quat_np)
        if world_centers.shape[0] == 0:
            viz.set_visibility(False)
            return

        # ── 步骤 6：按相对高度映射 8 色，在视口中绘制彩球 ─────────────────
        device = env.device
        translations = torch.from_numpy(world_centers).float().to(device)

        z_min, z_max = float(z_vals.min()), float(z_vals.max())
        z_range = max(z_max - z_min, 0.05)
        bin_idx = np.floor(8.0 * (z_vals - z_min) / z_range).clip(0, 7).astype(np.int32)
        marker_indices = torch.from_numpy(bin_idx).long().to(device)

        viz.visualize(translations=translations, marker_indices=marker_indices)
