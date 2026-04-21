# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from dataclasses import MISSING
from isaaclab.utils.noise import NoiseModel, NoiseModelCfg
import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
import numpy as np
import torch
from typing import Optional, Tuple
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.utils.noise import AdditiveGaussianNoiseCfg as Gnoise

from unitree_rl_lab.tasks.locomotion import mdp
from math import pi

from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG
from unitree_rl_lab.assets.robots.unitree import UNITREE_GO2_CFG_lab

##
# Pre-defined configs
##
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip


class BurstNoiseModel(NoiseModel):
    """Stateful per-env burst noise with random intervals."""

    # Provide a __name__ so callable_to_string can serialize instances
    __name__ = "BurstNoiseModel"
    
    # 类级别的共享状态字典
    _shared_states = {}

    def __init__(self, cfg, num_envs: int, device: str | torch.device):
        # ObservationManager passes (cfg, num_envs, device)
        super().__init__(cfg, num_envs, device)
        # Keep a public alias for convenience since the base stores it as _noise_model_cfg
        self.cfg = cfg
        self.device = torch.device(device)
        self.burst_bias = None
        print(f"Initializing BurstNoiseModel on device {self.device} for {num_envs} envs.")
        
        # 检查是否使用共享状态
        shared_id = getattr(cfg, "shared_id", None)
        
        if shared_id is not None:
            # 使用共享状态
            if shared_id not in self._shared_states:
                # 首次创建，初始化共享状态
                self._shared_states[shared_id] = {
                    "steps_until_burst": self._sample_range(self.cfg.interval_steps_range, num_envs, self.device),
                    "burst_steps_left": torch.zeros(num_envs, device=self.device, dtype=torch.long),
                }
            # 引用共享状态
            self.steps_until_burst = self._shared_states[shared_id]["steps_until_burst"]
            self.burst_steps_left = self._shared_states[shared_id]["burst_steps_left"]
            self.shared_id = shared_id
        else:
            # 独立状态
            self.steps_until_burst = self._sample_range(self.cfg.interval_steps_range, num_envs, self.device)
            self.burst_steps_left = torch.zeros(num_envs, device=self.device, dtype=torch.long)
            self.shared_id = None

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        if self.steps_until_burst is None or self.burst_steps_left is None:
            return data
        if self.burst_bias is None or self.burst_bias.shape != data.shape:
            self.burst_bias = torch.zeros_like(data)
        

        # if torch.rand(1).item() < 0.1:  # 1% 概率打印，避免刷屏
        #     print(f"[BurstDebug] shared_id={self.shared_id}")
        #     print(f"  steps_until_burst: min={self.steps_until_burst.min().item()}, max={self.steps_until_burst.max().item()}, mean={self.steps_until_burst.float().mean().item():.1f}")
        #     print(f"  burst_steps_left: min={self.burst_steps_left.min().item()}, max={self.burst_steps_left.max().item()}, sum={self.burst_steps_left.sum().item()}")
        
        
        data_noisy = self._apply_base_noise(data)

        bursting = self.burst_steps_left > 0

        if getattr(self.cfg, "normalize", False):
            norms = torch.linalg.norm(data_noisy, dim=-1, keepdim=True)
            data_noisy = data_noisy / torch.clamp(norms, min=1e-8)
            

        new_burst_steps = torch.where(bursting, self.burst_steps_left - 1, self.burst_steps_left)
        burst_end_mask = bursting & (new_burst_steps == 0)
        self.burst_steps_left = new_burst_steps

        self.steps_until_burst = torch.where(bursting, self.steps_until_burst, self.steps_until_burst - 1)

        start_mask = (~bursting) & (self.steps_until_burst <= 0)
        if start_mask.any():
            count = int(start_mask.sum().item())
            self.burst_steps_left[start_mask] = self._sample_range(self.cfg.burst_steps_range, count, self.device)
            self.steps_until_burst[start_mask] = 0
            new_bias = torch.randn_like(data_noisy[start_mask]) * float(self.cfg.burst_std)
            self.burst_bias[start_mask] = new_bias

        if bursting.any():
            data_noisy[bursting] = data_noisy[bursting] + self.burst_bias[bursting]
            print("INFO:bursting")
        else:
            print("INFO: not bursting")

        if burst_end_mask.any():
            count = int(burst_end_mask.sum().item())
            self.steps_until_burst[burst_end_mask] = self._sample_range(self.cfg.interval_steps_range, count, self.device)
            self.burst_bias[burst_end_mask] = 0.0

        if self.cfg.burst_clip is not None:
            low, high = self.cfg.burst_clip
            data_noisy = torch.clamp(data_noisy, min=low, max=high)

        return data_noisy

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
            count = self.burst_steps_left.shape[0]
        else:
            if isinstance(env_ids, slice):
                # slice(None) handled above; for partial slice, materialize indices
                env_ids = torch.arange(self.burst_steps_left.shape[0], device=self.device)[env_ids]
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
            count = env_ids.numel()
        if count == 0:
            return
        self.steps_until_burst[env_ids] = self._sample_range(self.cfg.interval_steps_range, count, self.device)
        self.burst_steps_left[env_ids] = 0
        if self.burst_bias is not None:
            self.burst_bias[env_ids] = 0.0

    def _apply_base_noise(self, data: torch.Tensor) -> torch.Tensor:
        cfg = getattr(self.cfg, "base_noise", None)
        if cfg is None:
            return data

        if hasattr(cfg, "n_min") and hasattr(cfg, "n_max"):
            noise = torch.rand_like(data) * (float(cfg.n_max) - float(cfg.n_min)) + float(cfg.n_min)
            return data + noise

        if hasattr(cfg, "std"):
            mean = float(getattr(cfg, "mean", 0.0))
            std = float(cfg.std)
            return data + torch.randn_like(data) * std + mean

        return data

    @staticmethod
    def _sample_range(bounds: Tuple[int, int], count: int, device: torch.device) -> torch.Tensor:
        low, high = bounds
        if high < low:
            high = low
        return torch.randint(low, high + 1, (count,), device=device, dtype=torch.long)


@configclass
class BurstNoiseModelCfg(NoiseModelCfg):
    #step
    interval_steps_range: Tuple[int, int] = (30, 60)
    burst_steps_range: Tuple[int, int] = (10, 25)
    burst_std: float = 1.0
    base_noise: Optional[object] = None
    burst_clip: Optional[Tuple[float, float]] = None
    noise_cfg: object = Gnoise(mean=0.0, std=0.0)
    shared_id: Optional[str] = None  # 用于标识共享组
    normalize: bool = False

    def __post_init__(self):
        try:
            super().__post_init__()
        except AttributeError:
            pass
        self.class_type = BurstNoiseModel



##
# Scene definition
##
def stitched_modular(difficulty, cfg):
    """
    Generates a stitched terrain based on the configured layout.
    cfg.sub_terrains should contain keys like "top_left", "top_right", etc.
    But standard cfg structure doesn't easily pass custom params per sub-terrain function call 
    unless we embed them in the cfg class or use a global/closure.
    
    We will use a fixed pattern based on the 'layout' attribute we add to the cfg.
    """
    # Grid size per sub-terrain
    quad_size = (cfg.size[0] / 2.0, cfg.size[1] / 2.0)
    STITCH_PLATFORM_FRACTION = 0.25
    platform_w = STITCH_PLATFORM_FRACTION * min(quad_size)
    
    # Define available sub-terrain configs
    # Flat
    flat_cfg = terrain_gen.HfRandomUniformTerrainCfg(
        noise_range=(0.0, 0.0), noise_step=0.01, border_width=0.0
    )
    # Stairs
    stairs_cfg = terrain_gen.MeshPyramidStairsTerrainCfg(
        step_height_range=(0.05, 0.23),
        step_width=0.3,
        platform_width=platform_w,
        border_width=0.0,
        holes=False,
    )
    # Gap
    # Go2 length is approx 0.7m. Max gap should be 0.5 * length = 0.35m
    gap_cfg = terrain_gen.MeshGapTerrainCfg(
        gap_width_range=(0.05, 0.35),
        platform_width=platform_w,
    )
    # Slope (Pyramid)
    slope_cfg = terrain_gen.HfPyramidSlopedTerrainCfg(
        slope_range=(0.1, 0.4),
        platform_width=platform_w,
        border_width=0.0,
    )
    # Rough
    rough_cfg = terrain_gen.HfRandomUniformTerrainCfg(
        noise_range=(0.01, 0.06),
        noise_step=0.01,
        border_width=0.0,
    )
    
    # Hurdles (Rails)
    hurdles_cfg = terrain_gen.MeshRailsTerrainCfg(
        rail_thickness_range=(0.05, 0.2),
        rail_height_range=(0.30, 0.05),
        platform_width=platform_w,
    )
    
    # Stepping Stones (Random Grid)
    stepping_stones_cfg = terrain_gen.MeshRandomGridTerrainCfg(
        grid_width=0.35,
        grid_height_range=(0.025, 0.1),
        platform_width=platform_w,
        holes=False,
    )
    
    # Map layout names to configs
    terrain_map = {
        "flat": flat_cfg,
        "stairs": stairs_cfg,
        "gap": gap_cfg,
        "slope": slope_cfg,
        "rough": rough_cfg,
        "hurdles": hurdles_cfg,
        "stepping_stones": stepping_stones_cfg,
    }
    
    # Layout selection: if cfg.layouts exists, pick one at random each call; else fall back to cfg.layout or default
    # layouts = [
    #     ["stepping_stones", "stairs", "rough", "slope"],
    #     ["slope", "stairs", "rough", "stepping_stones"],
    #     ["slope", "rough", "stairs", "stepping_stones"],
    #     ["rough", "slope", "stairs", "stairs"],
    # ]
    
    # layouts = [
    #     ["stepping_stones", "stairs", "rough", "hurdles"],
    #     ["slope", "stairs", "rough", "stepping_stones"],
    #     ["slope", "hurdles", "stairs", "stepping_stones"],
    #     ["rough", "slope", "hurdles", "stairs"],
    # ]
    
    # layouts = [
    #     ["stepping_stones", "rough", "stairs", "slope"],
    #     ["slope", "stairs", "rough", "stepping_stones"],
    #     ["slope", "stairs", "rough", "stepping_stones"],
        #     ["rough", "stairs", "slope", "stepping_stones"],
    # ]

    # layouts = [
    #     ["stairs", "rough", "stairs", "slope"],
    #     ["slope", "stairs", "rough", "rough"],
    #     ["slope", "stairs", "rough", "stairs"],
    #     ["rough", "stairs", "slope", "rough"],
    # ]

    layouts = [
        ["stepping_stones", "rough", "stairs", "slope"],
        ["slope", "stairs", "rough", "stepping_stones"],
        ["slope", "stairs", "rough", "stepping_stones"],
        ["rough", "stairs", "rough", "stepping_stones"],
    ]

    #layouts = [["flat", "flat", "flat", "flat"]]

    if layouts:
        layout = layouts[np.random.randint(len(layouts))]
        print("INFO: Using modular terrain layout:", layout)
    else:
        layout = getattr(cfg, "layout", ["flat", "stairs", "stairs", "hurdles"])
    # layout = getattr(cfg, "layout", ["flat", "stepping_stones", "gap", "hurdles"])
    
    base_cfgs = [terrain_map[name] for name in layout]
    
    # Offsets for 2x2 grid (Bottom-Left, Top-Left, Top-Right, Bottom-Right)
    # Note: offsets in original code were:
    # (0,0), (0,Y), (X,Y), (X,0) -> BL, TL, TR, BR
    STITCH_SEAM_OVERLAP = 0.0
    offsets = [
        (0.0, 0.0),
        (0.0, quad_size[1] - STITCH_SEAM_OVERLAP),
        (quad_size[0] - STITCH_SEAM_OVERLAP, quad_size[1] - STITCH_SEAM_OVERLAP),
        (quad_size[0] - STITCH_SEAM_OVERLAP, 0.0),
    ]
    
    meshes_out = []
    first_origin = None
    for base_cfg, (ox, oy) in zip(base_cfgs, offsets):
        seg_cfg = base_cfg.replace(size=quad_size, proportion=1.0)
        seg_meshes, seg_origin = seg_cfg.function(difficulty, seg_cfg)
        if first_origin is None:
            first_origin = np.array(seg_origin, dtype=float)
        for m in seg_meshes:
            m_copy = m.copy()
            T = np.eye(4)
            T[0, -1] = ox
            T[1, -1] = oy
            m_copy.apply_transform(T)
            meshes_out.append(m_copy)
    if first_origin is None:
        first_origin = np.zeros(3, dtype=float)
    return meshes_out, first_origin

# Config for Stair-Gap Combination (and others via layout modification)
STAIR_GAP_TERRAINS_SIZE = (8.0, 8.0)
MODULAR_TERRAINS_CFG = terrain_gen.TerrainGeneratorCfg(
    size=STAIR_GAP_TERRAINS_SIZE,
    border_width=2.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "modular_2x2": terrain_gen.SubTerrainBaseCfg(
            function=stitched_modular,
            proportion=1.0,
            size=STAIR_GAP_TERRAINS_SIZE,
        ),
    },
)

@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=MODULAR_TERRAINS_CFG,
        max_init_terrain_level=0,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    # robots
    robot: ArticulationCfg = MISSING
    # sensors
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.0, 0.7]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.1,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0), lin_vel_y=(-0.5, 0.5), ang_vel_z=(-1.0, 1.0), heading=(-math.pi, math.pi)
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True)


SHARED_INTERVAL_STEPS_RANGE = (180, 250)
SHARED_BURST_STEPS_RANGE = (50, 100)

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)

        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            scale=0.2,
            clip=(-100, 100),
            noise=Gnoise(mean=0.0, std=0.2),
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            clip=(-100, 100),
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        joint_pos_rel = ObsTerm(
            func=mdp.joint_pos_rel,
            clip=(-100, 100),
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel,
            scale=0.05,
            clip=(-100, 100),
            noise=Gnoise(mean=0.0, std=1.5),
        )
        joint_effort = ObsTerm(
            func=mdp.joint_effort,
            scale=0.01,
            clip=(-100, 100),
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, clip=(-100, 100), params={"command_name": "base_velocity"}
        )
        last_action = ObsTerm(func=mdp.last_action, clip=(-100, 100))

        def __post_init__(self):
            self.history_length = 3
            self.enable_corruption = True
            self.concatenate_terms = True
            self.flatten_history_dim = True
    
    # observation groups
    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for critic group."""
        
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, clip=(-100, 100))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, clip=(-100, 100))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, clip=(-100, 100))
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, clip=(-100, 100), params={"command_name": "base_velocity"}
        )
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, clip=(-100, 100))
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, clip=(-100, 100))
        joint_effort = ObsTerm(func=mdp.joint_effort, scale=0.01, clip=(-100, 100))
        last_action = ObsTerm(func=mdp.last_action, clip=(-100, 100))
        height_scanner = ObsTerm(func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 5.0),
        )
        material_friction = ObsTerm(
            func=mdp.rigid_body_material_friction,
            params={"asset_cfg": SceneEntityCfg("robot")},
            clip=(0.0, 2.0),
        )
        # external_wrench = ObsTerm(
        #     func=mdp.external_force_torque,
        #     params={"asset_cfg": SceneEntityCfg("robot", body_names="base")},
        #     clip=(-100.0, 100.0),
        # )

        #def __post_init__(self):
            #self.history_length = 5

    # privileged observations
    teacher: CriticCfg = CriticCfg()

@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 0.8),
            "dynamic_friction_range": (0.6, 0.6),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )
    


    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-5.0, 5.0),
            "operation": "add",
        },
    )

    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.01, 0.01)},
        },
    )

    

    # reset
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # -- task
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=1.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=0.5, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    # -- penalties
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=0.125,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*FOOT"),
            "command_name": "base_velocity",
            "threshold": 0.5,
        },
    )
    base_link_contact = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.5,
        params={
            "threshold": 1,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"),
        },
    )
    # -- optional penalties
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=0.0)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=0.0)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # check if robot is out of map
    map_edge_time_out = DoneTerm(
        func=mdp.position_out_of_terrain_bounds,
        time_out=True,
        params={"edge_margin": 0.05},
    )
    # check if robot is out of map
    void_fall_time_out = DoneTerm(
        func=mdp.root_height_below_minimum,
        time_out=True,
        params={"minimum_height": -0.2},
    )
    # base_contact = DoneTerm(
    #     func=mdp.illegal_contact,
    #     params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
    # )
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.8})


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)


##
# Environment configuration
##


@configclass
class LocomotionVelocityRoughEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""
    
    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 30.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False


class UnitreeGo2RoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.scene.robot = UNITREE_GO2_CFG_lab.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base"
        # scale down the terrains because the robot is small
        # self.scene.terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (0.025, 0.1)
        # self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_range = (0.01, 0.06)
        # self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_step = 0.01

        # reduce action scale
        self.actions.joint_pos.scale = 0.25

        # event
        self.events.push_robot = None
        self.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 3.0)
        self.events.add_base_mass.params["asset_cfg"].body_names = "base"
        self.events.base_external_force_torque.params["asset_cfg"].body_names = "base"
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }
        self.events.base_com = None

        # rewards
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = ".*_foot"
        self.rewards.feet_air_time.weight = 0.01
        #self.rewards.undesired_contacts = None
        self.rewards.dof_torques_l2.weight = -0.0002
        self.rewards.track_lin_vel_xy_exp.weight = 1.5
        self.rewards.track_ang_vel_z_exp.weight = 0.75
        self.rewards.dof_acc_l2.weight = -2.5e-7

        # #terminations
        # self.terminations.base_contact.params["sensor_cfg"].body_names = "base"


@configclass
class UnitreeGo2RoughEnvCfg_PLAY(UnitreeGo2RoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 10
            self.scene.terrain.terrain_generator.num_cols = 5
            #self.scene.terrain.terrain_generator.curriculum = False
            self.curriculum.terrain_levels = None
            level = 5
            self.scene.terrain.max_init_terrain_level = level

        # disable randomization for play
        #self.observations.policy.enable_corruption = False
        # remove random pushing event
        #self.events.base_external_force_torque = None
        #self.events.push_robot = None