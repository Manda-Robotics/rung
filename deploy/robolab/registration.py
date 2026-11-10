"""Depth-enabled registration shared by the probe and evaluation runner."""


def wrist_pose_from_flange(env, offset_pos, offset_quat, component):
    """Public fixed mount + live proprioception; Camera.data pose is stale under Fabric."""
    import numpy as np
    import torch
    from robolab.robots.droid import ee_pos, ee_quat

    from rung.contracts.frames import Pose

    positions, quaternions = ee_pos(env), ee_quat(env)
    mount = Pose(offset_pos, offset_quat)
    poses = [
        Pose(p, q).compose(mount)
        for p, q in zip(
            positions.detach().cpu().numpy(), quaternions.detach().cpu().numpy(), strict=True
        )
    ]
    values = np.stack([pose.pos if component == "pos" else pose.quat for pose in poses])
    return torch.as_tensor(values, device=env.device, dtype=positions.dtype)


def register(task: str, *, wrist_calibration: bool = False, right_shoulder: bool = False):
    """Register ``task`` with depth-enabled scene cameras and the DROID rel-IK action space.

    ``right_shoulder`` adds the second shoulder camera (RoboLab's ``WRIST_LEFT_RIGHT``
    preset) as a calibrated view; physics, scoring and the action space are unchanged.

    ``auto_env_registrations_rel_ik`` does not forward ``lazy_sensor_update``, and depth
    annotators only populate when sensors update eagerly, so the factory is called
    directly.
    """
    from robolab.constants import DEFAULT_TASK_SUBFOLDERS, TASK_DIR
    from robolab.core.environments.factory import auto_discover_and_create_cfgs
    from robolab.core.observations.observation_utils import (
        generate_image_obs_from_cameras,
        generate_obs_cfg,
    )
    from robolab.registrations.droid.camera_presets import WRIST_LEFT, WRIST_LEFT_RIGHT
    from robolab.robots.droid import (
        DroidCfg,
        DroidRelIKActionCfg,
        ProprioceptionObservationCfg,
        WristCameraCfg,
        contact_gripper,
    )
    from robolab.variations.backgrounds import HomeOfficeBackgroundCfg
    from robolab.variations.camera import EgocentricMirroredCameraCfg, with_depth
    from robolab.variations.lighting import SphereLightCfg

    preset = WRIST_LEFT_RIGHT if right_shoulder else WRIST_LEFT
    depth_cameras = [c if c is WristCameraCfg else with_depth(c) for c in preset]
    ImageObsCfg = generate_image_obs_from_cameras(depth_cameras)
    if wrist_calibration:
        from isaaclab.managers import ObservationTermCfg, SceneEntityCfg
        from robolab.core.observations.observation_utils import (
            camera_intrinsics,
        )

        # Metadata only: do not add depth annotators or change the wrist camera/robot.
        wrist_cfg = WristCameraCfg().wrist_cam
        if not wrist_cfg.prim_path.endswith("/Robotiq_2F_85/base_link/wrist_cam"):
            raise ValueError("wrist FK requires the declared base_link mount")
        if wrist_cfg.offset.convention != "opengl":
            raise ValueError("wrist FK requires canonical OpenGL mount configuration")
        ImageObsCfg.wrist_cam_K = ObservationTermCfg(
            func=camera_intrinsics, params={"sensor_cfg": SceneEntityCfg("wrist_cam")}
        )
        for component in ("pos", "quat"):
            setattr(
                ImageObsCfg,
                f"wrist_cam_{component}",
                ObservationTermCfg(
                    func=wrist_pose_from_flange,
                    params={
                        "offset_pos": tuple(wrist_cfg.offset.pos),
                        "offset_quat": tuple(wrist_cfg.offset.rot),
                        "component": component,
                    },
                ),
            )
    ViewportObsCfg = generate_image_obs_from_cameras([EgocentricMirroredCameraCfg])
    ObservationCfg = generate_obs_cfg(
        {
            "image_obs": ImageObsCfg(),
            "proprio_obs": ProprioceptionObservationCfg(),
            "viewport_cam": ViewportObsCfg(),
        }
    )
    scene_cameras = [c for c in depth_cameras if c is not WristCameraCfg]

    auto_discover_and_create_cfgs(
        task_dir=TASK_DIR,
        task_subdirs=DEFAULT_TASK_SUBFOLDERS,
        tasks=[task],
        pattern="*.py",
        env_prefix="",
        env_postfix="",
        observations_cfg=ObservationCfg(),
        actions_cfg=DroidRelIKActionCfg(),
        robot_cfg=DroidCfg,
        camera_cfg=[*scene_cameras, EgocentricMirroredCameraCfg],
        lighting_cfg=SphereLightCfg,
        background_cfg=HomeOfficeBackgroundCfg,
        contact_gripper=contact_gripper,
        dt=1 / (60 * 2),
        render_interval=8,
        decimation=8,
        seed=1,
        lazy_sensor_update=False,
    )
