#!/usr/bin/env python3
"""Run one task with Verified's native dashboard results and diagnostic videos.

Use suite.py for task manifests. The JSON status file is authoritative, since Isaac can
exit zero after an exception. --resume extends completed tasks; an interrupted task keeps
its evidence and needs a new output directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from rung.adapters.robolab.translate import RIGHT_SHOULDER_CAMERA, TRACKING_GAIN


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def revision(module):
    folder = Path(module.__file__).resolve().parent
    try:
        return subprocess.check_output(
            ["git", "-C", str(folder), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, OSError):
        # The upstream image installs the tree without .git and records the stack SHA in
        # .robolab-revision (deploy/robolab/image/upstream60 recipe).
        for candidate in (
            folder / "REVISION",
            folder.parent / "REVISION",
            folder / ".robolab-revision",
            folder.parent / ".robolab-revision",
        ):
            if candidate.is_file():
                return candidate.read_text().strip()
    return "unknown"


def source_digest(module):
    digest = hashlib.sha256()
    root = Path(module.__file__).resolve().parent
    for path in sorted(root.rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--task", default="BananaInBowlTask")
    parser.add_argument(
        "--episodes",
        "--num-runs",
        dest="episodes",
        type=int,
        default=2,
        help="Episodes per environment (upstream RoboLab calls these runs)",
    )
    parser.add_argument(
        "--max-upstream-resets",
        type=int,
        default=2,
        help="Upstream RoboLab only: early re-resets allowed per episode before it ends "
        "as infrastructure (upstream itself re-resets without a cap)",
    )
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--interface", choices=["l0", "l2", "direct"], default="l2")
    parser.add_argument(
        "--policy-instruction",
        help="Explicit actor-only task wording; native task/scorer unchanged",
    )
    parser.add_argument("--direct-prices", help="JSON TokenPrices with dated pricing source")
    parser.add_argument(
        "--motion-tools", choices=["relative", "absolute", "both"], default="relative"
    )
    parser.add_argument(
        "--public-overlay",
        action="store_true",
        help="Draw public rig/table coordinates and measured TCP on scene RGB",
    )
    parser.add_argument(
        "--table-projection",
        action="store_true",
        help="Offer geometry-only model-selected pixel/table ray intersection",
    )
    parser.add_argument(
        "--wrist-calibration",
        action="store_true",
        help="Add public wrist K/pose metadata; wrist remains RGB-only",
    )
    parser.add_argument("--stereo-triangulation", action="store_true")
    parser.add_argument(
        "--tcp-marker",
        action="store_true",
        help="Public calibration-only fingertip marker on both RGB views",
    )
    parser.add_argument(
        "--spatial-frame-cues",
        action="store_true",
        help="Explicit robot-root versus camera directional instruction",
    )
    parser.add_argument(
        "--tcp-frame-motions",
        action="store_true",
        help="Opt-in measured TCP-axis relative motion tool",
    )
    parser.add_argument(
        "--geometry-feedback",
        action="store_true",
        help="Opt-in public-calibration point reprojection receipts",
    )
    parser.add_argument(
        "--depth-probe",
        action="store_true",
        help="Opt-in probe_point: rig depth at a model-selected pixel (declared RGB-D input)",
    )
    parser.add_argument(
        "--action-sequence",
        action="store_true",
        help="Opt-in execute_sequence: guarded multi-step motion in one model turn",
    )
    parser.add_argument("--max-sequence-steps", type=int, default=6)
    parser.add_argument(
        "--look-tool",
        action="store_true",
        help="Opt-in look: guarded move of the wrist camera to a chosen viewpoint on a point",
    )
    parser.add_argument(
        "--camera-geometry-prompt",
        action="store_true",
        help="Write camera poses, intrinsics and the wrist mount into every prompt",
    )
    parser.add_argument(
        "--recent-frames",
        action="store_true",
        help="Show buffered frames from earlier control steps (see --history-steps)",
    )
    parser.add_argument(
        "--history-steps",
        default="14,7",
        help="Comma-separated control-step ages of the earlier frames, newest last",
    )
    parser.add_argument(
        "--right-shoulder",
        action="store_true",
        help="Register the right shoulder camera as a calibrated extra view",
    )
    parser.add_argument(
        "--image-scale",
        type=float,
        default=1.0,
        help="Resize factor for images shown to the model (0.3 gives 384x216 from 720p)",
    )
    parser.add_argument("--max-translation-m", type=float, default=0.10)
    parser.add_argument("--max-rotation-rad", type=float, default=0.50)
    parser.add_argument(
        "--max-speed-mps",
        type=float,
        default=0.10,
        help="Direct-interface linear speed cap; the DROID rig allows 0.30 (2 cm per 15 Hz step)",
    )
    parser.add_argument(
        "--max-ang-speed-rps",
        type=float,
        default=0.50,
        help="Direct-interface angular speed cap; the DROID rig allows 1.5",
    )
    parser.add_argument(
        "--no-terminal-tools",
        action="store_true",
        help="Remove done/give_up: the episode ends at native success or the native horizon",
    )
    parser.add_argument(
        "--table-extent",
        action="store_true",
        help="Declare the table footprint in the state and warn in receipts near or beyond "
        "its edges",
    )
    parser.add_argument(
        "--done-awaits-confirmation",
        action="store_true",
        help="done holds still and the episode continues unless the native grader confirms success",
    )
    parser.add_argument(
        "--sequence-grasp-guard",
        action="store_true",
        help="Stop a sequence when the jaw closes further on nothing while carrying (grasp lost)",
    )
    parser.add_argument("--model-max-tokens", type=int, default=4096)
    parser.add_argument(
        "--model-timeout-s",
        type=float,
        default=90,
        help="Explicit native API timeout; no retry on timeout",
    )
    parser.add_argument(
        "--campaign-profile",
        choices=[
            "legacy",
            "frontier-capability-dev",
            "matched-reset-qualification",
            "frontier-capability-eval",
        ],
        default="legacy",
    )
    parser.add_argument(
        "--start-hdf5", help="Stable Verified recording for matched single-env reset"
    )
    parser.add_argument("--start-demo", default="demo_0")
    parser.add_argument("--model", help="Omit for deterministic motion smoke")
    parser.add_argument("--effort")
    parser.add_argument("--probe", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--video-mode", choices=["all", "sensor", "viewport", "none"], default="all"
    )
    parser.add_argument("--max-dollars", type=float, default=0.60)
    parser.add_argument("--max-decisions", type=int, default=25)
    parser.add_argument(
        "--horizon-s",
        type=float,
        default=180.0,
        help="Harness horizon cap; the effective horizon is the smaller of this and the "
        "task's native episode length (see --native-horizon)",
    )
    parser.add_argument(
        "--native-horizon",
        action="store_true",
        help="Opt-in: the effective horizon is exactly the task's native episode length "
        "(env.max_episode_length x control step), with no harness cap. Replaces --horizon-s",
    )
    parser.add_argument(
        "--post-success-settle-s",
        type=float,
        default=0,
        help="Development-only diagnostic after native grading; 0 disables",
    )
    parser.add_argument(
        "--seed", type=int, help="Defaults to recorded seed for matched starts, else 1"
    )
    parser.add_argument(
        "--time-mode",
        choices=["realtime", "paused"],
        default="realtime",
        help="realtime: harness and model time passes in the simulation with the robot "
        "holding still, as on a real robot. paused: the simulation waits while the policy "
        "infers, as the official synchronous RoboLab runner does (upstream "
        "robolab/eval/episode.py); real inference latency is reported separately. Paused "
        "results are not comparable to realtime ones, so label which mode produced them",
    )
    parser.add_argument(
        "--result-class",
        choices=["leaderboard_submission"],
        help="Opt-in explicit result class. leaderboard_submission: an upstream RoboLab "
        "run under paused time, matching the official synchronous runner; requires "
        "--time-mode paused and the upstream tree",
    )
    parser.add_argument(
        "--tracking-gain",
        type=float,
        default=TRACKING_GAIN,
        help="Realised fraction of a commanded Cartesian step; commands are divided by it. "
        f"Default {TRACKING_GAIN} is the measured DROID rel-IK value; 1.0 disables "
        "compensation. Re-measure with calibrate_speed.py on a new controller",
    )
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.episodes < 1 or args.num_envs < 1:
        parser.error("episodes and num-envs must be positive")
    if args.max_upstream_resets < 0:
        parser.error("max-upstream-resets must be non-negative")
    if args.model_max_tokens < 1:
        parser.error("model-max-tokens must be positive")
    if args.native_horizon and args.horizon_s != 180.0:
        parser.error("--native-horizon replaces --horizon-s; pass one or the other")
    if args.result_class == "leaderboard_submission" and args.time_mode != "paused":
        parser.error(
            "leaderboard_submission requires --time-mode paused "
            "(the official runner is synchronous)"
        )
    if args.stereo_triangulation and not args.wrist_calibration:
        parser.error("stereo triangulation requires qualified wrist calibration")
    history_steps = ()
    if args.recent_frames:
        try:
            history_steps = tuple(int(x) for x in args.history_steps.split(",") if x.strip())
        except ValueError:
            parser.error("history-steps must be comma-separated integers")
        if not history_steps or any(h <= 0 for h in history_steps):
            parser.error("history-steps must be positive control-step ages")
    if not 0 < args.image_scale <= 1:
        parser.error("image-scale must be in (0, 1]")
    import math

    if not math.isfinite(args.model_timeout_s) or args.model_timeout_s <= 0:
        parser.error("model-timeout-s must be finite and positive")
    if not math.isfinite(args.post_success_settle_s) or not 0 <= args.post_success_settle_s <= 3:
        parser.error("post-success-settle-s must be between 0 and 3")
    if args.post_success_settle_s and (
        args.campaign_profile != "frontier-capability-dev" or args.num_envs != 1
    ):
        parser.error("post-success settling is restricted to single-env development")
    if (
        args.campaign_profile in ("frontier-capability-dev", "frontier-capability-eval")
        and args.interface != "direct"
    ):
        parser.error("frontier capability profiles require the direct interface")
    if args.campaign_profile == "frontier-capability-eval" and (
        not args.start_hdf5 or args.num_envs != 1 or args.episodes != 1
    ):
        parser.error("frontier-capability-eval requires one explicit recorded shared start")
    if args.campaign_profile == "matched-reset-qualification" and (
        args.model or not args.start_hdf5
    ):
        parser.error("matched-reset-qualification requires a recorded start and no live model")
    matched_start = None
    source_env_cfg = None
    if args.start_hdf5:
        if args.num_envs != 1 or args.episodes != 1 or args.resume:
            parser.error("matched development reset requires one env/episode and a fresh output")
        from rung.adapters.robolab.matched_start import load_start

        matched_start = load_start(Path(args.start_hdf5), args.start_demo, duplicate_tolerance=1e-6)
        if matched_start.provenance["source_task"] != args.task:
            parser.error("recording task differs from requested task")
        cfg_path = Path(args.start_hdf5).with_name("env_cfg.json")
        if not cfg_path.exists():
            parser.error("matched reset also requires source env_cfg.json")
        source_env_cfg = json.loads(cfg_path.read_text())
        recorded_seed = source_env_cfg.get("seed")
        if not isinstance(recorded_seed, int) or isinstance(recorded_seed, bool):
            parser.error("recorded seed missing or invalid")
        if args.seed is not None and args.seed != recorded_seed:
            parser.error("matched run seed must equal the recorded configuration seed")
        args.seed = recorded_seed
        matched_start.provenance["source_env_cfg_sha256"] = hashlib.sha256(
            cfg_path.read_bytes()
        ).hexdigest()
    if args.seed is None:
        args.seed = 1
    direct_prices = None
    direct_config = None
    if args.interface == "direct":
        if (
            not args.model
            or not args.model.startswith(("gemini-", "gpt-"))
            or (not args.direct_prices)
        ):
            parser.error("direct profile requires qualified native model and --direct-prices")
        from rung.backends.gemini_session import TokenPrices

        direct_prices = TokenPrices(**json.loads(Path(args.direct_prices).read_text()))
        from rung.interfaces.direct import DirectConfig

        direct_config = DirectConfig(
            motion_tools=args.motion_tools,
            public_overlay=args.public_overlay,
            table_projection=args.table_projection,
            stereo_triangulation=args.stereo_triangulation,
            tcp_marker=args.tcp_marker,
            spatial_frame_cues=args.spatial_frame_cues,
            tcp_frame_motions=args.tcp_frame_motions,
            geometry_feedback=args.geometry_feedback,
            depth_probe=args.depth_probe,
            action_sequence=args.action_sequence,
            max_sequence_steps=args.max_sequence_steps,
            sequence_grasp_guard=args.sequence_grasp_guard,
            table_extent=args.table_extent,
            done_awaits_confirmation=args.done_awaits_confirmation,
            camera_geometry_prompt=args.camera_geometry_prompt,
            recent_frames=args.recent_frames,
            extra_views=args.right_shoulder,
            image_scale=args.image_scale,
            max_translation_m=args.max_translation_m,
            max_rotation_rad=args.max_rotation_rad,
            max_speed_mps=args.max_speed_mps,
            max_ang_speed_rps=args.max_ang_speed_rps,
            terminal_tools=not args.no_terminal_tools,
            look_tool=args.look_tool,
        )
    elif (
        args.public_overlay
        or args.table_projection
        or args.stereo_triangulation
        or args.tcp_marker
        or args.spatial_frame_cues
        or args.tcp_frame_motions
        or args.geometry_feedback
        or args.depth_probe
        or args.action_sequence
        or args.sequence_grasp_guard
        or args.table_extent
        or args.done_awaits_confirmation
        or args.look_tool
        or args.camera_geometry_prompt
        or args.recent_frames
        or args.right_shoulder
        or args.image_scale != 1.0
        or args.no_terminal_tools
        or args.max_speed_mps != 0.10
        or args.max_ang_speed_rps != 0.50
        or args.motion_tools != "relative"
        or args.max_translation_m != 0.10
        or args.max_rotation_rad != 0.50
    ):
        parser.error("motion-tools/public-overlay switches require the direct interface")
    probe = json.loads(Path(args.probe).read_text())
    calibration = json.loads(Path(args.calibration).read_text())
    if not probe.get("passed") or not calibration.get("passed"):
        parser.error("passing adapter probe and independent TCP calibration required")
    if args.wrist_calibration and not probe.get("wrist_calibration"):
        parser.error("wrist calibration requires its own passing live metadata/mount probe")
    offset = float(calibration["tcp_offset_m"])
    if not 0.06 < offset < 0.20:
        parser.error("TCP calibration outside the gripper envelope")
    convention = next(
        f["measured"]["convention"]
        for f in probe["findings"]
        if f["name"] == "camera_convention_detected"
    )
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "status").mkdir(exist_ok=True)
    status_path = out / "status" / f"{args.task}.json"
    prior = json.loads(status_path.read_text()) if status_path.exists() else None
    if prior and (not args.resume or prior.get("state") != "complete"):
        parser.error("existing task incomplete or --resume missing; use a new output directory")
    status = dict(
        state="starting",
        task=args.task,
        expected_episodes=args.episodes * args.num_envs,
        completed_episodes=0,
    )
    write_json(status_path, status)
    app = env = client = None
    try:
        args.enable_cameras = args.headless = True
        app = AppLauncher(args).app
        import dataclasses

        import numpy as np
        import robolab
        import robolab.constants
        from registration import register
        from robolab.constants import set_output_dir
        from robolab.core.environments.factory import get_envs
        from robolab.core.environments.runtime import create_env
        from robolab.core.logging.results import init_experiment
        from robolab.eval.episode import run_episode
        from robolab.eval.summarize import summarize_run

        import rung
        from rung.adapters.robolab.client import RungPolicyClient
        from rung.adapters.robolab.compat import (
            GRADER_SOURCE,
            UPSTREAM,
            ReResetGuard,
            isaaclab_compat,
            robolab_tree,
        )
        from rung.contracts.action import TCPTarget
        from rung.contracts.frames import Pose
        from rung.contracts.trace import to_jsonable
        from rung.policy import Decision, Verdict

        # The fork's Isaac Lab 2/3 helpers when installed; Rung's 2.x shim on upstream.
        scene_state_from_isaaclab = isaaclab_compat().scene_state_from_isaaclab
        tree = robolab_tree()
        if tree == UPSTREAM and matched_start is not None:
            raise RuntimeError(
                "upstream RoboLab uses its own deterministic reset; drop --start-hdf5"
            )
        if args.result_class == "leaderboard_submission" and tree != UPSTREAM:
            raise RuntimeError("leaderboard_submission is only for the upstream RoboLab tree")
        config = dict(
            model=args.model,
            policy_instruction=args.policy_instruction,
            policy_instruction_sha256=None
            if args.policy_instruction is None
            else hashlib.sha256(args.policy_instruction.encode()).hexdigest(),
            post_success_settle_s=args.post_success_settle_s,
            model_max_tokens=args.model_max_tokens,
            model_timeout_s=args.model_timeout_s if args.interface == "direct" else None,
            campaign_profile=args.campaign_profile,
            result_class="qualification"
            if args.campaign_profile == "matched-reset-qualification"
            else "evaluation"
            if args.campaign_profile == "frontier-capability-eval"
            else "development"
            if args.campaign_profile != "legacy"
            else ("diagnostic" if args.time_mode == "paused" else "legacy"),
            interface=args.interface,
            effort=args.effort,
            num_envs=args.num_envs,
            max_dollars=args.max_dollars,
            max_decisions=args.max_decisions,
            max_model_calls=args.max_decisions if args.interface == "direct" else None,
            direct_prices=None if direct_prices is None else dataclasses.asdict(direct_prices),
            direct_config=None if direct_config is None else dataclasses.asdict(direct_config),
            horizon_s=args.horizon_s,
            seed=args.seed,
            video_mode=args.video_mode,
            time_mode=args.time_mode,
            wrist_calibration=args.wrist_calibration,
            tracking_gain=args.tracking_gain,
            tcp_offset_m=offset,
            convention=convention,
            rung_revision=revision(rung),
            rung_source_sha256=source_digest(rung),
            robolab_revision=revision(robolab),
            matched_start=None if matched_start is None else matched_start.provenance,
        )
        if tree == UPSTREAM:
            # Only upstream runs record the tree, so Verified configs stay byte-identical.
            config.update(robolab_tree=tree, max_upstream_resets=args.max_upstream_resets)
        # Opt-in fields are added only when set, so runs without them stay byte-identical.
        if args.native_horizon:
            # The horizon varies by task; the per-task value goes into provenance.json.
            config.update(horizon_s=None, native_horizon=True)
        if args.result_class:
            config.update(
                result_class=args.result_class,
                inference_timing="simulation paused during inference, matching the official "
                "synchronous RoboLab runner; real inference latency is reported separately",
            )
        config_path = out / "config.json"
        if config_path.exists() and json.loads(config_path.read_text()) != config:
            raise RuntimeError("configuration/source changed: use a new output directory")
        write_json(config_path, config)
        register(
            args.task, wrist_calibration=args.wrist_calibration, right_shoulder=args.right_shoulder
        )
        matches = get_envs(task=[args.task])
        if len(matches) != 1:
            raise RuntimeError(f"expected one environment for {args.task}, got {matches}")
        task_env = matches[0]
        scene = out / task_env
        scene.mkdir(exist_ok=True)
        set_output_dir(str(scene))
        write_json(
            scene / "provenance.json",
            dict(
                config,
                probe=probe,
                calibration=calibration,
                initial_state_matching="pending strict restore audit"
                if matched_start
                else "unaudited; resumed runs may use a different reset sequence",
                terminal_semantics="hold after policy completion until Verified termination"
                if tree != UPSTREAM
                else "hold after policy completion until upstream RoboLab termination",
            ),
        )
        results_file, results = init_experiment(str(out))
        completed = {int(row["run"]) for row in results if row["env_name"] == task_env}
        if prior and completed != set(range(len(completed))):
            raise RuntimeError("cannot resume non-contiguous episode results")
        if completed and max(completed) >= args.episodes:
            raise RuntimeError("episodes cannot be reduced on resume")
        if len(completed) < args.episodes:
            env, cfg = create_env(
                task_env,
                device=args.device,
                num_envs=args.num_envs,
                seed=args.seed,
                policy="rung",
                rendering_mode=source_env_cfg["sim"]["render"]["rendering_mode"]
                if source_env_cfg
                else None,
            )
            if source_env_cfg:
                for path, actual, expected in (
                    ("episode_length_s", cfg.episode_length_s, source_env_cfg["episode_length_s"]),
                    ("decimation", cfg.decimation, source_env_cfg["decimation"]),
                    ("sim.dt", cfg.sim.dt, source_env_cfg["sim"]["dt"]),
                    (
                        "sim.render_interval",
                        cfg.sim.render_interval,
                        source_env_cfg["sim"]["render_interval"],
                    ),
                ):
                    if actual != expected:
                        raise RuntimeError(f"matched configuration mismatch at {path}")
            # The evaluator ends the episode at its own step limit regardless of the harness
            # horizon. The effective horizon is the smaller of the two (or exactly the native
            # one under --native-horizon), so the harness clock, motion budgets and the time
            # reported to the model agree with the grader. It varies by task, so it goes into
            # per-task provenance, not the shared config, which must match across every task
            # in a sweep.
            step_s = float(cfg.sim.dt) * int(cfg.decimation)
            verified_horizon_s = float(env.max_episode_length) * step_s
            horizon_s = (
                verified_horizon_s
                if args.native_horizon
                else min(args.horizon_s, verified_horizon_s)
            )
            if matched_start:
                from rung.adapters.robolab.matched_start import install_matched_reset

                restore_audits = []

                def record_restore(audit):
                    restore_audits.append(audit)
                    write_json(scene / "matched-start-audits.json", restore_audits)

                install_matched_reset(env, matched_start, record_restore)
            provenance = json.loads((scene / "provenance.json").read_text())
            provenance.update(effective_horizon_s=horizon_s, verified_horizon_s=verified_horizon_s)
            if args.native_horizon:
                provenance.update(horizon_source="native_episode_length")
            write_json(scene / "provenance.json", provenance)

            class MotionSmoke:
                name = "deterministic-motion-smoke"
                interface = "scripted"

                def reset(self, instruction, capabilities):
                    self.calls = 0

                def act(self, obs, feedback):
                    self.calls += 1
                    if self.calls == 1:
                        return Decision(
                            "move",
                            "Lift the empty hand 3 cm",
                            targets=[
                                TCPTarget(
                                    Pose(obs.tcp.pos + np.array([0.0, 0.0, 0.03]), obs.tcp.quat),
                                    timeout_s=3.0,
                                )
                            ],
                        )
                    return Decision("done", "Motion smoke finished", verdict=Verdict.DONE)

            class AuditedClient(RungPolicyClient):
                def begin_episode(self, episode_idx):
                    super().begin_episode(episode_idx)
                    state = to_jsonable(
                        scene_state_from_isaaclab(env.scene.get_state(is_relative=True))
                    )
                    encoded = json.dumps(state, sort_keys=True, separators=(",", ":"))
                    (scene / f"initial-{episode_idx:04d}.json").write_text(encoded)
                    (scene / f"initial-{episode_idx:04d}.sha256").write_text(
                        hashlib.sha256(encoded.encode()).hexdigest() + "\n"
                    )

            if tree == UPSTREAM:
                guard = ReResetGuard(args.max_upstream_resets)

                class GuardedClient(AuditedClient):
                    def begin_episode(self, episode_idx):
                        guard.start()
                        super().begin_episode(episode_idx)

                    def infer_batch(self, obs, instruction, *, env_ids):
                        guard.observe(env, env_ids)
                        return super().infer_batch(obs, instruction, env_ids=env_ids)

                client_class = GuardedClient
            else:
                client_class = AuditedClient
            client = client_class(
                model=args.model,
                policy_instruction=args.policy_instruction,
                model_max_tokens=args.model_max_tokens,
                model_timeout_s=args.model_timeout_s,
                effort=args.effort,
                interface=args.interface,
                record_dir=scene / "rung",
                max_dollars=args.max_dollars,
                max_decisions=args.max_decisions,
                horizon_s=horizon_s,
                tcp_offset_m=offset,
                convention=convention,
                time_mode=args.time_mode,
                tracking_gain=args.tracking_gain,
                direct_prices=direct_prices,
                direct_config=direct_config,
                policy_factory=None if args.model else lambda emb: MotionSmoke(),
                history_steps=history_steps,
                extra_cameras=(RIGHT_SHOULDER_CAMERA,) if args.right_shoulder else (),
                grader_source=GRADER_SOURCE[tree],
            )
            for episode in range(len(completed), args.episodes):
                status.update(
                    state="running",
                    active_episode=episode,
                    completed_episodes=episode * args.num_envs,
                )
                write_json(status_path, status)
                grades, msgs, timing = run_episode(
                    env,
                    cfg,
                    episode,
                    client,
                    headless=True,
                    save_videos=args.video_mode != "none",
                    video_mode=args.video_mode,
                )
                client.record_grades(grades, episode_idx=episode)
                if matched_start:
                    from rung.adapters.robolab.matched_start import audit_state, load_start

                    exported = load_start(
                        scene / f"run_{episode}.hdf5", "demo_0", duplicate_tolerance=1e-6
                    )
                    export_audit = {
                        "state": audit_state(matched_start.state, exported.state),
                        "cameras": audit_state(matched_start.cameras, exported.cameras),
                        "exported_provenance": exported.provenance,
                    }
                    write_json(scene / "matched-export-audit.json", export_audit)
                    if (
                        not export_audit["state"]["passed"]
                        or not (export_audit["cameras"]["passed"])
                    ):
                        raise RuntimeError("exported HDF5 start differs from the restored start")
                results = summarize_run(
                    env_results=grades,
                    msgs=msgs,
                    env=env,
                    env_cfg=cfg,
                    num_envs=args.num_envs,
                    run_idx=episode,
                    run_name=f"{task_env}_{episode}",
                    task_env=task_env,
                    scene_output_dir=str(scene),
                    policy="rung",
                    episode_results=results,
                    episode_results_file=results_file,
                    enable_subtask_progress=robolab.constants.ENABLE_SUBTASK_PROGRESS_CHECKING,
                    timing=timing,
                    instruction_type="default",
                    extra_fields=dict(
                        model=args.model, interface=args.interface, effort=args.effort
                    ),
                )
                for row in results[-len(grades) :]:
                    row["rung"] = to_jsonable(
                        dataclasses.asdict(client.results[episode, row["env_id"]])
                    )
                temporary = Path(results_file + ".tmp")
                temporary.write_text("".join(json.dumps(to_jsonable(r)) + "\n" for r in results))
                temporary.replace(results_file)
                write_json(
                    scene / f"verified-{episode:04d}.json",
                    to_jsonable(dict(grades=grades, timing=timing)),
                )
                if args.video_mode != "none":
                    for eid in range(args.num_envs):
                        suffix = f"_{episode}" + (f"_env{eid}" if args.num_envs > 1 else "")
                        kinds = [""] if args.video_mode == "sensor" else ["_viewport"]
                        if args.video_mode == "all":
                            kinds = ["", "_viewport"]
                        for kind in kinds:
                            videos = list(scene.glob(f"*{suffix}{kind}.mp4"))
                            if not any(v.stat().st_size > 1024 for v in videos):
                                raise RuntimeError(
                                    f"missing {kind or 'sensor'} video for episode "
                                    f"{episode}, env {eid}"
                                )
                if args.post_success_settle_s:
                    from PIL import Image

                    from rung.adapters.robolab.settling import post_success_settling
                    from rung.adapters.robolab.translate import (
                        capabilities_from_observation,
                        to_observation,
                    )

                    def capture_settling(raw, index, elapsed, directory):
                        caps = capabilities_from_observation(raw, tcp_offset_m=offset)
                        obs = to_observation(
                            raw, t=elapsed, env_id=0, capabilities=caps, convention=convention
                        )
                        sample = dict(
                            tcp_xyz_m=obs.tcp.pos.tolist(),
                            gripper_opening_m=obs.gripper.opening_m,
                            gripper_travel=obs.gripper.travel,
                        )
                        if index % 5 == 0 or elapsed >= args.post_success_settle_s:
                            for label, frame in (("scene", obs.scene), ("wrist", obs.wrist)):
                                if frame is not None:
                                    image_path = directory / f"{index:04d}-{label}.png"
                                    Image.fromarray(frame.rgb).save(image_path)
                                    sample[label + "_image"] = image_path.name
                        return sample

                    diagnostic_dir = scene / f"post-success-settling-{episode:04d}"
                    try:
                        post_success_settling(
                            env,
                            grades,
                            seconds=args.post_success_settle_s,
                            step_s=cfg.sim.dt * cfg.decimation,
                            out=diagnostic_dir,
                            capture=capture_settling,
                        )
                    except Exception as exc:
                        write_json(
                            scene / f"settling-error-{episode:04d}.json",
                            dict(
                                state="failed_diagnostic",
                                error=str(exc),
                                native_grades_preserved=True,
                            ),
                        )
                env.reset_eval_state()
                print(f"EPISODE_COMPLETE {args.task} {episode}", flush=True)
        status.update(state="complete", completed_episodes=args.episodes * args.num_envs)
    except BaseException:
        import traceback

        status.update(state="failed", error=traceback.format_exc())
        print(status["error"], flush=True)
    finally:
        try:
            if client:
                client.close()
            if env:
                env.close()
        except BaseException:
            import traceback

            status.update(state="failed", cleanup_error=traceback.format_exc())
        write_json(status_path, status)
        print("RUN_STATUS " + json.dumps(status), flush=True)
        if app:
            app.close()


if __name__ == "__main__":
    main()
