"""Verified's per-step evaluation client, without an Isaac dependency."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from rung.adapters.robolab.bridge import RoboLabEmbodiment, RungInferenceClient, run_in_thread
from rung.adapters.robolab.compat import instruction_text
from rung.adapters.robolab.translate import (
    FLANGE_TO_FINGERTIP_M,
    SCENE_CAMERA,
    capabilities_from_observation,
)
from rung.contracts.trace import TraceWriter
from rung.loop import Limits, run_episode
from rung.record import Recorder


class RungPolicyClient:
    """Harness policy client for RoboLab-Verified: one action per environment per step.

    The evaluator must call begin_episode (RoboLab-Verified) or prepare_episode
    (upstream RoboLab v0.3.2) before infer_batch. Every episode gets fresh policy,
    memory, bridge, and trace state. The evaluator retains ownership of grading.
    """

    def __init__(
        self,
        instruction="",
        *,
        model=None,
        effort=None,
        interface="l2",
        camera=SCENE_CAMERA,
        record_dir=None,
        trace_path=None,
        max_decisions=25,
        max_dollars=2.0,
        horizon_s=180.0,
        tcp_offset_m=FLANGE_TO_FINGERTIP_M,
        convention="opengl",
        policy_factory: Callable | None = None,
        time_mode: str = "realtime",
        tracking_gain: float = 1.0,
        direct_prices=None,
        direct_config=None,
        model_max_tokens: int = 4096,
        model_timeout_s: float = 90,
        policy_instruction: str | None = None,
        history_steps: tuple[int, ...] = (),
        extra_cameras: tuple[str, ...] = (),
        grader_source: str = "RoboLab-Verified",
    ):
        if policy_instruction is not None and (
            interface != "direct"
            or not isinstance(policy_instruction, str)
            or not policy_instruction.strip()
            or len(policy_instruction) > 4096
        ):
            raise ValueError(
                "policy instruction requires direct interface and nonempty text <=4096 chars"
            )
        self.policy_instruction = policy_instruction
        if interface not in ("l0", "l2", "direct"):
            raise ValueError("interface must be l0, l2 or direct")
        if interface == "direct" and policy_factory is None and direct_prices is None:
            raise ValueError("direct interface requires explicit prices")
        self.direct_prices = direct_prices
        self.direct_config = direct_config
        self.model_max_tokens = model_max_tokens
        self.model_timeout_s = model_timeout_s
        self.instruction = instruction
        self.model, self.effort, self.interface = model, effort, interface
        self.camera, self.tcp_offset_m = camera, tcp_offset_m
        self.convention = convention
        self.record_dir, self.trace_path = record_dir, trace_path
        self.policy_factory = policy_factory
        self.time_mode = time_mode
        self.tracking_gain = tracking_gain
        self.history_steps = tuple(history_steps)
        self.extra_cameras = tuple(extra_cameras)
        self.limits = Limits(
            max_decisions=max_decisions,
            max_dollars=max_dollars,
            horizon_s=horizon_s,
            max_model_calls=max_decisions if interface == "direct" else None,
        )
        self.grader_source = grader_source
        self.episode_idx = None
        self._prepared_episode_idx = None
        self._sessions = {}
        self.results = {}
        self.errors = {}
        self.traces = {}
        self.recorders = {}
        self.table_estimates = {}

    def begin_episode(self, episode_idx: int) -> None:
        self.reset()
        self.episode_idx = episode_idx

    def prepare_episode(self, episode_idx: int, available_context=None) -> None:
        """Upstream v0.3.2's lifecycle hook (``robolab/eval/base_client.py``).

        Idempotent for the active episode, refuses another episode before ``reset``,
        and requests no context fields: the instruction arrives with each inference
        call, exactly as for legacy upstream clients.
        """
        if self._prepared_episode_idx is not None:
            if self._prepared_episode_idx == episode_idx:
                return
            raise RuntimeError("cannot prepare another episode before resetting the policy client")
        self.begin_episode(episode_idx)
        self._prepared_episode_idx = episode_idx

    def requested_context_fields(self, available_fields, *, episode_idx: int):
        return ()

    def _policy(self, emb):
        if self.policy_factory is not None:
            return self.policy_factory(emb)
        from rung.factory import build_backend
        from rung.interfaces import L0Config, L0Policy, L2Config, L2Policy

        if not self.model:
            raise ValueError("model must be set for a live backend")
        if self.interface == "direct":
            from rung.backends.gemini_session import GeminiSession
            from rung.backends.openai_session import OpenAISession
            from rung.interfaces.direct import DirectPolicy

            if self.model.startswith("gemini-"):
                session_class, options = GeminiSession, {"transport": "rest"}
            elif self.model.startswith("gpt-"):
                session_class, options = OpenAISession, {}
            else:
                raise ValueError("direct profile requires an explicitly qualified native model")
            return DirectPolicy(
                session_class(
                    self.model,
                    self.direct_prices,
                    effort=self.effort or "low",
                    max_tokens=self.model_max_tokens,
                    timeout_s=self.model_timeout_s,
                    trace_dir=Path(self.record_dir) / f"native-{self.episode_idx}-{emb.env_id}"
                    if self.record_dir
                    else None,
                    **options,
                ),
                self.direct_config,
                budgets={
                    "horizon_s": self.limits.horizon_s,
                    "max_model_calls": self.limits.max_model_calls,
                    "time_mode": self.time_mode,
                },
            )
        backend = build_backend(self.model)
        if self.interface == "l0":
            return L0Policy(backend, L0Config(effort=self.effort))
        from rung.executor import Executor
        from rung.factory import build_grounding
        from rung.grasp import AnalyticGraspPlanner
        from rung.perception import ObjectMemory, PerceptionService

        executor = Executor(
            emb,
            PerceptionService(build_grounding(self.model)),
            AnalyticGraspPlanner(emb.caps),
            ObjectMemory(),
            emb.instruction,
        )
        return L2Policy(
            backend, executor, L2Config(effort=self.effort, horizon_s=self.limits.horizon_s)
        )

    def _measure_table(self, obs, env_id, caps):
        """Rebuild the capabilities with the table height this scene actually has."""
        from rung.adapters.robolab.translate import to_observation
        from rung.perception.cloud import estimate_table_z

        try:
            first = to_observation(
                obs,
                t=0.0,
                env_id=env_id,
                camera=self.camera,
                capabilities=caps,
                convention=self.convention,
            )
            table_z, share = estimate_table_z(first.scene)
        except Exception:  # a scene without depth keeps the declared table
            return caps
        self.table_estimates[(self.episode_idx, env_id)] = (table_z, share)
        return capabilities_from_observation(
            obs, env_id=env_id, camera=self.camera, tcp_offset_m=self.tcp_offset_m, table_z=table_z
        )

    def _start(self, obs, instruction, env_id):
        key = (self.episode_idx, env_id)
        caps = capabilities_from_observation(
            obs, env_id=env_id, camera=self.camera, tcp_offset_m=self.tcp_offset_m
        )
        caps = self._measure_table(obs, env_id, caps)
        emb = RoboLabEmbodiment(
            caps,
            camera=self.camera,
            env_id=env_id,
            chunk_steps=1,
            convention=self.convention,
            time_mode=self.time_mode,
            tracking_gain=self.tracking_gain,
            history_steps=self.history_steps,
            extra_cameras=self.extra_cameras,
        )
        emb.instruction = instruction
        client = RungInferenceClient(emb, instruction, chunk_steps=1)
        policy = self._policy(emb)
        name = f"episode-{self.episode_idx:04d}-env-{env_id:03d}"
        folder = Path(self.record_dir) / name if self.record_dir else None
        trace_path = folder / "trace.jsonl" if folder else None
        if self.trace_path:
            base = Path(self.trace_path)
            trace_path = base.with_name(f"{base.stem}-{name}{base.suffix}")
        trace = self.traces[key] = TraceWriter(trace_path)
        table = self.table_estimates.get(key)
        trace.emit(
            "setup",
            0.0,
            name,
            env_id=env_id,
            model=self.model,
            interface=self.interface,
            table_z=None if table is None else round(float(table[0]), 4),
            table_share=None if table is None else round(float(table[1]), 3),
            tcp_offset_m=self.tcp_offset_m,
            camera=self.camera,
            camera_convention=self.convention,
            time_mode=self.time_mode,
            tracking_gain=self.tracking_gain,
            note="table height measured from the first depth image",
        )
        recorder = self.recorders[key] = Recorder(Path(self.record_dir)) if folder else None

        def drive():
            try:
                self.results[key] = run_episode(
                    emb,
                    policy,
                    task=instruction,
                    episode_id=name,
                    limits=self.limits,
                    trace=trace,
                    recorder=recorder,
                )
            except BaseException as exc:
                self.errors[key] = exc
            finally:
                emb.close()

        self._sessions[env_id] = (emb, client, run_in_thread(drive))

    def infer_batch(self, obs, instruction, *, env_ids: list[int]):
        if self.episode_idx is None:
            raise RuntimeError("begin_episode must precede infer_batch")
        if not isinstance(instruction, str):
            # Upstream passes TrackedInstructionFields; the model only ever sees text.
            instruction = instruction_text(instruction)
        result = {}
        for env_id in env_ids:
            key = (self.episode_idx, env_id)
            if env_id not in self._sessions:
                self._start(
                    obs,
                    self.policy_instruction
                    if self.policy_instruction is not None
                    else instruction or self.instruction,
                    env_id,
                )
            action = self._sessions[env_id][1].get_action(obs)[0]
            if key in self.errors:
                raise RuntimeError(f"harness failed for {key}") from self.errors[key]
            result[env_id] = {"action": action, "viz": None}
        return result

    def reset(self, *, env_id=None):
        if env_id is None:
            self._prepared_episode_idx = None
        ids = list(self._sessions) if env_id is None else [env_id]
        for eid in ids:
            session = self._sessions.get(eid)
            if session is None:
                continue
            emb, _, thread = session
            emb.close()
            thread.join(timeout=5)
            if thread.is_alive():
                raise RuntimeError("policy call still running; cannot start another episode")
            del self._sessions[eid]

    def close(self):
        self.reset()

    def record_grades(self, grades, *, episode_idx: int):
        """Join evaluator-only grades after reset has joined all policy threads."""
        if self._sessions:
            raise RuntimeError("reset must finish policy threads before grading")
        for grade in grades:
            key = (episode_idx, int(grade["env_id"]))
            if key not in self.results:
                raise RuntimeError(f"missing policy result for {key}")
            result = self.results[key]
            if result.success is not None:
                raise RuntimeError(f"episode already graded: {key}")
            result.success = bool(grade["success"])
            result.grade_detail = dict(grade, source=self.grader_source)
            self.traces[key].emit(
                "episode_end",
                result.sim_seconds,
                result.episode_id,
                env_id=key[1],
                success=result.success,
                grade=result.grade_detail,
                end_reason=result.end_reason.value,
            )
            recorder = self.recorders[key]
            if recorder:
                recorder.record_result(result)
