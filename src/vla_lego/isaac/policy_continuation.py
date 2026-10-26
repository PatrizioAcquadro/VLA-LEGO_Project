"""Actual fixed-policy observation windows followed by in-place teacher intervention."""

from __future__ import annotations

import json
import math
import socket
from pathlib import Path

from vla_lego.isaac.environment import TABLE_TOP
from vla_lego.isaac.evaluation import CONTACT_ZERO_N, POSITION_TOLERANCE_M, _footprint_inside
from vla_lego.pretrained.decisions import DECISION_NAMES, decode_decision
from vla_lego.pretrained.transport import receive_message, send_message


def released_prefix(task, states):
    """Current ordered support, never a historical maximum or a controller input."""
    if any(state["robot_contact_force_n"] >= CONTACT_ZERO_N for state in states):
        raise ValueError("Interruption handoff still holds a block; another recovery is required")
    if any(
        not all(math.isfinite(v) for v in (*state["block_pose_w"], *state["block_velocity_w"]))
        or (
            state["nonrobot_contact_force_n"] <= CONTACT_ZERO_N
            and not any(force[2] > CONTACT_ZERO_N for force in state["block_contact_forces_n"])
        )
        for state in states
    ):
        raise ValueError("Interruption handoff lacks finite physically supported pieces")
    bottom, previous, count = TABLE_TOP, None, 0
    for target in task.targets:
        index = next(i for i, p in enumerate(task.placements) if p.spec == target)
        state = states[index]
        support = (
            state["nonrobot_contact_force_n"]
            if previous is None
            else state["block_contact_forces_n"][previous][2]
        )
        if (
            abs(state["block_pose_w"][2] - bottom - target.dimensions_m[2] / 2)
            >= POSITION_TOLERANCE_M
            or support <= CONTACT_ZERO_N
            or not _footprint_inside(task, index, state["block_pose_w"])
        ):
            break
        bottom += target.dimensions_m[2]
        previous, count = index, count + 1
    return count


class PolicyIntervention:
    """The teacher may observe errors, but cannot assist while policy commands run."""

    def __init__(self, path: Path, expected_checkpoint: Path, instruction: str):
        self.connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.connection.settimeout(120)
        self.connection.connect(str(path))
        send_message(self.connection, {"prepare": True, "instruction": instruction})
        self.identity, _ = receive_message(self.connection)
        if (
            not self.identity.get("ready")
            or self.identity.get("execution_horizon") != 10
            or not self.identity.get("jaw_overlap")
            or Path(self.identity.get("checkpoint", "")).resolve() != expected_checkpoint.resolve()
        ):
            self.connection.close()
            raise ValueError("Policy window differs from the declared fixed reference/kernel")
        self.active = False
        self.last_keepalive = -math.inf

    def keepalive(self, time_s):
        if not self.active and time_s - self.last_keepalive >= 5:
            send_message(self.connection, {"prepare": True})
            receive_message(self.connection)
            self.last_keepalive = time_s

    def interrupted_progress(self, reference, case, task, *, completed_levels, ticks=1800):
        """A real policy resumes a correct partial tower; unsuccessful attempts stay failures."""
        import numpy as np

        from vla_lego.isaac.environment import ARM_JOINTS, JAW_JOINTS

        env, recorder = reference.env, reference.env.recorder
        if (
            case.row["correction_origin"] != "policy"
            or case.row["mechanism"] != "interrupted_progress"
            or case.row["partition"] != "train"
            or ticks < 1
            or not 1 <= completed_levels < 5
        ):
            raise ValueError("This policy handoff requires the declared TRAIN interruption case")
        before = [env.task_truth(i) for i in range(len(env.blocks))]
        if released_prefix(task, before) != completed_levels:
            raise ValueError("Conditional teacher prefix is not the supported partial tower")
        start_frame, start_time = recorder.count, env.time
        names = (*ARM_JOINTS, *JAW_JOINTS)
        ids = [env.robot.data.joint_names.index(name) for name in names]
        limits = env.robot.data.joint_pos_limits.torch[0, ids].cpu().numpy()
        chunk = []
        terminal, executed = "budget", 0
        next_index = next(
            i
            for i, placement in enumerate(task.placements)
            if placement.spec == task.targets[completed_levels]
        )
        acquired_next = False
        recorder.positive_imitation = False
        recorder.action_source = "policy"
        recorder.stage = "actual_policy_interruption_window"
        self.active = True
        try:
            with (reference.output_dir / "policy-window.jsonl").open("x") as stream:
                for _ in range(ticks):
                    if not chunk:
                        env.capture()
                        observation = env.observe()
                        send_message(
                            self.connection,
                            {
                                "state": observation["joint_position"].tolist(),
                                "state_names": list(observation["joint_names"]),
                                "velocity": observation["joint_velocity"].tolist(),
                                "image_shape": list(observation["head_rgb"].shape),
                                "instruction": task.instruction,
                                "observation_time_s": env.time,
                            },
                            observation["head_rgb"].tobytes(),
                        )
                        response, _ = receive_message(self.connection)
                        if response["action_representation"] != "absolute" or response.get(
                            "task_decision_names"
                        ) != list(DECISION_NAMES):
                            raise ValueError(
                                "Expected the retained component's served motor/decision interface"
                            )
                        chunk = response["actions"]
                        if len(chunk) != 10:
                            raise ValueError(
                                "The fixed policy must serve its declared execution horizon"
                            )
                    action = np.asarray(chunk.pop(0), dtype=np.float32)
                    if action.shape != (18,) or not np.isfinite(action).all():
                        raise ValueError("Non-finite or incompatible fixed-policy prediction")
                    decision = decode_decision(action[16:])
                    stream.write(
                        json.dumps(
                            {
                                "frame": recorder.count,
                                "time_s": env.time,
                                "raw_prediction": action.tolist(),
                                "decision": decision,
                                "applied": decision == "continue",
                            }
                        )
                        + "\n"
                    )
                    if decision != "continue":
                        terminal = decision
                        break
                    bounded = np.clip(action[:16], limits[:, 0], limits[:, 1])
                    env.set_joint_targets(dict(zip(names, bounded.tolist(), strict=True)))
                    env.step(10)
                    reference.sample("actual_policy_interruption_window")
                    current = env.task_truth(next_index)
                    acquired_next |= (
                        current["block_pose_w"][2] - before[next_index]["block_pose_w"][2] >= 0.07
                        and current["robot_contact_force_n"] > 0.5
                    )
                    executed += 1
        finally:
            self.active = False
            recorder.action_source = "teacher"
            recorder.positive_imitation = True
        after = [env.task_truth(i) for i in range(len(env.blocks))]
        remaining_prefix = released_prefix(task, after)
        if executed < 1 or not completed_levels <= remaining_prefix < 5:
            raise RuntimeError("Policy did not establish a recoverable incomplete interruption")
        if terminal == "budget" and (
            env.time - start_time < 60 or acquired_next or remaining_prefix != completed_levels
        ):
            raise RuntimeError(
                "A budget ending during useful progress is not a verified interruption"
            )
        source = {
            "mode": "in_place_policy_window_v1",
            "policy": self.identity,
            "teacher_prefix_frames": start_frame,
            "conditional_start": True,
            "autonomous_from_start": False,
            "start_frame": start_frame,
            "end_frame": recorder.count,
            "start_time_s": start_time,
            "end_time_s": env.time,
            "executed_motor_commands": executed,
            "source_terminal_decision": terminal,
            "physically_verified": True,
            "completed_levels_before": completed_levels,
            "completed_levels_at_intervention": remaining_prefix,
            "object_state_writes": False,
            "teacher_commands_during_policy_window": False,
            "error_basis": (
                "premature_terminal_on_supported_incomplete_task"
                if terminal != "budget"
                else "no_acquisition_or_ordered_progress_for_at_least60s"
            ),
        }
        recorder.admit_policy_intervention(source)
        return {
            "type": "interrupted_progress",
            "teacher_induced": False,
            "physically_verified": True,
            "completed_levels": remaining_prefix,
            "error_end_s": env.time,
            "policy_source": source,
        }

    def close(self):
        try:
            send_message(self.connection, {"close": True})
        finally:
            self.connection.close()
