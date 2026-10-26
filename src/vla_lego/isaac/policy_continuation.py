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


def released_prefix(task, states, *, held_index=None):
    """Current ordered support, never a historical maximum or a controller input."""
    if any(
        state["robot_contact_force_n"] >= CONTACT_ZERO_N
        for i, state in enumerate(states)
        if i != held_index
    ):
        raise ValueError("Interruption handoff still holds a block; another recovery is required")
    if any(
        not all(math.isfinite(v) for v in (*state["block_pose_w"], *state["block_velocity_w"]))
        or (
            i != held_index
            and state["nonrobot_contact_force_n"] <= CONTACT_ZERO_N
            and not any(force[2] > CONTACT_ZERO_N for force in state["block_contact_forces_n"])
        )
        for i, state in enumerate(states)
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

    def _window(self, reference, task, index, *, ticks, stop_when=None):
        """Execute only model commands; callers establish the actual error afterward."""
        import numpy as np

        from vla_lego.isaac.environment import ARM_JOINTS, JAW_JOINTS

        env, recorder = reference.env, reference.env.recorder
        before = [env.task_truth(i) for i in range(len(env.blocks))]
        start_frame, start_time = recorder.count, env.time
        names = (*ARM_JOINTS, *JAW_JOINTS)
        ids = [env.robot.data.joint_names.index(name) for name in names]
        limits = env.robot.data.joint_pos_limits.torch[0, ids].cpu().numpy()
        chunk = []
        terminal, executed = "budget", 0
        observed_error = None
        acquired_next = False
        recorder.positive_imitation = False
        recorder.action_source = "policy"
        recorder.stage = "actual_policy_error_window"
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
                    reference.sample("actual_policy_error_window")
                    current = env.task_truth(index)
                    acquired_next |= (
                        current["block_pose_w"][2] - before[index]["block_pose_w"][2] >= 0.07
                        and current["robot_contact_force_n"] > 0.5
                    )
                    executed += 1
                    if stop_when is not None:
                        observed_error = stop_when(
                            [env.task_truth(i) for i in range(len(env.blocks))]
                        )
                        if observed_error is not None:
                            terminal = "physical_intervention"
                            break
        finally:
            self.active = False
            recorder.action_source = "teacher"
            recorder.positive_imitation = True
        after = [env.task_truth(i) for i in range(len(env.blocks))]
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
            "physically_verified": False,
            "object_state_writes": False,
            "teacher_commands_during_policy_window": False,
            "observed_error": observed_error,
        }
        return source, before, after, acquired_next

    def missed_grasp(self, reference, case, task, index, *, ticks=1800):
        """Intervene after a measured empty close-and-lift, not mere inaction."""
        env, recorder = reference.env, reference.env.recorder
        if (
            case.row["correction_origin"] != "policy"
            or case.row["mechanism"] != "missed_grasp"
            or case.row["partition"] != "train"
            or not 0 <= case.error_level < case.cooperative_level
            or ticks < 1
        ):
            raise ValueError("This empty-grasp handoff precedes bimanual stabilization")
        initial = [env.task_truth(i) for i in range(len(env.blocks))]
        if released_prefix(task, initial) != case.error_level:
            raise ValueError("Missed-grasp window lacks its supported nominal prefix")
        attempts = {}
        acquired = False
        width = task.placements[index].spec.dimensions_m[1]

        def observe(states):
            nonlocal acquired
            # An actual low closed jaw followed by an empty 7 cm lift is consequential.
            # Waiting, harmless tracking error and a successful lift are not misses.
            acquired |= any(
                state["robot_contact_force_n"] > 0.5
                and state["block_pose_w"][2] - initial[i]["block_pose_w"][2] >= 0.07
                for i, state in enumerate(states)
            )
            if acquired:
                return None
            target = states[index]
            for side in ("left", "right"):
                tool = reference.ctrl.pose(side)[0, :3].tolist()
                jaw_id = env.robot.data.joint_names.index(f"{side}_WSG32_JAW_OPENING")
                jaw = float(env.robot.data.joint_pos.torch[0, jaw_id])
                if side not in attempts and (
                    jaw < width / 2 - 0.002
                    and math.dist(tool[:2], target["block_pose_w"][:2]) < 0.12
                    and 0 < tool[2] - target["block_pose_w"][2] < 0.12
                ):
                    attempts[side] = {
                        "side": side,
                        "close_tool_z_m": tool[2],
                        "close_time_s": env.time,
                    }
                attempt = attempts.get(side)
                if (
                    attempt is not None
                    and jaw < width / 2 - 0.002
                    and tool[2] - attempt["close_tool_z_m"] >= 0.07
                    and target["robot_contact_force_n"] < CONTACT_ZERO_N
                    and target["nonrobot_contact_force_n"] > CONTACT_ZERO_N
                    and abs(target["block_pose_w"][2] - initial[index]["block_pose_w"][2]) < 0.01
                ):
                    return {**attempt, "empty_lift_m": tool[2] - attempt["close_tool_z_m"]}
            return None

        source, _, after, _ = self._window(reference, task, index, ticks=ticks, stop_when=observe)
        if (
            source["executed_motor_commands"] < 1
            or not source.get("observed_error")
            or released_prefix(task, after) != case.error_level
        ):
            raise RuntimeError("Actual policy did not produce a recoverable empty grasp-and-lift")
        source.update(physically_verified=True, error_basis="measured_empty_close_and_lift")
        recorder.admit_policy_intervention(source)
        return {
            "type": "missed_grasp",
            "teacher_induced": False,
            "physically_verified": True,
            "target_level": case.error_level + 1,
            "error_end_s": env.time,
            "policy_source": source,
        }

    def interrupted_progress(self, reference, case, task, *, completed_levels, ticks=1800):
        """A real policy resumes a correct partial tower; unsuccessful attempts stay failures."""
        env, recorder = reference.env, reference.env.recorder
        if (
            case.row["correction_origin"] != "policy"
            or case.row["mechanism"] != "interrupted_progress"
            or case.row["partition"] != "train"
            or ticks < 1
            or not 1 <= completed_levels < 5
        ):
            raise ValueError("This policy handoff requires the declared TRAIN interruption case")
        if (
            released_prefix(task, [env.task_truth(i) for i in range(len(env.blocks))])
            != completed_levels
        ):
            raise ValueError("Conditional teacher prefix is not the supported partial tower")
        index = next(
            i for i, p in enumerate(task.placements) if p.spec == task.targets[completed_levels]
        )
        source, _before, after, acquired_next = self._window(reference, task, index, ticks=ticks)
        remaining_prefix = released_prefix(task, after)
        if source["executed_motor_commands"] < 1 or not completed_levels <= remaining_prefix < 5:
            raise RuntimeError("Policy did not establish a recoverable incomplete interruption")
        if source["source_terminal_decision"] == "budget" and (
            env.time - source["start_time_s"] < 60
            or acquired_next
            or remaining_prefix != completed_levels
        ):
            raise RuntimeError(
                "A budget ending during useful progress is not a verified interruption"
            )
        source.update(
            physically_verified=True,
            completed_levels_before=completed_levels,
            completed_levels_at_intervention=remaining_prefix,
            error_basis=(
                "premature_terminal_on_supported_incomplete_task"
                if source["source_terminal_decision"] != "budget"
                else "no_acquisition_or_ordered_progress_for_at_least60s"
            ),
        )
        recorder.admit_policy_intervention(source)
        return {
            "type": "interrupted_progress",
            "teacher_induced": False,
            "physically_verified": True,
            "completed_levels": remaining_prefix,
            "error_end_s": env.time,
            "policy_source": source,
        }

    def wrong_piece(self, reference, case, task, index, *, ticks=1800):
        """Catch a real bilateral lift of a piece different from the next requested one."""
        env, recorder = reference.env, reference.env.recorder
        if (
            case.row["correction_origin"] != "policy"
            or case.row["mechanism"] != "wrong_piece"
            or case.row["partition"] != "train"
            or not 0 <= case.error_level < case.cooperative_level
            or ticks < 1
        ):
            raise ValueError("This wrong-piece handoff precedes bimanual stabilization")
        initial = [env.task_truth(i) for i in range(len(env.blocks))]
        if released_prefix(task, initial) != case.error_level:
            raise ValueError("Wrong-piece window lacks its supported nominal prefix")
        jaws = {
            side: [
                i
                for i, path in enumerate(env.contact_robot_paths)
                if path.rsplit("/", 1)[-1]
                in (f"{side}_WSG32_NEGATIVE_JAW_LINK", f"{side}_WSG32_POSITIVE_JAW_LINK")
            ]
            for side in ("left", "right")
        }
        if any(len(ids) != 2 for ids in jaws.values()):
            raise ValueError("Wrong-piece detection requires both actual named jaws")

        def observe(states):
            for i, state in enumerate(states):
                if (
                    task.placements[i].spec == task.placements[index].spec
                    or state["block_pose_w"][2] - initial[i]["block_pose_w"][2] < 0.07
                ):
                    continue
                for side, ids in jaws.items():
                    forces = [
                        math.sqrt(sum(v * v for v in state["robot_contact_forces_n"][j]))
                        for j in ids
                    ]
                    if min(forces) > CONTACT_ZERO_N and sum(forces) > 0.5:
                        return {
                            "object_index": i,
                            "side": side,
                            "return_source_xy_m": initial[i]["block_pose_w"][:2],
                            "lift_m": state["block_pose_w"][2] - initial[i]["block_pose_w"][2],
                        }
            return None

        source, _, after, _ = self._window(reference, task, index, ticks=ticks, stop_when=observe)
        error = source.get("observed_error")
        if (
            source["executed_motor_commands"] < 1
            or not error
            or released_prefix(task, after, held_index=error["object_index"]) != case.error_level
        ):
            raise RuntimeError("Actual policy did not produce a recoverable wrong-piece lift")
        source.update(physically_verified=True, error_basis="actual_bilateral_wrong_piece_lift")
        recorder.admit_policy_intervention(source)
        return {
            "type": "wrong_piece",
            "teacher_induced": False,
            "physically_verified": True,
            "target_level": case.error_level + 1,
            "error_end_s": env.time,
            "policy_source": source,
            **error,
        }

    def slip_drop(self, reference, case, task, index, *, ticks=1800):
        """Observe a policy-caused loss from a valid held block; never issue a drop command."""
        env, recorder = reference.env, reference.env.recorder
        if (
            case.row["correction_origin"] != "policy"
            or case.row["mechanism"] != "slip_drop"
            or case.row["partition"] != "train"
            or not 0 <= case.error_level < case.cooperative_level
            or ticks < 1
        ):
            raise ValueError("This drop handoff precedes active bimanual stabilization")
        held = env.task_truth(index)
        height = task.placements[index].spec.dimensions_m[2]
        if (
            held["robot_contact_force_n"] <= 0.5
            or held["block_pose_w"][2] < TABLE_TOP + height / 2 + 0.07
        ):
            raise ValueError("Policy drop window lacks its physically acquired starting block")
        source, before, after, _ = self._window(reference, task, index, ticks=ticks)
        state = after[index]
        fall = before[index]["block_pose_w"][2] - state["block_pose_w"][2]
        if not (
            source["executed_motor_commands"] > 0
            and fall >= 0.07
            and abs(state["block_pose_w"][2] - TABLE_TOP - height / 2) < 0.01
            and state["robot_contact_force_n"] < CONTACT_ZERO_N
            and state["nonrobot_contact_force_n"] > CONTACT_ZERO_N
            and 0.25 <= state["block_pose_w"][0] <= 0.55
            and abs(state["block_pose_w"][1]) < 0.5
            and released_prefix(task, after) == case.error_level
        ):
            raise RuntimeError("Actual policy did not produce the declared reachable released drop")
        source.update(physically_verified=True, error_basis="actual_released_fall", fall_m=fall)
        recorder.admit_policy_intervention(source)
        return {
            "type": "slip_drop",
            "teacher_induced": False,
            "physically_verified": True,
            "target_level": case.error_level + 1,
            "fall_m": fall,
            "recovery_source_pose_w": state["block_pose_w"],
            "error_end_s": env.time,
            "policy_source": source,
        }

    def close(self):
        try:
            send_message(self.connection, {"close": True})
        finally:
            self.connection.close()
