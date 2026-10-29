"""Actual fixed-policy observation windows followed by in-place teacher intervention."""

from __future__ import annotations

import json
import math
import socket
from collections import deque
from pathlib import Path

from vla_lego.isaac.environment import TABLE_TOP
from vla_lego.isaac.evaluation import (
    CONTACT_ZERO_N,
    POSITION_TOLERANCE_M,
    _corners,
    _footprint_inside,
)
from vla_lego.pretrained.decisions import DECISION_NAMES, decode_decision
from vla_lego.pretrained.transport import receive_message, send_message


def reject_head_contact(diagnostics):
    """Abort collection on physical head contact; never steer the policy with truth."""
    if any(
        path.rsplit("/", 1)[-1] == "HEAD_LINK" and math.hypot(*force) > CONTACT_ZERO_N
        for path, force in zip(
            diagnostics["robot_contact_paths"], diagnostics["robot_contact_forces_n"], strict=True
        )
    ):
        raise RuntimeError("Actual policy contacted the head; preserve and exclude this attempt")


def released_prefix(task, states, *, held_index=None, supported_contacts=()):
    """Current ordered support, never a historical maximum or a controller input."""
    if any(
        state["robot_contact_force_n"] >= CONTACT_ZERO_N
        for i, state in enumerate(states)
        if i != held_index and i not in supported_contacts
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
        reject_head_contact(env.robot_diagnostics())
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
                    reject_head_contact(env.robot_diagnostics())
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

    def _pick_interruption(self, reference, case, task, source, after, *, acquired, progressed):
        """Preserve a real incomplete interruption independently of the requested error."""
        env, recorder = reference.env, reference.env.recorder
        if (
            not source.get("observed_error")
            and source["executed_motor_commands"] > 0
            and (
                source.get("source_terminal_decision") in ("complete", "abstain")
                or (
                    source.get("source_terminal_decision") == "budget"
                    and env.time - source["start_time_s"] >= 60
                )
            )
            and not acquired
            and not progressed
            and released_prefix(task, after) == case.error_level
        ):
            source.update(
                physically_verified=True,
                requested_mechanism=case.row["mechanism"],
                observed_mechanism="interrupted_progress",
                completed_levels_before=case.error_level,
                completed_levels_at_intervention=case.error_level,
                error_basis=(
                    "no_acquisition_or_ordered_progress_for_at_least60s"
                    if source["source_terminal_decision"] == "budget"
                    else "premature_terminal_on_supported_incomplete_task"
                ),
            )
            recorder.admit_policy_intervention(source)
            return {
                "type": "interrupted_progress",
                "teacher_induced": False,
                "physically_verified": True,
                "completed_levels": case.error_level,
                "error_end_s": env.time,
                "policy_source": source,
            }
        return None

    def missed_grasp(self, reference, case, task, index, *, ticks=1800):
        """Observe an empty lift or actual incomplete-task interruption without inducing it."""
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
        progressed = False
        width = task.placements[index].spec.dimensions_m[1]

        def observe(states):
            nonlocal acquired, progressed
            try:
                progressed |= released_prefix(task, states) > case.error_level
            except ValueError:
                pass  # A held or unsupported piece cannot establish ordered progress.
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
        interruption = self._pick_interruption(
            reference, case, task, source, after, acquired=acquired, progressed=progressed
        )
        if interruption is not None:
            return interruption
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

        acquired, progressed = False, False

        def observe(states):
            nonlocal acquired, progressed
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
            try:
                progressed |= released_prefix(task, states) > case.error_level
            except ValueError:
                pass
            acquired |= any(
                state["robot_contact_force_n"] > 0.5
                and state["block_pose_w"][2] - initial[i]["block_pose_w"][2] >= 0.07
                for i, state in enumerate(states)
            )
            return None

        source, _, after, _ = self._window(reference, task, index, ticks=ticks, stop_when=observe)
        error = source.get("observed_error")
        interruption = self._pick_interruption(
            reference, case, task, source, after, acquired=acquired, progressed=progressed
        )
        if interruption is not None:
            return interruption
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
        quiet = 0

        def dropped(states):
            state = states[index]
            fall = held["block_pose_w"][2] - state["block_pose_w"][2]
            if (
                fall >= 0.07
                and abs(state["block_pose_w"][2] - TABLE_TOP - height / 2) < 0.01
                and state["robot_contact_force_n"] < CONTACT_ZERO_N
                and state["nonrobot_contact_force_n"] > CONTACT_ZERO_N
                and math.hypot(*state["block_velocity_w"][:3]) < 0.02
                and 0.25 <= state["block_pose_w"][0] <= 0.55
                and abs(state["block_pose_w"][1]) < 0.5
                and released_prefix(task, states) == case.error_level
            ):
                return {"error_basis": "actual_released_fall", "fall_m": fall}
            return None

        def observe(states):
            nonlocal quiet
            event = dropped(states)
            quiet = quiet + 1 if event is not None else 0
            return event if quiet >= 3 else None

        source, before, after, _ = self._window(
            reference, task, index, ticks=ticks, stop_when=observe
        )
        state = after[index]
        fall = before[index]["block_pose_w"][2] - state["block_pose_w"][2]
        if source["executed_motor_commands"] < 1 or dropped(after) is None:
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

    def placement(self, reference, case, task, index, side, *, stabilizer_side=None, ticks=1800):
        """Require failed physical load transfer after actual policy placement motion."""
        env, recorder = reference.env, reference.env.recorder
        if (
            case.row["correction_origin"] != "policy"
            or case.row["mechanism"] != "placement"
            or case.row["partition"] != "train"
            or not 1 <= case.error_level < 5
            or ticks < 1
        ):
            raise ValueError("Placement handoff requires its declared incomplete TRAIN tower")
        indices = [
            next(i for i, p in enumerate(task.placements) if p.spec == target)
            for target in task.targets
        ]
        allowed = indices[: case.error_level] if stabilizer_side is not None else ()

        def gripping(state, hand):
            ids = [
                i
                for i, path in enumerate(env.contact_robot_paths)
                if path.rsplit("/", 1)[-1]
                in (f"{hand}_WSG32_NEGATIVE_JAW_LINK", f"{hand}_WSG32_POSITIVE_JAW_LINK")
            ]
            forces = [
                math.sqrt(sum(v * v for v in state["robot_contact_forces_n"][i])) for i in ids
            ]
            return len(forces) == 2 and min(forces) > CONTACT_ZERO_N and sum(forces) > 0.5

        initial = [env.task_truth(i) for i in range(len(env.blocks))]
        if (
            not gripping(initial[index], side)
            or released_prefix(task, initial, held_index=index, supported_contacts=allowed)
            != case.error_level
        ):
            raise ValueError(
                "Placement window lacks an acquired block and physically supported prefix"
            )
        expected_z = (
            TABLE_TOP
            + sum(target.dimensions_m[2] for target in task.targets[: case.error_level])
            + task.targets[case.error_level].dimensions_m[2] / 2
        )
        initial_xy = initial[index]["block_pose_w"][:2]
        max_z = initial[index]["block_pose_w"][2]
        descent, lateral = 0.0, 0.0
        unsupported_since = None
        below_support_since = None
        outside_support_since = None
        support_overhang = 0.0
        evidence = None
        recent = deque()
        released_quiet = 0
        progress_levels, progress_since = None, None
        progress_poses = {}
        progress_supported_s = 0.0
        progress_previous_time, progress_previous_supported = None, False
        remaining_acquired = False

        def observe(states):
            nonlocal max_z, descent, lateral, unsupported_since, evidence, released_quiet
            nonlocal below_support_since
            nonlocal outside_support_since, support_overhang
            nonlocal progress_levels, progress_since, remaining_acquired
            nonlocal progress_poses, progress_supported_s
            nonlocal progress_previous_time, progress_previous_supported
            state = states[index]
            pose = state["block_pose_w"]
            # A supported placement can receive incidental single-jaw contact
            # while the opposite hand continues stabilizing the lower layer.
            try:
                supported = released_prefix(task, states, supported_contacts=(*allowed, index))
            except ValueError:
                supported = None
            if any(gripping(state, hand) for hand in ("left", "right")):
                supported = None
            moved = any(
                math.dist(states[i]["block_pose_w"][:3], xyz) >= 0.02
                for i, xyz in progress_poses.items()
            )
            new_level = (
                supported is not None
                and supported > case.error_level
                and supported != len(progress_poses)
            )
            if moved or new_level:
                progress_poses, progress_since = {}, None
                progress_supported_s, progress_previous_supported = 0.0, False
            if not progress_poses and supported is not None and supported > case.error_level:
                progress_poses = {
                    i: tuple(states[i]["block_pose_w"][:3]) for i in indices[:supported]
                }
                progress_since = env.time
            valid_progress = supported is not None and supported == len(progress_poses)
            # Brief tracking/contact excursions pause evidence, rather than
            # erase it. Consequential 2 cm motion still starts a new interval.
            if valid_progress and progress_previous_supported:
                progress_supported_s += env.time - progress_previous_time
            progress_levels = supported
            progress_previous_time, progress_previous_supported = env.time, valid_progress
            for next_index in indices[case.error_level + 1 :]:
                remaining_acquired |= (
                    states[next_index]["block_pose_w"][2] - initial[next_index]["block_pose_w"][2]
                    >= 0.07
                    and states[next_index]["robot_contact_force_n"] > 0.5
                )
            recent.append((env.time, tuple(pose[:3])))
            while len(recent) > 1 and recent[1][0] <= env.time - 5:
                recent.popleft()
            max_z = max(max_z, pose[2])
            descent = max(descent, max_z - pose[2])
            lateral = max(lateral, math.dist(pose[:2], initial_xy))
            if gripping(state, side) and pose[2] <= expected_z - 0.02:
                if below_support_since is None:
                    below_support_since = env.time
            else:
                below_support_since = None
            support_corners = _corners(
                task.targets[case.error_level - 1].dimensions_m,
                states[indices[case.error_level - 1]]["block_pose_w"],
            )
            support_overhang = max(
                max(
                    min(c[a] for c in support_corners) - pose[a],
                    pose[a] - max(c[a] for c in support_corners),
                    0.0,
                )
                for a in (0, 1)
            )
            if gripping(state, side) and support_overhang >= 0.02:
                if outside_support_since is None:
                    outside_support_since = env.time
            else:
                outside_support_since = None
            if (
                state["robot_contact_force_n"] < CONTACT_ZERO_N
                and (
                    state["nonrobot_contact_force_n"] > CONTACT_ZERO_N
                    or any(f[2] > CONTACT_ZERO_N for f in state["block_contact_forces_n"])
                )
                and math.hypot(*state["block_velocity_w"][:3]) < 0.02
                and max(descent, lateral) >= 0.02
            ):
                try:
                    prefix = released_prefix(task, states, supported_contacts=allowed)
                except ValueError:
                    prefix = -1
                if 1 <= prefix <= case.error_level:
                    released_quiet += 1
                    if released_quiet >= 3:
                        return {
                            "mode": "released_failed_load_transfer",
                            "completed_levels": prefix,
                            "actual_descent_m": descent,
                            "actual_lateral_motion_m": lateral,
                        }
                else:
                    released_quiet = 0
            else:
                released_quiet = 0
            support = state["block_contact_forces_n"][indices[case.error_level - 1]][2]
            if (
                gripping(state, side)
                and support <= CONTACT_ZERO_N
                and pose[2] - expected_z > 0.01
                and math.dist(pose[:2], states[indices[0]]["block_pose_w"][:2]) < 0.10
            ):
                if unsupported_since is None:
                    unsupported_since = env.time
                evidence = {
                    "support_n": support,
                    "height_gap_m": pose[2] - expected_z,
                    "unsupported_duration_s": env.time - unsupported_since,
                    "actual_descent_m": descent,
                    "actual_lateral_motion_m": lateral,
                }
            else:
                unsupported_since, evidence = None, None
            return None  # Normal transport pauses are not early failure interventions.

        source, _, after, _ = self._window(reference, task, index, ticks=ticks, stop_when=observe)
        recent_motion = (
            math.sqrt(
                sum(
                    (max(p[axis] for _, p in recent) - min(p[axis] for _, p in recent)) ** 2
                    for axis in range(3)
                )
            )
            if recent
            else math.inf
        )
        stalled_after_placement = (
            source["source_terminal_decision"] == "budget"
            and progress_levels is not None
            and case.error_level < progress_levels < 5
            and progress_since is not None
            and progress_supported_s >= 60
            and not remaining_acquired
            and recent
            and env.time - recent[0][0] >= 5
            and recent_motion < 0.01
        )
        if source["source_terminal_decision"] in ("complete", "abstain") or stalled_after_placement:
            try:
                # Match the observer's supported single-jaw contact convention.
                completed = released_prefix(task, after, supported_contacts=(*allowed, index))
            except ValueError:
                completed = -1
            if (
                source["executed_motor_commands"] > 0
                and case.error_level < completed < 5
                and not any(gripping(after[index], hand) for hand in ("left", "right"))
            ):
                source.update(
                    physically_verified=True,
                    error_basis=(
                        "no_resume_after_supported_placement_for_at_least60s"
                        if stalled_after_placement
                        else "premature_terminal_after_actual_placement"
                    ),
                    observed_mechanism="interrupted_progress",
                    completed_levels_at_intervention=completed,
                    supported_progress_duration_s=progress_supported_s,
                )
                recorder.admit_policy_intervention(source)
                return {
                    "type": "interrupted_progress",
                    "teacher_induced": False,
                    "physically_verified": True,
                    "completed_levels": completed,
                    "recovery_mode": "rebuild_released_prefix",
                    "error_end_s": env.time,
                    "policy_source": source,
                }
        observed = source.get("observed_error")
        if observed and observed.get("mode") == "released_failed_load_transfer":
            if (
                source["executed_motor_commands"] < 1
                or released_prefix(task, after, supported_contacts=allowed)
                != observed["completed_levels"]
            ):
                raise RuntimeError("Released placement correction lost its supported prefix")
            source.update(physically_verified=True, error_basis="released_failed_load_transfer")
            recorder.admit_policy_intervention(source)
            return {
                "type": "placement",
                "teacher_induced": False,
                "physically_verified": True,
                "target_level": case.error_level + 1,
                "recovery_mode": "rebuild_released_prefix",
                "error_end_s": env.time,
                "policy_source": source,
                **observed,
            }
        completed = released_prefix(task, after, held_index=index, supported_contacts=allowed)
        stabilizer_lost = stabilizer_side is not None and not any(
            gripping(after[i], stabilizer_side) for i in allowed
        )
        # Wait for the policy's own terminal/budget before correcting held pieces.
        held_error = None
        if (
            source["executed_motor_commands"] > 0
            and below_support_since is not None
            and env.time - below_support_since >= 2
            and descent >= 0.02
            and completed == case.error_level
            and (
                source["source_terminal_decision"] in ("complete", "abstain")
                or (
                    source["source_terminal_decision"] == "budget"
                    and env.time - source["start_time_s"] >= 60
                )
            )
        ):
            observed = {
                "height_gap_m": after[index]["block_pose_w"][2] - expected_z,
                "below_support_duration_s": env.time - below_support_since,
                "actual_descent_m": descent,
                "actual_lateral_motion_m": lateral,
                "stabilizer_grasp_lost": stabilizer_lost,
            }
            held_error = "held_below_required_support", observed
        # Edge contact can carry some load while the hand still holds an
        # off-support piece. That is not a completed placement or an air stall.
        elif (
            source["executed_motor_commands"] > 0
            and outside_support_since is not None
            and env.time - outside_support_since >= 2
            and max(descent, lateral) >= 0.02
            and completed == case.error_level
            and (
                source["source_terminal_decision"] in ("complete", "abstain")
                or (
                    source["source_terminal_decision"] == "budget"
                    and env.time - source["start_time_s"] >= 60
                    and recent
                    and env.time - recent[0][0] >= 5
                    and recent_motion < 0.01
                )
            )
        ):
            observed = {
                "actual_support_overhang_m": support_overhang,
                "outside_support_duration_s": env.time - outside_support_since,
                "actual_descent_m": descent,
                "actual_lateral_motion_m": lateral,
                "recent_held_motion_m": recent_motion,
                "stabilizer_grasp_lost": stabilizer_lost,
            }
            held_error = "held_outside_support_after_policy_stop", observed
        if held_error is not None:
            basis, observed = held_error
            source.update(physically_verified=True, error_basis=basis, observed_error=observed)
            recorder.admit_policy_intervention(source)
            return {
                "type": "placement",
                "teacher_induced": False,
                "physically_verified": True,
                "target_level": case.error_level + 1,
                "recovery_mode": "return_held_then_rebuild",
                "completed_levels": completed,
                "object_index": index,
                "side": side,
                "return_source_xy_m": list(task.placements[index].xy_m),
                "error_end_s": env.time,
                "policy_source": source,
                **observed,
            }
        if (
            source["executed_motor_commands"] < 1
            or evidence is None
            or evidence["unsupported_duration_s"] < 2
            or max(descent, lateral) < 0.02
            or (
                source["source_terminal_decision"] == "budget"
                and (
                    env.time - source["start_time_s"] < 60
                    or not recent
                    or env.time - recent[0][0] < 5
                    or recent_motion >= 0.01
                )
            )
            or completed != case.error_level
        ):
            raise RuntimeError("Actual policy did not establish failed physical placement")
        source.update(
            physically_verified=True,
            error_basis="failed_actual_load_transfer",
            observed_error=evidence,
            recent_held_motion_m=recent_motion,
        )
        recorder.admit_policy_intervention(source)
        event = {
            "type": "placement",
            "teacher_induced": False,
            "physically_verified": True,
            "target_level": case.error_level + 1,
            "error_end_s": env.time,
            "policy_source": source,
            **evidence,
        }
        if stabilizer_lost:
            event.update(
                recovery_mode="return_held_then_rebuild",
                completed_levels=completed,
                object_index=index,
                side=side,
                return_source_xy_m=list(task.placements[index].xy_m),
                stabilizer_grasp_lost=True,
            )
        return event

    def close(self):
        try:
            send_message(self.connection, {"close": True})
        finally:
            self.connection.close()
