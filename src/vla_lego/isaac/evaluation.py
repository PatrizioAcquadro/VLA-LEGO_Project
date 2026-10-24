"""Provisional full-task scoring on privileged physics history, never policy inputs."""

from __future__ import annotations

import itertools
import math
from collections import deque

from vla_lego.isaac.environment import TABLE_TOP
from vla_lego.isaac.task import TowerTask

POSITION_TOLERANCE_M = 0.003
DWELL_DISPLACEMENT_M = 0.003
CONTACT_ZERO_N = 1e-6
PERIOD_S = 0.05
DWELL_S = 2.0
COOPERATION_S = 0.2
MARKER_INNER_HALF_M = 0.06 - 0.004 / 2


def _norm(vector) -> float:
    return math.sqrt(sum(float(v) ** 2 for v in vector))


def _corners(dimensions, pose):
    x, y, z, w = pose[3:]
    length = _norm((x, y, z, w))
    if not length > 0:
        return ()
    x, y, z, w = (v / length for v in (x, y, z, w))
    rotation = (
        (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
        (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
        (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
    )
    return tuple(
        tuple(
            pose[axis] + sum(r * v for r, v in zip(rotation[axis], corner, strict=True))
            for axis in range(3)
        )
        for corner in itertools.product(*[(-v / 2, v / 2) for v in dimensions])
    )


def _footprint_inside(task: TowerTask, index: int, pose: list[float]) -> bool:
    corners = _corners(task.placements[index].spec.dimensions_m, pose)
    if not corners:
        return False
    if task.build_region_xy_m is not None:
        # Physical tabletop support, independent of the teacher's chosen point.
        return all(
            0.25 <= corner[0] <= 0.95 and -0.55 <= corner[1] <= 0.55
            for corner in _corners(task.placements[index].spec.dimensions_m, pose)
        )
    return all(
        abs(corner[axis] - task.assembly_xy_m[axis])
        <= MARKER_INNER_HALF_M + 1e-6  # Float32 geometry tolerance, not a physical margin.
        for corner in corners
        for axis in (0, 1)
    )


class TowerEvaluator:
    """Require construction, ordered support, physical cooperation and released dwell.

    Thresholds are initial native component-calibration values. They are not
    measured print/hardware ranges or a frozen final qualification protocol.
    """

    def __init__(
        self,
        task: TowerTask,
        robot_contact_paths: list[str],
        *,
        motion_measurement: str = "pose_dwell",
    ):
        if motion_measurement not in ("solver_velocity", "pose_step_peaks", "pose_dwell"):
            raise ValueError("Unknown stability measurement protocol.")
        self.motion_measurement = motion_measurement
        self.task = task
        candidates = [
            [i for i, placement in enumerate(task.placements) if placement.spec == target]
            for target in task.targets
        ]
        self.assignments = [a for a in itertools.product(*candidates) if len(set(a)) == 5]
        self.jaws = {
            side: [
                i
                for i, path in enumerate(robot_contact_paths)
                if path.rsplit("/", 1)[-1]
                in (f"{side}_WSG32_NEGATIVE_JAW_LINK", f"{side}_WSG32_POSITIVE_JAW_LINK")
            ]
            for side in ("left", "right")
        }
        if any(len(ids) != 2 for ids in self.jaws.values()):
            raise ValueError("Evaluator requires both named jaws of each gripper.")
        bottom = TABLE_TOP
        self.expected_z = []
        for target in task.targets:
            self.expected_z.append(bottom + target.dimensions_m[2] / 2)
            bottom += target.dimensions_m[2]
        self.history: deque = deque(maxlen=round(DWELL_S / PERIOD_S) + 1)
        self.previous_time: float | None = None
        self.initial_loose: bool | None = None
        self.coordination: dict[tuple, dict] = {}
        self.events: dict[tuple, dict] = {}
        self.samples = 0
        self.clock_gaps = 0

    def _gripping(self, state: dict, side: str) -> bool:
        forces = [_norm(state["robot_contact_forces_n"][i]) for i in self.jaws[side]]
        return min(forces) > CONTACT_ZERO_N and sum(forces) > 0.5

    def _stable(self, state: dict) -> bool:
        if self.motion_measurement == "pose_dwell":
            return True  # Displacement is assessed over the complete dwell in result().
        if self.motion_measurement == "solver_velocity":
            return (
                _norm(state["block_velocity_w"][:3]) < 0.01
                and _norm(state["block_velocity_w"][3:]) < 0.1
            )
        if "motion_speed_max" not in state or "motion_window_s" not in state:
            raise ValueError("Pose-motion admission requires fresh physics-step measurements.")
        speeds = state["motion_speed_max"]
        return (
            len(speeds) == 2
            and all(math.isfinite(v) and v >= 0 for v in speeds)
            and math.isfinite(state["motion_window_s"])
            and abs(state["motion_window_s"] - PERIOD_S) < 1e-8
            and speeds[0] < 0.01
            and speeds[1] < 0.1
        )

    def update(
        self, time_s: float, blocks: list[dict], *, tool_positions: dict | None = None
    ) -> None:
        if len(blocks) != len(self.task.placements) or not math.isfinite(time_s):
            raise ValueError("Evaluator received a mismatched scene or invalid clock.")
        if self.previous_time is not None and time_s <= self.previous_time:
            raise ValueError("Evaluator samples must have strictly increasing times.")
        contiguous = (
            self.previous_time is not None and time_s - self.previous_time <= PERIOD_S + 1e-8
        )
        if not contiguous:
            if self.previous_time is not None:
                self.clock_gaps += 1
            self.coordination.clear()
        self.previous_time = time_s
        self.samples += 1
        finite = all(
            len(state["block_pose_w"]) == 7
            and len(state["block_velocity_w"]) == 6
            and all(math.isfinite(v) for v in (*state["block_pose_w"], *state["block_velocity_w"]))
            for state in blocks
        )
        if self.initial_loose is None:
            self.initial_loose = finite and all(
                abs(state["block_pose_w"][2] - TABLE_TOP - placement.spec.dimensions_m[2] / 2)
                < POSITION_TOLERANCE_M
                and state["robot_contact_force_n"] < CONTACT_ZERO_N
                and all(_norm(force) < CONTACT_ZERO_N for force in state["block_contact_forces_n"])
                for placement, state in zip(self.task.placements, blocks, strict=True)
            )
        released = all(state["robot_contact_force_n"] < CONTACT_ZERO_N for state in blocks)
        checks = []
        for assignment in self.assignments:
            aligned = []
            placed = []
            for level, index in enumerate(assignment):
                state = blocks[index]
                support = (
                    state["nonrobot_contact_force_n"]
                    if level == 0
                    else state["block_contact_forces_n"][assignment[level - 1]][2]
                )
                aligned.append(
                    finite
                    and abs(state["block_pose_w"][2] - self.expected_z[level])
                    < POSITION_TOLERANCE_M
                    and _footprint_inside(self.task, index, state["block_pose_w"])
                )
                placed.append(aligned[-1] and support > CONTACT_ZERO_N)
            stable = all(self._stable(blocks[index]) for index in assignment)
            site_valid = withdrawn = True
            if self.task.build_region_xy_m is not None:
                x0, x1, y0, y1 = self.task.build_region_xy_m
                base = blocks[assignment[0]]["block_pose_w"]
                site_valid = x0 <= base[0] <= x1 and y0 <= base[1] <= y1
                withdrawn = (
                    tool_positions is not None
                    and set(tool_positions) == {"left", "right"}
                    and all(
                        min(math.dist(position[:3], b["block_pose_w"][:3]) for b in blocks) >= 0.10
                        for position in tool_positions.values()
                    )
                )
            checks.append(
                {
                    "valid_build_site": site_valid,
                    "hands_withdrawn": withdrawn,
                    "finite": finite,
                    "ordered_support": all(placed),
                    "released": released,
                    "stable": stable,
                }
            )
            for upper_level in range(1, 5):
                upper = blocks[assignment[upper_level]]
                support = upper["block_contact_forces_n"][assignment[upper_level - 1]][2]
                for lower_level in range(upper_level):
                    lower = blocks[assignment[lower_level]]
                    for stabilizer, placer in (("left", "right"), ("right", "left")):
                        key = (assignment, lower_level, upper_level, stabilizer)
                        active = self._gripping(lower, stabilizer) and self._gripping(upper, placer)
                        if not active:
                            self.coordination.pop(key, None)
                            continue
                        progress = self.coordination.setdefault(
                            key, {"approach": False, "contact_start": None}
                        )
                        if (
                            # The stabilizing hand can carry the aligned lower
                            # piece before the incoming load restores its support.
                            all(placed[:lower_level])
                            and aligned[lower_level]
                            and support <= CONTACT_ZERO_N
                            and upper["block_pose_w"][2] - self.expected_z[upper_level]
                            >= POSITION_TOLERANCE_M
                        ):
                            progress["approach"] = True
                        if (
                            progress["approach"]
                            and all(placed[:upper_level])
                            and abs(
                                min(
                                    c[2]
                                    for c in _corners(
                                        self.task.placements[
                                            assignment[upper_level]
                                        ].spec.dimensions_m,
                                        upper["block_pose_w"],
                                    )
                                )
                                - max(
                                    c[2]
                                    for c in _corners(
                                        self.task.placements[
                                            assignment[upper_level - 1]
                                        ].spec.dimensions_m,
                                        blocks[assignment[upper_level - 1]]["block_pose_w"],
                                    )
                                )
                            )
                            < POSITION_TOLERANCE_M
                            and support > 0.05
                        ):
                            if progress["contact_start"] is None:
                                progress["contact_start"] = time_s
                            duration = time_s - progress["contact_start"]
                            if duration >= COOPERATION_S - 1e-8:
                                event_key = (lower_level, upper_level, stabilizer)
                                previous = self.events.get(event_key, {}).get("duration_s", 0)
                                if duration > previous:
                                    self.events[event_key] = {
                                        "lower_level": lower_level,
                                        "upper_level": upper_level,
                                        "stabilizer": stabilizer,
                                        "placer": placer,
                                        "contact_start_s": progress["contact_start"],
                                        "duration_s": duration,
                                    }
                        else:
                            progress["contact_start"] = None
        self.history.append((time_s, checks, tuple(tuple(b["block_pose_w"]) for b in blocks)))

    def _dwell_displacements(self) -> list[float]:
        if not self.history:
            return []
        displacements = []
        for index, placement in enumerate(self.task.placements):
            reference = _corners(placement.spec.dimensions_m, self.history[0][2][index])
            displacements.append(
                max(
                    (
                        _norm(tuple(b - a for a, b in zip(start, current, strict=True)))
                        for row in self.history
                        for start, current in zip(
                            reference,
                            _corners(placement.spec.dimensions_m, row[2][index]),
                            strict=False,
                        )
                    ),
                    default=math.inf,
                )
            )
        return displacements

    def result(self) -> dict:
        times = [row[0] for row in self.history]
        complete_dwell = (
            len(times) == self.history.maxlen
            and times[-1] - times[0] >= DWELL_S - 1e-8
            and all(b - a <= PERIOD_S + 1e-8 for a, b in zip(times, times[1:], strict=False))
        )
        displacements = (
            self._dwell_displacements() if self.motion_measurement == "pose_dwell" else []
        )
        passing = [
            assignment
            for i, assignment in enumerate(self.assignments)
            if complete_dwell
            and (
                self.motion_measurement != "pose_dwell"
                or all(displacements[index] <= DWELL_DISPLACEMENT_M for index in assignment)
            )
            and all(
                all(
                    value
                    for key, value in row[1][i].items()
                    # Forty motion intervals cover the same two seconds as
                    # forty-one geometry/release points; do not add 50 ms.
                    if key != "stable" or self.motion_measurement == "solver_velocity" or sample > 0
                )
                for sample, row in enumerate(self.history)
            )
        ]
        checks = {
            "requested_pieces_present": bool(self.assignments),
            "initially_loose": bool(self.initial_loose),
            "complete_dwell": complete_dwell,
            "ordered_supported_released_stable": bool(passing),
            "functional_cooperation": bool(self.events),
        }
        return {
            "passed": all(checks.values()),
            "motion_measurement": self.motion_measurement,
            "dwell_corner_displacement_m": displacements,
            "dwell_displacement_limit_m": (
                DWELL_DISPLACEMENT_M if self.motion_measurement == "pose_dwell" else None
            ),
            "checks": checks,
            "final_assignment_indices": list(passing[0]) if passing else None,
            "cooperation_events": list(self.events.values()),
            "samples": self.samples,
            "clock_gaps_outside_scored_intervals": self.clock_gaps,
            "last_sample_checks": self.history[-1][1] if self.history else [],
            "calibration": "provisional native simulation; print/hardware ranges unmeasured",
        }
