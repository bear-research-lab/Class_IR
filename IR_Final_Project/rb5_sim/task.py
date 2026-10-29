"""The pipetting task sequence and its validation.

The robot only knows the nominal layout. Validation compares what actually
happened (true pipette tip position, bulb contact force) against the task
regions.
"""

from dataclasses import dataclass, field

import mujoco
import numpy as np

from .controller import RB5
from .forces import ForceMonitor
from .env import (BEAKER_BOTTOM_Z, BEAKER_INNER_HALF, BEAKER_RIM_Z, HOLDER_SEAT_Z, HOLDER_TOP_Z, SOCKET_CLEARANCE,
                  Layout, WATER_BOTTOM_Z, WATER_INNER_HALF, WATER_SURFACE_Z)

# Pipette geometry: distance from tip to the grasp point (finger center) along the axis.
GRASP_ABOVE_TIP = 0.11

# Planned tip heights (world z, m).
TRANSIT_TIP_Z = 0.17               # clears the holder, tank and beaker walls
ASPIRATE_TIP_Z = WATER_SURFACE_Z - 0.025
DISPENSE_TIP_Z = 0.06
APPROACH_DISTANCE = 0.10           # pre-grasp stand-off along the approach direction
DISPENSE_HOLD = 0.5                # s, bulb held squeezed in the beaker
RETURN_TIP_Z = HOLDER_SEAT_Z + 0.001  # nominal tip height when the robot lets go in the holder

# Validation thresholds.
MIN_BULB_FORCE = 2.0               # N per side, bulb must be squeezed at least this hard
RELEASED_BULB_FORCE = 0.2          # N per side, bulb counts as released below this
BOTTOM_OUT_MARGIN = 0.0005         # m, compression this close to the max means the bulb is flat
MAX_ENV_FORCE = 10.0               # N, larger contact force from the environment = collision
TIP_MARGIN = 0.005                 # m, keep the tip this far from walls and bottom
MAX_INSERT_FORCE = 15.0            # N, peak contact force from the holder while returning
SEATED_TOLERANCE = 0.005           # m, returned tip must be within this of the socket floor


@dataclass
class Check:
    name: str
    passed: bool
    detail: str
    kind: str = "task"  # "task": must pass; "force": bulb force quality (scored by the reward)


@dataclass
class Report:
    checks: list[Check] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)  # extra numbers for evaluation

    def add(self, name: str, passed: bool, detail: str, kind: str = "task") -> None:
        self.checks.append(Check(name, passed, detail, kind))

    @property
    def passed(self) -> bool:
        return bool(self.checks) and all(c.passed for c in self.checks)

    @property
    def task_passed(self) -> bool:
        """All checks except the bulb force ones (which the reward scores instead)."""
        return bool(self.checks) and all(c.passed for c in self.checks if c.kind == "task")

    def __str__(self) -> str:
        return "\n".join(f"  [{'PASS' if c.passed else 'FAIL'}] {c.name}: {c.detail}" for c in self.checks)


def tip_at(robot: RB5, tip_xyz) -> np.ndarray:
    """Grasp-site target that puts the (nominal) pipette tip at tip_xyz."""
    return np.asarray(tip_xyz, dtype=float) + [0.0, 0.0, GRASP_ABOVE_TIP]


class Probe:
    """Reads the ground-truth values used for validation."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, forces: ForceMonitor):
        self.data = data
        self.forces = forces
        self.tip_id = model.site("tip").id
        self.holder_id = model.body("holder").id
        self.pipette_id = model.body("pipette").id
        self.socket_half = model.geom("shaft").size[0] + SOCKET_CLEARANCE

    @property
    def tip(self) -> np.ndarray:
        return self.data.site_xpos[self.tip_id].copy()

    @property
    def bulb_force(self) -> float:
        return self.forces.bulb_force(self.data)

    @property
    def bulb_compression(self) -> float:
        return self.forces.bulb_compression(self.data)

    def bulb_detail(self) -> str:
        return (f"{self.bulb_force:.2f} N per side, {self.bulb_compression * 1e3:.1f} mm compression "
                f"(min {MIN_BULB_FORCE} N)")

    def bulb_not_flat(self) -> bool:
        return self.bulb_compression < self.forces.wall_max - BOTTOM_OUT_MARGIN

    def tip_in_holder(self) -> np.ndarray:
        """True tip position in the (true) holder frame."""
        d = self.data
        rot = d.xmat[self.holder_id].reshape(3, 3)
        return rot.T @ (d.site_xpos[self.tip_id] - d.xpos[self.holder_id])

    def tilt_in_holder(self) -> float:
        """Angle between the pipette axis and the holder axis [deg]."""
        d = self.data
        a = d.xmat[self.pipette_id].reshape(3, 3)[:, 2]
        b = d.xmat[self.holder_id].reshape(3, 3)[:, 2]
        return float(np.degrees(np.arccos(np.clip(a @ b, -1.0, 1.0))))


def inside(tip: np.ndarray, center_xy, half: float, z_lo: float, z_hi: float) -> bool:
    dx, dy = np.abs(tip[:2] - np.asarray(center_xy))
    return (max(dx, dy) <= half - TIP_MARGIN) and (z_lo + TIP_MARGIN <= tip[2] <= z_hi)


def run_pipetting(robot: RB5, layout: Layout, probe: Probe, log=print) -> Report:
    report = Report()
    holder_tip = np.array([*layout.holder, HOLDER_SEAT_Z])
    water_xy, beaker_xy = np.array(layout.water), np.array(layout.beaker)
    approach = np.array([APPROACH_DISTANCE, 0.0, 0.0])  # hand approaches along +x

    def phase(name: str) -> None:
        probe.forces.phase = name
        log(name)

    phase("settling")
    robot.wait(0.5)

    # 1. Grasp the pipette from its holder (nominal pose only).
    phase("1. grasp pipette")
    robot.open_gripper(0.1)
    robot.release_bulb(0.1)
    robot.move_to(tip_at(robot, holder_tip) - approach, duration=3.0, joint_space=True)
    robot.move_to(tip_at(robot, holder_tip), duration=2.0)
    robot.close_gripper()
    robot.move_to(tip_at(robot, [*layout.holder, TRANSIT_TIP_Z]), duration=2.0, settle=0.5)
    lifted = probe.tip[2] > HOLDER_SEAT_Z + 0.1
    report.add("pipette picked up", lifted, f"tip z = {probe.tip[2] * 1e3:.0f} mm")
    if not lifted:
        return report

    # 2. Move to the water source and aspirate: squeeze, dip, release.
    phase("2. aspirate")
    robot.move_to(tip_at(robot, [*water_xy, TRANSIT_TIP_Z]), duration=2.5)
    robot.squeeze_bulb()
    report.add("aspirate: bulb squeezed before dipping",
               probe.bulb_force >= MIN_BULB_FORCE and probe.bulb_not_flat(), probe.bulb_detail(), "force")
    probe.forces.event = "aspirate hold"   # bulb held squeezed while dipping
    robot.move_to(tip_at(robot, [*water_xy, ASPIRATE_TIP_Z]), duration=1.5, settle=0.3)
    probe.forces.event = ""
    tip = probe.tip
    robot.release_bulb()
    report.add("aspirate: tip under water at release",
               inside(tip, water_xy, WATER_INNER_HALF, WATER_BOTTOM_Z, WATER_SURFACE_Z),
               _tip_detail(tip, water_xy))
    report.add("aspirate: bulb released", probe.bulb_force <= RELEASED_BULB_FORCE,
               f"{probe.bulb_force:.2f} N per side")

    # 3. Move to the beaker.
    phase("3. move to beaker")
    robot.move_to(tip_at(robot, [*water_xy, TRANSIT_TIP_Z]), duration=1.5)
    robot.move_to(tip_at(robot, [*beaker_xy, TRANSIT_TIP_Z]), duration=3.0)
    robot.move_to(tip_at(robot, [*beaker_xy, DISPENSE_TIP_Z]), duration=1.5, settle=0.3)

    # 4. Dispense: squeeze the bulb with the tip inside the beaker.
    phase("4. dispense")
    robot.squeeze_bulb()
    probe.forces.event = "dispense hold"
    robot.wait(DISPENSE_HOLD)
    probe.forces.event = ""
    tip = probe.tip
    report.add("dispense: bulb squeezed",
               probe.bulb_force >= MIN_BULB_FORCE and probe.bulb_not_flat(), probe.bulb_detail(), "force")
    report.add("dispense: tip inside beaker",
               inside(tip, beaker_xy, BEAKER_INNER_HALF, BEAKER_BOTTOM_Z, BEAKER_RIM_Z),
               _tip_detail(tip, beaker_xy))
    robot.release_bulb()
    phase("retreat")
    robot.move_to(tip_at(robot, [*beaker_xy, TRANSIT_TIP_Z]), duration=1.5)

    # Collisions: between pick-up and the return, nothing but the hand should push on the pipette.
    peaks = probe.forces.peaks_by_phase()
    env_peak = max(peaks[p]["env"] for p in peaks if p[0] in "234" or p == "retreat")
    report.add("no collisions after pick-up", env_peak <= MAX_ENV_FORCE,
               f"peak environment force {env_peak:.1f} N (max {MAX_ENV_FORCE} N)")

    # 5. Return the pipette: straight down into the nominal socket, let go, back out.
    phase("5. return pipette")
    robot.move_to(tip_at(robot, [*layout.holder, TRANSIT_TIP_Z]), duration=3.0, settle=0.3)
    robot.move_to(tip_at(robot, [*layout.holder, HOLDER_TOP_Z + 0.015]), duration=1.5)
    robot.move_to(tip_at(robot, [*layout.holder, RETURN_TIP_Z]), duration=2.5, settle=0.3)  # slow insert
    robot.open_gripper()
    robot.move_to(tip_at(robot, [*layout.holder, RETURN_TIP_Z]) - approach, duration=1.5)
    insert_force = probe.forces.peaks_by_phase()["5. return pipette"]["env"]
    phase("finish")
    robot.move_by([0.0, 0.0, 0.1], duration=1.5)
    robot.wait(0.5)

    tip_h, tilt = probe.tip_in_holder(), probe.tilt_in_holder()
    lateral = float(np.max(np.abs(tip_h[:2])))
    height = float(tip_h[2] - HOLDER_SEAT_Z)
    seated = lateral <= probe.socket_half + 0.0005 and height <= SEATED_TOLERANCE
    report.metrics.update(return_height=height, return_tilt=tilt, insert_force=insert_force)
    report.add("return: pipette seated in holder", seated,
               f"tip {height * 1e3:.1f} mm above the socket floor (max {SEATED_TOLERANCE * 1e3:.0f} mm), "
               f"{lateral * 1e3:.1f} mm off-center, leaning {tilt:.1f} deg")
    report.add("return: insertion force", insert_force <= MAX_INSERT_FORCE,
               f"peak {insert_force:.1f} N from the holder (max {MAX_INSERT_FORCE} N)")
    log("done")
    return report


def _tip_detail(tip: np.ndarray, target_xy) -> str:
    err = (tip[:2] - np.asarray(target_xy)) * 1e3
    return (f"tip xy error = ({err[0]:+.1f}, {err[1]:+.1f}) mm "
            f"[{np.linalg.norm(err):.1f} mm], z = {tip[2] * 1e3:.0f} mm")
