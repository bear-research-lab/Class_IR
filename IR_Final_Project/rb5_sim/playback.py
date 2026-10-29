"""Shared pieces of part1_fk.py and part2_ik.py: the scene, and frames drawn in the viewer."""

from collections import deque
from contextlib import contextmanager
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from .controller import RB5, Sim
from .env import DEFAULT_HAND, build_model
from .layout_md import load_layouts

AXIS_RGBA = ((0.9, 0.15, 0.15, 1.0), (0.15, 0.8, 0.15, 1.0), (0.2, 0.35, 0.95, 1.0))  # x red, y green, z blue


@contextmanager
def open_robot(kinematics, viewer: bool, layouts: Path = Path("layouts.md"), hand: Path = DEFAULT_HAND,
               playback: float = 1.0):
    """The task scene (first layout, no holder error) with the robot; yields (robot, viewer or None)."""
    model = build_model(next(iter(load_layouts(layouts).values())), hand=hand)
    data = mujoco.MjData(model)
    if not viewer:
        yield RB5(Sim(model, data), kinematics=kinematics), None
        return
    with mujoco.viewer.launch_passive(model, data) as v:
        yield RB5(Sim(model, data, viewer=v, playback=playback), kinematics=kinematics), v


def wait_for_close(robot: RB5, viewer) -> None:
    if viewer is None:
        return
    print("Close the viewer window to exit.")
    while viewer.is_running():
        robot.sim.step()


class FrameMarkers:
    """Sim callback that draws, in the viewer:

    - the grasp-site frame from YOUR forward_kinematics at the commanded joint angles
      (long axes) and the trail of its origin,
    - fixed frames, e.g. IK targets (short axes), and line segments between points.

    The text panel shows how far your FK frame is from the simulator's grasp site.
    """

    def __init__(self, viewer, robot: RB5, frames=(), segments=(), every: int = 10, trail: int = 300):
        self.viewer, self.robot = viewer, robot
        self.frames = list(frames)        # [(pos, rot)]
        self.segments = list(segments)    # [(p0, p1)]
        self.every = every
        self.trail = deque(maxlen=trail)
        self._count = 0

    def __call__(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        self._count += 1
        if self._count % self.every:
            return
        q = self.robot.q_cmd
        pos, rot = self.robot.fk(q)
        true_pos, true_rot = self.robot.sim_fk(q)
        self.trail.append(pos)

        scn = self.viewer.user_scn
        scn.ngeom = 0
        for p0, p1 in self.segments:
            _line(scn, p0, p1, 0.0015, (1.0, 0.8, 0.2, 0.6))
        for p, r in self.frames:
            _frame(scn, p, r, 0.04, 0.002)
        for p in self.trail:
            _add(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [0.003, 0, 0], p, (1.0, 0.4, 0.9, 0.8))
        _frame(scn, pos, rot, 0.09, 0.004)

        dp = np.linalg.norm(pos - true_pos) * 1e3
        dr = np.degrees(np.arccos(np.clip((np.trace(rot @ true_rot.T) - 1) / 2, -1, 1)))
        self.viewer.set_texts((
            mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
            "time\njoints [deg]\nyour FK [mm]\nFK - sim",
            f"{data.time:.2f} s\n{' '.join(f'{a:.0f}' for a in np.degrees(q))}\n"
            f"{pos[0] * 1e3:.1f}, {pos[1] * 1e3:.1f}, {pos[2] * 1e3:.1f}\n{dp:.2f} mm, {dr:.2f} deg",
        ))


def _add(scn, kind, size, pos, rgba, mat=np.eye(3)):
    if scn.ngeom >= scn.maxgeom:
        return None
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, kind, np.asarray(size, float), np.asarray(pos, float), np.asarray(mat, float).ravel(),
                        np.asarray(rgba, np.float32))
    scn.ngeom += 1
    return g


def _line(scn, p0, p1, width, rgba):
    g = _add(scn, mujoco.mjtGeom.mjGEOM_CAPSULE, [0, 0, 0], np.zeros(3), rgba)
    if g is not None:
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, width, np.asarray(p0, float), np.asarray(p1, float))


def _frame(scn, pos, rot, length, width):
    for k in range(3):
        _line(scn, pos, pos + length * rot[:, k], width, AXIS_RGBA[k])
