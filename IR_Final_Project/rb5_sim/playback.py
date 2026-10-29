"""Shared pieces of part1_fk.py and part2_ik.py: the scene, and frames drawn in the viewer."""

from collections import deque
from contextlib import contextmanager, nullcontext
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from .controller import RB5, Sim
from .env import DEFAULT_HAND, build_model
from .layout_md import load_layouts
from .video import VideoRecorder

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


def recording(robot: RB5, path: Path | None, markers: "FrameMarkers | None" = None):
    """Context manager that records the run to `path` (MP4) while active; no-op without a path."""
    if path is None:
        return nullcontext()
    video = VideoRecorder(robot.sim.model, path, overlays=[markers.draw] if markers else [])
    robot.sim.callbacks.append(video)
    return video


def wait_for_close(robot: RB5, viewer) -> None:
    if viewer is None:
        return
    print("Close the viewer window to exit.")
    while viewer.is_running():
        robot.sim.step()


class FrameMarkers:
    """Sim callback that draws, in the viewer and in --video recordings:

    - the grasp-site frame from YOUR forward_kinematics at the commanded joint angles
      (long axes) and the trail of its origin,
    - fixed frames, e.g. IK targets (short axes), and line segments between points.

    The viewer's text panel shows how far your FK frame is from the simulator's grasp site.
    """

    def __init__(self, robot: RB5, viewer=None, frames=(), segments=(), every: int = 10, trail: int = 300,
                 show_fk: bool = True):
        """show_fk=False: forward_kinematics isn't written yet; draw only the fixed frames."""
        self.robot, self.viewer = robot, viewer
        self.frames = list(frames)        # [(pos, rot)]
        self.segments = list(segments)    # [(p0, p1)]
        self.every = every
        self.show_fk = show_fk
        self.trail = deque(maxlen=trail)
        self.current = None               # (pos, rot) of your FK frame
        self._count = 0

    def __call__(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        self._count += 1
        if (self._count - 1) % self.every:
            return
        q = self.robot.q_cmd
        if not self.show_fk:
            if self.viewer is not None:
                self.viewer.user_scn.ngeom = 0
                self.draw(self.viewer.user_scn)
                self.viewer.set_texts((
                    mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
                    "time\njoints [deg]\nyour FK",
                    f"{data.time:.2f} s\n{' '.join(f'{a:.0f}' for a in np.degrees(q))}\nnot implemented yet",
                ))
            return
        pos, rot = self.robot.fk(q)
        self.current = pos, rot
        self.trail.append(pos)
        if self.viewer is None:
            return

        self.viewer.user_scn.ngeom = 0
        self.draw(self.viewer.user_scn)
        true_pos, true_rot = self.robot.sim_fk(q)
        dp = np.linalg.norm(pos - true_pos) * 1e3
        dr = np.degrees(np.arccos(np.clip((np.trace(rot @ true_rot.T) - 1) / 2, -1, 1)))
        self.viewer.set_texts((
            mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
            "time\njoints [deg]\nyour FK [mm]\nFK - sim",
            f"{data.time:.2f} s\n{' '.join(f'{a:.0f}' for a in np.degrees(q))}\n"
            f"{pos[0] * 1e3:.1f}, {pos[1] * 1e3:.1f}, {pos[2] * 1e3:.1f}\n{dp:.2f} mm, {dr:.2f} deg",
        ))

    def draw(self, scn: mujoco.MjvScene) -> None:
        """Add the markers to a scene (after the geoms already in it)."""
        for p0, p1 in self.segments:
            _line(scn, p0, p1, 0.0015, (1.0, 0.8, 0.2, 0.6))
        for p, r in self.frames:
            _frame(scn, p, r, 0.04, 0.002)
        for p in self.trail:
            _add(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [0.003, 0, 0], p, (1.0, 0.4, 0.9, 0.8))
        if self.current is not None:
            _frame(scn, *self.current, 0.09, 0.004)


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
