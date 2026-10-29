"""RB5 arm control: smooth Cartesian/joint moves and hand commands.

The arm's kinematics come from student_kinematics.py (Parts 1 and 2): every move
uses its forward_kinematics() and inverse_kinematics().

Motion commands are blocking: they step the simulation until the move is done.
Every command only uses targets you give it; the controller never looks at the
pipette or holder pose.
"""

import time

import mujoco
import numpy as np

from .kinematics import load_kinematics, pose_matrix

ARM_JOINTS = ("base", "shoulder", "elbow", "wrist1", "wrist2", "wrist3")

# Starting configuration: grasp site at (0.40, 0, 0.35), hand pointing forward (+x).
# IK is always seeded from the current command, so this also picks the IK branch:
# elbow up, wrist above the hand (the other branches hit the bench at low targets).
HOME_Q = np.array([0.5457, -0.492, 2.2764, 1.3572, -1.0251, 1.5708])

# Hand orientation used for the whole task: approach (hand z) along +x,
# pipette axis (hand x) pointing up, so the pipette hangs vertically.
HAND_FORWARD = np.array([
    [0.0, 0.0, 1.0],
    [0.0, -1.0, 0.0],
    [1.0, 0.0, 0.0],
])

# Hand commands: the ctrl values the controller sends to the hand's "gripper" and
# "squeezer" actuators. Every hand defines them in its XML, e.g.
#   <custom><numeric name="gripper_open" data="0"/> ...</custom>
# so the meaning (travel, force, ...) is up to the hand design.
HAND_COMMANDS = ("gripper_open", "gripper_close", "squeeze_release", "squeeze_press")


class ViewerClosed(Exception):
    pass


def min_jerk(s: float) -> float:
    s = min(max(s, 0.0), 1.0)
    return s * s * s * (10.0 - 15.0 * s + 6.0 * s * s)


def mat_to_quat(mat: np.ndarray) -> np.ndarray:
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(mat, dtype=float).ravel())
    return q


def slerp(q0: np.ndarray, q1: np.ndarray, s: float) -> np.ndarray:
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1, dot = -q1, -dot
    if dot > 0.9995:
        q = q0 + s * (q1 - q0)
        return q / np.linalg.norm(q)
    theta = np.arccos(dot)
    return (np.sin((1.0 - s) * theta) * q0 + np.sin(s * theta) * q1) / np.sin(theta)


def pose_error(pos: np.ndarray, quat: np.ndarray, cur_pos: np.ndarray, cur_mat: np.ndarray) -> np.ndarray:
    """6D error [dp, dw] in the world frame, from current pose to target pose."""
    cur_quat = mat_to_quat(cur_mat)
    neg, dq, rot = np.zeros(4), np.zeros(4), np.zeros(3)
    mujoco.mju_negQuat(neg, cur_quat)
    mujoco.mju_mulQuat(dq, quat, neg)
    mujoco.mju_quat2Vel(rot, dq, 1.0)
    return np.concatenate([pos - cur_pos, rot])


class Sim:
    """Steps physics and (optionally) keeps a passive viewer in sync."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, viewer=None, realtime: bool = True,
                 playback: float = 1.0):
        """playback: viewer speed relative to real time (0.5 = half speed). Only affects
        how fast it is shown, never the physics."""
        self.model, self.data = model, data
        self.viewer = viewer
        self.realtime = realtime and viewer is not None
        self.playback = playback
        self.callbacks = []  # called as f(model, data) after every step

    def step(self) -> None:
        start = time.perf_counter()
        mujoco.mj_step(self.model, self.data)
        for cb in self.callbacks:
            cb(self.model, self.data)
        if self.viewer is not None:
            if not self.viewer.is_running():
                raise ViewerClosed
            self.viewer.sync()
            if self.realtime:
                remaining = self.model.opt.timestep / self.playback - (time.perf_counter() - start)
                if remaining > 0:
                    time.sleep(remaining)


class RB5:
    """Controls the arm through the hand's "grasp" site, plus the gripper and bulb squeezer."""

    def __init__(self, sim: Sim, site: str = "grasp", time_scale: float = 1.0, kinematics=None):
        """time_scale: multiplies every move/grasp/squeeze duration (2 = robot twice as slow).
        kinematics: module with forward_kinematics/inverse_kinematics (default: student_kinematics.py)."""
        self.sim = sim
        self.time_scale = time_scale
        self.kin = load_kinematics() if kinematics is None else kinematics
        m = sim.model
        self.cmd = {name: float(m.numeric(name).data[0]) for name in HAND_COMMANDS}
        self.site_id = m.site(site).id
        self.flange_id = m.site("tcp").id
        self.qadr = np.array([m.joint(j).qposadr[0] for j in ARM_JOINTS])
        self.dofadr = np.array([m.joint(j).dofadr[0] for j in ARM_JOINTS])
        self.arm_act = np.array([m.actuator(j).id for j in ARM_JOINTS])
        self.gripper_act = m.actuator("gripper").id
        self.squeezer_act = m.actuator("squeezer").id
        self.q_lo = m.jnt_range[[m.joint(j).id for j in ARM_JOINTS], 0]
        self.q_hi = m.jnt_range[[m.joint(j).id for j in ARM_JOINTS], 1]
        self._fk_data = mujoco.MjData(m)  # scratch data so sim_fk never touches the simulation
        self.q_cmd = sim.data.qpos[self.qadr].copy()
        # The hand's grasp site in the flange frame: the `tool` passed to the kinematics.
        flange, grasp = self.sim_fk(self.q_cmd, site="tcp"), self.sim_fk(self.q_cmd)
        self.tool = np.linalg.inv(pose_matrix(*flange)) @ pose_matrix(*grasp)

    # ---- state -----------------------------------------------------------------------

    def reset(self, q: np.ndarray = HOME_Q) -> None:
        d = self.sim.data
        d.qpos[self.qadr] = q
        d.qvel[self.dofadr] = 0.0
        self.q_cmd = np.array(q, dtype=float)
        d.ctrl[self.arm_act] = self.q_cmd
        d.ctrl[self.gripper_act] = self.cmd["gripper_open"]
        d.ctrl[self.squeezer_act] = self.cmd["squeeze_release"]
        mujoco.mj_forward(self.sim.model, d)

    def grasp_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Measured pose of the grasp site: (position, 3x3 rotation)."""
        d = self.sim.data
        return d.site_xpos[self.site_id].copy(), d.site_xmat[self.site_id].reshape(3, 3).copy()

    # ---- kinematics ------------------------------------------------------------------

    def fk(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Grasp site pose at joint angles q from the kinematics module: (position, 3x3 rotation)."""
        T = np.asarray(self.kin.forward_kinematics(np.array(q, dtype=float), self.tool), dtype=float)
        return T[:3, 3].copy(), T[:3, :3].copy()

    def ik(self, pos, rot, q_init=None, max_iters: int = 200, tol: float = 1e-6) -> tuple[np.ndarray, float]:
        """Joint angles for a grasp site pose from the kinematics module: (q, final error norm)."""
        q_init = self.q_cmd if q_init is None else q_init
        q, err = self.kin.inverse_kinematics(pose_matrix(pos, rot), np.array(q_init, dtype=float), self.tool.copy(),
                                             max_iters=max_iters, tol=tol)
        return np.asarray(q, dtype=float), float(err)

    def sim_fk(self, q: np.ndarray, site: str | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Ground truth from the simulator: pose of the grasp site (or another site) at q."""
        m, d = self.sim.model, self._fk_data
        sid = self.site_id if site is None else m.site(site).id
        d.qpos[self.qadr] = q
        mujoco.mj_kinematics(m, d)
        return d.site_xpos[sid].copy(), d.site_xmat[sid].reshape(3, 3).copy()

    # ---- motion ----------------------------------------------------------------------

    def _run(self, duration: float, update) -> None:
        dt = self.sim.model.opt.timestep
        n = max(1, int(round(duration * self.time_scale / dt)))
        for i in range(n):
            update((i + 1) / n)
            self.sim.step()

    def _command_arm(self, q: np.ndarray) -> None:
        self.q_cmd = q
        self.sim.data.ctrl[self.arm_act] = q

    def move_joints(self, q_target, duration: float, settle: float = 0.2) -> None:
        q0, q1 = self.q_cmd.copy(), np.asarray(q_target, dtype=float)
        self._run(duration, lambda s: self._command_arm(q0 + min_jerk(s) * (q1 - q0)))
        self.wait(settle)

    def move_to(self, pos, rot=HAND_FORWARD, duration: float = 2.0, settle: float = 0.2,
                joint_space: bool = False) -> None:
        """Move the grasp site to (pos, rot).

        joint_space=False: straight line in Cartesian space (use near objects).
        joint_space=True: solve IK once and interpolate joints (use for big moves).
        """
        pos, rot = np.asarray(pos, dtype=float), np.asarray(rot, dtype=float)
        if joint_space:
            q, err = self.ik(pos, rot)
            if err > 1e-4:
                raise RuntimeError(f"IK failed for target {pos} (error {err:.2e})")
            self.move_joints(q, duration, settle)
            return

        p0, r0 = self.fk(self.q_cmd)
        q0, q1 = mat_to_quat(r0), mat_to_quat(rot)

        def update(s: float) -> None:
            a = min_jerk(s)
            r = np.zeros(9)
            mujoco.mju_quat2Mat(r, slerp(q0, q1, a))
            q, _ = self.ik(p0 + a * (pos - p0), r.reshape(3, 3), self.q_cmd, max_iters=10)
            self._command_arm(q)

        self._run(duration, update)
        self.wait(settle)

    def move_by(self, delta, duration: float = 1.5, settle: float = 0.2) -> None:
        """Straight-line relative move of the grasp site, keeping orientation."""
        pos, rot = self.fk(self.q_cmd)
        self.move_to(pos + np.asarray(delta, dtype=float), rot, duration, settle)

    def wait(self, duration: float) -> None:
        self._run(duration, lambda s: None)

    # ---- hand ------------------------------------------------------------------------

    def open_gripper(self, duration: float = 0.6) -> None:
        self.sim.data.ctrl[self.gripper_act] = self.cmd["gripper_open"]
        self.wait(duration)

    def close_gripper(self, duration: float = 0.8) -> None:
        self.sim.data.ctrl[self.gripper_act] = self.cmd["gripper_close"]
        self.wait(duration)

    def squeeze_bulb(self, duration: float = 0.6) -> None:
        self.sim.data.ctrl[self.squeezer_act] = self.cmd["squeeze_press"]
        self.wait(duration)

    def release_bulb(self, duration: float = 0.6) -> None:
        self.sim.data.ctrl[self.squeezer_act] = self.cmd["squeeze_release"]
        self.wait(duration)
