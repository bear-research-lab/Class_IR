"""Parts 1 and 2: forward and inverse kinematics of the RB5 (your code goes here).

Part 1: forward_kinematics()    test: uv run part1_fk.py
Part 2: inverse_kinematics()    test: uv run part2_ik.py

The robot controller calls these two functions for every move, so the pipetting task
(Part 3) runs on your kinematics.

Rules: use NumPy only. Don't import mujoco or rb5_sim in this file. You may add your own
helper functions. Don't change the function signatures or the provided helpers.

Frames and units: meters and radians. Poses are 4x4 homogeneous transforms.
  base frame    the robot base (= the world frame of the simulation)
  flange frame  the RB5 tool flange: z out of the flange (site "tcp" in rb5_850e.xml)
  tool frame    a frame fixed to the flange, given by `tool` = its pose in the flange
                frame. The robot passes the hand's "grasp" site here.
"""

import numpy as np

# Joint limits [rad], the same for all 6 joints.
Q_LO = np.full(6, -6.2832)
Q_HI = np.full(6, 6.2832)


# ---- provided helpers ----------------------------------------------------------------

def rotation_vector(R: np.ndarray) -> np.ndarray:
    """Rotation vector (axis * angle [rad]) of the rotation matrix R."""
    cos = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    angle = np.arccos(cos)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if angle < 1e-9:
        return 0.5 * v
    if angle > np.pi - 1e-6:  # sin(angle) ~ 0: take the axis from R + I instead
        B = (R + np.eye(3)) / 2.0
        k = int(np.argmax(np.diag(B)))
        axis = B[:, k] / np.sqrt(B[k, k])
        return angle * axis
    return angle / (2.0 * np.sin(angle)) * v


def pose_error(target: np.ndarray, current: np.ndarray) -> np.ndarray:
    """6D error from the current pose to the target pose (both 4x4), in the base frame.

    [0:3] position difference [m]; [3:6] rotation vector [rad] that turns the current
    orientation into the target orientation.
    """
    dp = target[:3, 3] - current[:3, 3]
    dw = rotation_vector(target[:3, :3] @ current[:3, :3].T)
    return np.concatenate([dp, dw])


# ---- Part 1: forward kinematics ------------------------------------------------------

def forward_kinematics(q: np.ndarray, tool: np.ndarray = np.eye(4)) -> np.ndarray:
    """Pose of the tool frame in the base frame.

    q:    (6,) joint angles [rad]: base, shoulder, elbow, wrist1, wrist2, wrist3
    tool: (4, 4) pose of the tool frame in the flange frame (identity = the flange itself)

    Returns the (4, 4) homogeneous transform T_base_tool.
    """
    # TODO: build T_base_flange from the joint angles and the link geometry of the RB5
    # (see PART1_FK.md), then return T_base_flange @ tool.
    raise NotImplementedError("Part 1: implement forward_kinematics() in student_kinematics.py")


# ---- Part 2: inverse kinematics ------------------------------------------------------

def inverse_kinematics(target: np.ndarray, q_init: np.ndarray, tool: np.ndarray = np.eye(4),
                       max_iters: int = 200, tol: float = 1e-6) -> tuple[np.ndarray, float]:
    """Joint angles that put the tool frame at the target pose.

    target:    (4, 4) target pose of the tool frame in the base frame
    q_init:    (6,) starting joint angles; start here and stay on the same IK branch
    tool:      (4, 4) pose of the tool frame in the flange frame
    max_iters: maximum number of iterations
    tol:       stop when the norm of pose_error(target, current) is below this

    Returns (q, err): the (6,) solution inside [Q_LO, Q_HI] and the norm of the 6D pose
    error at q.
    """
    q = np.array(q_init, dtype=float)

    # TODO: implement an iterative IK solver, e.g. damped least squares:
    #   1. current pose: forward_kinematics(q, tool)
    #   2. 6D error: pose_error(target, current); stop if its norm < tol
    #   3. the 6x6 Jacobian J of the tool frame at q (write your own function for it)
    #   4. a joint step dq from J and the error
    #   5. update q (limit the step size, keep q inside [Q_LO, Q_HI])
    raise NotImplementedError("Part 2: implement inverse_kinematics() in student_kinematics.py")
