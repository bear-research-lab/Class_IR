"""Parts 1 and 2: the same kinematics with MuJoCo's functions (your code goes here).

In student_kinematics.py you derive the forward kinematics and the Jacobian yourself.
Here you get the same quantities from MuJoCo, the way you would in practice, and the
test scripts check both against each other:

Part 1: mujoco_fk()          test: uv run part1_fk.py
Part 2: mujoco_jacobian()    test: uv run part2_ik.py

Rules: use mujoco and NumPy. Don't import rb5_sim. Don't change the function signatures.

The scripts pass you the simulation model and a scratch MjData of your own, so changing
its qpos never moves the simulated robot.

MuJoCo functions and fields you will need (https://mujoco.readthedocs.io):
  model.joint(name).qposadr[0]   index of a joint's angle in data.qpos
  model.joint(name).dofadr[0]    index of a joint's velocity (= Jacobian column)
  model.site(name).id            index of a site
  mujoco.mj_kinematics(m, d)     positions and orientations of all bodies and sites
                                 from data.qpos
  data.site_xpos[i]              site position (3,) in the world (= base) frame
  data.site_xmat[i]              site rotation (9,), row by row
  mujoco.mj_comPos(m, d)         must run (after mj_kinematics) before mj_jacSite
  mujoco.mj_jacSite(m, d, jacp, jacr, i)   linear and angular Jacobians of site i,
                                 each 3 x model.nv
The model holds more than the arm (fingers, pipette, ...), so the arm joints are not the
first entries of qpos and the Jacobian has more than 6 columns.
"""

import mujoco
import numpy as np

# The arm joints in rb5_850e.xml, in the order of q.
ARM_JOINTS = ("base", "shoulder", "elbow", "wrist1", "wrist2", "wrist3")


# ---- Part 1 --------------------------------------------------------------------------

def mujoco_fk(model: mujoco.MjModel, data: mujoco.MjData, q: np.ndarray, site: str = "tcp") -> np.ndarray:
    """Pose of a site at joint angles q, from MuJoCo.

    q:    (6,) joint angles [rad], in the order of ARM_JOINTS
    site: "tcp" is the flange, "grasp" is the hand's grasp site

    Returns the (4, 4) homogeneous transform of the site in the base frame.
    """
    # TODO: 1. write q into data.qpos at the arm joints' addresses
    #       2. run MuJoCo's forward kinematics
    #       3. build the 4x4 matrix from the site's position and rotation
    raise NotImplementedError("Part 1: implement mujoco_fk() in student_mujoco.py")


# ---- Part 2 --------------------------------------------------------------------------

def mujoco_jacobian(model: mujoco.MjModel, data: mujoco.MjData, q: np.ndarray, site: str = "grasp") -> np.ndarray:
    """Geometric Jacobian of a site at joint angles q, from MuJoCo.

    Returns the (6, 6) Jacobian: rows 0-2 linear velocity, rows 3-5 angular velocity,
    both in the base frame; column i is joint ARM_JOINTS[i].
    """
    # TODO: 1. write q into data.qpos and run mj_kinematics, then mj_comPos
    #       2. get the linear and angular Jacobians of the site with mj_jacSite
    #       3. stack them and keep only the columns of the 6 arm joints
    raise NotImplementedError("Part 2: implement mujoco_jacobian() in student_mujoco.py")
