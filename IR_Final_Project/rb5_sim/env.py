"""Build the pipetting task model: scene + RB5 + hand, with the holder pose error applied."""

import warnings
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

ASSETS = Path(__file__).resolve().parent.parent / "assets"

# Geometry shared with assets/pipetting/task_scene.xml (meters).
HOLDER_SEAT_Z = 0.02      # pipette tip height in the holder frame
WATER_INNER_HALF = 0.06   # water source inner half-width
WATER_BOTTOM_Z = 0.005
WATER_SURFACE_Z = 0.05
BEAKER_INNER_HALF = 0.04
BEAKER_BOTTOM_Z = 0.005
BEAKER_RIM_Z = 0.10


@dataclass(frozen=True)
class Layout:
    """Nominal object positions (x, y) on the bench in the robot base frame [m]."""

    holder: tuple[float, float]
    water: tuple[float, float]
    beaker: tuple[float, float]  # the box the water is dispensed into

    def __str__(self) -> str:
        fmt = lambda p: f"({p[0] * 1e3:.0f}, {p[1] * 1e3:.0f})"  # noqa: E731
        return f"holder {fmt(self.holder)}, water {fmt(self.water)}, dispense {fmt(self.beaker)} mm"


@dataclass(frozen=True)
class HolderError:
    """Holder pose error that the robot is NOT told about."""

    dx: float = 0.0         # m
    dy: float = 0.0         # m
    tilt_deg: float = 0.0   # tilt magnitude
    tilt_dir_deg: float = 0.0  # direction the holder leans toward, measured from +x

    @staticmethod
    def sample(rng: np.random.Generator, max_xy: float = 0.005, max_tilt_deg: float = 2.0) -> "HolderError":
        return HolderError(
            dx=rng.uniform(-max_xy, max_xy),
            dy=rng.uniform(-max_xy, max_xy),
            tilt_deg=rng.uniform(0.0, max_tilt_deg),
            tilt_dir_deg=rng.uniform(0.0, 360.0),
        )

    def quat(self) -> np.ndarray:
        # Leaning toward direction d means rotating about the horizontal axis z x d.
        d = np.deg2rad(self.tilt_dir_deg)
        axis = np.array([-np.sin(d), np.cos(d), 0.0])
        q = np.zeros(4)
        mujoco.mju_axisAngle2Quat(q, axis, np.deg2rad(self.tilt_deg))
        return q

    def __str__(self) -> str:
        return (f"dx={self.dx * 1e3:+.1f} mm, dy={self.dy * 1e3:+.1f} mm, "
                f"tilt={self.tilt_deg:.2f} deg toward {self.tilt_dir_deg:.0f} deg")


@dataclass(frozen=True)
class PipetteSpec:
    """Pipette thickness. The defaults are the pipette in task_scene.xml."""

    shaft_radius: float = 0.004   # m
    bulb_radius: float = 0.009    # m

    # Ranges used by sample(), in meters.
    SHAFT_RANGE = (0.003, 0.005)
    BULB_RANGE = (0.008, 0.011)

    @staticmethod
    def sample(rng: np.random.Generator, shaft_range=SHAFT_RANGE, bulb_range=BULB_RANGE) -> "PipetteSpec":
        return PipetteSpec(shaft_radius=rng.uniform(*shaft_range), bulb_radius=rng.uniform(*bulb_range))

    def __str__(self) -> str:
        return f"shaft {self.shaft_radius * 2e3:.1f} mm dia, bulb {self.bulb_radius * 2e3:.1f} mm dia"


SOCKET_CLEARANCE = 0.001   # holder socket half-width = shaft radius + this
SOCKET_CHAMFER = 0.001     # 45 deg lead-in at the top of the socket (0 = sharp edge)
HOLDER_TOP_Z = 0.06        # top face of the holder block
# Glass shaft on a plastic holder. MuJoCo uses the larger of the two geoms' friction, so
# the shaft is 0.3 too. Must stay below tan(45 deg) = 1 or the tip wedges in the chamfer.
HOLDER_FRICTION = 0.3
_HOLDER_HALF = 0.025       # holder block half-width
_BULB_WALL_RADIUS = 0.004  # capsule radius of each bulb wall segment (task_scene.xml)


def _apply_pipette(spec: mujoco.MjSpec, pipette: PipetteSpec) -> None:
    r = pipette.shaft_radius
    shaft = spec.geom("shaft")
    shaft.size[0] = r
    shaft.fromto = [0, 0, r, 0, 0, 0.16]  # keep the tip at the body origin

    spec.geom("bulb_core").size[0] = pipette.bulb_radius - 0.003
    d = pipette.bulb_radius - _BULB_WALL_RADIUS
    for name, axis in (("px", [d, 0]), ("nx", [-d, 0]), ("py", [0, d]), ("ny", [0, -d])):
        spec.body(f"bulb_wall_{name}").pos = [*axis, 0.18]

    _add_socket(spec.body("holder"), r + SOCKET_CLEARANCE)


def _add_socket(holder: mujoco.MjsBody, w: float) -> None:
    """Socket walls around a square hole of half-width w, from the base top (z = 0.02)
    to the block top (z = 0.06), with a SOCKET_CHAMFER x 45 deg lead-in at the top."""
    H, c, t = _HOLDER_HALF, SOCKET_CHAMFER, 0.002
    z0, z1 = HOLDER_SEAT_Z, HOLDER_TOP_Z
    zc = z1 - c  # where the chamfer starts
    rgba = [0.78, 0.74, 0.66, 1]
    friction = [HOLDER_FRICTION, 0.02, 0.001]

    def box(name, lo, hi, quat=(1, 0, 0, 0)):
        lo, hi = np.array(lo), np.array(hi)
        holder.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX, size=(hi - lo) / 2,
                        pos=(hi + lo) / 2, quat=quat, rgba=rgba, friction=friction)

    for sign, side in ((1, "p"), (-1, "n")):
        # walls normal to x span the full block in y; walls normal to y fit between them
        for axis in ("x", "y"):
            e = np.array([1.0, 0, 0]) if axis == "x" else np.array([0, 1.0, 0])
            a = np.array([0, 1.0, 0]) if axis == "x" else np.array([1.0, 0, 0])
            span_low = H if axis == "x" else w
            span_top = H if axis == "x" else w + c
            name = f"socket_{side}{axis}"

            def slab(n, d_in, d_out, span, za, zb):
                corners = [sign * d_in * e + span * a, sign * d_out * e - span * a]
                lo = np.minimum(*corners)
                hi = np.maximum(*corners)
                box(n, [lo[0], lo[1], za], [hi[0], hi[1], zb])

            slab(name, w, H, span_low, z0, zc)                 # lower wall
            if c > 0:
                slab(name + "_rim", w + c, H, span_top, zc, z1)  # top rim outside the chamfer
                # 45 deg chamfer plate
                out = sign * e
                u = (out + [0, 0, 1]) / np.sqrt(2)             # up the slope
                n = (-out + [0, 0, 1]) / np.sqrt(2)            # surface normal (into the hole)
                rot = np.column_stack([u, np.cross(n, u), n])
                q = np.zeros(4)
                mujoco.mju_mat2Quat(q, rot.ravel())
                mid = out * (w + c / 2) + [0, 0, zc + c / 2] - n * t / 2
                holder.add_geom(name=name + "_chamfer", type=mujoco.mjtGeom.mjGEOM_BOX,
                                size=[c * np.sqrt(2) / 2, span_top, t / 2], pos=mid, quat=q, rgba=rgba,
                                friction=friction)


DEFAULT_HAND = ASSETS.parent / "hands" / "basic" / "hand.xml"

# What every hand must provide (see PART3_HAND.md).
HAND_SITES = ("grasp",)
HAND_ACTUATORS = ("gripper", "squeezer")
HAND_NUMERICS = ("gripper_open", "gripper_close", "squeeze_release", "squeeze_press")


def load_hand(path: str | Path = DEFAULT_HAND) -> mujoco.MjSpec:
    """Load a hand MJCF and check it provides the interface the controller uses."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"hand file not found: {path}")
    hand = mujoco.MjSpec.from_file(str(path))
    names = lambda items: {x.name for x in items}  # noqa: E731
    missing = ([f"site '{n}'" for n in HAND_SITES if n not in names(hand.sites)]
               + [f"actuator '{n}'" for n in HAND_ACTUATORS if n not in names(hand.actuators)]
               + [f"numeric '{n}'" for n in HAND_NUMERICS if n not in names(hand.numerics)])
    if missing:
        raise ValueError(f"{path} is missing: {', '.join(missing)} (see PART3_HAND.md)")
    if len(hand.worldbody.bodies) != 1:
        raise ValueError(f"{path}: the worldbody must contain exactly one (root) body, the hand")
    return hand


_EMPTY_PREFIX_WARNING = "Attaching a spec with an empty prefix"


def build_model(layout: Layout, error: HolderError = HolderError(),
                pipette: PipetteSpec = PipetteSpec(), hand: str | Path = DEFAULT_HAND) -> mujoco.MjModel:
    # Robot and hand are attached without name prefixes (so names stay "grasp", "base", ...);
    # MuJoCo warns about that because the merged model can't be saved back to XML.
    previous = mujoco.get_mju_user_warning()
    mujoco.set_mju_user_warning(lambda msg: None if _EMPTY_PREFIX_WARNING in msg else print("MuJoCo:", msg))
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=_EMPTY_PREFIX_WARNING)
            return _build(layout, error, pipette, load_hand(hand))
    finally:
        mujoco.set_mju_user_warning(previous)


def _build(layout: Layout, error: HolderError, pipette: PipetteSpec, hand: mujoco.MjSpec) -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_file(str(ASSETS / "pipetting" / "task_scene.xml"))
    robot = mujoco.MjSpec.from_file(str(ASSETS / "rb5_850e" / "rb5_850e.xml"))

    # The robot's keyframes only cover the arm joints; they don't fit the full scene.
    for key in list(robot.keys):
        robot.delete(key)
    robot.attach(hand, site=robot.site("tcp"), prefix="")
    spec.attach(robot, frame=spec.worldbody.add_frame(), prefix="")
    _apply_pipette(spec, pipette)

    # Holder: nominal pose + error. The pipette starts seated in it.
    holder_pos = np.array([layout.holder[0] + error.dx, layout.holder[1] + error.dy, 0.0])
    holder_quat = error.quat()
    spec.body("holder").pos = holder_pos
    spec.body("holder").quat = holder_quat

    seat = np.zeros(3)
    mujoco.mju_rotVecQuat(seat, np.array([0.0, 0.0, HOLDER_SEAT_Z + 0.0005]), holder_quat)
    spec.body("pipette").pos = holder_pos + seat
    spec.body("pipette").quat = holder_quat

    spec.body("water_source").pos = [*layout.water, 0.0]
    spec.body("beaker").pos = [*layout.beaker, 0.0]

    return spec.compile()
