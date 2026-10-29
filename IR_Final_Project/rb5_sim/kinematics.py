"""Load the kinematics module (student_kinematics.py) and read/write trajectory files.

Trajectory files are plain text, one waypoint per line, as `key:value` pairs separated
by spaces. `#` starts a comment. T is the time [s] at which the robot reaches the
waypoint; the first line is where the robot starts.

Joint file (Part 1), angles in degrees:

    T:0 theta1:31.3 theta2:-28.2 theta3:130.4 theta4:77.8 theta5:-58.7 theta6:90
    T:2 theta1:40   theta2:-20   theta3:120   theta4:80   theta5:-60   theta6:90

Pose file (Part 2), the hand's grasp point in mm and its orientation as ZYZ Euler
angles in degrees, R = Rz(alpha) Ry(beta) Rz(gamma):

    T:0 x:400 y:0   z:350 alpha:0 beta:90 gamma:180
    T:2 x:500 y:100 z:300

alpha/beta/gamma may be left out on a line: the default (0, 90, 180) is the task
orientation, hand pointing forward (+x) with the pipette axis vertical.
"""

import ast
import importlib.util
import re
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import numpy as np

DEFAULT_KINEMATICS = Path(__file__).resolve().parent.parent / "student_kinematics.py"
DEFAULT_MUJOCO = Path(__file__).resolve().parent.parent / "student_mujoco.py"
FORBIDDEN_IMPORTS = ("mujoco", "rb5_sim")   # student_kinematics.py: NumPy only
FORBIDDEN_IMPORTS_MUJOCO = ("rb5_sim",)     # student_mujoco.py: MuJoCo and NumPy
JOINT_KEYS = tuple(f"theta{i}" for i in range(1, 7))
POSITION_KEYS = ("x", "y", "z")
EULER_KEYS = ("alpha", "beta", "gamma")
DEFAULT_EULER_DEG = (0.0, 90.0, 180.0)


def load_kinematics(path: str | Path = DEFAULT_KINEMATICS) -> ModuleType:
    """Import a kinematics file (default: student_kinematics.py) as a module."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"kinematics file not found: {path}")
    spec = importlib.util.spec_from_file_location(f"kinematics_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_student_file(path: str | Path, mujoco_allowed: bool = False) -> ModuleType:
    """Check a student file's imports against the rules, then load it."""
    forbidden = FORBIDDEN_IMPORTS_MUJOCO if mujoco_allowed else FORBIDDEN_IMPORTS
    bad = forbidden_imports(path, forbidden)
    if bad:
        allowed = "mujoco and NumPy are" if mujoco_allowed else "only NumPy is"
        raise SystemExit(f"{path} imports {', '.join(bad)}: {allowed} allowed (see PART1_FK.md)")
    return load_kinematics(path)


def forbidden_imports(path: str | Path = DEFAULT_KINEMATICS, forbidden=FORBIDDEN_IMPORTS) -> list[str]:
    """Names of modules imported by the file that the rules don't allow."""
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
        elif isinstance(node, ast.Call) and getattr(node.func, "id", "") == "__import__":
            names.append("__import__(...)")
    return sorted({n for n in names if n.split(".")[0] in forbidden or n.startswith("__import__")})


# ---- rotations --------------------------------------------------------------------

def _rz(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _ry(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def euler_zyz_to_matrix(alpha: float, beta: float, gamma: float) -> np.ndarray:
    """R = Rz(alpha) Ry(beta) Rz(gamma), angles in radians."""
    return _rz(alpha) @ _ry(beta) @ _rz(gamma)


def matrix_to_euler_zyz(R: np.ndarray) -> tuple[float, float, float]:
    """Inverse of euler_zyz_to_matrix, with beta in [0, pi]."""
    beta = float(np.arctan2(np.hypot(R[0, 2], R[1, 2]), R[2, 2]))
    if np.sin(beta) < 1e-9:  # beta = 0 or pi: only alpha +- gamma is defined; use gamma = 0
        if R[2, 2] > 0:
            return float(np.arctan2(R[1, 0], R[0, 0])), beta, 0.0
        return float(np.arctan2(-R[0, 1], -R[0, 0])), beta, 0.0
    alpha = float(np.arctan2(R[1, 2], R[0, 2]))
    gamma = float(np.arctan2(R[2, 1], -R[2, 0]))
    return alpha, beta, gamma


def pose_matrix(pos, rot) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = rot, pos
    return T


# ---- trajectory files ---------------------------------------------------------------

@dataclass
class Waypoint:
    t: float
    value: np.ndarray   # joint file: (6,) angles [rad]; pose file: (4, 4) pose [m]
    line: int


def _parse_line(text: str, where: str) -> dict[str, float]:
    fields = {}
    for token in re.split(r"[\s,]+", text.strip()):
        if not token:
            continue
        key, sep, value = token.partition(":")
        if not sep:
            raise ValueError(f"{where}: expected key:value, got '{token}'")
        try:
            fields[key.strip().lower()] = float(value)
        except ValueError:
            raise ValueError(f"{where}: '{value}' is not a number (in '{token}')") from None
    return fields


def _read(path: str | Path, required: tuple[str, ...], optional: tuple[str, ...]) -> list[tuple[dict, int]]:
    path = Path(path)
    rows, known = [], {"t", *required, *optional}
    for n, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        text = raw.split("#", 1)[0]
        if not text.strip():
            continue
        where = f"{path}:{n}"
        fields = _parse_line(text, where)
        missing = [k for k in ("t", *required) if k not in fields]
        if missing:
            raise ValueError(f"{where}: missing {', '.join(k.upper() if k == 't' else k for k in missing)}")
        unknown = [k for k in fields if k not in known]
        if unknown:
            raise ValueError(f"{where}: unknown key(s) {', '.join(unknown)}")
        if rows and fields["t"] <= rows[-1][0]["t"]:
            raise ValueError(f"{where}: T must increase from line to line")
        rows.append((fields, n))
    if not rows:
        raise ValueError(f"{path}: no waypoints")
    return rows


def read_joint_file(path: str | Path) -> list[Waypoint]:
    rows = _read(path, JOINT_KEYS, ())
    return [Waypoint(f["t"], np.radians([f[k] for k in JOINT_KEYS]), n) for f, n in rows]


def read_pose_file(path: str | Path) -> list[Waypoint]:
    rows = _read(path, POSITION_KEYS, EULER_KEYS)
    out = []
    for f, n in rows:
        given = [k for k in EULER_KEYS if k in f]
        if given and len(given) != 3:
            raise ValueError(f"{path}:{n}: give all of alpha, beta, gamma or none")
        euler = [f[k] for k in EULER_KEYS] if given else DEFAULT_EULER_DEG
        pos = np.array([f[k] for k in POSITION_KEYS]) * 1e-3
        out.append(Waypoint(f["t"], pose_matrix(pos, euler_zyz_to_matrix(*np.radians(euler))), n))
    return out


def format_pose_line(t: float, T: np.ndarray) -> str:
    x, y, z = T[:3, 3] * 1e3
    a, b, g = np.degrees(matrix_to_euler_zyz(T[:3, :3]))
    return f"T:{t:g} x:{x:.2f} y:{y:.2f} z:{z:.2f} alpha:{a:.3f} beta:{b:.3f} gamma:{g:.3f}"


def format_joint_line(t: float, q: np.ndarray) -> str:
    return f"T:{t:g} " + " ".join(f"{k}:{a:.3f}" for k, a in zip(JOINT_KEYS, np.degrees(q)))


def write_pose_file(path: str | Path, rows: list[tuple[float, np.ndarray]], header: str = "") -> None:
    _write(path, [format_pose_line(t, T) for t, T in rows], header)


def write_joint_file(path: str | Path, rows: list[tuple[float, np.ndarray]], header: str = "") -> None:
    _write(path, [format_joint_line(t, q) for t, q in rows], header)


def _write(path: str | Path, lines: list[str], header: str) -> None:
    lines = [f"# {h}" for h in header.splitlines()] + lines
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
