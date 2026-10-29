"""Read task layouts (object positions) from a Markdown file.

Each Markdown table is one layout, named after the closest heading above it.
Rows name the objects; columns give x and y in the robot base frame:

    ## Practice 1

    | object         | x (mm) | y (mm) |
    |----------------|-------:|-------:|
    | pipette holder |    500 |   -250 |
    | water box      |    550 |      0 |
    | dispense box   |    500 |    250 |

Units come from the column header: "(mm)" or "(m)"; plain "x"/"y" means mm.
Object names are matched by keyword: "holder", "water", and one of
"dispense", "beaker", "extra", "target" for the box that receives the water.
Text outside tables is ignored.
"""

import re
from pathlib import Path

from .env import Layout

_OBJECTS = {
    "holder": ("holder",),
    "water": ("water",),
    "beaker": ("dispense", "beaker", "extra", "target"),
}
_LABELS = {"holder": "pipette holder", "water": "water box", "beaker": "dispense box"}


def _cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _is_separator(cells: list[str]) -> bool:
    return all(re.fullmatch(r":?-{3,}:?", c) for c in cells if c)


def _object_key(name: str) -> str | None:
    name = name.lower()
    for key, words in _OBJECTS.items():
        if any(w in name for w in words):
            return key
    return None


def _parse_table(title: str, rows: list[list[str]], where: str) -> Layout:
    header = [h.lower() for h in rows[0]]
    cols, scale = {}, {}
    for i, h in enumerate(header):
        m = re.match(r"^\s*([xy])\b\s*(?:\((mm|m)\))?", h)
        if m:
            cols[m.group(1)] = i
            scale[m.group(1)] = 1.0 if m.group(2) == "m" else 1e-3
    if set(cols) != {"x", "y"}:
        raise ValueError(f"{where}: table '{title}' needs x and y columns, got {rows[0]}")

    found: dict[str, tuple[float, float]] = {}
    for r in rows[1:]:
        if _is_separator(r):
            continue
        key = _object_key(r[0])
        if key is None:
            raise ValueError(f"{where}: unknown object '{r[0]}' in '{title}' "
                             f"(use holder / water / dispense box)")
        try:
            found[key] = (float(r[cols["x"]]) * scale["x"], float(r[cols["y"]]) * scale["y"])
        except (ValueError, IndexError) as e:
            raise ValueError(f"{where}: bad coordinates for '{r[0]}' in '{title}': {r}") from e

    missing = [label for key, label in _LABELS.items() if key not in found]
    if missing:
        raise ValueError(f"{where}: layout '{title}' is missing: {', '.join(missing)}")
    return Layout(**found)


def load_layouts(path: str | Path) -> dict[str, Layout]:
    """Return {layout name: Layout} in file order."""
    path = Path(path)
    layouts: dict[str, Layout] = {}
    title, table = path.stem, []

    def flush() -> None:
        if table:
            name = title if title not in layouts else f"{title} ({len(layouts) + 1})"
            layouts[name] = _parse_table(name, table, str(path))
            table.clear()

    for line in path.read_text(encoding="utf-8").splitlines():
        if line.lstrip().startswith("|"):
            table.append(_cells(line))
            continue
        flush()
        heading = re.match(r"^\s*#+\s+(.*\S)", line)
        if heading:
            title = heading.group(1)
    flush()

    if not layouts:
        raise ValueError(f"{path}: no layout tables found")
    return layouts
