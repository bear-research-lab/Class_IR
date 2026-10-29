"""Live graphs of bulb squeeze force and pipette tip z displacement.

Two ways to show them, both used as Sim step callbacks:

    sim.callbacks.append(ViewerOverlay(viewer, model, forces))           # inside the viewer
    sim.callbacks.append(graph := GraphWindow(model, forces)); ...; graph.close()  # own window

ViewerOverlay also writes a text panel (phase, force, compression, tip z) in the
viewer's bottom-left corner; pass plots=False to show only the text.
"""

import multiprocessing as mp
import queue
from collections import deque

import mujoco
import numpy as np

from .forces import ForceMonitor


class _Sampler:
    """Samples (time, bulb force, tip dz) every `every` physics steps."""

    def __init__(self, model: mujoco.MjModel, forces: ForceMonitor, every: int):
        self.forces = forces
        self.tip_id = model.site("tip").id
        self.every = every
        self._count = 0
        self._z0 = None

    def sample(self, data: mujoco.MjData):
        self._count += 1
        if self._count % self.every:
            return None
        tip_z = float(data.site_xpos[self.tip_id][2])
        if self._z0 is None:
            self._z0 = tip_z
        return float(data.time), self.forces.bulb_force(data), (tip_z - self._z0) * 1e3, tip_z


# ---- graphs drawn inside the MuJoCo viewer --------------------------------------------

def _figure(title: str, yformat: str, rgb) -> mujoco.MjvFigure:
    fig = mujoco.MjvFigure()
    mujoco.mjv_defaultFigure(fig)
    fig.title = title
    fig.xformat = "%.0f"
    fig.yformat = yformat
    fig.linergb[0] = rgb
    fig.linewidth = 2.0
    fig.gridsize = [6, 3]
    fig.flg_extend = 0          # y range is set explicitly each update
    fig.figurergba[3] = 0.6     # semi-transparent background
    fig.panergba[3] = 0.6
    return fig


def _nice_range(lo: float, hi: float, step: float) -> list[float]:
    """Round [lo, hi] outward to multiples of step (min 2 steps tall)."""
    lo, hi = np.floor(lo / step) * step, np.ceil(hi / step) * step
    if hi - lo < 2 * step:
        hi = lo + 2 * step
    return [float(lo), float(hi)]


def _padded_max(ys: np.ndarray) -> float:
    return float(ys.max() * 1.1) if len(ys) else 0.0


class ViewerOverlay:
    def __init__(self, viewer, model: mujoco.MjModel, forces: ForceMonitor, plots: bool = True,
                 window: float = 10.0, every: int = 10):
        self.viewer = viewer
        self.forces = forces
        self.plots = plots
        self.window = window          # seconds of history shown
        self.sampler = _Sampler(model, forces, every)
        n = min(int(window / (model.opt.timestep * every)) + 1, 1000)
        self.t, self.f, self.z = deque(maxlen=n), deque(maxlen=n), deque(maxlen=n)
        self.fig_force = _figure("Bulb force per side [N]  (x: time [s])", "%.1f", [1.0, 0.55, 0.3])
        self.fig_z = _figure("Pipette tip dz [mm]  (x: time [s])", "%.0f", [0.3, 0.7, 1.0])

    def __call__(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        s = self.sampler.sample(data)
        if s is None:
            return
        time, force, dz, tip_z = s
        self.viewer.set_texts((
            mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
            "phase\nbulb force\nbulb compression\ntip z",
            f"{self.forces.phase}\n{force:.2f} N\n{self.forces.bulb_compression(data) * 1e3:.1f} mm\n"
            f"{tip_z * 1e3:.0f} mm ({dz:+.0f})",
        ))
        if not self.plots:
            return

        self.t.append(time)
        self.f.append(force)
        self.z.append(dz)
        t = np.fromiter(self.t, float)
        f, z = np.fromiter(self.f, float), np.fromiter(self.z, float)
        for fig, ys, yrange in ((self.fig_force, f, _nice_range(0.0, max(3.0, _padded_max(f)), 1.0)),
                                (self.fig_z, z, _nice_range(min(0.0, z.min()), max(0.0, z.max()), 50.0))):
            n = len(t)
            fig.linepnt[0] = n
            fig.linedata[0, 0:2 * n:2] = t
            fig.linedata[0, 1:2 * n:2] = ys
            fig.range[0] = [max(0.0, t[-1] - self.window), max(self.window, t[-1])]
            fig.range[1] = yrange

        vp = self.viewer.viewport
        if vp is None or vp.width == 0:
            return
        # Flush against the right edge, stacked at the bottom.
        w, h = vp.width * 2 // 5, vp.height * 3 // 10
        left = vp.left + vp.width - w
        self.viewer.set_figures([
            (mujoco.MjrRect(left, vp.bottom + h, w, h), self.fig_force),
            (mujoco.MjrRect(left, vp.bottom, w, h), self.fig_z),
        ])


# ---- graphs in a separate matplotlib window -------------------------------------------

def _graph_window_main(q: mp.Queue) -> None:
    """Runs in its own process: draws everything received on the queue."""
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    fig, (ax_f, ax_z) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
    fig.canvas.manager.set_window_title("RB5 pipetting - live graphs")
    (line_f,) = ax_f.plot([], [], color="tab:orange", lw=1.8)
    (line_z,) = ax_z.plot([], [], color="tab:blue", lw=1.8)
    ax_f.set_ylabel("bulb force per side [N]")
    ax_z.set_ylabel("tip z displacement [mm]")
    ax_z.set_xlabel("time [s]")
    for ax in (ax_f, ax_z):
        ax.grid(alpha=0.3)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    t, f, z = [], [], []
    state = {"phase": None}

    def update(_):
        changed = False
        while True:
            try:
                item = q.get_nowait()
            except queue.Empty:
                break
            time, force, dz, phase = item
            if phase != state["phase"]:
                state["phase"] = phase
                for ax in (ax_f, ax_z):
                    ax.axvline(time, color="0.6", ls="--", lw=0.8)
                ax_f.text(time, 1.01, f" {phase}", transform=ax_f.get_xaxis_transform(),
                          fontsize=8, rotation=20, ha="left", va="bottom")
            t.append(time)
            f.append(force)
            z.append(dz)
            changed = True
        if changed:
            line_f.set_data(t, f)
            line_z.set_data(t, z)
            for ax in (ax_f, ax_z):
                ax.relim()
                ax.autoscale_view()
            ax_f.set_ylim(bottom=0, top=max(3.0, max(f) * 1.15))
            fig.suptitle(f"phase: {state['phase']}   |   bulb force {f[-1]:.2f} N   |   tip dz {z[-1]:+.0f} mm",
                         fontsize=10)
        return line_f, line_z

    _anim = FuncAnimation(fig, update, interval=100, cache_frame_data=False)  # noqa: F841 (keep alive)
    plt.show()


class GraphWindow:
    """Streams samples to a matplotlib window running in a separate process."""

    def __init__(self, model: mujoco.MjModel, forces: ForceMonitor, every: int = 10):
        self.forces = forces
        self.sampler = _Sampler(model, forces, every)
        self.queue: mp.Queue = mp.Queue()
        self.process = mp.Process(target=_graph_window_main, args=(self.queue,))
        self.process.start()

    def __call__(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        s = self.sampler.sample(data)
        if s is not None and self.process.is_alive():
            time, force, dz, _ = s
            self.queue.put((time, force, dz, self.forces.phase))

    def wait_closed(self) -> None:
        """Block until the user closes the graph window."""
        self.process.join()
