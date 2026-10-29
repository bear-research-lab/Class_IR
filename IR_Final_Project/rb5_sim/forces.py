"""Force monitoring for the pipette: grip, bulb squeeze and environment contacts.

Attach a ForceMonitor to the Sim (sim.callbacks.append(monitor)) and it records,
every step:
  grip             sum of contact normal forces from the hand on the pipette shaft [N]
  bulb_pads        sum of contact normal forces from the hand on the bulb [N]
  bulb_force       bulb squeeze force per side, from the wall springs [N]
  bulb_comp        bulb compression per side [m]
  env              sum of contact normal forces from everything that is not the
                   hand or robot (holder, tank, beaker, bench) [N]; spikes mean a collision

"Hand" is everything attached to the RB5's last link, so this works for any hand
design without knowing its body names.
"""

import csv
from collections import defaultdict
from pathlib import Path

import mujoco
import numpy as np

BULB_WALL_PAIRS = (("bulb_px", "bulb_nx"), ("bulb_py", "bulb_ny"))
CHANNELS = ("grip", "bulb_pads", "bulb_force", "bulb_comp", "env")


class ForceMonitor:
    def __init__(self, model: mujoco.MjModel):
        self.model = model
        self.phase = "start"
        self.event = ""  # set by the task while the bulb is held squeezed (used by the reward)
        self.time: list[float] = []
        self.phases: list[str] = []
        self.events: list[str] = []
        self.series: dict[str, list[float]] = {c: [] for c in CHANNELS}

        pipette = model.body("pipette").id
        self.pipette_bodies = {b for b in range(model.nbody) if model.body_rootid[b] == pipette}
        self.bulb_bodies = {model.body(f"bulb_wall_{s}").id for s in ("px", "nx", "py", "ny")}
        self.robot_bodies = {b for b in range(model.nbody) if self._has_ancestor(b, model.body("link1").id)}
        last_link = model.body("link6").id
        self.hand_bodies = {b for b in self.robot_bodies if self._has_ancestor(b, last_link)}

        self.wall_qadr = [[model.joint(j).qposadr[0] for j in pair] for pair in BULB_WALL_PAIRS]
        self.wall_k = float(model.jnt_stiffness[model.joint("bulb_px").id])
        self.wall_max = float(model.jnt_range[model.joint("bulb_px").id, 1])
        self._f = np.zeros(6)

    def _has_ancestor(self, body: int, ancestor: int) -> bool:
        while body != 0:
            if body == ancestor:
                return True
            body = self.model.body_parentid[body]
        return False

    def bulb_compression(self, data: mujoco.MjData) -> float:
        """Compression per side of the most-squeezed wall pair [m]."""
        return max(float(np.mean(data.qpos[adr])) for adr in self.wall_qadr)

    def bulb_force(self, data: mujoco.MjData) -> float:
        """Squeeze force per side from the wall springs [N]."""
        return self.wall_k * self.bulb_compression(data)

    def __call__(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        sums = defaultdict(float)
        for i in range(data.ncon):
            c = data.contact[i]
            b1, b2 = model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2]
            in1, in2 = b1 in self.pipette_bodies, b2 in self.pipette_bodies
            if in1 == in2:
                continue
            own, other = (b1, b2) if in1 else (b2, b1)
            mujoco.mj_contactForce(model, data, i, self._f)
            if other in self.hand_bodies:
                sums["bulb_pads" if own in self.bulb_bodies else "grip"] += self._f[0]
            elif other not in self.robot_bodies:
                sums["env"] += self._f[0]

        self.time.append(float(data.time))
        self.phases.append(self.phase)
        self.events.append(self.event)
        for ch in ("grip", "bulb_pads", "env"):
            self.series[ch].append(sums[ch])
        comp = self.bulb_compression(data)
        self.series["bulb_comp"].append(comp)
        self.series["bulb_force"].append(self.wall_k * comp)

    # ---- reporting -----------------------------------------------------------------

    def peaks_by_phase(self) -> dict[str, dict[str, float]]:
        phases = np.array(self.phases)
        out = {}
        for phase in dict.fromkeys(self.phases):
            mask = phases == phase
            out[phase] = {ch: float(np.max(np.asarray(v)[mask])) for ch, v in self.series.items()}
        return out

    def summary(self) -> str:
        rows = [f"  {'phase':<22}{'grip':>8}{'bulb F':>8}{'bulb mm':>9}{'env':>8}   (peak N)"]
        for phase, p in self.peaks_by_phase().items():
            rows.append(f"  {phase:<22}{p['grip']:8.1f}{p['bulb_force']:8.2f}"
                        f"{p['bulb_comp'] * 1e3:9.1f}{p['env']:8.1f}")
        return "\n".join(rows)

    def save_csv(self, path: str | Path) -> None:
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["time", "phase", "event", *CHANNELS])
            for i, t in enumerate(self.time):
                w.writerow([f"{t:.4f}", self.phases[i], self.events[i],
                            *(f"{self.series[c][i]:.5g}" for c in CHANNELS)])

    def plot(self, path: str | Path) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        t = np.array(self.time)
        fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
        axes[0].plot(t, self.series["grip"], label="hand on shaft")
        axes[0].set_ylabel("grip contact force [N]")
        axes[1].plot(t, self.series["bulb_force"], label="bulb squeeze force (spring)")
        axes[1].plot(t, np.array(self.series["bulb_pads"]) / 2, "--", label="hand on bulb, contact force / 2")
        ax_c = axes[1].twinx()
        ax_c.plot(t, np.array(self.series["bulb_comp"]) * 1e3, color="0.5", lw=0.8)
        ax_c.set_ylabel("compression [mm]", color="0.4")
        ax_c.axhline(self.wall_max * 1e3, color="0.5", ls=":", lw=0.8)
        axes[1].set_ylabel("bulb force per side [N]")
        axes[2].plot(t, self.series["env"], color="tab:red", label="holder / tank / beaker / bench")
        axes[2].set_ylabel("environment force [N]")
        axes[2].set_xlabel("time [s]")

        # shade task phases
        starts = [0] + [i for i in range(1, len(self.phases)) if self.phases[i] != self.phases[i - 1]]
        for k, i0 in enumerate(starts):
            i1 = starts[k + 1] if k + 1 < len(starts) else len(t) - 1
            for ax in axes:
                if k % 2:
                    ax.axvspan(t[i0], t[i1], color="0.93", zorder=0)
            axes[0].text(t[i0], 1.02, self.phases[i0], transform=axes[0].get_xaxis_transform(),
                         fontsize=8, rotation=20, ha="left", va="bottom")
        for ax in axes:
            ax.legend(loc="upper right", fontsize=8)
            ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
