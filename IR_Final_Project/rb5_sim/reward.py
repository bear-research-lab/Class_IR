"""Reward for one pipetting experiment: did the task succeed, and how consistent
was the force applied to the bulb?

    reward = gate * exp(-(rms_error / FORCE_TOLERANCE)^2)        in [0, 1]

gate       1 if every task check passed (picked up, tip under water at release,
           tip inside the dispense box, bulb released, no collisions, pipette
           returned and seated in the holder with at most 15 N), else 0.
rms_error  RMS of (bulb squeeze force - TARGET_FORCE) over every sample while the
           bulb is held squeezed (the aspirate dip and the dispense hold). It
           penalizes both the wrong force level and a force that wobbles.

With the defaults: on target -> 1.0, 0.25 N off -> 0.78, 0.5 N off -> 0.37,
1 N off -> 0.02. A hand that applies the same force no matter how thick the
pipette is scores high across all experiments.
"""

from dataclasses import dataclass

import numpy as np

from .forces import ForceMonitor
from .task import Report

TARGET_FORCE = 2.5      # N per side
FORCE_TOLERANCE = 0.5   # N; RMS error at which the force term drops to exp(-1) = 0.37


@dataclass
class ExperimentScore:
    reward: float
    task_passed: bool
    force_rms_error: float   # N
    mean_force: float        # N, during squeeze holds
    force_std: float         # N, during squeeze holds
    max_compression: float   # m
    bulb_bottomed_out: bool
    insert_force: float      # N, peak holder force while returning the pipette


def score_experiment(report: Report, forces: ForceMonitor) -> ExperimentScore:
    events = np.array(forces.events)
    force = np.array(forces.series["bulb_force"])
    comp = np.array(forces.series["bulb_comp"])
    held = events != ""

    if held.any():
        f = force[held]
        rms = float(np.sqrt(np.mean((f - TARGET_FORCE) ** 2)))
        mean, std = float(f.mean()), float(f.std())
        max_comp = float(comp[held].max())
    else:  # the task never got to squeezing
        rms, mean, std, max_comp = float("nan"), float("nan"), float("nan"), 0.0

    gate = report.task_passed and held.any()
    reward = float(np.exp(-(rms / FORCE_TOLERANCE) ** 2)) if gate else 0.0
    return ExperimentScore(
        reward=reward,
        task_passed=report.task_passed,
        force_rms_error=rms,
        mean_force=mean,
        force_std=std,
        max_compression=max_comp,
        bulb_bottomed_out=max_comp >= forces.wall_max - 0.0005,
        insert_force=report.metrics.get("insert_force", float("nan")),
    )
