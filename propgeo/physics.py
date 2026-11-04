"""Acoustic propagation + Doppler model for a moving source and a fixed microphone.

Model (moving source, fixed observer, uniform wind w in a still atmosphere):

    sound received at time t_rx was emitted at tau, where
        t_rx = tau + r(tau) / (c + w . n)
    and the observed/emitted frequency ratio is
        f_obs / f_src = (c + w . n) / (c - (v - w) . n)

with n the unit vector from source (at tau) to observer, v the aircraft's
ground velocity, r the slant range. With w = 0 this reduces to the classic
f = f0 * c / (c - v_r).

The solver works with log-frequency, so a curve's unknown source frequency
(blade-passing frequency x harmonic number) is a constant additive offset.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

C_DEFAULT = 343.0


def speed_of_sound(temp_c: float) -> float:
    return 331.3 * float(np.sqrt(1.0 + temp_c / 273.15))


@dataclass
class Kinematics:
    """Uniformly resampled aircraft position/velocity in local ENU metres."""

    t0: float
    dt: float
    pos: np.ndarray  # (n, 3)
    vel: np.ndarray  # (n, 3)
    icao24: str = ""
    callsign: str = ""
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.pos = np.ascontiguousarray(self.pos, float)
        self.vel = np.ascontiguousarray(self.vel, float)
        if len(self.pos) < 2:
            raise ValueError("Kinematics needs at least two samples")

    @property
    def t_end(self) -> float:
        return self.t0 + self.dt * (len(self.pos) - 1)

    def at(self, t):
        """Linear interpolation of position & velocity. Returns pos, vel, valid."""
        t = np.asarray(t, float)
        n = len(self.pos)
        idx = (t - self.t0) / self.dt
        valid = (idx >= 0.0) & (idx <= n - 1)
        idx = np.clip(idx, 0.0, n - 1 - 1e-9)
        i = idx.astype(np.int64)
        f = (idx - i)[..., None]
        p = self.pos[i] * (1.0 - f) + self.pos[i + 1] * f
        v = self.vel[i] * (1.0 - f) + self.vel[i + 1] * f
        return p, v, valid


def propagate(kin: Kinematics, obs, t_rx, c: float = C_DEFAULT, wind=None, iters: int = 3):
    """Doppler log-ratio seen at `obs` for sound received at absolute times `t_rx`.

    `obs` must broadcast against `t_rx[..., None]` (last axis = ENU xyz).
    Returns (log_ratio, slant_range, emission_time, valid) broadcast to the
    common shape.
    """
    obs = np.asarray(obs, float)
    t_rx = np.asarray(t_rx, float)
    w = np.zeros(3) if wind is None else np.array([wind[0], wind[1], 0.0], float)
    # Newton on g(tau) = tau + r(tau)/c_eff - t_rx, g' = 1 - (v . n)/c_eff.
    # Start from the straight-line guess tau = t_rx - r(t_rx)/c.
    p, _, _ = kin.at(t_rx)
    d = obs - p
    tau = t_rx - np.sqrt(np.einsum("...i,...i->...", d, d)) / c
    for _ in range(iters):
        p, v, _ = kin.at(tau)
        d = obs - p
        r = np.sqrt(np.einsum("...i,...i->...", d, d))
        n = d / r[..., None]
        ceff = c + n @ w
        g = tau + r / ceff - t_rx
        gp = 1.0 - np.einsum("...i,...i->...", v, n) / ceff
        tau = tau - g / gp
    p, v, valid = kin.at(tau)
    d = obs - p
    r = np.sqrt(np.einsum("...i,...i->...", d, d))
    n = d / r[..., None]
    wn = n @ w
    log_ratio = np.log(c + wn) - np.log(c - np.einsum("...i,...i->...", v, n) + wn)
    return log_ratio, r, tau, valid
