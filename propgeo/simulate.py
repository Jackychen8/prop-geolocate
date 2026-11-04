"""Ground-truth simulator: flight geometry -> microphone audio + ADS-B feed.

The audio synthesis is physically exact for the model in physics.py: each
sample is evaluated at its true emission time (retarded time), so Doppler,
propagation delay and 1/r spreading all fall out naturally. Realism knobs:
blade-passing harmonics with per-harmonic decay, slow RPM drift (governor
hunting), slow amplitude flutter, atmospheric absorption, broadband engine
noise, pink background noise, optional speech babble and mains hum, and
unmodelled wind.

The ADS-B side adds position noise, 25 ft altitude quantisation, irregular
sampling, and distractor aircraft that are in the feed but not in the audio.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

import numpy as np
from scipy.signal import butter, sosfilt

from .adsb import Track
from .geo import LocalFrame
from .physics import C_DEFAULT, Kinematics, propagate

FT = 0.3048


@dataclass
class SimAircraft:
    icao24: str
    callsign: str
    pos0: list  # ENU metres at scenario t_start
    vel: list  # ENU m/s
    f0: float = 0.0  # blade-passing frequency, Hz (0 => silent distractor)
    n_harm: int = 8
    decay_db: float = 3.0
    level_db: float = 0.0
    audible: bool = True

    def position(self, t_rel):
        t_rel = np.asarray(t_rel, float)[..., None]
        return np.asarray(self.pos0) + np.asarray(self.vel) * t_rel


@dataclass
class Scenario:
    lat: float
    lon: float
    obs_alt: float
    t_start: float
    duration: float
    aircraft: list = field(default_factory=list)
    wind: tuple = (0.0, 0.0)
    c: float = C_DEFAULT
    seed: int = 0

    @property
    def frame(self) -> LocalFrame:
        return LocalFrame(self.lat, self.lon, 0.0)

    def to_json(self) -> str:
        d = asdict(self)
        return json.dumps(d, indent=2)

    @staticmethod
    def from_json(s: str) -> "Scenario":
        d = json.loads(s)
        d["aircraft"] = [SimAircraft(**a) for a in d["aircraft"]]
        d["wind"] = tuple(d["wind"])
        return Scenario(**d)

    def kinematics(self, ac: SimAircraft, pad: float = 120.0, dt: float = 0.02) -> Kinematics:
        t = np.arange(-pad, self.duration + pad, dt)
        pos = ac.position(t)
        vel = np.broadcast_to(np.asarray(ac.vel, float), pos.shape)
        return Kinematics(self.t_start - pad, dt, pos, vel, ac.icao24, ac.callsign)


def _hex(rng) -> str:
    return "".join(rng.choice(list("0123456789abcdef"), 6))


def random_scenario(
    seed: int = 0,
    n_aircraft: int = 2,
    n_distractors: int = 4,
    duration: float = 60.0,
    cpa_range_m: tuple = (250.0, 2500.0),
    agl_range_m: tuple = (300.0, 1500.0),
    speed_range: tuple = (45.0, 85.0),
    base_latlon: tuple = (37.45, -122.10),
    wind_speed: float = 0.0,
    t_start: float = 1_780_000_000.0,
) -> Scenario:
    """Random geometry with every audible aircraft's closest approach inside the clip."""
    rng = np.random.default_rng(seed)
    lat = base_latlon[0] + rng.uniform(-0.3, 0.3)
    lon = base_latlon[1] + rng.uniform(-0.3, 0.3)
    obs_alt = float(rng.uniform(0, 300))
    wdir = rng.uniform(0, 2 * np.pi)
    scn = Scenario(lat, lon, obs_alt, t_start + float(rng.integers(0, 86400)), duration,
                   wind=(wind_speed * np.cos(wdir), wind_speed * np.sin(wdir)), seed=seed)
    for _ in range(n_aircraft):
        hdg = rng.uniform(0, 2 * np.pi)
        u = np.array([np.sin(hdg), np.cos(hdg)])  # heading as compass bearing
        perp = np.array([u[1], -u[0]]) * rng.choice([-1, 1])
        d = rng.uniform(*cpa_range_m)
        spd = rng.uniform(*speed_range)
        agl = rng.uniform(*agl_range_m)
        t_cpa = rng.uniform(0.25, 0.75) * duration
        p_cpa = np.r_[perp * d, obs_alt + agl]
        vz = rng.normal(0, 1.0)  # gentle climb/descent
        vel = np.r_[u * spd, vz]
        pos0 = p_cpa - vel * t_cpa
        scn.aircraft.append(
            SimAircraft(_hex(rng), f"N{rng.integers(100, 999)}{chr(65 + rng.integers(0, 26))}",
                        pos0.tolist(), vel.tolist(), f0=float(rng.uniform(65, 140)),
                        n_harm=int(rng.integers(6, 13)), decay_db=float(rng.uniform(2, 5)),
                        level_db=float(rng.uniform(-3, 3)))
        )
    for _ in range(n_distractors):
        hdg = rng.uniform(0, 2 * np.pi)
        u = np.array([np.sin(hdg), np.cos(hdg)])
        if rng.random() < 0.5:  # airliner overhead-ish
            spd, alt, d = rng.uniform(180, 250), rng.uniform(6000, 11000), rng.uniform(0, 15000)
        else:  # far-away GA traffic
            spd, alt, d = rng.uniform(45, 85), rng.uniform(500, 2500), rng.uniform(5000, 15000)
        perp = np.array([u[1], -u[0]])
        t_cpa = rng.uniform(-60, duration + 60)
        vel = np.r_[u * spd, 0.0]
        pos0 = np.r_[perp * d, obs_alt + alt] - vel * t_cpa
        scn.aircraft.append(SimAircraft(_hex(rng), f"X{rng.integers(100, 999)}", pos0.tolist(), vel.tolist(),
                                        audible=False))
    return scn


# --------------------------------------------------------------------------
# audio
# --------------------------------------------------------------------------

def _lowpass_noise(rng, n, fs, fc):
    sos = butter(2, fc, fs=fs, output="sos")
    x = sosfilt(sos, rng.standard_normal(n))
    return x / (np.std(x) + 1e-12)


def _pink(rng, n):
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.arange(len(spec), dtype=float)
    f[0] = 1.0
    x = np.fft.irfft(spec / np.sqrt(f), n)
    return x / np.std(x)


def _babble(rng, n, sr, talkers=3):
    out = np.zeros(n)
    for _ in range(talkers):
        pos = int(rng.uniform(0, 0.3) * sr)
        base = rng.uniform(95, 230)
        f1, f2 = rng.uniform(300, 800), rng.uniform(900, 2500)
        while pos < n:
            dur = int(rng.uniform(0.1, 0.45) * sr)
            seg_t = np.arange(dur) / sr
            f_start = base * rng.uniform(0.85, 1.15)
            f_end = f_start * rng.uniform(0.8, 1.25)
            f = np.linspace(f_start, f_end, dur)
            ph = 2 * np.pi * np.cumsum(f) / sr
            seg = np.zeros(dur)
            for k in range(1, int(3500 / max(f_start, f_end))):
                fk = k * f_start
                amp = np.exp(-((fk - f1) / 250) ** 2) + 0.6 * np.exp(-((fk - f2) / 400) ** 2) + 0.05
                seg += amp * np.sin(k * ph + rng.uniform(0, 2 * np.pi))
            seg *= np.hanning(dur)
            end = min(n, pos + dur)
            out[pos:end] += seg[: end - pos]
            pos = end + int(rng.uniform(0.03, 0.4) * sr)
        _ = seg_t
    return out / (np.std(out) + 1e-12)


def synthesize(
    scn: Scenario,
    sr: int = 16000,
    snr_db: float = 10.0,
    rpm_drift: float = 0.001,
    babble_db: float | None = None,
    hum: bool = False,
    seed: int | None = None,
) -> np.ndarray:
    """Render the microphone signal.

    snr_db: total aircraft power / broadband pink-noise power over the clip.
    rpm_drift: std of slow (~3 s correlation) fractional RPM wander.
    babble_db: speech babble level relative to aircraft power (None = off).
    """
    rng = np.random.default_rng(scn.seed + 1000 if seed is None else seed)
    n = int(round(scn.duration * sr))
    t_abs = scn.t_start + np.arange(n) / sr
    obs = np.array([0.0, 0.0, scn.obs_alt])
    wind = scn.wind if any(scn.wind) else None
    planes = np.zeros(n)
    for ac in scn.aircraft:
        if not ac.audible or ac.f0 <= 0:
            continue
        kin = scn.kinematics(ac)
        _, r, tau, _ = propagate(kin, obs, t_abs, c=scn.c, wind=wind, iters=5)
        # propeller tone directivity: loudest broadside, ~-10 dB fore/aft, slightly
        # asymmetric. This makes the solver's simple 1/r level model deliberately wrong.
        p_e, v_e, _ = kin.at(tau)
        dvec = obs - p_e
        cos_t = np.einsum("ij,ij->i", dvec, v_e) / (np.linalg.norm(dvec, axis=1) * np.linalg.norm(v_e, axis=1) + 1e-9)
        directivity = (0.3 + 0.7 * (1 - cos_t**2)) * (1 - 0.2 * cos_t)
        # source phase as a function of emission time, with slow RPM wander
        tg = kin.t0 + np.arange(len(kin.pos)) * kin.dt
        drift = 1.0 + rpm_drift * _lowpass_noise(rng, len(tg), 1 / kin.dt, 0.3)
        phase_g = 2 * np.pi * np.cumsum(ac.f0 * drift) * kin.dt
        phase = np.interp(tau, tg, phase_g)
        flutter = 10 ** (1.5 * _lowpass_noise(rng, n, sr, 1.0) / 20)
        sig = np.zeros(n)
        for k in range(1, ac.n_harm + 1):
            fk = k * ac.f0
            absorb_db_per_m = 5.0e-3 * (fk / 1000.0) ** 2 + 2e-5
            amp = 10 ** ((-ac.decay_db * (k - 1) - absorb_db_per_m * r) / 20)
            sig += amp * np.sin(k * phase + rng.uniform(0, 2 * np.pi))
        engine = 0.15 * _lowpass_noise(rng, n, sr, 600.0)
        planes += 10 ** (ac.level_db / 20) * flutter * (directivity * sig + engine) * (1000.0 / r)
    p_planes = np.mean(planes**2) + 1e-20
    noise = _pink(rng, n) * np.sqrt(p_planes / 10 ** (snr_db / 10))
    out = planes + noise
    if babble_db is not None:
        out += _babble(rng, n, sr) * np.sqrt(p_planes * 10 ** (babble_db / 10))
    if hum:
        tt = np.arange(n) / sr
        h = sum(np.sin(2 * np.pi * 60 * k * tt) / k for k in (1, 2, 3, 5))
        out += h * np.sqrt(p_planes) * 0.5
    return (0.25 * out / (np.std(out) + 1e-12)).astype(np.float32)


# --------------------------------------------------------------------------
# ADS-B
# --------------------------------------------------------------------------

def adsb_feed(
    scn: Scenario,
    interval_s: float = 1.0,
    pos_noise_m: float = 8.0,
    span_s: float = 300.0,
    seed: int | None = None,
) -> list[Track]:
    """What an ADS-B receiver network would have logged for the scenario."""
    rng = np.random.default_rng(scn.seed + 2000 if seed is None else seed)
    frame = scn.frame
    tracks = []
    for ac in scn.aircraft:
        t_rel = np.arange(-span_s + rng.uniform(0, interval_s), scn.duration + span_s, interval_s)
        t_rel = t_rel + rng.uniform(-0.15, 0.15, len(t_rel)) * min(interval_s, 1.0)
        keep = rng.random(len(t_rel)) > 0.05  # 5% dropped messages
        t_rel = np.sort(t_rel[keep])
        p = ac.position(t_rel)
        e = p[:, 0] + rng.normal(0, pos_noise_m, len(t_rel))
        nn = p[:, 1] + rng.normal(0, pos_noise_m, len(t_rel))
        alt_q = np.round(p[:, 2] / (25 * FT)) * (25 * FT)
        lat, lon, h = frame.to_geodetic(e, nn, alt_q)
        tracks.append(Track(ac.icao24, ac.callsign, scn.t_start + t_rel, lat, lon, h))
    return tracks


def truth_doppler(scn: Scenario, ac: SimAircraft, t_rel: np.ndarray):
    """Ground-truth log Doppler ratio for an aircraft at clip-relative times."""
    kin = scn.kinematics(ac)
    obs = np.array([0.0, 0.0, scn.obs_alt])
    ld, r, _, _ = propagate(kin, obs, scn.t_start + np.asarray(t_rel), c=scn.c,
                            wind=scn.wind if any(scn.wind) else None, iters=5)
    return ld, r
