"""Doppler curves + ADS-B tracks -> microphone position.

Unknowns: observer east/north (x, y), audio clock offset delta (optional),
which aircraft produced each curve, and each curve's at-rest frequency.

For a trial (x, y, delta) and aircraft a, physics.propagate predicts the log
Doppler ratio along the curve's timestamps. The at-rest frequency is a free
additive constant in log space, so it is profiled out by weighted demeaning.
Each curve takes whichever aircraft fits it best (or is declared unexplained
at a fixed cost, so a car or an aircraft without ADS-B can't drag the fit).

Search: dense coarse grid over (x, y, delta) with a deliberately loose noise
scale, then multi-start robust least-squares (Huber) refinement from the best
local minima, re-assigning curves to aircraft between refinements. A single
aircraft on a straight line leaves a mirror ambiguity across its track; that
is detected and reported rather than hidden.
"""
from __future__ import annotations

import time as _time
from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import minimum_filter
from scipy.optimize import least_squares

from .adsb import Track, build_kinematics
from .extract import DopplerCurve
from .geo import LocalFrame, haversine_m
from .physics import Kinematics, propagate


@dataclass
class LocateConfig:
    search_radius_m: float = 8000.0
    auto_radius_m: float = 12000.0  # used instead when no search centre is given
    coarse_step_m: float = 300.0
    time_window_s: float = 0.0  # +/- uncertainty of the audio start time
    time_step_s: float = 1.5
    observer_alt_m: float = 0.0  # observer height, same datum as ADS-B altitudes
    sound_speed: float = 343.0
    wind_en: tuple | None = None  # known wind (east, north) m/s, if any
    sigma_logf: float = 0.0015
    coarse_sigma_logf: float = 0.006
    huber_k: float = 2.5
    max_range_m: float = 12000.0
    coarse_hop_s: float = 1.0
    fine_hop_s: float = 0.3
    top_k: int = 12
    adsb_dt: float = 0.1
    unexplained_z: float = 4.0
    ambiguity_dchi2: float = 16.0
    ambiguity_cost_ratio: float = 1.5  # any distinct solution within this cost ratio is "acceptable"
    use_level: bool = False  # fit loudness envelope to -20log10(range); benchmarked: no net gain
    sys_floor_m: float = 10.0  # unmodelled systematic error (ADS-B position accuracy etc.), added in quadrature  # also fit the tone's level envelope to -20 log10(range)
    sigma_level_db: float = 3.0
    chunk: int = 96


@dataclass
class CurveFit:
    curve_index: int
    icao24: str | None
    callsign: str
    rest_freq_hz: float | None
    rms_logf: float | None
    t: np.ndarray = field(repr=False, default=None)
    f_obs: np.ndarray = field(repr=False, default=None)
    f_pred: np.ndarray = field(repr=False, default=None)


@dataclass
class Candidate:
    lat: float
    lon: float
    x: float
    y: float
    time_offset_s: float
    cost: float
    dchi2: float = 0.0


@dataclass
class LocateResult:
    ok: bool
    message: str = ""
    lat: float = float("nan")
    lon: float = float("nan")
    time_offset_s: float = 0.0
    sigma_major_m: float = float("nan")
    sigma_minor_m: float = float("nan")
    ellipse_angle_deg: float = 0.0  # major-axis direction, degrees clockwise from north
    r95_m: float = float("nan")
    rms_logf: float = float("nan")
    reduced_chi2: float = float("nan")
    ambiguous: bool = False
    fits: list = field(default_factory=list)
    candidates: list = field(default_factory=list)
    alternates: list = field(default_factory=list)
    center: tuple = (float("nan"), float("nan"))
    heat_lat: np.ndarray = field(repr=False, default=None)
    heat_lon: np.ndarray = field(repr=False, default=None)
    heat_cost: np.ndarray = field(repr=False, default=None)
    n_aircraft: int = 0
    elapsed_s: float = 0.0

    def summary(self) -> dict:
        return {
            "ok": self.ok,
            "message": self.message,
            "lat": self.lat,
            "lon": self.lon,
            "time_offset_s": round(self.time_offset_s, 2),
            "r95_m": round(self.r95_m, 1) if np.isfinite(self.r95_m) else None,
            "sigma_major_m": round(self.sigma_major_m, 1) if np.isfinite(self.sigma_major_m) else None,
            "sigma_minor_m": round(self.sigma_minor_m, 1) if np.isfinite(self.sigma_minor_m) else None,
            "rms_logf": self.rms_logf,
            "reduced_chi2": self.reduced_chi2,
            "ambiguous": self.ambiguous,
            "alternates": [
                {"lat": a.lat, "lon": a.lon, "dchi2": round(a.dchi2, 1)} for a in self.alternates
            ],
            "fits": [
                {
                    "curve": f.curve_index,
                    "icao24": f.icao24,
                    "callsign": f.callsign,
                    "rest_freq_hz": None if f.rest_freq_hz is None else round(f.rest_freq_hz, 2),
                    "rms_logf": None if f.rms_logf is None else round(f.rms_logf, 5),
                }
                for f in self.fits
            ],
            "n_aircraft_considered": self.n_aircraft,
            "elapsed_s": round(self.elapsed_s, 2),
        }


@dataclass
class _Bins:
    t: np.ndarray
    y: np.ndarray
    w: np.ndarray
    lv: np.ndarray | None = None  # mean tone level per bin (dB)


def _bin(c: DopplerCurve, hop: float) -> _Bins:
    idx = np.floor((c.t - c.t[0]) / hop + 1e-9).astype(int)
    n = idx.max() + 1
    w = np.bincount(idx, c.weight, n)
    ok = w > 0
    t = np.bincount(idx, c.weight * c.t, n)[ok] / w[ok]
    y = np.bincount(idx, c.weight * c.logf, n)[ok] / w[ok]
    w = w[ok]
    w = np.clip(w / np.median(w), 0.05, 5.0)
    lv = None
    if c.level_db is not None:
        lv = np.bincount(idx, c.level_db, n)[ok] / np.bincount(idx, None, n)[ok]
    return _Bins(t, y, w / w.mean(), lv)


def _huber(z, k):
    a = np.abs(z)
    return np.where(a <= k, 0.5 * z * z, k * a - 0.5 * k * k)


def _capped(z, k, zu):
    """Huber loss capped at the 'unexplained' level: a point that is way off
    (e.g. a stretch where the tracker followed another plane's harmonic)
    costs exactly what an unexplained point costs, no more."""
    return np.minimum(_huber(z, k), _huber(zu, k))


def _capped_loss_factory(k: float, zu: float):
    """scipy least_squares callable loss matching _capped (z = (e/k)^2)."""
    u2 = (zu / k) ** 2

    def loss(z):
        z = np.asarray(z, float)
        sz = np.sqrt(np.maximum(z, 1e-12))
        rho = np.where(z <= 1, z, np.where(z <= u2, 2 * sz - 1, 2 * np.sqrt(u2) - 1))
        d1 = np.where(z <= 1, 1.0, np.where(z <= u2, 1 / sz, 0.0))
        d2 = np.where((z > 1) & (z <= u2), -0.5 * sz ** -3, 0.0)
        return np.vstack([rho, d1, d2])

    return loss


class _Model:
    def __init__(self, kins: list[Kinematics], curves: list[DopplerCurve], audio_start: float,
                 frame: LocalFrame, cfg: LocateConfig):
        self.kins, self.cfg, self.frame, self.t0 = kins, cfg, frame, audio_start
        self.coarse = [_bin(c, cfg.coarse_hop_s) for c in curves]
        self.fine = [_bin(c, cfg.fine_hop_s) for c in curves]
        self.wind = cfg.wind_en if cfg.wind_en and any(cfg.wind_en) else None

    def obs(self, x, y):
        x, y = np.asarray(x, float), np.asarray(y, float)
        z = self.frame.ground_up(x, y, self.cfg.observer_alt_m)
        return np.stack(np.broadcast_arrays(x, y, z), axis=-1)

    def _use_level(self, b: _Bins) -> bool:
        return self.cfg.use_level and b.lv is not None

    def unexplained(self, b: _Bins) -> float:
        n = np.sum(b.w) + (len(b.lv) if self._use_level(b) else 0)
        return float(n * _huber(self.cfg.unexplained_z, self.cfg.huber_k))

    # ---- coarse grid -----------------------------------------------------
    def coarse_costs(self, pts: np.ndarray, deltas: np.ndarray):
        """Per-curve min-over-aircraft cost on the (point, delta) grid."""
        cfg = self.cfg
        sig, k = cfg.coarse_sigma_logf, cfg.huber_k
        J, G, D = len(self.coarse), len(pts), len(deltas)
        best = np.empty((J, G, D))
        arg = np.full((J, G, D), -1, int)
        for j, b in enumerate(self.coarse):
            unexpl_pt = _huber(cfg.unexplained_z, k)
            best[j] = self.unexplained(b)
            t_abs = self.t0 + deltas[:, None] + b.t[None, :]  # (D, N)
            for a, kin in enumerate(self.kins):
                for g0 in range(0, G, cfg.chunk):
                    obs = pts[g0:g0 + cfg.chunk, None, None, :]
                    ld, r, _, valid = propagate(kin, obs, t_abs, cfg.sound_speed, self.wind, iters=2)
                    valid &= r < cfg.max_range_m
                    res = b.y - ld
                    wv = b.w * valid
                    m = np.sum(wv * res, -1) / np.maximum(np.sum(wv, -1), 1e-9)
                    rho = _capped((res - m[..., None]) / sig, k, cfg.unexplained_z)
                    rho = np.where(valid, rho, unexpl_pt)
                    cost = np.sum(b.w * rho, -1)
                    if self._use_level(b):
                        lres = b.lv + 20 * np.log10(r)
                        lm = np.sum(valid * lres, -1) / np.maximum(np.sum(valid, -1), 1)
                        lrho = _capped((lres - lm[..., None]) / cfg.sigma_level_db, k, cfg.unexplained_z)
                        cost = cost + np.sum(np.where(valid, lrho, unexpl_pt), -1)
                    sl = slice(g0, g0 + cfg.chunk)
                    better = cost < best[j, sl]
                    best[j, sl] = np.where(better, cost, best[j, sl])
                    arg[j, sl] = np.where(better, a, arg[j, sl])
        return best, arg

    # ---- fine model ------------------------------------------------------
    def curve_residuals(self, j: int, a: int, x, y, delta, sigma=None):
        cfg = self.cfg
        b = self.fine[j]
        sig = sigma or cfg.sigma_logf
        ld, r, _, valid = propagate(self.kins[a], self.obs(x, y), self.t0 + delta + b.t,
                                    cfg.sound_speed, self.wind, iters=3)
        valid &= r < cfg.max_range_m
        res = b.y - ld
        wv = b.w * valid
        m = np.sum(wv * res) / max(np.sum(wv), 1e-9)
        e = np.sqrt(b.w) * (res - m) / sig
        e = np.where(valid, e, cfg.unexplained_z)
        if self._use_level(b):
            lres = b.lv + 20 * np.log10(r)
            lm = np.sum(valid * lres) / max(np.sum(valid), 1)
            sl = cfg.sigma_level_db * (sig / cfg.sigma_logf)
            el = np.where(valid, (lres - lm) / sl, cfg.unexplained_z)
            e = np.concatenate([e, el])
        return e, m, ld, valid

    def curve_cost(self, j, a, x, y, delta) -> float:
        e, *_ = self.curve_residuals(j, a, x, y, delta)
        return float(np.sum(_capped(e, self.cfg.huber_k, self.cfg.unexplained_z)))

    def assign(self, x, y, delta):
        out, total = [], 0.0
        for j, b in enumerate(self.fine):
            costs = [self.curve_cost(j, a, x, y, delta) for a in range(len(self.kins))]
            best_a = int(np.argmin(costs)) if costs else -1
            unexpl = self.unexplained(b)
            if best_a < 0 or costs[best_a] >= unexpl:
                out.append(-1)
                total += unexpl
            else:
                out.append(best_a)
                total += costs[best_a]
        return out, total

    def refine(self, x0, y0, d0, assign, free_delta: bool, sigma=None):
        cfg = self.cfg
        pairs = [(j, a) for j, a in enumerate(assign) if a >= 0]
        if not pairs:
            return None

        def fun(p):
            d = p[2] if free_delta else d0
            return np.concatenate([self.curve_residuals(j, a, p[0], p[1], d, sigma)[0] for j, a in pairs])

        p0 = [x0, y0] + ([d0] if free_delta else [])
        R = cfg.search_radius_m * 1.25
        lo = [-R, -R] + ([-cfg.time_window_s - 2 * cfg.time_step_s] if free_delta else [])
        hi = [R, R] + ([cfg.time_window_s + 2 * cfg.time_step_s] if free_delta else [])
        p0 = np.clip(p0, np.array(lo) + 1e-6, np.array(hi) - 1e-6)
        sol = least_squares(fun, p0, loss=_capped_loss_factory(cfg.huber_k, cfg.unexplained_z),
                            f_scale=cfg.huber_k, bounds=(lo, hi),
                            x_scale=[100.0, 100.0] + ([1.0] if free_delta else []), max_nfev=200)
        return sol


def _select_kinematics(tracks, frame, t_lo, t_hi, cfg) -> list[Kinematics]:
    kins = []
    for tr in tracks:
        kin = build_kinematics(tr, frame, t_lo, t_hi, dt=cfg.adsb_dt)
        if kin is None:
            continue
        p, _, valid = kin.at(np.linspace(t_lo, t_hi, 20))
        if not valid.any():
            continue
        horiz = np.hypot(p[valid, 0], p[valid, 1])
        if horiz.min() <= cfg.search_radius_m + cfg.max_range_m:
            kins.append(kin)
    return kins


def auto_center(tracks: list[Track], t_mid: float, max_alt_m: float = 3000.0):
    """Fallback search centre: median position of low-flying aircraft mid-clip."""
    lats, lons = [], []
    for tr in tracks:
        if tr.t[0] <= t_mid <= tr.t[-1]:
            alt = np.interp(t_mid, tr.t, tr.alt)
            if alt <= max_alt_m:
                lats.append(np.interp(t_mid, tr.t, tr.lat))
                lons.append(np.interp(t_mid, tr.t, tr.lon))
    if not lats:
        return None
    return float(np.median(lats)), float(np.median(lons))


def locate(
    curves: list[DopplerCurve],
    tracks: list[Track],
    audio_start_unix: float,
    center: tuple | None = None,
    cfg: LocateConfig | None = None,
) -> LocateResult:
    cfg = cfg or LocateConfig()
    t_start = _time.time()
    if not curves:
        return LocateResult(False, "no Doppler curves extracted from the audio")
    dur = max(float(c.t[-1]) for c in curves)
    if center is None:
        # no area hint: low traffic can sit well off to one side, so search wider
        from dataclasses import replace

        cfg = replace(cfg, search_radius_m=max(cfg.search_radius_m, cfg.auto_radius_m))
        center = auto_center(tracks, audio_start_unix + dur / 2)
        if center is None:
            return LocateResult(False, "no low-altitude ADS-B traffic during the clip to centre a search on")
    frame = LocalFrame(center[0], center[1], 0.0)
    t_lo = audio_start_unix - cfg.time_window_s - cfg.max_range_m / cfg.sound_speed - 5
    t_hi = audio_start_unix + dur + cfg.time_window_s + 5
    kins = _select_kinematics(tracks, frame, t_lo, t_hi, cfg)
    if not kins:
        return LocateResult(False, "no ADS-B tracks overlap the clip time / search area", center=center)
    model = _Model(kins, curves, audio_start_unix, frame, cfg)

    # coarse grid
    R, h = cfg.search_radius_m, cfg.coarse_step_m
    ax = np.arange(-R, R + 1e-6, h)
    gx, gy = np.meshgrid(ax, ax)
    inside = np.hypot(gx, gy) <= R
    pts = model.obs(gx[inside], gy[inside])
    free_delta = cfg.time_window_s > 0
    deltas = (np.arange(-cfg.time_window_s, cfg.time_window_s + 1e-9, cfg.time_step_s)
              if free_delta else np.array([0.0]))
    best, arg = model.coarse_costs(pts, deltas)
    total = best.sum(0)  # (G, D)
    d_idx = np.argmin(total, 1)
    flat = total[np.arange(len(pts)), d_idx]
    heat = np.full(gx.shape, np.nan)
    heat[inside] = flat

    # candidate local minima on the heatmap
    hf = np.where(np.isnan(heat), np.inf, heat)
    is_min = (hf == minimum_filter(hf, size=3, mode="nearest")) & np.isfinite(hf)
    cand_cells = np.argwhere(is_min)
    cand_cells = cand_cells[np.argsort(hf[is_min])][: cfg.top_k]
    lin_index = -np.ones(gx.shape, int)
    lin_index[inside] = np.arange(len(pts))

    refined = []
    for (iy, ix) in cand_cells:
        g = lin_index[iy, ix]
        x, y, d = gx[iy, ix], gy[iy, ix], deltas[d_idx[g]]
        assign = [int(arg[j, g, d_idx[g]]) for j in range(len(curves))]
        # stage 1: loose noise scale widens the basin of attraction
        sol = model.refine(x, y, d, assign, free_delta, sigma=cfg.coarse_sigma_logf)
        if sol is not None:
            x, y = sol.x[:2]
            d = sol.x[2] if free_delta else d
            assign, _ = model.assign(x, y, d)
        # stage 2: true noise scale, re-assigning curves until stable
        for _ in range(3):
            sol = model.refine(x, y, d, assign, free_delta)
            if sol is None:
                break
            x, y = sol.x[:2]
            d = sol.x[2] if free_delta else d
            new_assign, cost = model.assign(x, y, d)
            if new_assign == assign:
                break
            assign = new_assign
        if sol is None:
            continue
        _, cost = model.assign(x, y, d)
        refined.append((cost, x, y, d, assign, sol))
    if not refined:
        return LocateResult(False, "no curve could be matched to any ADS-B aircraft", center=center,
                            n_aircraft=len(kins))
    refined.sort(key=lambda r: r[0])

    # polish: with a free clock the capped loss has flat stretches where the
    # optimizer can stall a second or two off; re-start the best few from
    # shifted clock offsets and keep whatever improves.
    if free_delta:
        polished = []
        for cost, x, y, d, assign, sol in refined[:4]:
            best = (cost, x, y, d, assign, sol)
            for dd in (-1.5, -0.75, 0.75, 1.5):
                d1 = float(np.clip(d + dd, -cfg.time_window_s, cfg.time_window_s))
                s1 = model.refine(x, y, d1, assign, True)
                if s1 is None:
                    continue
                a1, c1 = model.assign(s1.x[0], s1.x[1], s1.x[2])
                if c1 < best[0]:
                    best = (c1, s1.x[0], s1.x[1], s1.x[2], a1, s1)
            polished.append(best)
        refined = sorted(polished + refined[4:], key=lambda r: r[0])

    # dedupe nearby solutions
    uniq = []
    for r in refined:
        if all(np.hypot(r[1] - u[1], r[2] - u[2]) > 50 for u in uniq):
            uniq.append(r)
    cost0, x0, y0, d0, assign0, sol0 = uniq[0]

    # uncertainty: Gauss-Newton covariance from a central-difference Jacobian of
    # the inlier residuals (scipy's robust-scaled Jacobian zeroes outlier rows and
    # can go singular), scaled by the empirical reduced chi-square.
    free_delta_used = free_delta
    pairs = [(j, a) for j, a in enumerate(assign0) if a >= 0]

    def resid(px, py, pd):
        return np.concatenate([model.curve_residuals(j, a, px, py, pd)[0] for j, a in pairs])

    e0 = resid(x0, y0, d0)
    inl = np.abs(e0) <= cfg.huber_k
    steps = [1.0, 1.0] + ([0.05] if free_delta_used else [])
    cols = []
    for i, hstep in enumerate(steps):
        dp = np.zeros(3)
        dp[i] = hstep
        ep = resid(x0 + dp[0], y0 + dp[1], d0 + dp[2])
        em = resid(x0 - dp[0], y0 - dp[1], d0 - dp[2])
        cols.append((ep - em) / (2 * hstep))
    Jm = np.stack(cols, axis=1)[inl]
    m, p = int(inl.sum()), Jm.shape[1]
    red = float(np.sum(e0[inl] ** 2) / max(m - p, 1))
    cov = np.linalg.pinv(Jm.T @ Jm) * max(red, 1e-6)
    # residuals are strongly time-correlated (RPM wander, track-smoothing error),
    # so the independent-sample covariance is overconfident. Inflate by the AR(1)
    # variance factor (1+rho)/(1-rho) estimated from the Doppler residuals.
    rhos, wts = [], []
    for j, a in pairs:
        e_d = model.curve_residuals(j, a, x0, y0, d0)[0][: len(model.fine[j].t)]
        e_d = e_d[np.abs(e_d) <= cfg.huber_k]
        if len(e_d) > 10:
            e_d = e_d - e_d.mean()
            den = float(np.sum(e_d * e_d))
            if den > 0:
                rhos.append(float(np.sum(e_d[1:] * e_d[:-1]) / den))
                wts.append(len(e_d))
    rho = float(np.clip(np.average(rhos, weights=wts), 0.0, 0.98)) if rhos else 0.0
    cov = cov * (1 + rho) / (1 - rho)
    cov[:2, :2] += np.eye(2) * cfg.sys_floor_m**2
    evals, evecs = np.linalg.eigh(cov[:2, :2])
    evals = np.clip(evals, 0, None)
    s_minor, s_major = np.sqrt(evals)
    major = evecs[:, 1]
    angle = float(np.degrees(np.arctan2(major[0], major[1])) % 180)
    r95 = 2.45 * s_major

    lat, lon, _ = frame.to_geodetic(x0, y0, 0.0)
    cands = []
    for cost, x, y, d, _, _ in uniq:
        la, lo, _ = frame.to_geodetic(x, y, 0.0)
        dchi2 = 2 * (cost - cost0) / max(red, 1e-6)
        cands.append(Candidate(float(la), float(lo), float(x), float(y), float(d), float(cost), float(dchi2)))
    sep = max(3 * r95 if np.isfinite(r95) else 0.0, 250.0)
    alternates = [c for c in cands[1:]
                  if np.hypot(c.x - x0, c.y - y0) > sep
                  and (c.dchi2 < cfg.ambiguity_dchi2 or c.cost <= cfg.ambiguity_cost_ratio * cost0)]

    fits, rms_all = [], []
    for j, a in enumerate(assign0):
        b = model.fine[j]
        if a < 0:
            fits.append(CurveFit(j, None, "", None, None, b.t, np.exp(b.y), None))
            continue
        _, mshift, ld, valid = model.curve_residuals(j, a, x0, y0, d0)
        if not valid.any():
            fits.append(CurveFit(j, None, "", None, None, b.t, np.exp(b.y), None))
            continue
        res = (b.y - ld - mshift)[valid]
        rms = float(np.sqrt(np.average(res**2, weights=b.w[valid])))
        rms_all.append(res)
        fits.append(CurveFit(j, kins[a].icao24, kins[a].callsign, float(np.exp(mshift)), rms,
                             b.t, np.exp(b.y), np.exp(ld + mshift)))

    hlat, hlon, _ = frame.to_geodetic(gx, gy, 0.0)
    return LocateResult(
        ok=True,
        message="ambiguous: alternate solution(s) fit nearly as well" if alternates else "ok",
        lat=float(lat), lon=float(lon), time_offset_s=float(d0),
        sigma_major_m=float(s_major), sigma_minor_m=float(s_minor), ellipse_angle_deg=angle,
        r95_m=float(r95),
        rms_logf=float(np.sqrt(np.mean(np.concatenate(rms_all) ** 2))) if rms_all else float("nan"),
        reduced_chi2=float(red), ambiguous=bool(alternates), fits=fits, candidates=cands,
        alternates=alternates, center=tuple(center), heat_lat=hlat, heat_lon=hlon, heat_cost=heat,
        n_aircraft=len(kins), elapsed_s=_time.time() - t_start,
    )


def error_m(res: LocateResult, lat: float, lon: float) -> float:
    return float(haversine_m(res.lat, res.lon, lat, lon))
