"""Audio -> Doppler curves.

Pipeline:
  1. resample to a low working rate (propeller tones live under ~1.5 kHz)
  2. high-resolution STFT (0.5 s Hann, 4x zero padding, 0.1 s hop)
  3. whiten against a running median noise floor, pick spectral peaks,
     refine each to sub-bin precision with parabolic interpolation
  4. link peaks into partial tracks frame-to-frame
  5. group partials that are harmonics of the same source: harmonics k1*f0 and
     k2*f0 share the Doppler factor, so their log-frequencies differ by a
     constant. Grouping averages away per-partial noise.
  6. drop stationary tones (mains hum, appliances): no sweep, no information

Each output DopplerCurve is log(frequency) of its reference partial vs time.
Because the solver treats the source frequency as a free constant, it does not
matter which harmonic a curve is expressed in, so octave errors are harmless.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import find_peaks, resample_poly


@dataclass
class ExtractConfig:
    work_sr: int = 4000
    fmin: float = 40.0
    fmax: float = 1500.0
    win_s: float = 0.5
    hop_s: float = 0.1
    zero_pad: int = 4
    floor_width_hz: float = 40.0
    peak_db: float = 7.0
    max_peaks: int = 40
    track_rel_tol: float = 0.006  # allowed relative jump per hop (after slope prediction)
    track_abs_tol_hz: float = 0.8
    max_gap_frames: int = 4
    min_track_s: float = 2.5
    min_track_sweep: float = 0.004  # tracks flatter than this are treated as stationary tones
    group_min_overlap_s: float = 3.0
    group_max_std: float = 0.0025
    min_curve_s: float = 5.0
    min_curve_sweep: float = 0.01
    min_single_partial_s: float = 10.0  # lone partials are often crossing artefacts
    min_single_partial_sweep: float = 0.03  # ...and flat ones fit any distant aircraft
    snr_cap_db: float = 35.0
    collision_rel: float = 0.015  # partials this close in pitch may have swapped sources...
    collision_frames: int = 5  # ...within this many frames; cut them there


@dataclass(eq=False)
class Partial:
    frames: np.ndarray
    freqs: np.ndarray
    snr_db: np.ndarray


@dataclass
class DopplerCurve:
    t: np.ndarray  # seconds from start of audio (frame centres)
    logf: np.ndarray  # natural log of reference-partial frequency
    weight: np.ndarray  # relative inverse variance
    ref_freq: float
    n_partials: int
    level_db: np.ndarray | None = None  # tone level above noise floor (dB, arbitrary offset)

    @property
    def freq(self):
        return np.exp(self.logf)

    @property
    def sweep(self) -> float:
        return float(np.percentile(self.logf, 97) - np.percentile(self.logf, 3))


@dataclass
class ExtractResult:
    curves: list
    partials: list
    frame_times: np.ndarray
    spec_freqs: np.ndarray
    spec_db: np.ndarray  # (frames, freqs) whitened dB, for display
    config: ExtractConfig = field(default_factory=ExtractConfig)


def _stft_db(x, sr, cfg):
    L = int(round(cfg.win_s * sr))
    H = int(round(cfg.hop_s * sr))
    nfft = 1 << int(np.ceil(np.log2(L * cfg.zero_pad)))
    if len(x) < L:
        x = np.pad(x, (0, L - len(x)))
    n_frames = 1 + (len(x) - L) // H
    idx = np.arange(L)[None, :] + H * np.arange(n_frames)[:, None]
    frames = x[idx] * np.hanning(L)[None, :]
    spec = np.fft.rfft(frames, nfft, axis=1)
    freqs = np.fft.rfftfreq(nfft, 1 / sr)
    band = (freqs >= cfg.fmin * 0.8) & (freqs <= cfg.fmax * 1.05)
    db = 20 * np.log10(np.abs(spec[:, band]) + 1e-12)
    times = (np.arange(n_frames) * H + L / 2) / sr
    return times, freqs[band], db


def _noise_floor(db, df, cfg):
    dec = 4
    w = max(3, int(cfg.floor_width_hz / df / dec) | 1)
    coarse = median_filter(db[:, ::dec], size=(1, w), mode="nearest")
    xs = np.arange(db.shape[1])
    return np.stack([np.interp(xs, xs[::dec], row) for row in coarse])


def _pick_peaks(white, db, freqs, cfg):
    df = freqs[1] - freqs[0]
    lobe = max(2, int(0.5 * (2.0 / cfg.win_s) / df))
    out = []
    lo = np.searchsorted(freqs, cfg.fmin)
    hi = np.searchsorted(freqs, cfg.fmax)
    for i in range(white.shape[0]):
        row = white[i]
        pk, props = find_peaks(row[lo:hi], height=cfg.peak_db, distance=lobe)
        pk = pk + lo
        if len(pk) > cfg.max_peaks:
            keep = np.argsort(props["peak_heights"])[-cfg.max_peaks:]
            pk = np.sort(pk[keep])
        pk = pk[(pk > 0) & (pk < len(freqs) - 1)]
        a, b, c = db[i, pk - 1], db[i, pk], db[i, pk + 1]
        den = a - 2 * b + c
        delta = np.where(np.abs(den) > 1e-9, 0.5 * (a - c) / den, 0.0)
        delta = np.clip(delta, -0.5, 0.5)
        out.append((freqs[pk] + delta * df, row[pk]))
    return out


def _track(peaks, cfg) -> list[Partial]:
    active: list[dict] = []
    done: list[dict] = []
    for fi, (pf, ps) in enumerate(peaks):
        keep = []
        for tr in active:
            (done if fi - tr["frames"][-1] > cfg.max_gap_frames + 1 else keep).append(tr)
        active = keep
        used = np.zeros(len(pf), bool)
        if active and len(pf):
            preds, tols = [], []
            for tr in active:
                gap = fi - tr["frames"][-1]
                fr = tr["freqs"]
                if len(fr) >= 3:
                    k = min(6, len(fr))
                    slope = (fr[-1] - fr[-k]) / (tr["frames"][-1] - tr["frames"][-k])
                else:
                    slope = 0.0
                pred = fr[-1] + slope * gap
                preds.append(pred)
                tols.append((cfg.track_rel_tol * pred + cfg.track_abs_tol_hz) * np.sqrt(gap))
            preds, tols = np.array(preds), np.array(tols)
            dist = np.abs(pf[None, :] - preds[:, None]) / tols[:, None]
            pairs = np.argwhere(dist < 1.0)
            order = np.argsort(dist[pairs[:, 0], pairs[:, 1]])
            t_used = np.zeros(len(active), bool)
            for ti, pi in pairs[order]:
                if t_used[ti] or used[pi]:
                    continue
                t_used[ti] = used[pi] = True
                tr = active[ti]
                tr["frames"].append(fi)
                tr["freqs"].append(float(pf[pi]))
                tr["snr"].append(float(ps[pi]))
        for pi in np.flatnonzero(~used):
            active.append({"frames": [fi], "freqs": [float(pf[pi])], "snr": [float(ps[pi])]})
    done.extend(active)
    min_len = int(cfg.min_track_s / cfg.hop_s)
    return [
        Partial(np.array(t["frames"]), np.array(t["freqs"]), np.array(t["snr"]))
        for t in done
        if len(t["frames"]) >= min_len
    ]


class _Acc:
    """Weighted running combination of partials in reference-partial log units."""

    def __init__(self, n_frames: int, cfg: ExtractConfig):
        self.acc = np.zeros(n_frames)
        self.wsum = np.zeros(n_frames)
        self.lacc = np.zeros(n_frames)
        self.lcnt = np.zeros(n_frames)
        self.cfg = cfg
        self.n = 0
        self.ref_freq = None

    def add(self, p: Partial, off: float, level_off: float = 0.0):
        snr_lin = 10 ** (np.minimum(p.snr_db, self.cfg.snr_cap_db) / 10)
        w = snr_lin * (p.freqs / 100.0) ** 2
        np.add.at(self.acc, p.frames, w * (np.log(p.freqs) + off))
        np.add.at(self.wsum, p.frames, w)
        np.add.at(self.lacc, p.frames, p.snr_db + level_off)
        np.add.at(self.lcnt, p.frames, 1.0)
        self.n += 1
        if self.ref_freq is None:
            self.ref_freq = float(np.median(p.freqs))

    @property
    def est(self):
        out = np.full(len(self.acc), np.nan)
        g = self.wsum > 0
        out[g] = self.acc[g] / self.wsum[g]
        return out

    @property
    def est_level(self):
        out = np.full(len(self.lacc), np.nan)
        g = self.lcnt > 0
        out[g] = self.lacc[g] / self.lcnt[g]
        return out

    def curve(self, hop: float) -> DopplerCurve:
        g = np.flatnonzero(self.wsum > 0)
        return DopplerCurve(t=g * hop, logf=self.acc[g] / self.wsum[g], weight=self.wsum[g],
                            ref_freq=self.ref_freq, n_partials=self.n,
                            level_db=self.lacc[g] / self.lcnt[g])


def _subset(p: Partial, mask: np.ndarray) -> Partial:
    return Partial(p.frames[mask], p.freqs[mask], p.snr_db[mask])


def _split_runs(p: Partial, cfg) -> list[Partial]:
    if len(p.frames) == 0:
        return []
    cuts = np.flatnonzero(np.diff(p.frames) > cfg.max_gap_frames + 1) + 1
    min_len = int(cfg.min_track_s / cfg.hop_s)
    out = []
    for idx in np.split(np.arange(len(p.frames)), cuts):
        if len(idx) >= min_len:
            out.append(Partial(p.frames[idx], p.freqs[idx], p.snr_db[idx]))
    return out


def _sweep(p: Partial) -> float:
    return float(np.ptp(np.log(p.freqs)))


def _cut_at_collisions(partials: list[Partial], cfg) -> list[Partial]:
    """Split partials wherever another partial passes within `collision_rel`.

    A frame-to-frame tracker can only jump from one aircraft's harmonic to
    another's where the two nearly collide. Cutting there removes the possible
    swap; the grouping step re-links the true continuation through the other
    harmonics, which cross at different times.
    """
    if len(partials) < 2:
        return partials
    n_frames = max(int(p.frames.max()) for p in partials) + 1
    occ: list[list[tuple[float, int]]] = [[] for _ in range(n_frames)]
    for pid, p in enumerate(partials):
        for f, fr in zip(p.frames, p.freqs):
            occ[f].append((fr, pid))
    W = cfg.collision_frames
    out = []
    for pid, p in enumerate(partials):
        bad = np.zeros(len(p.frames), bool)
        for i, (f, fr) in enumerate(zip(p.frames, p.freqs)):
            lo, hi = max(0, f - W), min(n_frames, f + W + 1)
            for g in range(lo, hi):
                if any(q != pid and abs(fq - fr) <= cfg.collision_rel * fr for fq, q in occ[g]):
                    bad[i] = True
                    break
        if not bad.any():
            out.append(p)
        else:
            out.extend(_split_runs(_subset(p, ~bad), cfg))
    return out


def _group_curves(partials: list[Partial], n_frames: int, cfg) -> list[DopplerCurve]:
    """Greedy grouping of partials into per-source Doppler curves.

    Seeds with the longest/sweepiest partial, then repeatedly absorbs any
    partial that tracks the group's *combined* curve (constant log offset) over
    enough overlapping frames. Only the agreeing frames are absorbed when a
    partial only partly agrees (e.g. it jumped to another aircraft's tone at a
    crossing); the remainder goes back into the pool.
    """
    thr = cfg.group_max_std
    min_ov = int(cfg.group_min_overlap_s / cfg.hop_s)
    pool = list(partials)
    curves = []
    while pool:
        pool.sort(key=lambda p: -(len(p.frames) * max(_sweep(p), 1e-4) * np.median(p.freqs) ** 0.5))
        grp = _Acc(n_frames, cfg)
        grp.add(pool.pop(0), 0.0)
        changed = True
        while changed:
            changed = False
            est = grp.est
            for p in list(pool):
                e = est[p.frames]
                have = ~np.isnan(e)
                if have.sum() < min_ov:
                    continue
                lp = np.log(p.freqs)
                diff = e[have] - lp[have]
                off = float(np.median(diff))
                inl = np.abs(diff - off) < 2.5 * thr
                if inl.sum() < min_ov:
                    continue
                if np.ptp(lp[have][inl]) < 3 * thr and inl.sum() * cfg.hop_s < 10:
                    continue  # too flat to tell co-movement from coincidence
                if np.std(diff[inl]) > thr:
                    continue
                off = float(np.mean(diff[inl]))
                take = np.zeros(len(p.frames), bool)
                have_idx = np.flatnonzero(have)
                if inl.mean() >= 0.9:
                    take[:] = True
                    take[have_idx[~inl]] = False
                else:
                    take[have_idx[inl]] = True
                lev = grp.est_level[p.frames]
                lmask = take & ~np.isnan(lev)
                loff = float(np.median(lev[lmask] - p.snr_db[lmask])) if lmask.any() else 0.0
                pool.remove(p)
                grp.add(_subset(p, take), off, loff)
                pool.extend(_split_runs(_subset(p, ~take), cfg))
                changed = True
                est = grp.est
        curves.append(grp.curve(cfg.hop_s))
    return curves


def extract_curves(audio: np.ndarray, sr: int, cfg: ExtractConfig | None = None) -> ExtractResult:
    cfg = cfg or ExtractConfig()
    x = np.asarray(audio, float)
    if sr != cfg.work_sr:
        g = np.gcd(int(sr), int(cfg.work_sr))
        x = resample_poly(x, cfg.work_sr // g, sr // g)
    x = x / (np.std(x) + 1e-12)
    times, freqs, db = _stft_db(x, cfg.work_sr, cfg)
    df = freqs[1] - freqs[0]
    white = db - _noise_floor(db, df, cfg)
    peaks = _pick_peaks(white, db, freqs, cfg)
    partials = _track(peaks, cfg)
    t_off = times[0]  # frame-centre time of frame 0

    moving = [p for p in _cut_at_collisions(partials, cfg) if np.ptp(np.log(p.freqs)) >= cfg.min_track_sweep]
    curves = []
    for c in _group_curves(moving, len(times), cfg):
        c.t = c.t + t_off
        dur = c.t[-1] - c.t[0]
        min_dur = cfg.min_curve_s if c.n_partials > 1 else max(cfg.min_curve_s, cfg.min_single_partial_s)
        min_sweep = cfg.min_curve_sweep if c.n_partials > 1 else max(cfg.min_curve_sweep, cfg.min_single_partial_sweep)
        if dur >= min_dur and c.sweep >= min_sweep:
            curves.append(c)
    curves.sort(key=lambda c: -float(np.sum(c.weight > 0) * c.sweep))
    return ExtractResult(curves, partials, times, freqs, white, cfg)
