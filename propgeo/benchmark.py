"""Monte-Carlo test of the claim: '<60 s of audio -> location within ~100 m,
surviving phone compression / noise suppression / MP3'.

Every condition runs the same seeds, so rows are directly comparable. Each
trial: random geometry -> synthesize -> degrade -> extract -> locate, with the
searcher given only a rough area (centre offset ~2 km from truth, 8 km radius)
and ADS-B that includes distractor aircraft not present in the audio.
"""
from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

from .audio_io import degrade
from .extract import extract_curves
from .geo import haversine_m
from .locate import LocateConfig, error_m, locate
from .simulate import adsb_feed, random_scenario, synthesize

BASE = dict(
    n_aircraft=2, snr_db=10.0, babble_db=None, hum=False, degrade="clean", time_err=0.0, window=0.0,
    adsb_dt=1.0, adsb_noise=8.0, wind=0.0, drift=0.001, elev_err=0.0, use_level=False, center_err_m=2000.0,
    duration=60.0,
)

CONDITIONS: dict[str, dict] = {
    "baseline: 2 planes, clean, clock known": {},
    "1 plane": dict(n_aircraft=1),
    "1 plane + level cue": dict(n_aircraft=1, use_level=True),
    "3 planes": dict(n_aircraft=3),
    "30 s clip": dict(duration=30.0),
    "faint: SNR -10 dB": dict(snr_db=-10.0),
    "very faint: SNR -20 dB": dict(snr_db=-20.0),
    "speech babble + mains hum": dict(babble_db=6.0, hum=True),
    "MP3 128k": dict(degrade="mp3_128k"),
    "MP3 32k": dict(degrade="mp3_32k"),
    "phone chain (HPF200+AGC+AAC64k)": dict(degrade="aac_phone"),
    "noise suppression (afftdn)": dict(degrade="denoise"),
    "Opus VoIP 16k": dict(degrade="opus_voip_16k"),
    "clock unknown +/-30 s": dict(time_err=25.0, window=30.0),
    "ADS-B 10 s sampling": dict(adsb_dt=10.0),
    "ADS-B pos noise 30 m": dict(adsb_noise=30.0),
    "unmodelled wind 6 m/s": dict(wind=6.0),
    "RPM drift 0.5%": dict(drift=0.005),
    "observer elevation off by 50 m": dict(elev_err=50.0),
    "2 planes + level cue": dict(use_level=True),
    "realistic: phone+babble+SNR0+clock+/-30s": dict(
        degrade="aac_phone", babble_db=3.0, snr_db=0.0, time_err=25.0, window=30.0, drift=0.002),
}


def run_trial(cond: str, params: dict, seed: int) -> dict:
    p = {**BASE, **params}
    t0 = time.time()
    scn = random_scenario(seed, n_aircraft=p["n_aircraft"], wind_speed=p["wind"], duration=p["duration"])
    sr = 16000
    audio = synthesize(scn, sr=sr, snr_db=p["snr_db"], rpm_drift=p["drift"], babble_db=p["babble_db"], hum=p["hum"])
    audio = degrade(audio, sr, p["degrade"])
    tracks = adsb_feed(scn, interval_s=p["adsb_dt"], pos_noise_m=p["adsb_noise"])
    rng = np.random.default_rng(seed + 77)
    toff = rng.uniform(-p["time_err"], p["time_err"]) if p["time_err"] else 0.0
    ang = rng.uniform(0, 2 * np.pi)
    cdist = abs(rng.normal(0, p["center_err_m"]))
    cdist = min(cdist, 5000.0)
    dlat = cdist * np.cos(ang) / 111_320.0
    dlon = cdist * np.sin(ang) / (111_320.0 * np.cos(np.deg2rad(scn.lat)))
    center = (scn.lat + dlat, scn.lon + dlon)
    cfg = LocateConfig(time_window_s=p["window"], observer_alt_m=scn.obs_alt + p["elev_err"], use_level=p["use_level"])
    ex = extract_curves(audio, sr)
    res = locate(ex.curves, tracks, scn.t_start + toff, center=center, cfg=cfg)
    row = dict(condition=cond, seed=seed, n_curves=len(ex.curves), ok=res.ok, message=res.message,
               elapsed_s=time.time() - t0)
    if res.ok:
        err = error_m(res, scn.lat, scn.lon)
        cand_err = min(float(haversine_m(c.lat, c.lon, scn.lat, scn.lon)) for c in [res, *res.alternates])
        audible = {a.icao24 for a in scn.aircraft if a.audible}
        row.update(error_m=err, r95_m=res.r95_m, ambiguous=res.ambiguous, best_candidate_err_m=cand_err,
                   clock_err_s=abs(res.time_offset_s + toff),
                   wrong_aircraft=any(f.icao24 and f.icao24 not in audible for f in res.fits))
    return row


def run(conditions: list[str] | None = None, trials: int = 20, workers: int | None = None, log=print) -> pd.DataFrame:
    names = conditions or list(CONDITIONS)
    workers = workers or max(1, (os.cpu_count() or 2) - 2)
    jobs = [(n, CONDITIONS[n], s) for n in names for s in range(trials)]
    rows = []
    with ProcessPoolExecutor(workers) as ex:
        futs = {ex.submit(run_trial, *j): j for j in jobs}
        for i, f in enumerate(as_completed(futs), 1):
            try:
                rows.append(f.result())
            except Exception as exc:  # record crashes as failures, don't hide them
                n, _, s = futs[f]
                rows.append(dict(condition=n, seed=s, ok=False, message=f"crash: {exc!r}"))
            if i % 20 == 0 or i == len(jobs):
                log(f"  {i}/{len(jobs)} trials done")
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for name in dict.fromkeys(df["condition"]):
        g = df[df["condition"] == name]
        okg = g[g["ok"] == True]  # noqa: E712
        err = okg["error_m"] if "error_m" in okg else pd.Series(dtype=float)
        n = len(g)
        out.append({
            "condition": name,
            "trials": n,
            "within 100 m": f"{100 * (err < 100).sum() / n:.0f}%",
            "median err (m)": f"{err.median():.0f}" if len(err) else "-",
            "p90 err (m)": f"{err.quantile(0.9):.0f}" if len(err) else "-",
            "flagged ambiguous": f"{100 * okg['ambiguous'].sum() / n:.0f}%" if len(okg) else "-",
            "truth in candidates (<100 m)": f"{100 * (okg['best_candidate_err_m'] < 100).sum() / n:.0f}%" if len(okg) else "-",
            "err <= reported r95": f"{100 * (okg['error_m'] <= okg['r95_m']).sum() / max(len(okg), 1):.0f}%" if len(okg) else "-",
            "no solution": f"{100 * (n - len(okg)) / n:.0f}%",
            "s/trial": f"{g['elapsed_s'].median():.1f}" if "elapsed_s" in g else "-",
        })
    return pd.DataFrame(out)
