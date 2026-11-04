"""Command line: simulate | locate | benchmark | capture."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np


def _parse_time(s: str) -> float:
    try:
        return float(s)
    except ValueError:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def cmd_simulate(a):
    from .adsb import save_tracks
    from .audio_io import degrade, save_wav
    from .simulate import adsb_feed, random_scenario, synthesize

    os.makedirs(a.out, exist_ok=True)
    scn = random_scenario(a.seed, n_aircraft=a.aircraft, wind_speed=a.wind, duration=a.duration)
    audio = synthesize(scn, snr_db=a.snr, babble_db=a.babble, hum=a.hum)
    audio = degrade(audio, 16000, a.degrade)
    wav = os.path.join(a.out, "audio.wav")
    save_wav(wav, audio, 16000)
    # phone-like container with creation_time metadata, to exercise that path
    iso = datetime.fromtimestamp(scn.t_start, timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000000Z")
    m4a = os.path.join(a.out, "clip.m4a")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", wav, "-c:a", "aac", "-b:a", "128k",
                    "-metadata", f"creation_time={iso}", m4a], check=True)
    save_tracks(adsb_feed(scn, interval_s=a.adsb_dt), os.path.join(a.out, "adsb.csv"))
    with open(os.path.join(a.out, "truth.json"), "w") as f:
        f.write(scn.to_json())
    print(f"wrote {a.out}/audio.wav, clip.m4a (creation_time={iso}), adsb.csv, truth.json")
    print(f"truth: {scn.lat:.6f}, {scn.lon:.6f}  elev {scn.obs_alt:.0f} m")
    print(f"try:  python -m propgeo locate {m4a} --adsb {a.out}/adsb.csv --elev {scn.obs_alt:.0f} "
          f"--center {scn.lat + 0.015:.4f},{scn.lon - 0.01:.4f}")


def cmd_locate(a):
    from .adsb import load_tracks
    from .audio_io import creation_time, load_audio
    from .extract import extract_curves
    from .locate import LocateConfig, locate

    start = _parse_time(a.start) if a.start else creation_time(a.audio)
    if start is None:
        sys.exit("no --start given and no creation_time metadata in the file")
    audio = load_audio(a.audio, 16000)
    ex = extract_curves(audio, 16000)
    print(f"extracted {len(ex.curves)} Doppler curve(s) from {len(audio) / 16000:.1f} s of audio", file=sys.stderr)
    center = tuple(float(v) for v in a.center.split(",")) if a.center else None
    cfg = LocateConfig(search_radius_m=a.radius, time_window_s=a.window, observer_alt_m=a.elev,
                       use_level=a.level)
    res = locate(ex.curves, load_tracks(a.adsb), start, center=center, cfg=cfg)
    out = res.summary()
    if a.truth:
        from .geo import haversine_m
        from .simulate import Scenario

        scn = Scenario.from_json(open(a.truth).read())
        out["error_m"] = round(float(haversine_m(res.lat, res.lon, scn.lat, scn.lon)), 1)
    print(json.dumps(out, indent=2))


def cmd_benchmark(a):
    from . import benchmark as B

    names = list(B.CONDITIONS)
    if a.conditions:
        keys = [k.strip().lower() for k in a.conditions.split(",")]
        names = [n for n in names if any(k in n.lower() for k in keys)]
    print(f"running {len(names)} condition(s) x {a.trials} trials", file=sys.stderr)
    df = B.run(names, a.trials, a.workers, log=lambda m: print(m, file=sys.stderr))
    summ = B.summarize(df)
    os.makedirs(a.out, exist_ok=True)
    df.to_csv(os.path.join(a.out, "trials.csv"), index=False)
    md = summ.to_markdown(index=False)
    with open(os.path.join(a.out, "summary.md"), "w") as f:
        f.write(md + "\n")
    print(md)


def cmd_capture(a):
    from .adsb import capture_live

    capture_live(a.lat, a.lon, a.radius_km, a.duration, a.interval, a.source, a.out)
    print(f"wrote {a.out}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="propgeo", description="Propeller-Doppler acoustic geolocation")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("simulate", help="generate a synthetic clip + ADS-B + ground truth")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--aircraft", type=int, default=2)
    s.add_argument("--duration", type=float, default=60.0)
    s.add_argument("--snr", type=float, default=10.0)
    s.add_argument("--babble", type=float, default=None, help="speech babble level dB re aircraft")
    s.add_argument("--hum", action="store_true")
    s.add_argument("--wind", type=float, default=0.0)
    s.add_argument("--adsb-dt", type=float, default=1.0)
    s.add_argument("--degrade", default="clean")
    s.add_argument("--out", default="out/sim")
    s.set_defaults(fn=cmd_simulate)

    s = sub.add_parser("locate", help="locate a recording")
    s.add_argument("audio")
    s.add_argument("--adsb", required=True, help="ADS-B CSV (OpenSky state-vector layout)")
    s.add_argument("--start", help="recording start (unix or ISO-8601); default: file metadata")
    s.add_argument("--window", type=float, default=0.0, help="+/- seconds of clock uncertainty")
    s.add_argument("--center", help="lat,lon of rough search area (default: auto from traffic)")
    s.add_argument("--radius", type=float, default=8000.0)
    s.add_argument("--elev", type=float, default=0.0, help="observer height in the ADS-B altitude datum (m)")
    s.add_argument("--level", action="store_true", help="also fit the loudness envelope (off by default)")
    s.add_argument("--truth", help="truth.json from `simulate`, to report error")
    s.set_defaults(fn=cmd_locate)

    s = sub.add_parser("benchmark", help="Monte-Carlo accuracy/robustness test")
    s.add_argument("--trials", type=int, default=20)
    s.add_argument("--workers", type=int, default=None)
    s.add_argument("--conditions", help="comma-separated substrings to select conditions")
    s.add_argument("--out", default="out/benchmark")
    s.set_defaults(fn=cmd_benchmark)

    s = sub.add_parser("capture", help="log live ADS-B around you while you record a test clip")
    s.add_argument("--lat", type=float, required=True)
    s.add_argument("--lon", type=float, required=True)
    s.add_argument("--radius-km", type=float, default=15.0)
    s.add_argument("--duration", type=float, default=180.0)
    s.add_argument("--interval", type=float, default=2.0)
    s.add_argument("--source", choices=["adsb.lol", "opensky"], default="adsb.lol")
    s.add_argument("--out", default="out/live_adsb.csv")
    s.set_defaults(fn=cmd_capture)

    a = p.parse_args(argv)
    np.seterr(all="ignore")
    a.fn(a)


if __name__ == "__main__":
    main()
