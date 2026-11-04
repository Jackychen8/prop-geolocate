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

    lat, lon, radius = a.lat, a.lon, a.radius_km
    if not a.exact:
        # don't send your exact position to a third-party API: round to 0.1 deg
        # (~11 km) and widen the radius to compensate
        lat, lon, radius = round(lat, 1), round(lon, 1), radius + 12.0
        print(f"querying around rounded point {lat}, {lon} with radius {radius:.0f} km (use --exact to disable)")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    capture_live(lat, lon, radius, a.duration, a.interval, a.source, a.out)
    print(f"wrote {a.out}")


def _locate_clip(audio, tracks, start, center, cfg):
    from .extract import extract_curves
    from .locate import locate

    ex = extract_curves(audio, 16000)
    return ex, locate(ex.curves, tracks, start, center=center, cfg=cfg)


def cmd_field_test(a):
    """Real-recording test -> shareable figures + numbers."""
    from . import report as R
    from .adsb import load_tracks
    from .audio_io import creation_time, degrade, load_audio
    from .geo import haversine_m
    from .locate import LocateConfig

    private = not a.show_coords
    start = _parse_time(a.start) if a.start else creation_time(a.clip)
    if start is None:
        sys.exit("no --start and no creation_time metadata in the clip")
    truth = tuple(float(v) for v in a.truth.split(",")) if a.truth else None
    center = tuple(float(v) for v in a.center.split(",")) if a.center else None
    tracks = load_tracks(a.adsb)
    audio = load_audio(a.clip, 16000)
    cfg = LocateConfig(search_radius_m=a.radius, time_window_s=a.window, observer_alt_m=a.elev, use_level=a.level)
    print(f"clip {len(audio) / 16000:.0f} s, {len(tracks)} aircraft in ADS-B log, clock search +/-{a.window:.0f} s",
          file=sys.stderr)
    ex, res = _locate_clip(audio, tracks, start, center, cfg)
    print(f"extracted {len(ex.curves)} Doppler curve(s)", file=sys.stderr)
    out = R.ensure_dir(a.out)
    R.fig_spectrogram(ex, os.path.join(out, "1_spectrogram.png"))
    if not res.ok:
        print(json.dumps({"ok": False, "message": res.message}, indent=2))
        print(f"spectrogram written to {out}/1_spectrogram.png; check whether any curves are visible", file=sys.stderr)
        return
    R.fig_fit(res, os.path.join(out, "2_doppler_fit.png"), private=private)
    tw = (start - 120, start + len(audio) / 16000 + 120)
    R.fig_map(res, tracks, os.path.join(out, "3_map.png"), truth=truth, private=private, t_window=tw)

    robustness = []
    for chain in [c for c in (a.also_degrade or "").split(",") if c]:
        _, r2 = _locate_clip(degrade(audio, 16000, chain), tracks, start, center, cfg)
        row = {"chain": chain, "ok": r2.ok, "r95_m": round(r2.r95_m, 1) if r2.ok else None, "error_m": None}
        if r2.ok and truth:
            row["error_m"] = round(float(haversine_m(r2.lat, r2.lon, *truth)), 1)
        robustness.append(row)
        print(f"  {chain}: {row}", file=sys.stderr)
    summary = R.write_results(res, os.path.join(out, "result.json"), os.path.join(out, "result.md"),
                              truth=truth, private=private,
                              extra={"clip_s": round(len(audio) / 16000, 1), "robustness": robustness})
    print(json.dumps(summary, indent=2))
    print(f"\nfigures + result.md in {out}/ ({'privacy mode' if private else 'COORDINATES INCLUDED'})",
          file=sys.stderr)


def cmd_figures(a):
    """Simulation demo figures + benchmark chart for the README / posts."""
    import pandas as pd

    from . import report as R
    from .audio_io import degrade
    from .locate import LocateConfig
    from .simulate import adsb_feed, random_scenario, synthesize

    out = R.ensure_dir(a.out)
    scn = random_scenario(a.seed, n_aircraft=2)
    audio = degrade(synthesize(scn, snr_db=0.0, babble_db=3.0, hum=True), 16000, "aac_phone")
    tracks = adsb_feed(scn)
    cfg = LocateConfig(time_window_s=0.0, observer_alt_m=scn.obs_alt)
    ex, res = _locate_clip(audio, tracks, scn.t_start, (scn.lat + 0.012, scn.lon - 0.016), cfg)
    R.fig_spectrogram(ex, os.path.join(out, "sim_spectrogram.png"),
                      title="Simulated phone clip: 2 planes, speech, mains hum, AAC")
    R.fig_fit(res, os.path.join(out, "sim_doppler_fit.png"))
    R.fig_map(res, tracks, os.path.join(out, "sim_map.png"), truth=(scn.lat, scn.lon),
              t_window=(scn.t_start - 120, scn.t_start + 180))
    if os.path.exists(a.trials):
        df = pd.read_csv(a.trials)
        keep = [
            "baseline: 2 planes, clean, clock known", "3 planes", "1 plane", "30 s clip",
            "MP3 32k", "phone chain (HPF200+AGC+AAC64k)", "noise suppression (afftdn)", "Opus VoIP 16k",
            "speech babble + mains hum", "faint: SNR -10 dB", "very faint: SNR -20 dB",
            "clock unknown +/-30 s", "ADS-B 10 s sampling", "RPM drift 0.5%",
            "field-test path: no area hint, clock +/-15 s", "realistic: phone+babble+SNR0+clock+/-30s",
        ]
        keep = [k for k in keep if k in set(df["condition"])]
        R.fig_benchmark(df, os.path.join(out, "benchmark.png"), keep)
    print(f"wrote figures to {out}/")


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
    s.add_argument("--exact", action="store_true", help="send exact lat/lon to the API (default: rounded)")
    s.set_defaults(fn=cmd_capture)

    s = sub.add_parser("field-test", help="real recording -> shareable figures + numbers")
    s.add_argument("clip")
    s.add_argument("--adsb", required=True)
    s.add_argument("--truth", help="lat,lon where you actually stood (kept local; used only to compute error)")
    s.add_argument("--start", help="recording start (unix or ISO-8601); default: file metadata")
    s.add_argument("--window", type=float, default=15.0, help="+/- seconds of clock uncertainty")
    s.add_argument("--center", help="lat,lon of the rough search area (default: auto from low traffic)")
    s.add_argument("--radius", type=float, default=8000.0)
    s.add_argument("--elev", type=float, default=0.0,
                   help="ground height above the WGS84 ellipsoid (m) = elevation MSL + geoid height")
    s.add_argument("--level", action="store_true")
    s.add_argument("--also-degrade", default="mp3_32k,aac_phone,opus_voip_16k",
                   help="re-run on re-encoded copies of your clip (comma list, '' to skip)")
    s.add_argument("--show-coords", action="store_true", help="include lat/lon and callsigns (NOT for posting)")
    s.add_argument("--out", default="out/field")
    s.set_defaults(fn=cmd_field_test)

    s = sub.add_parser("figures", help="demo + benchmark figures for README/posts")
    s.add_argument("--seed", type=int, default=3)
    s.add_argument("--trials", default="out/benchmark/trials.csv")
    s.add_argument("--out", default="docs/img")
    s.set_defaults(fn=cmd_figures)

    a = p.parse_args(argv)
    np.seterr(all="ignore")
    a.fn(a)


if __name__ == "__main__":
    main()
