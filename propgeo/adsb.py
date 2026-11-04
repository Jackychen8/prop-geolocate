"""ADS-B flight-track loading, smoothing, and live capture.

Input format is the OpenSky `state_vectors_data4` layout (one row per
position report) but common aliases from dump1090/readsb/adsb.lol exports
are accepted too:

    time, icao24, lat, lon, geoaltitude|baroaltitude (metres), callsign, onground

Altitudes in feet are accepted in columns named alt_ft / alt_geom_ft / alt_baro_ft.
"""
from __future__ import annotations

import csv
import io
import json
import time as _time
import urllib.request
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.interpolate import make_smoothing_spline

from .geo import LocalFrame
from .physics import Kinematics

FT = 0.3048

_ALIASES = {
    "time": ["time", "timestamp", "t", "unix", "unixtime", "ts", "time_position"],
    "icao24": ["icao24", "icao", "hex", "icao_address", "address"],
    "lat": ["lat", "latitude"],
    "lon": ["lon", "lng", "long", "longitude"],
    "geoaltitude": ["geoaltitude", "geo_altitude", "alt_geom_m", "altitude_geo"],
    "baroaltitude": ["baroaltitude", "baro_altitude", "alt_baro_m", "altitude"],
    "alt_geom_ft": ["alt_geom_ft", "alt_geom"],
    "alt_baro_ft": ["alt_baro_ft", "alt_baro", "alt_ft", "altitude_ft"],
    "callsign": ["callsign", "flight"],
    "onground": ["onground", "on_ground"],
}


@dataclass
class Track:
    icao24: str
    callsign: str
    t: np.ndarray  # unix seconds
    lat: np.ndarray
    lon: np.ndarray
    alt: np.ndarray  # metres (geometric if available, else barometric)


def _pick(df: pd.DataFrame, key: str):
    cols = {c.lower().strip(): c for c in df.columns}
    for alias in _ALIASES[key]:
        if alias in cols:
            return df[cols[alias]]
    return None


def tracks_from_dataframe(df: pd.DataFrame) -> list[Track]:
    t = _pick(df, "time")
    if t is None:
        raise ValueError("ADS-B table needs a time column")
    if not np.issubdtype(t.dtype, np.number):
        t = pd.to_datetime(t, utc=True).astype("int64") / 1e9
    icao = _pick(df, "icao24")
    lat, lon = _pick(df, "lat"), _pick(df, "lon")
    if icao is None or lat is None or lon is None:
        raise ValueError("ADS-B table needs icao24, lat and lon columns")

    alt = None
    for key, scale in (("geoaltitude", 1.0), ("alt_geom_ft", FT), ("baroaltitude", 1.0), ("alt_baro_ft", FT)):
        col = _pick(df, key)
        if col is not None:
            vals = pd.to_numeric(col, errors="coerce") * scale
            alt = vals if alt is None else alt.fillna(vals)
    if alt is None:
        raise ValueError("ADS-B table needs an altitude column")

    cs = _pick(df, "callsign")
    og = _pick(df, "onground")
    clean = pd.DataFrame(
        {
            "t": pd.to_numeric(t, errors="coerce"),
            "icao24": icao.astype(str).str.strip().str.lower(),
            "lat": pd.to_numeric(lat, errors="coerce"),
            "lon": pd.to_numeric(lon, errors="coerce"),
            "alt": alt,
            "callsign": cs.fillna("").astype(str).str.strip() if cs is not None else "",
        }
    )
    if og is not None:
        on_ground = og.astype(str).str.lower().isin(["true", "1", "yes"])
        clean = clean[~on_ground.values]
    clean = clean.dropna(subset=["t", "lat", "lon", "alt"])

    tracks = []
    for icao24, g in clean.groupby("icao24"):
        g = g.sort_values("t").drop_duplicates("t")
        if len(g) < 2:
            continue
        callsign = next((c for c in g["callsign"] if c), "")
        tracks.append(
            Track(icao24, callsign, g["t"].to_numpy(float), g["lat"].to_numpy(float),
                  g["lon"].to_numpy(float), g["alt"].to_numpy(float))
        )
    return tracks


def load_tracks(src) -> list[Track]:
    """Load from a path, file-like object, or CSV text."""
    if isinstance(src, str) and "\n" in src:
        src = io.StringIO(src)
    return tracks_from_dataframe(pd.read_csv(src))


def tracks_to_dataframe(tracks: list[Track]) -> pd.DataFrame:
    rows = []
    for tr in tracks:
        for i in range(len(tr.t)):
            rows.append((tr.t[i], tr.icao24, tr.lat[i], tr.lon[i], tr.alt[i], tr.alt[i], tr.callsign, False))
    return pd.DataFrame(
        rows, columns=["time", "icao24", "lat", "lon", "geoaltitude", "baroaltitude", "callsign", "onground"]
    )


def save_tracks(tracks: list[Track], path) -> None:
    tracks_to_dataframe(tracks).to_csv(path, index=False, float_format="%.7f")


def build_kinematics(
    track: Track, frame: LocalFrame, t_min: float, t_max: float, dt: float = 0.1, margin: float = 90.0
) -> Kinematics | None:
    """Smooth a track (noisy positions -> position + velocity) over [t_min, t_max].

    ADS-B gives positions only every ~0.5-10 s with ~5-30 m noise; finite
    differences would give velocity errors of 10+ m/s, which is fatal for a
    Doppler model. A GCV-tuned smoothing spline per axis gives sub-m/s
    velocities on straight-ish legs.
    """
    sel = (track.t >= t_min - margin) & (track.t <= t_max + margin)
    t = track.t[sel]
    if len(t) < 2 or t[0] > t_max or t[-1] < t_min:
        # no overlap with the window we need; partial coverage is allowed and
        # uncovered emission times are flagged invalid by Kinematics.at()
        return None
    e, n, u = frame.to_enu(track.lat[sel], track.lon[sel], track.alt[sel])
    xyz = np.stack([e, n, u], axis=1)
    tg = np.arange(t[0], t[-1] + 1e-9, dt)
    if len(tg) < 2:
        return None
    if len(t) >= 5:
        pos = np.empty((len(tg), 3))
        vel = np.empty((len(tg), 3))
        for k in range(3):
            try:
                spl = make_smoothing_spline(t - t[0], xyz[:, k])
            except Exception:
                spl = None
            if spl is None:
                pos[:, k] = np.interp(tg, t, xyz[:, k])
                vel[:, k] = np.gradient(pos[:, k], dt)
            else:
                pos[:, k] = spl(tg - t[0])
                vel[:, k] = spl.derivative()(tg - t[0])
    else:
        pos = np.stack([np.interp(tg, t, xyz[:, k]) for k in range(3)], axis=1)
        vel = np.gradient(pos, dt, axis=0)
    return Kinematics(float(tg[0]), dt, pos, vel, icao24=track.icao24, callsign=track.callsign)


# --------------------------------------------------------------------------
# Live capture (for recording your own test clips)
# --------------------------------------------------------------------------

def _get_json(url: str, auth: tuple[str, str] | None = None, timeout: float = 15.0):
    req = urllib.request.Request(url, headers={"User-Agent": "propgeo/0.1 (research)"})
    if auth:
        import base64

        tok = base64.b64encode(f"{auth[0]}:{auth[1]}".encode()).decode()
        req.add_header("Authorization", f"Basic {tok}")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def _poll_adsb_lol(lat: float, lon: float, radius_km: float):
    nm = max(1, int(round(radius_km / 1.852)))
    data = _get_json(f"https://api.adsb.lol/v2/point/{lat:.5f}/{lon:.5f}/{nm}")
    now = float(data.get("now", _time.time() * 1000)) / 1000.0
    rows = []
    for ac in data.get("ac", []):
        if "lat" not in ac or "lon" not in ac:
            continue
        alt_geom = ac.get("alt_geom")
        alt_baro = ac.get("alt_baro")
        if alt_baro == "ground":
            continue
        alt_ft = alt_geom if isinstance(alt_geom, (int, float)) else alt_baro
        if not isinstance(alt_ft, (int, float)):
            continue
        rows.append(
            {
                "time": now - float(ac.get("seen_pos", 0.0)),
                "icao24": ac.get("hex", "").lower(),
                "lat": ac["lat"],
                "lon": ac["lon"],
                "geoaltitude": alt_ft * FT,
                "baroaltitude": (alt_baro * FT) if isinstance(alt_baro, (int, float)) else None,
                "callsign": (ac.get("flight") or "").strip(),
                "onground": False,
                "category": ac.get("category", ""),
                "type": ac.get("t", ""),
            }
        )
    return rows


def _poll_opensky(lat: float, lon: float, radius_km: float, auth=None):
    dlat = radius_km / 111.0
    dlon = radius_km / (111.0 * max(0.1, np.cos(np.deg2rad(lat))))
    url = (
        "https://opensky-network.org/api/states/all?"
        f"lamin={lat - dlat:.4f}&lamax={lat + dlat:.4f}&lomin={lon - dlon:.4f}&lomax={lon + dlon:.4f}"
    )
    data = _get_json(url, auth)
    rows = []
    for s in data.get("states") or []:
        if s[5] is None or s[6] is None or s[8]:
            continue
        alt = s[13] if s[13] is not None else s[7]
        if alt is None:
            continue
        rows.append(
            {
                "time": s[3] or s[4],
                "icao24": s[0],
                "lat": s[6],
                "lon": s[5],
                "geoaltitude": s[13],
                "baroaltitude": s[7],
                "callsign": (s[1] or "").strip(),
                "onground": False,
            }
        )
    return rows


def capture_live(
    lat: float,
    lon: float,
    radius_km: float = 15.0,
    duration_s: float = 120.0,
    interval_s: float = 2.0,
    source: str = "adsb.lol",
    out_csv=None,
    auth=None,
    log=print,
) -> pd.DataFrame:
    """Poll a live ADS-B feed for the area while you record a clip.

    `adsb.lol` needs no key and refreshes ~1 s. OpenSky anonymous access is
    limited to 10 s resolution and a daily credit budget, so use interval>=10.
    """
    poll = _poll_adsb_lol if source == "adsb.lol" else (lambda a, b, c: _poll_opensky(a, b, c, auth))
    rows, seen = [], set()
    t_end = _time.time() + duration_s
    while _time.time() < t_end:
        try:
            for r in poll(lat, lon, radius_km):
                key = (r["icao24"], round(float(r["time"]), 2))
                if key not in seen:
                    seen.add(key)
                    rows.append(r)
            log(f"[{_time.strftime('%H:%M:%S')}] {len(rows)} reports, {len({k[0] for k in seen})} aircraft")
        except Exception as exc:  # network hiccups shouldn't kill a capture
            log(f"poll failed: {exc}")
        _time.sleep(interval_s)
    df = pd.DataFrame(rows)
    if out_csv is not None:
        df.to_csv(out_csv, index=False, quoting=csv.QUOTE_MINIMAL)
    return df
