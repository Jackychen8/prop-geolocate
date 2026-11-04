"""WGS84 geodetic <-> local East/North/Up conversions.

Everything inside the solver works in a local ENU frame (metres). We use the
exact ECEF route rather than a flat-earth approximation because a 10 km
search radius with equirectangular projection already costs ~15 m of
distortion, which is not negligible against a 100 m accuracy target.
"""
from __future__ import annotations

import numpy as np

WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)
R_MEAN = 6371008.8


def geodetic_to_ecef(lat, lon, h):
    lat = np.deg2rad(np.asarray(lat, float))
    lon = np.deg2rad(np.asarray(lon, float))
    h = np.asarray(h, float)
    s, c = np.sin(lat), np.cos(lat)
    n = WGS84_A / np.sqrt(1.0 - WGS84_E2 * s * s)
    x = (n + h) * c * np.cos(lon)
    y = (n + h) * c * np.sin(lon)
    z = (n * (1.0 - WGS84_E2) + h) * s
    return x, y, z


def ecef_to_geodetic(x, y, z):
    x, y, z = (np.asarray(v, float) for v in (x, y, z))
    lon = np.arctan2(y, x)
    p = np.hypot(x, y)
    lat = np.arctan2(z, p * (1.0 - WGS84_E2))
    h = np.zeros_like(p)
    for _ in range(6):
        s = np.sin(lat)
        n = WGS84_A / np.sqrt(1.0 - WGS84_E2 * s * s)
        h = p / np.cos(lat) - n
        lat = np.arctan2(z, p * (1.0 - WGS84_E2 * n / (n + h)))
    return np.rad2deg(lat), np.rad2deg(lon), h


class LocalFrame:
    """East/North/Up tangent-plane frame anchored at (lat0, lon0, h0)."""

    def __init__(self, lat0: float, lon0: float, h0: float = 0.0):
        self.lat0, self.lon0, self.h0 = float(lat0), float(lon0), float(h0)
        self._p0 = np.array(geodetic_to_ecef(lat0, lon0, h0), float)
        la, lo = np.deg2rad(lat0), np.deg2rad(lon0)
        sl, cl, so, co = np.sin(la), np.cos(la), np.sin(lo), np.cos(lo)
        # rows: east, north, up
        self._rot = np.array(
            [[-so, co, 0.0], [-sl * co, -sl * so, cl], [cl * co, cl * so, sl]]
        )

    def to_enu(self, lat, lon, h):
        x, y, z = geodetic_to_ecef(lat, lon, h)
        d = np.stack([x, y, z], axis=-1) - self._p0
        enu = d @ self._rot.T
        return enu[..., 0], enu[..., 1], enu[..., 2]

    def to_geodetic(self, e, n, u):
        enu = np.stack(np.broadcast_arrays(*(np.asarray(v, float) for v in (e, n, u))), axis=-1)
        d = enu @ self._rot + self._p0
        return ecef_to_geodetic(d[..., 0], d[..., 1], d[..., 2])

    def ground_up(self, e, n, h_obs: float):
        """Up-coordinate of a point at ellipsoidal height h_obs above (e, n).

        Uses the spherical curvature drop, accurate to centimetres at 10 km.
        """
        e, n = np.asarray(e, float), np.asarray(n, float)
        return (h_obs - self.h0) - (e * e + n * n) / (2.0 * R_MEAN)


def haversine_m(lat1, lon1, lat2, lon2):
    p1, p2 = np.deg2rad(lat1), np.deg2rad(lat2)
    dp = p2 - p1
    dl = np.deg2rad(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R_MEAN * np.arcsin(np.sqrt(a))
