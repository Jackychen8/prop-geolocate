import io

import numpy as np
import pytest

from propgeo.adsb import Track, build_kinematics, load_tracks, save_tracks
from propgeo.audio_io import degrade, have_ffmpeg
from propgeo.extract import extract_curves
from propgeo.geo import LocalFrame, haversine_m
from propgeo.locate import LocateConfig, error_m, locate
from propgeo.physics import Kinematics, propagate
from propgeo.simulate import adsb_feed, random_scenario, synthesize, truth_doppler

SR = 16000


# ---------------------------------------------------------------- geometry
def test_enu_roundtrip_sub_millimetre():
    fr = LocalFrame(37.4, -122.1, 0.0)
    e, n, u = np.array([0.0, 8000.0, -5000.0]), np.array([0.0, -3000.0, 7000.0]), np.array([0.0, 900.0, 11000.0])
    lat, lon, h = fr.to_geodetic(e, n, u)
    e2, n2, u2 = fr.to_enu(lat, lon, h)
    assert np.max(np.abs(np.r_[e - e2, n - n2, u - u2])) < 1e-3


# ---------------------------------------------------------------- physics
def _straight_pass(speed=60.0, alt=500.0, offset=800.0):
    t = np.arange(-200, 200, 0.05)
    pos = np.stack([speed * t, np.full_like(t, offset), np.full_like(t, alt)], 1)
    vel = np.tile([speed, 0.0, 0.0], (len(t), 1))
    return Kinematics(t[0], 0.05, pos, vel)


def test_doppler_sign_and_emission_time():
    kin = _straight_pass()
    t_rx = np.array([-30.0, 30.0])
    ld, r, tau, valid = propagate(kin, np.zeros(3), t_rx, c=343.0)
    assert valid.all()
    assert ld[0] > 0 > ld[1]  # approaching: higher pitch; receding: lower
    np.testing.assert_allclose(t_rx - tau, r / 343.0, atol=1e-6)
    # far approach tends to c/(c - v)
    ld_far, *_ = propagate(kin, np.zeros(3), np.array([-150.0]), c=343.0)
    assert abs(np.exp(ld_far[0]) - 343 / (343 - 60)) < 0.01


# ---------------------------------------------------------------- ADS-B
def test_adsb_csv_roundtrip_and_aliases():
    tr = Track("abc123", "N1", np.arange(10.0), np.linspace(37, 37.01, 10), np.full(10, -122.0), np.full(10, 500.0))
    buf = io.StringIO()
    save_tracks([tr], buf)
    back = load_tracks(io.StringIO(buf.getvalue()))
    assert len(back) == 1 and back[0].icao24 == "abc123"
    np.testing.assert_allclose(back[0].alt, 500.0)
    # readsb-style columns, altitude in feet
    csv = "timestamp,hex,latitude,longitude,alt_baro\n0,ABC,37,-122,1000\n1,ABC,37.001,-122,1000\n"
    t2 = load_tracks(io.StringIO(csv))
    assert t2[0].icao24 == "abc" and abs(t2[0].alt[0] - 304.8) < 1e-6


def test_spline_velocity_from_noisy_positions():
    scn = random_scenario(5, n_aircraft=1, n_distractors=0)
    ac = scn.aircraft[0]
    tr = adsb_feed(scn, interval_s=1.0, pos_noise_m=10.0)[0]
    kin = build_kinematics(tr, scn.frame, scn.t_start, scn.t_start + 60)
    _, v, valid = kin.at(scn.t_start + np.arange(0, 60, 1.0))
    err = np.linalg.norm(v[valid, :2] - np.asarray(ac.vel)[:2], axis=1)
    assert np.median(err) < 1.5  # m/s; raw differencing would be ~14 m/s


# ---------------------------------------------------------------- extraction
def test_extraction_matches_truth_and_harmonic_numbers():
    scn = random_scenario(0, n_aircraft=2)
    ex = extract_curves(synthesize(scn), SR)
    aud = [a for a in scn.aircraft if a.audible]
    good = 0
    for c in ex.curves[:2]:
        best = min(aud, key=lambda a: np.std(c.logf - truth_doppler(scn, a, c.t)[0]))
        ld, _ = truth_doppler(scn, best, c.t)
        r = c.logf - ld
        harm = np.exp(np.median(r)) / best.f0
        if np.std(r) < 0.003 and abs(harm - round(harm)) < 0.02:
            good += 1
    assert good == 2


def test_stationary_tones_are_ignored():
    t = np.arange(30 * SR) / SR
    hum = sum(np.sin(2 * np.pi * 60 * k * t) / k for k in (1, 2, 3, 5)) + 0.1 * np.random.default_rng(0).standard_normal(len(t))
    assert extract_curves(hum.astype(np.float32), SR).curves == []


# ---------------------------------------------------------------- end to end
def _run(seed, n_aircraft=2, degrade_name="clean", window=0.0, time_err=0.0, **synth):
    scn = random_scenario(seed, n_aircraft=n_aircraft)
    audio = degrade(synthesize(scn, **synth), SR, degrade_name)
    ex = extract_curves(audio, SR)
    center = (scn.lat + 0.012, scn.lon - 0.015)  # rough area, ~2 km off
    cfg = LocateConfig(time_window_s=window, observer_alt_m=scn.obs_alt)
    res = locate(ex.curves, adsb_feed(scn), scn.t_start + time_err, center=center, cfg=cfg)
    return scn, res


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_two_planes_clean_within_50m(seed):
    scn, res = _run(seed)
    assert res.ok and not res.ambiguous
    assert error_m(res, scn.lat, scn.lon) < 50


@pytest.mark.skipif(not have_ffmpeg(), reason="ffmpeg required")
@pytest.mark.parametrize("chain", ["mp3_64k", "aac_phone", "denoise"])
def test_survives_lossy_chains_within_100m(chain):
    scn, res = _run(1, degrade_name=chain, snr_db=0.0)
    assert res.ok
    assert error_m(res, scn.lat, scn.lon) < 100


def test_unknown_clock_is_recovered():
    scn, res = _run(2, window=30.0, time_err=-17.0)
    assert res.ok and error_m(res, scn.lat, scn.lon) < 100
    assert abs(res.time_offset_s - 17.0) < 1.5


def test_single_plane_is_reported_ambiguous_not_overclaimed():
    """One straight pass can't distinguish mirror images / other aircraft:
    the solver must flag it, and the truth must be among its candidates."""
    hits = 0
    for seed in range(4):
        scn, res = _run(seed, n_aircraft=1)
        assert res.ok
        cands = [res, *res.alternates]
        best = min(haversine_m(c.lat, c.lon, scn.lat, scn.lon) for c in cands)
        if res.ambiguous and best < 100:
            hits += 1
    assert hits >= 3
