"""Streamlit UI:  streamlit run app.py"""
from __future__ import annotations

import io
import os
import tempfile
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from propgeo.adsb import load_tracks, tracks_to_dataframe
from propgeo.audio_io import DEGRADATIONS, creation_time, degrade, load_audio
from propgeo.extract import extract_curves
from propgeo.geo import haversine_m
from propgeo.locate import LocateConfig, locate
from propgeo.simulate import adsb_feed, random_scenario, synthesize

SR = 16000
st.set_page_config(page_title="Prop-Doppler geolocation", layout="wide")
ss = st.session_state

st.title("Propeller-Doppler acoustic geolocation")
st.caption(
    "Extracts the Doppler pitch curves of passing propeller aircraft from a clip's background audio, "
    "then searches for the microphone position whose predicted Doppler curves (from ADS-B tracks) match. "
    "Use the simulator for ground-truth tests, or upload a real clip plus an ADS-B log of the same time and area."
)

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    mode = st.radio("Input", ["Simulated scenario", "My recording"], horizontal=True)
    if mode == "Simulated scenario":
        seed = st.number_input("Scenario seed", 0, 10_000, 3)
        n_ac = st.slider("Audible propeller planes", 1, 3, 2)
        dur = st.slider("Clip length (s)", 20, 90, 60, 5)
        snr = st.slider("SNR (aircraft vs background, dB)", -20, 20, 5)
        deg = st.selectbox("Audio chain", list(DEGRADATIONS), index=list(DEGRADATIONS).index("aac_phone"))
        babble = st.checkbox("Speech babble + mains hum", value=True)
        clock_err = st.slider("Clip clock error (s)", 0, 30, 0)
        adsb_dt = st.select_slider("ADS-B report interval (s)", [0.5, 1.0, 2.0, 5.0, 10.0], 1.0)
        if st.button("Generate scenario", type="primary", use_container_width=True):
            with st.spinner("Synthesising audio + ADS-B…"):
                scn = random_scenario(int(seed), n_aircraft=n_ac, duration=float(dur))
                audio = synthesize(scn, snr_db=float(snr), babble_db=3.0 if babble else None, hum=babble)
                audio = degrade(audio, SR, deg)
                rng = np.random.default_rng(int(seed) + 5)
                toff = float(rng.uniform(-clock_err, clock_err)) if clock_err else 0.0
                ss.update(
                    audio=audio, tracks=adsb_feed(scn, interval_s=adsb_dt), truth=(scn.lat, scn.lon),
                    start=scn.t_start + toff, window=float(max(clock_err * 1.2, 0)),
                    center=(scn.lat + 0.012, scn.lon - 0.016), elev=scn.obs_alt, scenario=scn,
                    extract=None, result=None,
                )
    else:
        up = st.file_uploader("Audio or video clip", type=["wav", "mp3", "m4a", "mp4", "mov", "aac", "ogg", "flac", "webm"])
        adsb_up = st.file_uploader("ADS-B CSV (OpenSky state-vector layout)", type=["csv"])
        st.caption("No ADS-B log? Record outside while running `python -m propgeo capture --lat .. --lon ..`.")
        meta_t = None
        if up is not None:
            if ss.get("upload_name") != up.name:
                with tempfile.NamedTemporaryFile(suffix=os.path.splitext(up.name)[1], delete=False) as f:
                    f.write(up.getbuffer())
                    path = f.name
                ss.update(audio=load_audio(path, SR), upload_name=up.name, meta_t=creation_time(path),
                          extract=None, result=None, truth=None, scenario=None)
                os.unlink(path)
            meta_t = ss.get("meta_t")
        default_t = datetime.fromtimestamp(meta_t, timezone.utc).isoformat() if meta_t else ""
        t_txt = st.text_input("Recording start (ISO-8601 or unix)", default_t,
                              help="Pre-filled from file metadata when present. Some phones store the END time.")
        window = st.slider("Clock uncertainty ± (s)", 0, 60, 15 if meta_t else 30)
        c_txt = st.text_input("Rough area centre (lat, lon)", "", help="Blank = centre on low-flying traffic")
        radius = st.slider("Search radius (km)", 2, 20, 8)
        elev = st.number_input("Ground elevation (m, same datum as ADS-B altitude)", -100.0, 5000.0, 0.0)
        if adsb_up is not None:
            ss["tracks"] = load_tracks(io.StringIO(adsb_up.getvalue().decode()))
        try:
            ss["start"] = float(t_txt) if t_txt.replace(".", "").isdigit() else datetime.fromisoformat(
                t_txt.replace("Z", "+00:00")).timestamp()
        except ValueError:
            ss["start"] = None
        ss["window"] = float(window)
        ss["center"] = tuple(float(v) for v in c_txt.split(",")) if c_txt.strip() else None
        ss["elev"] = float(elev)
        ss["radius"] = radius * 1000.0

    st.divider()
    use_level = st.checkbox("Use loudness envelope as extra cue", value=False,
                            help="Off by default: in the benchmark it did not help (see Benchmark tab).")

tab1, tab2, tab3 = st.tabs(["1 · Audio → Doppler curves", "2 · Location", "3 · Benchmark"])

if ss.get("audio") is None:
    tab1.info("Generate a scenario or upload a clip in the sidebar.")
    st.stop()

# ------------------------------------------------------------------ tab 1
with tab1:
    audio = ss["audio"]
    buf = io.BytesIO()
    from scipy.io import wavfile

    wavfile.write(buf, SR, (np.clip(audio / (np.abs(audio).max() + 1e-9), -1, 1) * 32767).astype(np.int16))
    st.audio(buf.getvalue(), format="audio/wav")
    if ss.get("extract") is None:
        with st.spinner("Extracting tonal partials and Doppler curves…"):
            ss["extract"] = extract_curves(audio, SR)
    ex = ss["extract"]
    fmax_show = st.slider("Spectrogram max frequency (Hz)", 200, 1500, 800, 50)
    fsel = ex.spec_freqs <= fmax_show
    spec = ex.spec_db[:, fsel]
    step = max(1, spec.shape[1] // 500)
    fig = go.Figure(go.Heatmap(x=ex.frame_times, y=ex.spec_freqs[fsel][::step], z=spec[:, ::step].T,
                               colorscale="Greys", zmin=0, zmax=30, showscale=False))
    palette = ["#e4572e", "#29335c", "#f3a712", "#669bbc", "#a8c686", "#8e44ad"]
    for i, c in enumerate(ex.curves):
        # draw every harmonic of the curve that fits on screen, for visual checking
        f = c.freq
        fig.add_trace(go.Scatter(x=c.t, y=f, mode="lines", line=dict(color=palette[i % 6], width=2),
                                 name=f"curve {i} ({c.n_partials} partials)"))
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10), xaxis_title="time (s)",
                      yaxis_title="Hz", legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Whitened spectrogram (dB above local noise floor). Coloured lines = extracted Doppler curves, "
               "drawn at their reference partial. Stationary tones (hum, fans) are rejected.")
    st.dataframe(pd.DataFrame([{
        "curve": i, "span (s)": f"{c.t[0]:.1f}–{c.t[-1]:.1f}", "ref partial (Hz)": round(c.ref_freq, 1),
        "partials merged": c.n_partials, "pitch sweep": f"{100 * c.sweep:.1f}%",
    } for i, c in enumerate(ex.curves)]), hide_index=True)

# ------------------------------------------------------------------ tab 2
with tab2:
    tracks = ss.get("tracks")
    if not tracks:
        st.info("Upload an ADS-B CSV in the sidebar.")
        st.stop()
    if ss.get("start") is None:
        st.warning("Enter the recording start time in the sidebar.")
        st.stop()
    cols = st.columns(4)
    cols[0].metric("Doppler curves", len(ex.curves))
    cols[1].metric("ADS-B aircraft in file", len(tracks))
    cols[2].metric("Clock search ±", f"{ss['window']:.0f} s")
    if st.button("Locate", type="primary"):
        cfg = LocateConfig(time_window_s=ss["window"], observer_alt_m=ss.get("elev", 0.0),
                           search_radius_m=ss.get("radius", 8000.0), use_level=use_level)
        with st.spinner("Grid search + robust refinement…"):
            ss["result"] = locate(ex.curves, tracks, ss["start"], center=ss.get("center"), cfg=cfg)
    res = ss.get("result")
    if res is None:
        st.stop()
    if not res.ok:
        st.error(res.message)
        st.stop()
    truth = ss.get("truth")
    st.markdown(f"**Estimate:** `{res.lat:.6f}, {res.lon:.6f}`")
    m = st.columns(4)
    m[0].metric("95% radius", f"{res.r95_m:.0f} m")
    m[1].metric("Clock offset found", f"{res.time_offset_s:+.1f} s")
    if truth:
        m[2].metric("True error", f"{float(haversine_m(res.lat, res.lon, *truth)):.0f} m")
    m[3].metric("Solve time", f"{res.elapsed_s:.1f} s")
    if res.ambiguous:
        st.warning(f"Ambiguous: {len(res.alternates)} alternate location(s) fit nearly as well "
                   "(typical with a single aircraft: mirror image across its track, or another aircraft "
                   "that could have produced the same curve). Alternates are shown in orange.")

    # map
    fmap = go.Figure()
    H = res.heat_cost
    ok = np.isfinite(H)
    hmin = np.nanmin(H)
    show = ok & (H <= np.nanpercentile(H, 4))
    fmap.add_trace(go.Scattermap(lat=res.heat_lat[show], lon=res.heat_lon[show], mode="markers",
                                 marker=dict(size=9, color=-H[show], colorscale="YlOrRd", opacity=0.45),
                                 name="coarse fit (darker = better)", hoverinfo="skip"))
    t0, t1 = ss["start"] - 60, ss["start"] + float(ex.frame_times[-1]) + 60
    used = {f.icao24 for f in res.fits if f.icao24}
    for tr in tracks:
        sel = (tr.t >= t0) & (tr.t <= t1)
        if sel.sum() < 2 or haversine_m(tr.lat[sel].mean(), tr.lon[sel].mean(), res.lat, res.lon) > 25000:
            continue
        fmap.add_trace(go.Scattermap(lat=tr.lat[sel], lon=tr.lon[sel], mode="lines",
                                     line=dict(width=4 if tr.icao24 in used else 1.5,
                                               color="#29335c" if tr.icao24 in used else "#9aa5b1"),
                                     name=f"{tr.callsign or tr.icao24}{' ✓' if tr.icao24 in used else ''}"))
    ang = np.linspace(0, 2 * np.pi, 60)
    dlat = res.r95_m * np.cos(ang) / 111_320
    dlon = res.r95_m * np.sin(ang) / (111_320 * np.cos(np.deg2rad(res.lat)))
    fmap.add_trace(go.Scattermap(lat=res.lat + dlat, lon=res.lon + dlon, mode="lines", fill="toself",
                                 line=dict(color="#e4572e"), name="95% radius"))
    fmap.add_trace(go.Scattermap(lat=[res.lat], lon=[res.lon], mode="markers",
                                 marker=dict(size=14, color="#e4572e"), name="estimate"))
    if res.alternates:
        fmap.add_trace(go.Scattermap(lat=[a.lat for a in res.alternates], lon=[a.lon for a in res.alternates],
                                     mode="markers", marker=dict(size=12, color="#f3a712"), name="alternates"))
    if truth:
        fmap.add_trace(go.Scattermap(lat=[truth[0]], lon=[truth[1]], mode="markers",
                                     marker=dict(size=14, color="#2a9d8f", symbol="circle"), name="truth"))
    fmap.update_layout(map=dict(style="open-street-map", center=dict(lat=res.lat, lon=res.lon), zoom=11),
                       height=560, margin=dict(l=0, r=0, t=0, b=0), legend=dict(bgcolor="rgba(255,255,255,0.8)"))
    st.plotly_chart(fmap, use_container_width=True)

    # curve fits
    st.subheader("Observed vs predicted Doppler curves")
    ff = go.Figure()
    for f in res.fits:
        name = f.callsign or f.icao24 or "unexplained"
        ff.add_trace(go.Scatter(x=f.t, y=f.f_obs, mode="markers", marker=dict(size=4), name=f"curve {f.curve_index} observed"))
        if f.f_pred is not None:
            ff.add_trace(go.Scatter(x=f.t, y=f.f_pred, mode="lines", name=f"→ {name} predicted"))
    ff.update_layout(height=360, xaxis_title="time in clip (s)", yaxis_title="Hz (reference partial)",
                     margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(ff, use_container_width=True)
    st.dataframe(pd.DataFrame(res.summary()["fits"]), hide_index=True)

# ------------------------------------------------------------------ tab 3
with tab3:
    path = os.path.join(os.path.dirname(__file__), "out", "benchmark", "summary.md")
    if os.path.exists(path):
        st.markdown(open(path).read())
        st.caption("Generated by `python -m propgeo benchmark --trials 20`. Each row: same 20 random scenarios, "
                   "searcher given a ~2 km-off rough area and ADS-B that includes non-audible distractor aircraft.")
    else:
        st.info("Run `python -m propgeo benchmark` to populate this tab.")
