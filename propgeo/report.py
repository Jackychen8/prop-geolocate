"""Static, shareable figures + numbers (PNG / JSON / Markdown).

Privacy mode (default) is meant for posting results publicly about a recording
made where *you* were standing:
  * the map is drawn in metres relative to the true position: no basemap,
    no lat/lon;
  * aircraft are labelled "Aircraft A/B/...", not by callsign or ICAO hex;
  * no absolute timestamps.
Callsign + time is enough for anyone to pull the same ADS-B tracks from an
archive and redo the fit, so all three matter.
"""
from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .geo import LocalFrame  # noqa: E402

# reference palette (validated: first three categorical slots, all-pairs, light surface)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
OTHER = "#b9b7ae"


def _style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
        "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 11,
        "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "legend.frameon": False, "lines.linewidth": 2,
    })


def _titles(ax, title, subtitle=None):
    ax.set_title(title, pad=24 if subtitle else 8)
    if subtitle:
        ax.text(0, 1.015, subtitle, transform=ax.transAxes, color=INK2, fontsize=9.5, va="bottom")


def _color(i: int) -> str:
    return SERIES[i] if i < len(SERIES) else OTHER


def aircraft_labels(res, private: bool) -> dict:
    """Map icao24 -> display label, in order of first use."""
    out = {}
    for f in res.fits:
        if f.icao24 and f.icao24 not in out:
            k = len(out)
            out[f.icao24] = f"Aircraft {chr(65 + k)}" if private else (f.callsign or f.icao24)
    return out


def fig_spectrogram(ex, path, fmax=800.0, title="What the microphone heard"):
    _style()
    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=160)
    sel = ex.spec_freqs <= fmax
    ax.imshow(np.clip(ex.spec_db[:, sel].T, 0, 30), origin="lower", aspect="auto", cmap="Greys",
              extent=[ex.frame_times[0], ex.frame_times[-1], ex.spec_freqs[sel][0], ex.spec_freqs[sel][-1]],
              vmin=0, vmax=30, interpolation="nearest")
    ax.grid(False)
    shown = [c for c in ex.curves if np.median(c.freq) <= fmax][:3]
    for i, c in enumerate(shown):
        ax.plot(c.t, c.freq, color=_color(i), lw=1.6, alpha=0.95, label=f"Doppler curve {i + 1}")
        j = int(np.argmax(c.t))
        ax.annotate(f"curve {i + 1}", (c.t[j], c.freq[j]), xytext=(6, 0), textcoords="offset points",
                    color=INK, fontsize=10, va="center", annotation_clip=False)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("frequency (Hz)")
    _titles(ax, title, "dark lines = propeller harmonics, bending from high to low pitch as each plane passes "
                       "(Doppler); coloured = extracted")
    if shown:
        ax.legend(loc="upper right", labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_fit(res, path, private=True):
    _style()
    labels = aircraft_labels(res, private)
    order = list(labels)
    fig, ax = plt.subplots(figsize=(10, 4.6), dpi=160)
    for f in res.fits:
        if f.f_pred is None or f.icao24 not in labels:
            continue
        i = order.index(f.icao24)
        norm = f.rest_freq_hz
        ax.plot(f.t, 100 * (f.f_obs / norm - 1), "o", ms=3.2, color=_color(i), alpha=0.55,
                markeredgewidth=0)
        ax.plot(f.t, 100 * (f.f_pred / norm - 1), "-", lw=2, color=_color(i), label=f"{labels[f.icao24]}")
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xlabel("time in clip (s)")
    ax.set_ylabel("pitch shift (%)")
    ax.set_title("Measured pitch (dots) vs. predicted from flight tracks (lines)")
    ax.legend(loc="upper right", labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _label_inside(ax, e, n, text, span):
    inside = np.flatnonzero((np.abs(e) < 0.92 * span) & (np.abs(n) < 0.92 * span))
    if len(inside):
        k = inside[-1]
        ax.annotate(text, (e[k], n[k]), xytext=(4, 6), textcoords="offset points", color=INK, fontsize=10, zorder=6)


def fig_map(res, tracks, path, truth=None, private=True, t_window=None, span_km=None):
    """Two panels: overview in km with the flight tracks, and a zoom in metres.
    Origin = truth if given, else the estimate."""
    _style()
    o_lat, o_lon = truth if truth else (res.lat, res.lon)
    fr = LocalFrame(o_lat, o_lon, 0.0)
    labels = aircraft_labels(res, private)
    order = list(labels)
    ee, en, _ = fr.to_enu(res.lat, res.lon, 0.0)
    ee, en = float(ee), float(en)
    alts = [tuple(float(v) for v in fr.to_enu(a.lat, a.lon, 0.0)[:2]) for a in res.alternates]
    span = span_km or max(3.0, 2.2 * np.hypot(ee, en) / 1000, *(1.3 * np.hypot(*a) / 1000 for a in alts))
    fig, (ax, az) = plt.subplots(1, 2, figsize=(13, 6.6), dpi=160, gridspec_kw=dict(width_ratios=[1, 1]))

    # --- overview (km)
    for tr in tracks:
        sel = (tr.t >= t_window[0]) & (tr.t <= t_window[1]) if t_window is not None else np.ones(len(tr.t), bool)
        if sel.sum() < 2:
            continue
        e, n, _ = fr.to_enu(tr.lat[sel], tr.lon[sel], 0.0 * tr.alt[sel])
        e, n = e / 1000, n / 1000
        if np.min(np.hypot(e, n)) > 2.5 * span:
            continue
        if tr.icao24 in labels:
            i = order.index(tr.icao24)
            ax.plot(e, n, color=_color(i), lw=2.2, solid_capstyle="round", zorder=3)
            _label_inside(ax, e, n, labels[tr.icao24], span)
        else:
            ax.plot(e, n, color=OTHER, lw=1, zorder=2)
    ax.plot(ee / 1000, en / 1000, "o", ms=9, color=SERIES[1], markeredgecolor=SURFACE, markeredgewidth=2, zorder=7)
    for a in alts:
        ax.plot(a[0] / 1000, a[1] / 1000, "o", ms=9, mfc="none", mec=SERIES[1], mew=2, zorder=7)
        ax.annotate("alternate", (a[0] / 1000, a[1] / 1000), xytext=(8, 6), textcoords="offset points",
                    color=INK2, fontsize=9)
    if truth:
        ax.plot(0, 0, "D", ms=8, color=SERIES[2], markeredgecolor=SURFACE, markeredgewidth=2, zorder=8)
    ax.annotate("microphone", (ee / 1000, en / 1000), xytext=(10, -16), textcoords="offset points",
                color=INK, fontsize=10, zorder=8)
    ax.set_xlim(-span, span)
    ax.set_ylim(-span, span)
    ax.set_aspect("equal")
    ax.set_xlabel("east (km)")
    ax.set_ylabel("north (km)")
    who = "true position" if truth else "estimate"
    _titles(ax, "Flight tracks during the clip",
            f"km from the {who}" + ("; coordinates & callsigns withheld" if private else ""))

    # --- zoom (m)
    z = max(100.0, 2.2 * res.r95_m, 1.6 * np.hypot(ee, en))
    ang = np.linspace(0, 2 * np.pi, 160)
    az.fill(ee + res.r95_m * np.cos(ang), en + res.r95_m * np.sin(ang), color=SERIES[1], alpha=0.16, lw=0)
    az.plot(ee + res.r95_m * np.cos(ang), en + res.r95_m * np.sin(ang), color=SERIES[1], lw=1)
    az.plot(ee, en, "o", ms=11, color=SERIES[1], markeredgecolor=SURFACE, markeredgewidth=2, zorder=7)
    az.annotate(f"estimate\n95% radius {res.r95_m:.0f} m", (ee, en), xytext=(12, -26), textcoords="offset points",
                color=INK, fontsize=10)
    if truth:
        err = float(np.hypot(ee, en))
        az.plot(0, 0, "D", ms=10, color=SERIES[2], markeredgecolor=SURFACE, markeredgewidth=2, zorder=8)
        az.annotate("true position", (0, 0), xytext=(-12, 12), textcoords="offset points", ha="right",
                    color=INK, fontsize=10)
        az.plot([0, ee], [0, en], color=INK2, lw=1, ls=(0, (3, 3)), zorder=6)
        _titles(az, f"Error: {err:.0f} m", "metres from the true position")
    else:
        _titles(az, "Estimate", "metres; shaded = 95% confidence radius")
    az.set_xlim(-z, z)
    az.set_ylim(-z, z)
    az.set_aspect("equal")
    az.set_xlabel("east (m)")
    az.set_ylabel("north (m)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_benchmark(trials_df, path, conditions=None, title=None):
    """Horizontal bars: share of trials located within 100 m, per condition."""
    _style()
    df = trials_df.copy()
    names = conditions or list(dict.fromkeys(df["condition"]))
    vals = []
    for n in names:
        g = df[df["condition"] == n]
        vals.append(100.0 * float((g["error_m"].fillna(np.inf) < 100).sum()) / max(len(g), 1))
    fig, ax = plt.subplots(figsize=(10, 0.42 * len(names) + 1.4), dpi=160)
    y = np.arange(len(names))[::-1]
    ax.set_axisbelow(True)
    ax.barh(y, vals, height=0.62, color=SERIES[0], edgecolor=SURFACE, linewidth=2)
    for yi, v in zip(y, vals):
        ax.text(v + 1.2, yi, f"{v:.0f}%", va="center", color=INK, fontsize=10)
    ax.set_yticks(y, names, color=INK)
    ax.set_xlim(0, 112)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("trials located within 100 m (%)")
    n_tr = int(df.groupby("condition").size().median())
    ax.set_title(title or f"Simulated stress test ({n_tr} random scenarios per row)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_results(res, path_json, path_md, truth=None, private=True, extra: dict | None = None):
    labels = aircraft_labels(res, private)
    out = {
        "ok": res.ok,
        "r95_m": round(res.r95_m, 1),
        "ambiguous": res.ambiguous,
        "n_alternates": len(res.alternates),
        "clock_offset_s": round(res.time_offset_s, 2),
        "n_curves": len(res.fits),
        "aircraft_used": list(labels.values()),
        "solve_time_s": round(res.elapsed_s, 1),
    }
    if truth:
        fr = LocalFrame(*truth, 0.0)
        e, n, _ = fr.to_enu(res.lat, res.lon, 0.0)
        out["error_m"] = round(float(np.hypot(e, n)), 1)
        out["error_east_north_m"] = [round(float(e), 1), round(float(n), 1)]
        cand = [out["error_m"]]
        for a in res.alternates:
            ae, an, _ = fr.to_enu(a.lat, a.lon, 0.0)
            cand.append(float(np.hypot(ae, an)))
        out["nearest_candidate_error_m"] = round(min(cand), 1)
    if not private:
        out["lat"], out["lon"] = res.lat, res.lon
    if extra:
        out.update(extra)
    with open(path_json, "w") as f:
        json.dump(out, f, indent=2)
    lines = ["# Field test result", ""]
    if "error_m" in out:
        lines.append(f"- **Error vs. true position: {out['error_m']:.0f} m** (reported 95% radius {out['r95_m']:.0f} m)")
    else:
        lines.append(f"- Reported 95% radius: {out['r95_m']:.0f} m (no ground truth given)")
    lines += [
        f"- Doppler curves used: {out['n_curves']}, from {len(labels)} aircraft",
        f"- Clock offset recovered: {out['clock_offset_s']:+.1f} s",
        f"- Ambiguous: {'yes, ' + str(out['n_alternates']) + ' alternate(s)' if res.ambiguous else 'no'}"
        + (f" (nearest candidate {out['nearest_candidate_error_m']:.0f} m from truth)"
           if res.ambiguous and "nearest_candidate_error_m" in out else ""),
        f"- Solve time: {out['solve_time_s']} s",
    ]
    if extra and extra.get("robustness"):
        lines += ["", "| audio chain | error (m) | r95 (m) |", "|---|---:|---:|"]
        for row in extra["robustness"]:
            e = "-" if row.get("error_m") is None else f"{row['error_m']:.0f}"
            r = "-" if row.get("r95_m") is None else f"{row['r95_m']:.0f}"
            lines.append(f"| {row['chain']} | {e} | {r} |")
    with open(path_md, "w") as f:
        f.write("\n".join(lines) + "\n")
    return out


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p
