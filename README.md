# propgeo: acoustic geolocation from propeller-aircraft Doppler

Research and test implementation of the technique in the TechRadar report: recover a
recording's location from the Doppler pitch curves of passing propeller aircraft
in its background audio, matched against public ADS-B flight tracks.

```
audio ─► STFT ─► tonal peaks ─► partial tracks ─► harmonic grouping ─► Doppler curves ─┐
                                                                                       ├─► grid search + robust LSQ ─► lat/lon ± r95
ADS-B CSV ─► per-aircraft smoothing spline (pos + vel) ─► retarded-time Doppler model ┘     (+ clock offset, curve→aircraft
                                                                                              assignment, ambiguity check)
```

## Quick start

```bash
uv venv .venv && uv pip install --python .venv/bin/python -e '.[test]'   # needs ffmpeg on PATH
.venv/bin/streamlit run app.py                                            # UI
.venv/bin/python -m pytest -q                                             # 14 tests, ~1-2 min
.venv/bin/python -m propgeo simulate --seed 11 --out out/sim             # synthetic clip + ADS-B + truth
.venv/bin/python -m propgeo locate out/sim/clip.m4a --adsb out/sim/adsb.csv --elev 180 --window 5 --truth out/sim/truth.json
.venv/bin/python -m propgeo benchmark --trials 20                        # Monte-Carlo claim test, ~10 min
```

### Real recordings
To try it on yourself, record a minute or more of outdoor audio while propeller traffic passes, and log ADS-B
at the same time:

```bash
python -m propgeo capture --lat 37.45 --lon -122.10 --duration 300     # adsb.lol live feed, no key
python -m propgeo locate my_clip.m4a --adsb out/live_adsb.csv --window 10 --center 37.45,-122.10 --elev 30
```
Historical ADS-B (for older clips) needs an archive such as OpenSky's Trino history (research account).
Any CSV in OpenSky `state_vectors_data4` layout works, and readsb/adsb.lol column names are accepted.
**Not yet validated on a real recording**: every number below comes from simulation.

## How it works

* **Physics** (`physics.py`): each sample heard at time t was emitted at retarded time τ with
  `t = τ + r(τ)/c`, solved by Newton iteration. The Doppler ratio is `(c + w·n)/(c − (v−w)·n)`, including optional wind.
* **Extraction** (`extract.py`): a 0.5 s Hann STFT with 4× zero padding, peaks whitened against a median noise
  floor, sub-bin parabolic refinement, then frame-to-frame partial tracking with slope prediction.
  Harmonics of one propeller share a Doppler factor, so their log-frequencies differ by a constant. They are grouped
  greedily against the group's combined curve, absorbing only the frames that agree. This handles tracks that jump
  between aircraft where tones cross. Stationary tones (hum, HVAC) are rejected.
* **Solver** (`locate.py`): unknowns are observer x/y, clock offset δ, and which aircraft made each curve.
  Each curve's at-rest frequency is profiled out by demeaning in log space. A coarse grid over (x, y, δ) feeds
  two-stage Huber least squares (loose σ, then true σ) from the 12 best local minima, re-assigning curves
  between rounds. A curve no aircraft explains costs a fixed amount, so cars and non-ADS-B planes can't drag
  the fit. Uncertainty is Gauss-Newton, inflated for time-correlated residuals (AR(1)), plus a 10 m ADS-B floor.
  **Ambiguity**: any distinct solution with an acceptable fit is reported as an alternate, not hidden.
* **Simulator** (`simulate.py`): exact retarded-time synthesis of blade-passing harmonics, with RPM wander,
  amplitude flutter, propeller directivity, atmospheric absorption, engine noise, pink noise, speech babble
  and mains hum. The ADS-B side adds position noise, 25 ft altitude quantisation, dropouts, and 4 distractor
  aircraft that appear in ADS-B but not in the audio.

## Results (simulation, 20 seeds per row, same seeds across rows)

| condition                                |   trials | within 100 m   |   median err (m) |   p90 err (m) | flagged ambiguous   | truth in candidates (<100 m)   | err <= reported r95   | no solution   |   s/trial |
|:-----------------------------------------|---------:|:---------------|-----------------:|--------------:|:--------------------|:-------------------------------|:----------------------|:--------------|----------:|
| 1 plane                                  |       20 | 45%            |             1068 |          8673 | 55%                 | 70%                            | 45%                   | 0%            |       3.1 |
| baseline: 2 planes, clean, clock known   |       20 | 100%           |               10 |            20 | 5%                  | 100%                           | 95%                   | 0%            |       5.2 |
| 1 plane + level cue                      |       20 | 45%            |              866 |          9127 | 75%                 | 60%                            | 50%                   | 0%            |       3.2 |
| 30 s clip                                |       20 | 70%            |               44 |          3293 | 20%                 | 75%                            | 65%                   | 0%            |       3.4 |
| 3 planes                                 |       20 | 100%           |               12 |            35 | 10%                 | 100%                           | 90%                   | 0%            |       7.5 |
| very faint: SNR -20 dB                   |       20 | 5%             |              506 |          1017 | 10%                 | 5%                             | 33%                   | 85%           |       2.7 |
| faint: SNR -10 dB                        |       20 | 75%            |               25 |          1653 | 20%                 | 85%                            | 70%                   | 0%            |       5.4 |
| speech babble + mains hum                |       20 | 100%           |               13 |            32 | 10%                 | 100%                           | 95%                   | 0%            |       6   |
| MP3 128k                                 |       20 | 95%            |               11 |            33 | 10%                 | 95%                            | 90%                   | 0%            |       5.8 |
| MP3 32k                                  |       20 | 100%           |               12 |            36 | 0%                  | 100%                           | 95%                   | 0%            |       5.9 |
| phone chain (HPF200+AGC+AAC64k)          |       20 | 95%            |               11 |            47 | 10%                 | 95%                            | 90%                   | 0%            |       5.9 |
| noise suppression (afftdn)               |       20 | 100%           |               10 |            26 | 0%                  | 100%                           | 95%                   | 0%            |       6.4 |
| Opus VoIP 16k                            |       20 | 95%            |               13 |            43 | 10%                 | 95%                            | 90%                   | 0%            |       6.7 |
| ADS-B 10 s sampling                      |       20 | 80%            |               26 |           127 | 15%                 | 85%                            | 85%                   | 0%            |       4.9 |
| clock unknown +/-30 s                    |       20 | 85%            |               43 |           136 | 15%                 | 85%                            | 60%                   | 0%            |      40.1 |
| ADS-B pos noise 30 m                     |       20 | 95%            |               35 |            68 | 15%                 | 95%                            | 85%                   | 0%            |       5.6 |
| unmodelled wind 6 m/s                    |       20 | 95%            |               18 |            41 | 15%                 | 95%                            | 90%                   | 0%            |       5.5 |
| RPM drift 0.5%                           |       20 | 80%            |               50 |           338 | 90%                 | 80%                            | 50%                   | 0%            |       5.5 |
| observer elevation off by 50 m           |       20 | 100%           |               31 |            68 | 10%                 | 100%                           | 75%                   | 0%            |       5.5 |
| 2 planes + level cue                     |       20 | 100%           |               12 |            28 | 0%                  | 100%                           | 90%                   | 0%            |       5.6 |
| realistic: phone+babble+SNR0+clock+/-30s |       20 | 70%            |               64 |           130 | 20%                 | 70%                            | 55%                   | 0%            |      36.8 |

Search prior in every row: a centre about 2 km off the truth (σ, capped at 5 km) and an 8 km radius.
"truth in candidates" counts the estimate plus any reported alternates.

### What this says about the claim
* **"Within ~100 m" holds in simulation when ≥2 planes are heard and the clip's clock is known.** Baseline median
  error is 10 m (p90 20 m), and 3 planes do about as well. With the realistic combination (phone chain, speech, SNR 0 dB,
  clock unknown ±30 s), 70% of trials land within 100 m and the median is 64 m.
* **"Survives compression" holds.** MP3 at 32k and 128k, a phone chain (200 Hz high-pass + AGC + AAC 64k), spectral
  noise suppression and Opus 16k VoIP all stay at a 10–13 m median. These codecs preserve strong tonal partials,
  and the fit uses harmonics up to 1.5 kHz, so losing the fundamental to a high-pass costs nothing.
* **"Less than 60 seconds" is fragile.** With 30 s clips only 70% land within 100 m: often a plane's closest approach
  falls outside the clip.
* **One plane is not enough.** A single straight pass fits a mirror-image location and, often, *another ADS-B
  aircraft* at another place equally well. Only 45% are top-1 within 100 m. The solver flags ambiguity and lists
  candidates instead of over-claiming. A tight prior region (a few km) or a second plane resolves it. The
  loudness-envelope cue (`--level`) did not fix this, because the alias reproduces the same geometry.
* **The biggest real-world error sources** in this model are an unknown clock (along-track position trades
  off against time), RPM wander of 0.5% (it looks like Doppler), sparse ADS-B (10 s OpenSky-anonymous sampling),
  wrong ground elevation, and faintness (at −20 dB broadband SNR the tones are mostly undetectable).
* **Calibration:** the reported 95% radius contains the truth 90–95% of the time in nominal conditions, but only
  50–60% with an unknown clock or RPM drift, so treat r95 as optimistic there.

## Limitations / honest caveats
* Validated only in simulation. The simulator and solver share a physics model (straight rays, uniform sound
  speed). Real-world refraction, ground reflection (Lloyd's mirror), turbulence and directivity will cost accuracy.
* Observer elevation must be in the same datum as the ADS-B altitude: geometric altitude is height above the
  WGS84 ellipsoid (geoid separation is about −30 m in California). A 50 m error costs ~30 m of position.
* Wind is not estimated, but a known wind can be passed in (`LocateConfig.wind_en`).
* Clock-unknown searches take about 40 s on a laptop. Everything else takes about 5 s.

## Layout
`propgeo/` = `geo` (WGS84/ENU), `physics`, `adsb` (load/smooth/live capture), `audio_io` (ffmpeg decode +
degradation chains), `simulate`, `extract`, `locate`, `benchmark`, `cli`. `app.py` is the Streamlit UI and
`tests/` the pytest suite.
