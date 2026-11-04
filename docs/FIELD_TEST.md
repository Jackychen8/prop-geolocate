# Field test: try it on yourself

Goal: one real recording, a log of ADS-B traffic for the same minutes, and your true position (kept
private) to measure the error against. Budget about 30 minutes on site.

## 1. Pick a spot and a time
* **Not your home.** You'll be posting results, so pick a park or trailhead you don't mind being near.
* You need **two or more propeller planes passing within ~1-3 km** while you record. One plane alone is
  ambiguous (see README). The best spots are 1-3 km from a small general-aviation airport's traffic
  pattern, or under a flight-school practice area. Weekend late mornings are busiest.
* Before you go, open <https://globe.adsb.lol> and look for small, low (under ~3,000 ft) planes
  circling in the area. Small-plane coverage depends on volunteer receivers, so check that they show up there.

## 2. Start logging ADS-B (laptop, any machine with internet)
Start this a couple of minutes **before** you record and let it run until a couple of minutes after:
```bash
python -m propgeo capture --lat 37.4 --lon -122.1 --duration 1200 --out out/field_adsb.csv
```
Only a point rounded to 0.1° (~11 km) is sent to the adsb.lol API, with the radius widened to compensate.
`--exact` disables the rounding.

## 3. Record
* Phone voice memo or video, **3-5 minutes**, phone held still, mic shielded from wind (your body
  helps). Talking is fine.
* Note the start time to the second, as a fallback if the file has no timestamp.
* **Truth:** drop a pin in your maps app where you stand, or read the GPS coordinates. Keep it private.

## 4. Elevation (matters: 50 m wrong ≈ 30 m position error)
ADS-B geometric altitude is height above the WGS84 ellipsoid, so pass
`--elev = ground elevation above sea level + geoid height`. Look up the geoid height for your spot
(e.g. NOAA's GEOID18 tool in the US; about −32 m around the SF Bay Area).

## 5. Run it
Copy the original file to the laptop (AirDrop or cable keep the original; messaging apps re-encode).
```bash
python -m propgeo field-test my_clip.m4a --adsb out/field_adsb.csv \
    --truth 37.41234,-122.10987 --elev -20 --out out/field
```
* The search area is guessed from the low-flying traffic, so the tool is **not** told where you are.
  `--truth` is used only to score the result. To model an attacker who knows the city, add `--center lat,lon`
  with a point a few km off.
* The clock is searched ±15 s around the file's timestamp (`--window`). Phone clocks are network-synced,
  so if you know the timestamp is the start of the recording, `--window 3` is faster and more accurate.
  The clock/along-track trade-off is the main source of 100-300 m misses in simulation. Some Android phones
  stamp the *end* of a recording; if results look off, pass `--start` yourself.
* By default the clip is also re-run through MP3 32k, a phone AAC chain and Opus 16k (`--also-degrade`).
  For a real-world compression test, send the clip to yourself via WhatsApp/Telegram/X, download it,
  and run `field-test` on that copy too.

## 6. What's safe to post
`out/field/` is written in **privacy mode**:
* the map is in metres relative to you, with no basemap or coordinates;
* aircraft are "Aircraft A/B", not callsigns;
* no timestamps.

Post the PNGs and the numbers in `result.md`. **Don't post** the raw audio together with the date and
time, the ADS-B CSV, callsigns, or `--show-coords` output. Any of those lets someone redo the fit and find
where you were, which is the whole point of the project.

## Troubleshooting
| Symptom | Likely cause |
|---|---|
| `extracted 0 Doppler curve(s)` | Planes too far or quiet. Look at `1_spectrogram.png` for sloping lines; record closer or longer. |
| `no low-altitude ADS-B traffic` | The capture didn't cover the recording time, or no plane in range had ADS-B. |
| `ambiguous: yes` | Usually only one plane fit. `result.md` gives the nearest candidate's error; a city-level `--center` or a second plane resolves it. |
| Error of a few hundred metres | Check `--elev`, the clock (`--start`), and whether the ADS-B log had 1 s updates or sparser. |
