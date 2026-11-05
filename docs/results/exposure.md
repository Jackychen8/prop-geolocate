# How often does the propeller-Doppler attack have what it needs?

The solver needs **two or more ADS-B propeller aircraft passing within a few km** during the clip (one plane
is ambiguous). Two measurements, both reproducible with `tools/exposure/`.

## 1. Bay Area traffic, from the open adsb.lol archive

Source: [adsb.lol `globe_history`](https://github.com/adsblol/globe_history_2026), one release per day back
to 2023 (ODbL, no account needed). Two UTC days: Wed 2026-11-04 (Tue 16:00 to Wed 16:00 Pacific) and
Sun 2026-11-01 (Sat 17:00 to Sun 16:00 Pacific). Propeller = fixed-wing piston/turboprop by ICAO type
designator (tar1090-db type table). "Overhead" = within 3 km horizontally and below 3,000 ft above ground.
TIS-B targets (radar rebroadcasts, no ICAO address) are excluded. Coverage depends on volunteer receivers,
so these are lower bounds.

Share of daytime (07:00-19:00) clips with **2+ propeller aircraft overhead**:

| Site | Nearest public airport | 1-min clip | 3-min clip | 5-min clip |
|---|---|---|---|---|
| Oakland, ~3 km from OAK | OAK 3.1 km | 20-25% | 30-34% | 39-41% |
| Hayward downtown | HWD 3.8 km | 8-14% | 26-29% | 38-39% |
| Palo Alto downtown | PAO 4.4 km | 3% | 6-10% | 12-16% |
| Downtown SF | OAK 17.5 km | 0-1% | 0-2% | 0-3% |
| Rural Central Valley (south of Merced) | LSN 15.6 km | 0% | 0% | 0% |

* Night (19:00-07:00): at most 0.6% of 1-minute windows at any site.
* Within 5 km instead of 3 km, Palo Alto rises to 27-43% for a 1-minute clip (the PAO pattern sits ~4 km away).
* Traffic is bursty: Oakland had 97% of minutes with 2+ props in one Sunday hour (several trainers in the North
  Field pattern), and near 0% in others.
* Downtown SF had a helicopter within 3 km in 37% of Wednesday daytime minutes (mostly two news-type
  helicopters circling for hours). Helicopters are tonal but are not supported or tested by this tool.

Only two days in November, one rural site. Treat as indicative.

## 2. US population near an airport

Sources: FAA NASR airport data (28-day cycle effective 2026-10-29, `APT_BASE.csv`) and Census 2020
population-weighted block-group centroids (`CenPop2020_Mean_BG.txt`). 50 states + DC, 331,449,281 people.

| Airport set | Airports | ≤3 km | ≤5 km | ≤10 km |
|---|---|---|---|---|
| NPIAS (FAA national system, public use) | 3,414 | 5.2% | **15.4%** | 47.9% |
| All public-use airports | 4,721 | 6.1% | 17.7% | 53.5% |
| Towered airports | 712 | 2.7% | 9.1% | 32.8% |
| All open airports incl. private strips | 12,591 | 9.1% | 25.4% | 68.6% |

Living near an airport is not the same as being under its traffic (see section 1: Palo Alto vs Oakland), so
section 2 is an upper-bound framing and section 1 is the stronger evidence.

## Reproduce
```bash
cat v2026.11.04-planes-readsb-prod-0.tar.a? | python tools/exposure/extract_region.py region_2026-11-04.pkl
python tools/exposure/traffic.py REF_DIR region_2026-11-04.pkl      # REF_DIR: icao_types.json + apt/APT_BASE.csv
python tools/exposure/pop_near_airports.py REF_DIR                  # + cenpop_bg.txt
```
