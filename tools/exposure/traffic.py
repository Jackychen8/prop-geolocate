"""Per-site, per-local-hour share of 60 s windows with >=1 / >=2 propeller aircraft overhead."""
import sys, json, pickle, csv, collections
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import numpy as np

REF = sys.argv[1]; PKLS = sys.argv[2:]
TZ = ZoneInfo('America/Los_Angeles')
SITES = {  # name: (lat, lon, ground elevation ft MSL)
    'Downtown SF':                 (37.7897, -122.4000, 50),
    'Oakland, ~3 km from OAK':     (37.7350, -122.1900, 30),
    'Hayward downtown (HWD 4 km)': (37.6688, -122.0808, 110),
    'Palo Alto downtown (PAO 3 km)': (37.4443, -122.1598, 30),
    'Rural Central Valley':        (37.1000, -120.7000, 120),
}
RADII_KM = (3.0, 5.0); MAX_AGL_FT = 3000; STEP = 5.0; MAXGAP = 60.0
types = json.load(open(f'{REF}/icao_types.json'))

def klass(ac):
    d = (types.get(ac['t'] or '', {}) or {}).get('desc') or ''
    if len(d) == 3:
        if d[0] in 'HG': return 'helicopter'
        if d[2] == 'J': return 'jet'
        if d[2] in 'PTE': return 'prop'
    if ac['cat'] == 'A1' or ac['cat'] == 'A2': return 'unknown light (A1/A2)'
    if ac['cat'] == 'A7': return 'helicopter'
    return 'other/unknown'

# nearest public-use airport to each site, for context
apt = [a for a in csv.DictReader(open(f'{REF}/apt/APT_BASE.csv', encoding='latin-1'))
       if a['SITE_TYPE_CODE']=='A' and a['ARPT_STATUS']=='O' and a['FACILITY_USE_CODE']=='PU']
def hav(la1, lo1, la2, lo2):
    la1, lo1, la2, lo2 = map(np.radians, (la1, lo1, la2, lo2))
    return 2*6371.0088*np.arcsin(np.sqrt(np.sin((la2-la1)/2)**2 + np.cos(la1)*np.cos(la2)*np.sin((lo2-lo1)/2)**2))
alat = np.array([float(a['LAT_DECIMAL']) for a in apt]); alon = np.array([float(a['LONG_DECIMAL']) for a in apt])
for s, (la, lo, _) in SITES.items():
    d = hav(la, lo, alat, alon); i = np.argsort(d)[:2]
    print(f"{s:32s} nearest public airports: " + ', '.join(f"{apt[j]['ARPT_ID']} {d[j]:.1f} km" for j in i))

results = {}
for pkl in PKLS:
    acs = pickle.load(open(pkl, 'rb'))
    day = pkl.split('_')[-1].split('.')[0]
    t0 = datetime.fromisoformat(day).replace(tzinfo=timezone.utc).timestamp()
    nmin = 1440
    print(f"\n=== {day} UTC day ({datetime.fromtimestamp(t0, TZ):%a %H:%M} -> {datetime.fromtimestamp(t0+86400, TZ):%a %H:%M} local): {len(acs)} low-alt aircraft in region")
    print('classes:', collections.Counter(klass(a) for a in acs).most_common())
    # presence[site][radius][class] -> list of sets per minute
    pres = {s: {r: collections.defaultdict(lambda: [set() for _ in range(nmin)]) for r in RADII_KM} for s in SITES}
    for ac in acs:
        k = klass(ac); p = ac['pts']; p = p[np.argsort(p[:,0])]
        ok = ~np.isnan(p[:,3]) & (p[:,3] > 0)
        p = p[ok]
        if len(p) < 2: continue
        brk = np.where((np.diff(p[:,0]) > MAXGAP))[0] + 1
        for seg in np.split(p, brk):
            if len(seg) < 2: continue
            tg = np.arange(np.ceil(seg[0,0]/STEP)*STEP, seg[-1,0], STEP)
            if len(tg) == 0: continue
            la = np.interp(tg, seg[:,0], seg[:,1]); lo = np.interp(tg, seg[:,0], seg[:,2]); al = np.interp(tg, seg[:,0], seg[:,3])
            mi = ((tg - t0) // 60).astype(int); inday = (mi >= 0) & (mi < nmin)
            for s, (sla, slo, sel) in SITES.items():
                dk = hav(sla, slo, la, lo); low = (al - sel) <= MAX_AGL_FT
                for r in RADII_KM:
                    hit = inday & low & (dk <= r)
                    for m in np.unique(mi[hit]): pres[s][r][k][m].add(ac['icao'])
    hours = np.array([datetime.fromtimestamp(t0 + 60*m + 30, TZ).hour for m in range(nmin)])
    results[day] = (pres, hours)

def counts(pres_r, classes, nmin=1440):
    return np.array([len(set().union(*(pres_r[c][m] for c in classes))) for m in range(nmin)])

PROP = ['prop']; PROP_PLUS = ['prop', 'unknown light (A1/A2)']
for day, (pres, hours) in results.items():
    day_m = (hours >= 7) & (hours < 19)
    print(f"\n--- {day}: share of 60 s windows, propeller fixed-wing, <= {MAX_AGL_FT} ft AGL")
    print(f"{'site':32s} {'r':>4s} | {'>=1 day':>8s} {'>=2 day':>8s} | {'>=1 night':>9s} {'>=2 night':>9s} | {'>=2 day incl. A1/A2 unknown':>27s} | {'heli >=1 day':>12s}")
    for s in SITES:
        for r in RADII_KM:
            c = counts(pres[s][r], PROP); cp = counts(pres[s][r], PROP_PLUS); ch = counts(pres[s][r], ['helicopter'])
            print(f"{s:32s} {r:3.0f}k | {np.mean(c[day_m]>=1):8.1%} {np.mean(c[day_m]>=2):8.1%} | {np.mean(c[~day_m]>=1):9.1%} {np.mean(c[~day_m]>=2):9.1%} | {np.mean(cp[day_m]>=2):27.1%} | {np.mean(ch[day_m]>=1):12.1%}")
    # hourly profile at 3 km
    print(f"\nhourly share of minutes with >=2 props within 3 km ({day}):")
    print('hour ' + ' '.join(f'{h:>4d}' for h in range(24)))
    for s in SITES:
        c = counts(pres[s][3.0], PROP)
        print(f"{s[:20]:20s} " + ' '.join(f"{np.mean(c[hours==h]>=2)*100:4.0f}" for h in range(24)))
pickle.dump({d: ({s: {r: {k: [list(x) for x in v] for k, v in pr.items()} for r, pr in rr.items()} for s, rr in p.items()}, h.tolist()) for d, (p, h) in results.items()},
            open('traffic_results.pkl', 'wb'))
