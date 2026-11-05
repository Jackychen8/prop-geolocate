import csv, sys, numpy as np
from scipy.spatial import cKDTree
ref = sys.argv[1]
R = 6371.0088
def xyz(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    return np.c_[np.cos(la)*np.cos(lo), np.cos(la)*np.sin(lo), np.sin(la)]
apt = list(csv.DictReader(open(f'{ref}/apt/APT_BASE.csv', encoding='latin-1')))
bg = list(csv.DictReader(open(f'{ref}/cenpop_bg.txt', encoding='utf-8-sig')))
pop = np.array([float(b['POPULATION']) for b in bg])
P = xyz(np.array([float(b['LATITUDE']) for b in bg]), np.array([float(b['LONGITUDE']) for b in bg]))
# 50 states + DC (exclude PR and territories) for a "US" headline
us = np.array([b['STATEFP'] not in ('72',) for b in bg])
print(f"block groups {len(bg)}, population (50 states+DC) {pop[us].sum():,.0f}")
def is_open_airport(a): return a['SITE_TYPE_CODE'] == 'A' and a['ARPT_STATUS'] == 'O' and a['COUNTRY_CODE'] == 'US'
sets = {
  'NPIAS airports (FAA national system, public-use)': [a for a in apt if is_open_airport(a) and a['FACILITY_USE_CODE']=='PU' and a['NASP_CODE'].strip()],
  'All public-use airports':                          [a for a in apt if is_open_airport(a) and a['FACILITY_USE_CODE']=='PU'],
  'Towered airports (ATCT)':                          [a for a in apt if is_open_airport(a) and a['TWR_TYPE_CODE'].startswith('ATCT')],
  'All airports incl. private strips':                [a for a in apt if is_open_airport(a)],
}
rows = []
for name, A in sets.items():
    Q = xyz(np.array([float(a['LAT_DECIMAL']) for a in A]), np.array([float(a['LONG_DECIMAL']) for a in A]))
    d, _ = cKDTree(Q).query(P)
    dkm = 2*R*np.arcsin(np.clip(d/2, 0, 1))
    out = [name, len(A)]
    for km in (3, 5, 10):
        out.append(pop[us & (dkm <= km)].sum() / pop[us].sum())
    rows.append(out)
print(f"{'airport set':52s} {'n':>6s} {'<=3km':>7s} {'<=5km':>7s} {'<=10km':>7s}")
for r in rows: print(f"{r[0]:52s} {r[1]:6d} {r[2]:7.1%} {r[3]:7.1%} {r[4]:7.1%}")
