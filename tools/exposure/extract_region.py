"""Stream an adsb.lol globe_history day (tar on stdin) and keep low-altitude traces in a region."""
import sys, tarfile, gzip, json, re, pickle
from multiprocessing import Pool
import numpy as np

LAT0, LAT1, LON0, LON1 = 36.5, 38.6, -123.2, -119.8      # Bay Area + northern San Joaquin Valley
PAT = re.compile(rb'\[[0-9.]+,3[6-8]\.[0-9]+,-1(?:19|2[0-3])\.')

def work(raw):
    if raw[:2] == b'\x1f\x8b':
        raw = gzip.decompress(raw)
    if not PAT.search(raw):
        return None
    d = json.loads(raw)
    ts0 = d['timestamp']
    rows, cat = [], None
    for p in d['trace']:
        alt = p[3]
        if alt == 'ground': alt = 0
        if p[8] and cat is None: cat = p[8].get('category')
        rows.append((ts0 + p[0], p[1], p[2], np.nan if alt is None else alt, p[4] if p[4] is not None else np.nan, p[6]))
    a = np.array(rows, dtype=float)
    m = (a[:,1] >= LAT0) & (a[:,1] <= LAT1) & (a[:,2] >= LON0) & (a[:,2] <= LON1) & ~(a[:,3] > 6000)
    if not m.any():
        return None
    # keep points inside the region plus one neighbour either side for interpolation
    keep = m | np.r_[m[1:], False] | np.r_[False, m[:-1]]
    return dict(icao=d['icao'], t=d.get('t'), desc=d.get('desc'), r=d.get('r'), cat=cat, pts=a[keep].astype(np.float64))

def members(tf):
    for mbr in tf:
        if mbr.isfile() and '/traces/' in mbr.name:
            yield tf.extractfile(mbr).read()

if __name__ == '__main__':
    out = sys.argv[1]
    tf = tarfile.open(fileobj=sys.stdin.buffer, mode='r|')
    kept, n = [], 0
    with Pool(8) as pool:
        for res in pool.imap_unordered(work, members(tf), chunksize=64):
            n += 1
            if res: kept.append(res)
            if n % 50000 == 0: print(f'{n} traces scanned, {len(kept)} kept', file=sys.stderr, flush=True)
    pickle.dump(kept, open(out, 'wb'))
    print(f'done: {n} traces scanned, {len(kept)} kept -> {out}', file=sys.stderr)
