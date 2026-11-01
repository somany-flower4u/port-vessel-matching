import numpy as np, pandas as pd
CAT = None
def _cat():
    global CAT
    if CAT is None:
        CAT = pd.read_csv(__file__.rsplit('/', 1)[0] + '/vessel_catalog.csv').set_index('vessel_id')
    return CAT
def _digits(x, rng, mode):
    s = str(int(x))
    if len(s) < 3:
        return x
    if mode == 'digit':
        i = rng.randint(len(s)); d = str((int(s[i]) + 1 + rng.randint(9)) % 10)
        if i == 0 and d == '0': d = '1'
        s = s[:i] + d + s[i+1:]
    elif mode == 'swap':
        i = rng.randint(len(s) - 1); s = s[:i] + s[i+1] + s[i] + s[i+2:]
    elif mode == 'drop':
        i = rng.randint(len(s)); s = s[:i] + s[i+1:]
    return float(int(s)) if s and int(s) > 0 else x
def apply(kind, row, rng):
    r = dict(row); g, ig = r['gross_tonnage'], r['international_gross_tonnage']
    if kind == 'none':
        return r
    if kind in ('digit', 'swap', 'drop'):
        if pd.notna(g):
            new = _digits(g, rng, kind)
            if pd.notna(ig) and ig == g: r['international_gross_tonnage'] = new
            r['gross_tonnage'] = new
    elif kind == 'x20':
        f = rng.uniform(1.1, 1.3) if rng.rand() < .5 else rng.uniform(0.75, 0.9)
        for c in ('gross_tonnage', 'international_gross_tonnage'):
            if pd.notna(r[c]): r[c] = round(r[c] * f)
    elif kind == 'dwt':
        d = _cat().loc[r['target_vessel_id'], 'deadweight_tonnage']
        if pd.notna(d): r['gross_tonnage'] = r['international_gross_tonnage'] = float(d)
    elif kind == 'nat':
        c = _cat().loc[r['target_vessel_id']]
        if c['flag_name'] == 'South Korea' and pd.notna(c['gross_tonnage']):
            r['gross_tonnage'] = float(round(c['gross_tonnage'] * np.exp(rng.uniform(-0.37, -0.03))))
            r['international_gross_tonnage'] = np.nan
    elif kind == 'nat2':
        c = _cat().loc[r['target_vessel_id']]
        if c['flag_name'] == 'South Korea' and pd.notna(c['gross_tonnage']):
            v = float(round(c['gross_tonnage'] * np.exp(rng.uniform(-0.37, -0.03))))
            r['gross_tonnage'] = r['international_gross_tonnage'] = v
    elif kind == 'gt_only':
        r['international_gross_tonnage'] = np.nan
    elif kind == 'igt_only':
        r['gross_tonnage'] = np.nan
    return r
