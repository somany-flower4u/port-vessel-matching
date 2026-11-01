#!/bin/bash
# e2e.sh PKGDIR DATADIR OUTCSV : run a package like the server does (data/test.csv + catalog) and score vs DATADIR/../truth.csv if present
set -e
PKG=$1; DATA=$2; OUT=$3
start=$(date +%s)
${PY:-python} "$PKG/script.py" --data-dir "$DATA" --output "$OUT"
echo "runtime $(( $(date +%s) - start )) s"
TRUTH="$(dirname "$DATA")/truth.csv"
if [ -f "$TRUTH" ]; then ${PY:-python} - "$OUT" "$TRUTH" "$DATA/test.csv" <<'PY'
import sys, pandas as pd
o = pd.read_csv(sys.argv[1]); t = pd.read_csv(sys.argv[2]); x = pd.read_csv(sys.argv[3], encoding="utf-8-sig")
m = o.merge(t, on="sample_id", suffixes=("", "_true")).merge(x[["sample_id", "vessel_name"]], on="sample_id")
ko = m.vessel_name.str.contains("[가-힣]", regex=True, na=False)
err = m.target_vessel_id != m.target_vessel_id_true
print(f"rows {len(m)} errors {err.sum()} (ko {err[ko].sum()}/{ko.sum()}, en {err[~ko].sum()}/{(~ko).sum()})")
PY
fi
