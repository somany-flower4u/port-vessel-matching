#!/bin/bash
# build_en2.sh NAME NOTE [epochs=15] [reps=4] [syn=20000] [tag=C] [MINE=2,4,..,epochs-2 | none]
# v27af package (v27a Hangul + v27f Latin code) with a NEW Latin DE trained by round4/de_en2.py (v2 hard-negative mining; 6 models one at a time) + v19 Latin ranker recipe -> work/NAME
set -e
SP=${SEOKCHEON_WS:?set SEOKCHEON_WS to the workspace root (see training/README.md)}
R=$SP/agents/round4; L=$SP/agents/latin-de; W=$SP/work
PY=${PY:-python}; TPY=${TPY:-python}
NAME=$1; NOTE=$2; EP=${3:-15}; REPS=${4:-4}; SYN=${5:-20000}; TAG=${6:-C}
MINE=${7:-$(m=""; for ((e=2; e<=EP-2; e+=2)); do m="$m${m:+,}$e"; done; echo "${m:-none}")}
mkdir -p $R/$NAME/folds_en $R/logs
echo "== $NAME start $(date) ep=$EP reps=$REPS syn=$SYN tag=$TAG mine=$MINE"
cd $L
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 VECLIB_MAXIMUM_THREADS=3 OPENBLAS_NUM_THREADS=3
for F in 0 1 2 3 4 -1; do
  FN=$([ "$F" = "-1" ] && echo full || echo f$F)
  if [ -s out/de_en_${FN}${TAG}_weights.npz ] && [ -s out/de_en_${FN}${TAG}_cat.npy ]; then echo "skip $F"; else
  echo "de $F $(date)"
  $TPY $R/de_en2.py $F $EP 3 $REPS $SYN $TAG $MINE > $R/logs/${NAME}_de_$F.log 2>&1
  fi
  if [ "$F" != "-1" ]; then cp out/de_en_f${F}${TAG}_weights.npz $R/$NAME/folds_en/de_f${F}_weights.npz; cp out/de_en_f${F}${TAG}_cat.npy $R/$NAME/folds_en/de_f${F}_cat.npy; fi
done
cd $W
if [ ! -e $NAME ]; then
  rsync -a --exclude __pycache__ v27af/ $NAME/
  $PY - $NAME "$NOTE" <<'PY'
import sys; p=sys.argv[1]+"/script.py"; s=open(p,encoding="utf-8").read().split("\n")
assert s[0].startswith('"""DACON inference entry point: vessel matching (v27af:')
s[0]='"""DACON inference entry point: vessel matching (%s: Hangul path = v27a (stronger v2 character dual encoder); Latin path = v27f code with a retrained Latin character dual encoder (%s) and Latin rankers retrained with the v19 recipe on the new fold DEs).' % (sys.argv[1], sys.argv[2])
open(p,"w",encoding="utf-8").write("\n".join(s))
PY
fi
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd $L
$PY export_de.py out/de_en_full${TAG}_weights.npz out/de_en_full${TAG}_cat.npy $W/$NAME > $R/logs/${NAME}_export.log 2>&1
tail -2 $R/logs/${NAME}_export.log
echo "pairs $(date)"
NPROC=3 P_TON_ERR=0.1 VM_EN_DE_FOLD_DIR=$R/$NAME/folds_en $PY train14_ende.py $W/$NAME en orig,en_noise,en_heavy,en_abbr,en_abbr,orig@ton,en_noise@ton $R/$NAME/pairs_en > $R/logs/${NAME}_pairs.log 2>&1
tail -1 $R/logs/${NAME}_pairs.log
echo "fit $(date)"
$PY fit_en.py $R/$NAME/pairs_en both $W/$NAME 4 > $R/logs/${NAME}_fit.log 2>&1
grep "OOF\|final" $R/logs/${NAME}_fit.log
echo "== $NAME BUILT $(date)"
