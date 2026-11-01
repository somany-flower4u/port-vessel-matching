#!/bin/bash
# build_ko.sh NAME DOCSTRING_NOTE [DE args: epochs threads syn tag mine domreps]  (env DOM_ALL=1 optional)
# build_ko5_gpu.sh: identical to build_ko5.sh except that the 6 DE models are trained with dualenc_v2dom6_gpu.py
# (override with DE_SCRIPT) on DEVICE=${DEVICE:-cuda} (exported to the trainer); DE threads default 2. Everything after the
# DE step (export, pairs, dom, fit) is unchanged and runs on the CPU.
# v27a package + new Hangul DE (6 models one at a time) + v20 ranker recipe + v27f Latin path -> work/NAME
set -e
SP=${SEOKCHEON_WS:?set SEOKCHEON_WS to the workspace root (see training/README.md)}
R=$SP/agents/round4; T=$SP/agents/de-v2-full; W=$SP/work
PY=${PY:-python}; TPY=${TPY:-python}
NAME=$1; NOTE=$2; shift 2
EP=${1:-20}; TH=${2:-2}; SYN=${3:-12000}; TAG=${4:-v2d}; MINE=${5:-2,4,6,8,10,12,14,16,18}; DR=${6:-2}
export DEVICE=${DEVICE:-cuda}
export DE_OUT=$R/$NAME/out; mkdir -p $DE_OUT $R/$NAME/folds $R/logs
echo "== $NAME start $(date) DOM_ALL=${DOM_ALL:-0} ep=$EP syn=$SYN mine=$MINE domreps=$DR device=$DEVICE de_threads=$TH script=${DE_SCRIPT:-dualenc_v2dom6_gpu.py}"
cd $R
for F in -1 0 1 2 3 4; do
  FN=$([ "$F" = "-1" ] && echo full || echo f$F)
  if [ -s $DE_OUT/de_${FN}${TAG}.pt ] && [ -s $DE_OUT/de_${FN}${TAG}_cat.npy ] && [ -s $DE_OUT/de_${FN}${TAG}_timing.json ]; then echo "skip $F"; continue; fi
  echo "de $F $(date)"
  $TPY ${DE_SCRIPT:-dualenc_v2dom6_gpu.py} $F $EP $TH $SYN $TAG $MINE $DR > $R/logs/${NAME}_de_$F.log 2>&1
done
echo "export $(date)"
cd $W
if [ ! -e $NAME ]; then
  rsync -a --exclude __pycache__ v27a/ $NAME/
  rm -rf $NAME/model/en; rsync -a --exclude __pycache__ v27f/model/en/ $NAME/model/en/
  $PY - $NAME "$NOTE" <<'PY'
import sys; p=sys.argv[1]+"/script.py"; s=open(p,encoding="utf-8").read().split("\n")
assert s[0].startswith('"""DACON inference entry point: vessel matching (v27a:')
s[0]='"""DACON inference entry point: vessel matching (%s: v27a Hangul path with a retrained character dual encoder (%s), rankers retrained with the v20 recipe on the new fold DEs; Latin path = v27f (Latin character dual encoder, v19-recipe rankers)).' % (sys.argv[1], sys.argv[2])
open(p,"w",encoding="utf-8").write("\n".join(s))
PY
fi
cd $R
for k in 0 1 2 3 4; do $TPY export_de4.py fold $k $TAG $R/$NAME/folds; done
$TPY export_de4.py full $TAG $W/$NAME
cd $W
export PYTHONPATH=$T/capcpu VM_PROC_CAP=3
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export VM_KO_DE_FOLD_DIR=$R/$NAME/folds
echo "pairs $(date)"
$PY train14_de.py $NAME ko orig,ko_noise,ko_heavy,ko_noise pairs_$NAME.npz > $R/logs/${NAME}_pairs.log 2>&1
echo "dom $(date)"
DOM_REPS=1 $PY domtrain.py build $NAME dom_$NAME.npz > $R/logs/${NAME}_dom.log 2>&1
echo "fit $(date)"
EXCLUDE_KINDS=rename,dwt $PY fit20.py $NAME pairs_$NAME.npz dom_$NAME.npz 1 final > $R/logs/${NAME}_fit.log 2>&1
tail -2 $R/logs/${NAME}_fit.log
echo "== $NAME BUILT $(date)"
