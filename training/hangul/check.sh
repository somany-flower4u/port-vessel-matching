#!/bin/bash
# check.sh NAME : run the package like the server on domtest (Hangul) and cmp9 (Latin), log to logs/NAME_check.log
set -e
SP=${SEOKCHEON_WS:?set SEOKCHEON_WS to the workspace root (see training/README.md)}
R=$SP/agents/round4; W=$SP/work; NAME=$1
export PYTHONPATH=$SP/agents/de-v2-full/capcpu VM_PROC_CAP=3
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
{ echo "== check $NAME $(date)"
  bash $W/e2e.sh $W/$NAME $W/domtest/data $R/$NAME.domtest.csv | tail -2
  bash $W/e2e.sh $W/$NAME $SP/cmp9/data $R/$NAME.cmp9.csv | tail -2
  echo "== done $(date)"; } > $R/logs/${NAME}_check.log 2>&1
cat $R/logs/${NAME}_check.log
