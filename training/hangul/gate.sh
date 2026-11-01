#!/bin/bash
# gate.sh NAME FOLDDIR : OOF rank-1 rate of the fold DEs on the 382 real train Hangul names (a8 gate; v26b 0.545, v27a 0.683, v28a 0.675)
R=${SEOKCHEON_WS:?set SEOKCHEON_WS to the workspace root (see training/README.md)}/agents/round4
OMP_NUM_THREADS=1 ${PY:-python} $R/hm/a8g.py "$1=$2" > $R/logs/$1_gate.log 2>&1; cat $R/logs/$1_gate.log
