#!/bin/bash
cd /home/sjlee/Research/drift_monitor
export PYTHONPATH=$PWD NUMBA_NUM_THREADS=4 OMP_NUM_THREADS=4
F=/home/sjlee/Papers/mfg-agent/access_revision_2026-11-11/exp_E4/followup
P=~/.venvs/drift-rev/bin/python; S=revision/experiments/E4/fu_pfab.py
while pgrep -f "fu_pfab.py boardcal 1" >/dev/null; do sleep 10; done
w(){ until [ $(free -g | awk '/^메모리|^Mem/{print $7}') -ge 62 ]; do sleep 60; done; }
for t in 0 2; do w; $P $S boardcal $t > $F/boardcal_t$t.log 2>&1; done
for t in 0 1 2; do w; $P $S influence $t > $F/influence_t$t.log 2>&1; done
for t in 0 1 2; do w; $P $S recentred $t > $F/recentred_t$t.log 2>&1; done
echo DONE > $F/fu_done.flag
