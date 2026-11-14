#!/bin/bash
# sequential chain (<=4 threads at a time)
cd /home/sjlee/Research/drift_monitor
export PYTHONPATH=$PWD NUMBA_NUM_THREADS=4 OMP_NUM_THREADS=4
O=/home/sjlee/Papers/mfg-agent/access_revision_2026-11-11/exp_E4
for t in 0 1 2; do
  until [ $(free -g | awk '/^메모리|^Mem/{print $7}') -ge 62 ]; do sleep 60; done
  ~/.venvs/drift-rev/bin/python revision/experiments/E4/pfab_02_replay.py $t > $O/pfab/02_replay_t$t.log 2>&1
done
echo CHAIN_DONE > $O/chain_done.flag
