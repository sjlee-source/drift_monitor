#!/bin/bash
# sequential, resumable: skips settings whose JSON exists
cd /home/sjlee/Research/drift_monitor
OUT=/home/sjlee/Papers/mfg-agent/access_revision_2026-11-11/exp_E1/raw
for i in $(seq 0 23); do
  if [ -f $OUT/setting_$(printf %02d $i).json ]; then echo "skip $i"; continue; fi
  while [ $(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo) -lt 62 ]; do echo "low mem, wait"; sleep 60; done
  PYTHONPATH=$PWD NUMBA_NUM_THREADS=${NT:-3} OMP_NUM_THREADS=1 ~/.venvs/drift-rev/bin/python -m revision.experiments.E1.run_setting $i --out $OUT || echo "FAILED $i"
done
echo ALLDONE
