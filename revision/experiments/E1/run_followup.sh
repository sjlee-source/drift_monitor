#!/bin/bash
cd /home/sjlee/Research/drift_monitor
OUT=/home/sjlee/Papers/mfg-agent/access_revision_2026-11-11/exp_E1/followup
run(){ while [ $(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo) -lt 62 ]; do echo "low mem, wait"; sleep 60; done
  PYTHONPATH=$PWD NUMBA_NUM_THREADS=4 OMP_NUM_THREADS=1 ~/.venvs/drift-rev/bin/python -m revision.experiments.E1.run_followup $1 $2 || echo "FAILED $1 $2"; }
for k in 0 1 2 3; do ls $OUT/repair_${k}_*.json >/dev/null 2>&1 && continue; run repair $k; done
for j in $(seq 0 17); do ls $OUT/fineq_$(printf %02d $j)_*.json >/dev/null 2>&1 && continue; run fineq $j; done
echo ALLDONE
