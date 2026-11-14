#!/bin/bash
# amendment 2026-11-13b: uniform 100k-calibration rerun (main grid, then fine-q grid); resumable
cd /home/sjlee/Research/drift_monitor
OUT=/home/sjlee/Papers/mfg-agent/access_revision_2026-11-11/exp_E1/raw_cal100k
FU=/home/sjlee/Papers/mfg-agent/access_revision_2026-11-11/exp_E1/followup_cal100k
waitmem(){ while [ $(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo) -lt 62 ]; do echo "low mem, wait"; sleep 60; done; }
export PYTHONPATH=$PWD NUMBA_NUM_THREADS=4 OMP_NUM_THREADS=1
for i in $(seq 0 23); do
  [ -f $OUT/setting_$(printf %02d $i).json ] && continue; waitmem
  ~/.venvs/drift-rev/bin/python -m revision.experiments.E1.run_setting100k $i --out $OUT || echo "FAILED main $i"
done
for j in $(seq 0 17); do
  ls $FU/fineq_$(printf %02d $j)_*.json >/dev/null 2>&1 && continue; waitmem
  ~/.venvs/drift-rev/bin/python -m revision.experiments.E1.run_followup fineq100k $j || echo "FAILED fineq $j"
done
echo ALLDONE
