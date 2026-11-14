import os
import sys
from pathlib import Path

os.environ.setdefault('NUMBA_NUM_THREADS', '8')
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(k, '1')
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))
THEORY = Path('/home/sjlee/Papers/mfg-agent/access_revision_2026-11-11/theory')
if THEORY.exists():
    sys.path.insert(0, str(THEORY))
