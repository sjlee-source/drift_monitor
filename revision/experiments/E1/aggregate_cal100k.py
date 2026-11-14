"""Amendment 2026-11-13b: aggregate the uniform 100k-calibration rerun with the same code paths as the 20k
version, then copy tables to exp_E1/*_cal100k.csv and compare with the 20k version."""
import os, shutil, subprocess, glob
import numpy as np, pandas as pd
from revision import calibration as C
from revision.experiments.E1 import aggregate as A, aggregate_followup as F, e1_common as E

E1 = '/home/sjlee/Papers/mfg-agent/access_revision_2026-11-11/exp_E1'
W = os.path.join(E1, 'cal100k')
FAM = 1008


def main():
    os.makedirs(W, exist_ok=True)
    if not os.path.exists(os.path.join(W, 'raw')):
        os.symlink(os.path.join(E1, 'raw_cal100k'), os.path.join(W, 'raw'))
    shutil.copy(os.path.join(E1, 'cusum_exact_dp.csv'), W)
    A.ROOT, A.RAW, A.FAMILY = W, os.path.join(W, 'raw'), FAM
    A.main()
    F.ROOT, F.FU = W, os.path.join(E1, 'followup_cal100k')
    F.main()
    # fine-q validation: recompute simultaneous upper bound with family 1008
    fv = pd.read_csv(os.path.join(W, 'fa_validation_fineq.csv'))
    g = fv.path != 'const05'
    fv.loc[g, 'upper_one_sided'] = [C.clopper_pearson(k, n, 0.95, 'upper', FAM)[1] for k, n in zip(fv[g].alarms, fv[g].n)]
    fv.loc[g, 'family'] = FAM; fv.loc[g, 'pass_gate'] = fv.loc[g, 'upper_one_sided'] <= 0.03
    fv.to_csv(os.path.join(W, 'fa_validation_fineq.csv'), index=False)
    env = dict(os.environ, E1_ROOT=W, PYTHONPATH='/home/sjlee/Research/drift_monitor', OMP_NUM_THREADS='1')
    subprocess.run([os.path.expanduser('~/.venvs/drift-rev/bin/python'), '-m', 'revision.experiments.E1.crn_paired'], env=env,
                   check=True, stdout=subprocess.DEVNULL)
    subprocess.run(['/usr/bin/python3', '/home/sjlee/Research/drift_monitor/revision/experiments/E1/plot_e1.py'],
                   env=dict(env, E1_FIG=os.path.join(E1, 'figures', 'cal100k')), check=True)
    for p in glob.glob(os.path.join(W, '*.csv')):
        b = os.path.basename(p)[:-4]
        if b != 'cusum_exact_dp':
            shutil.copy(p, os.path.join(E1, f'{b}_cal100k.csv'))
    # ---------------- comparison with the 20k version
    rows = []
    o = pd.read_csv(os.path.join(E1, 'metrics_long.csv')); n = pd.read_csv(os.path.join(W, 'metrics_long.csv'))
    m = o.merge(n, on=['variant', 'family', 'pi', 'r1', 'q', 'd', 'D'], suffixes=('_20k', '_100k'))
    m['dpD'] = m.pD_100k - m.pD_20k
    for v, g in m.groupby('variant'):
        i = g.dpD.abs().idxmax()
        rows.append(dict(item='pD', variant=v, family=g.family.iloc[0], max_abs_dpD=g.dpD.abs().max(), mean_dpD=g.dpD.mean(),
                         at=f"pi={g.pi[i]} r1={g.r1[i]} q={g.q[i]} d={g.d[i]} D={g.D[i]}"))
    fo = pd.read_csv(os.path.join(E1, 'metrics_fineq.csv')); fn = pd.read_csv(os.path.join(W, 'metrics_fineq.csv'))
    fm = fo.merge(fn, on=['variant', 'pi', 'r1', 'q', 'd', 'D'], suffixes=('_20k', '_100k'))
    fm['dpD'] = fm.pD_100k - fm.pD_20k
    for v, g in fm.groupby('variant'):
        rows.append(dict(item='pD_fineq', variant=v, max_abs_dpD=g.dpD.abs().max(), mean_dpD=g.dpD.mean()))
    cmpd = pd.DataFrame(rows)
    cmpd.to_csv(os.path.join(E1, 'compare_20k_vs_100k_pD.csv'), index=False)
    ro = pd.read_csv(os.path.join(E1, 'ranking_repaired_with_without_ddm.csv'))
    rn = pd.read_csv(os.path.join(W, 'ranking_repaired_with_without_ddm.csv'))
    rr = ro.merge(rn, on=['ddm_included', 'D', 'monitor'], suffixes=('_20k', '_100k'))
    for s in ('_20k', '_100k'):
        rr['order' + s] = rr.groupby(['ddm_included', 'D'])['mean_rank_all' + s].rank(method='min')
    rr.to_csv(os.path.join(E1, 'compare_20k_vs_100k_ranking.csv'), index=False)
    qo = pd.read_csv(os.path.join(E1, 'qmin_interp_fineq.csv')); qn = pd.read_csv(os.path.join(W, 'qmin_interp_fineq.csv'))
    qq = qo.merge(qn, on=['variant', 'd', 'D'], suffixes=('_20k', '_100k'))
    qq.to_csv(os.path.join(E1, 'compare_20k_vs_100k_qmin_interp.csv'), index=False)
    so = pd.read_csv(os.path.join(E1, 'qmin_summary.csv')); sn = pd.read_csv(os.path.join(W, 'qmin_summary.csv'))
    ss = so.merge(sn, on=['variant', 'family', 'pi', 'r1', 'd'], suffixes=('_20k', '_100k'))
    ss[(ss.q_min_point_20k != ss.q_min_point_100k) | (ss.q_min_lower_20k != ss.q_min_lower_100k)].to_csv(
        os.path.join(E1, 'compare_20k_vs_100k_qmin_grid_changes.csv'), index=False)
    print('comparison written')


if __name__ == '__main__':
    main()
