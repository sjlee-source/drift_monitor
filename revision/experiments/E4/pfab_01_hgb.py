"""E4 Pfab step 1: re-fit the pilot HGB identically (pfab_pilot/protocol.json), reproduce 16/140/217,
write the calib and eval candidate streams. Run: ~/.venvs/smt-pub/bin/python -I pfab_01_hgb.py
"""
import os
os.environ['OMP_NUM_THREADS'] = '4'
import json
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
PP = '/home/sjlee/Papers/mfg-agent/access_revision_2026-11-11/pfab_pilot/protocol.json'
OUT = '/home/sjlee/Papers/mfg-agent/access_revision_2026-11-11/exp_E4/pfab/'
P = json.load(open(PP))
D = '/home/sjlee/Dataset/pfab_aoi_2024/'
df = pd.read_csv(D + 'dataset.csv').drop(columns=['Unnamed: 0'])
df['timestamp'] = pd.to_datetime(df.timestamp)
df = df.sort_values('timestamp', kind='stable').reset_index(drop=True)
df['board'] = df.groupby('timestamp').ngroup()
N = len(df)
def cut(f):
    i = int(f * N); b = df.board.iloc[i]
    return int(df.index[df.board > b][0]) if df.board.iloc[i - 1] == b else i
c1, c2 = cut(.30), cut(.40)
df['split'] = np.where(df.index < c1, 'train', np.where(df.index < c2, 'calib', 'eval'))
assert df.groupby('board').split.nunique().max() == 1
cat = ['inspection_type', 'meta_feat1', 'meta_feat2', 'meta_feat3', 'meta_feat4']
feats = [c for c in df.columns if c.startswith('inspection_feat')] + cat
X = df[feats].copy()
for c in cat: X[c] = X[c].astype('category')
tr = df.split == 'train'; ca = df.split == 'calib'; ev = df.split == 'eval'
m = HistGradientBoostingClassifier(categorical_features='from_dtype', random_state=0,
                                   **{k: v for k, v in P['model']['params'].items() if k != 'random_state'})
m.fit(X[tr], df.loc[tr, 'class'])
df['s'] = m.predict_proba(X)[:, 1]
cd = df.loc[ca & (df['class'] == 1), 's'].values
res = {'n_calib_defects': int(len(cd))}
E = df[ev]
ORIG = (E.timestamp.min().normalize() - pd.Timedelta(days=E.timestamp.min().dayofweek))
df['week'] = ((df.timestamp - ORIG) // pd.Timedelta(days=7)).astype(int)
df['week_start'] = (ORIG + pd.to_timedelta(df.week * 7, unit='D')).dt.strftime('%m-%d')
for a in [0.01, 0.05, 0.10]:
    t = float(np.quantile(cd, a, method='inverted_cdf'))
    df[f'release_{a:g}'] = (df.s < t).astype(np.int8)
    res[f'thr_{a:g}'] = t
    for nm, mask in [('calib', ca), ('eval', ev)]:
        dd = df[mask & (df['class'] == 1)]
        res[f'{nm}_miss_{a:g}'] = int(dd[f'release_{a:g}'].sum())
print(res)
assert (res['eval_miss_0.01'], res['eval_miss_0.05'], res['eval_miss_0.1']) == (16, 140, 217), 'pilot not reproduced'
res['reproduced_pilot_16_140_217'] = True
res['sklearn'] = __import__('sklearn').__version__
json.dump(res, open(OUT + 'hgb_summary.json', 'w'), indent=1)
cols = ['timestamp', 'board', 'split', 'week', 'week_start', 'inspection_type', 'class', 's',
        'release_0.01', 'release_0.05', 'release_0.1']
df.loc[ca | ev, cols].to_csv(OUT + 'stream_calib_eval.csv.gz', index_label='row')
