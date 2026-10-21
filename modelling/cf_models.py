# %% [markdown]
# # ALS, personalised PageRank, association rules and content-based filtering
#
# Every method is fitted on the history of a snapshot, scores the pool programmes, keeps repeats and fills empty places from the popularity ranking. Settings are chosen on S21 and checked on S16.

# %%
# Environment, data, snapshots and programme attributes
import json
import os
import platform
import sys
import time

import numpy as np
import pandas as pd
import scipy

ROOT = os.path.abspath('..')
sys.path.insert(0, os.path.join(ROOT, 'src'))
from sweettv import baselines, cf, data, recs, snapshot  # noqa: E402

RAW, CACHE = os.path.join(ROOT, 'data', 'raw'), os.path.join(ROOT, 'data', 'cache')
SUB_DIR = os.path.join(ROOT, 'submissions')
pd.set_option('display.width', 250)
print(platform.platform(), '| python', platform.python_version(), '| numpy', np.__version__, '| pandas', pd.__version__,
      '| scipy', scipy.__version__)
T0 = time.time()
views = data.load_views(RAW, CACHE)
sched = data.load_schedule(RAW, CACHE)
test_users = pd.Index(data.load_submission(RAW).user_id)
snaps = {name: snapshot.build(views, sched, cut, end, with_target=wt)
         for name, (cut, end, wt) in {'S16': (16, 20, True), 'S21': (21, 30, True), 'S31': (31, 42, False)}.items()}
fallbacks = {name: baselines.popularity_ranking(s) for name, s in snaps.items()}
inter = {name: cf.Interactions.from_snapshot(s) for name, s in snaps.items()}
lab = sched[sched.labelled]
prog = lab.groupby('tv_show_id').agg(title=('tv_show_title', 'first'), category=('tv_show_category', 'first'),
                                     genre_1=('tv_show_genre_1', 'first'), genre_2=('tv_show_genre_2', 'first'),
                                     genre_3=('tv_show_genre_3', 'first'), year=('year_of_production', 'first'),
                                     director=('director', 'first'), actors=('actors', 'first'))
for name, m in inter.items():
    print(name, 'matrix', m.X.shape, 'non-zeros', m.X.nnz)


def evaluate(name, scores):
    s = snaps[name]
    r = recs.top5(scores, s.target.index, fallbacks[name], s.pool)
    out = recs.evaluate(r, s.target, {'test users': test_users} if name == 'S21' else None)
    out['filled'] = recs.filled_share(scores, s.target.index, s.pool)
    return out


def fit(method, name, **kw):
    t = time.time()
    m = inter[name]
    if method == 'als':
        sc = cf.als(m, **kw)
    elif method == 'ppr':
        sc = cf.ppr(m, **kw)
    elif method == 'rules':
        sc = cf.association_rules(m, **kw)
    else:
        sc = cf.content(m, prog, sorted(snaps[name].pool))
    return sc, time.time() - t


# %%
# ALS: factors and confidence scale at reg 1, then the regularisation at the best pair
rows = []


def als_row(factors, alpha, reg):
    sc, sec = fit('als', 'S21', factors=factors, alpha=alpha, reg=reg)
    return {'factors': factors, 'alpha': alpha, 'reg': reg, 'seconds': round(sec, 1),
            **{f'S21 {k}': v for k, v in evaluate('S21', sc.long(snaps['S21'].pool, 50)).items()}}


for factors in [32, 64, 128]:
    for alpha in [3.0, 10.0, 30.0]:
        rows.append(als_row(factors, alpha, 1.0))
top = max(rows, key=lambda r: r['S21 all'])
for reg in [0.1, 10.0, 100.0]:
    rows.append(als_row(top['factors'], top['alpha'], reg))
# the first pass ended on the edges (fewest factors, smallest alpha, largest reg), so the grid is extended past them
for factors in [16, 32]:
    for alpha in [0.3, 1.0, 3.0]:
        for reg in [30.0, 100.0, 300.0, 1000.0, 3000.0]:
            if not any(r['factors'] == factors and r['alpha'] == alpha and r['reg'] == reg for r in rows):
                rows.append(als_row(factors, alpha, reg))
als_grid = pd.DataFrame(rows).sort_values('S21 all', ascending=False)
display(als_grid.round(4))
ALS = als_grid.iloc[0][['factors', 'alpha', 'reg']].to_dict()
ALS['factors'] = int(ALS['factors'])

# %%
# Personalised PageRank: walk probability and edge weighting
rows = []
for weighting in ['cosine', 'count']:
    for alpha in [0.3, 0.5, 0.7, 0.85]:
        sc, sec = fit('ppr', 'S21', alpha=alpha, weighting=weighting)
        rows.append({'weighting': weighting, 'alpha': alpha, 'seconds': round(sec, 1),
                     **{f'S21 {k}': v for k, v in evaluate('S21', sc.long(snaps['S21'].pool, 50)).items()}})
ppr_grid = pd.DataFrame(rows).sort_values('S21 all', ascending=False)
display(ppr_grid.round(4))
PPR = ppr_grid.iloc[0][['alpha', 'weighting']].to_dict()

# %%
# Association rules: minimum support of a rule
rows = []
for ms in [3, 10, 30, 100]:
    sc, sec = fit('rules', 'S21', min_support=ms)
    rows.append({'min support': ms, 'seconds': round(sec, 1),
                 **{f'S21 {k}': v for k, v in evaluate('S21', sc.long(snaps['S21'].pool, 50)).items()}})
rules_grid = pd.DataFrame(rows).sort_values('S21 all', ascending=False)
display(rules_grid.round(4))
RULES = {'min_support': int(rules_grid.iloc[0]['min support'])}

# %%
# All methods with the selected settings on S21 and S16, then fitted on S31 for submissions
SETTINGS = {'als': ALS, 'ppr': PPR, 'rules': RULES, 'content': {}}
print(SETTINGS)
summary, s31_scores = {}, {}
for method, kw in SETTINGS.items():
    for name in ['S16', 'S21', 'S31']:
        sc, sec = fit(method, name, **kw)
        if name == 'S31':
            s31_scores[method] = sc
            summary[(method, 'S31 seconds')] = round(sec, 1)
            continue
        res = evaluate(name, sc.long(snaps[name].pool, 50))
        for k, v in res.items():
            summary[(method, f'{name} {k}')] = v
        summary[(method, f'{name} seconds')] = round(sec, 1)
table = pd.Series(summary).unstack()
display(table.round(4))

# %%
# Write one submission per method
s = snaps['S31']
for method, sc in s31_scores.items():
    r = recs.top5(sc.long(s.pool, 50), test_users, fallbacks['S31'], s.pool)
    recs.write_submission(r, test_users, os.path.join(SUB_DIR, f'{method}.csv'))
print(sorted(os.listdir(SUB_DIR)))
with open(os.path.join(ROOT, 'modelling', 'cf_results.json'), 'w') as f:
    json.dump({'settings': SETTINGS, 'als_grid': als_grid.to_dict(orient='records'), 'ppr_grid': ppr_grid.to_dict(orient='records'),
               'rules_grid': rules_grid.to_dict(orient='records'), 'summary': {f'{k[0]} | {k[1]}': v for k, v in summary.items()}},
              f, indent=1, default=str)
print(f'total runtime {time.time() - T0:.0f}s')
