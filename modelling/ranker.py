# %% [markdown]
# # Candidate generation and LightGBM re-ranking
#
# The ranker is trained on S16 (history weeks 11 to 15, labels from weeks 16 to 20) with early stopping on 20% of its users, compared with the single methods on S21 and refitted on S16 and S21 together for the test window.

# %%
# Environment, data, snapshots and the collaborative models of every snapshot
import json
import os
import platform
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = os.path.abspath('..')
sys.path.insert(0, os.path.join(ROOT, 'src'))
from sweettv import baselines, cf, data, features, ranker, recs, snapshot  # noqa: E402

RAW, CACHE = os.path.join(ROOT, 'data', 'raw'), os.path.join(ROOT, 'data', 'cache')
SUB_DIR = os.path.join(ROOT, 'submissions')
pd.set_option('display.width', 250)
pd.set_option('display.max_columns', 40)
print(platform.platform(), '| python', platform.python_version(), '| numpy', np.__version__, '| pandas', pd.__version__,
      '| lightgbm', lgb.__version__)
T0 = time.time()
views = data.load_views(RAW, CACHE)
sched = data.load_schedule(RAW, CACHE)
test_users = pd.Index(data.load_submission(RAW).user_id)
SETTINGS = json.load(open(os.path.join(ROOT, 'modelling', 'cf_results.json')))['settings']
print('collaborative settings chosen in cf_models:', SETTINGS)
lab = sched[sched.labelled]
prog = lab.groupby('tv_show_id').agg(title=('tv_show_title', 'first'), category=('tv_show_category', 'first'),
                                     genre_1=('tv_show_genre_1', 'first'), genre_2=('tv_show_genre_2', 'first'),
                                     genre_3=('tv_show_genre_3', 'first'), year=('year_of_production', 'first'),
                                     director=('director', 'first'), actors=('actors', 'first'))
snaps = {name: snapshot.build(views, sched, cut, end, with_target=wt)
         for name, (cut, end, wt) in {'S16': (16, 20, True), 'S21': (21, 30, True), 'S31': (31, 42, False)}.items()}
fallbacks = {name: baselines.popularity_ranking(s) for name, s in snaps.items()}


def cf_models(name):
    m = cf.Interactions.from_snapshot(snaps[name])
    als_kw = dict(SETTINGS['als'])
    als_kw['factors'] = int(als_kw['factors'])
    return {'als': cf.als(m, **als_kw), 'ppr': cf.ppr(m, **SETTINGS['ppr']), 'rules': cf.association_rules(m, **SETTINGS['rules']),
            'content': cf.content(m, prog, sorted(snaps[name].pool)), 'habit': features.habit_scores(snaps[name], views)}


users = {'S16': snaps['S16'].target.index, 'S21': snaps['S21'].target.index, 'S31': test_users}
tables = {}
for name in ['S16', 'S21', 'S31']:
    t = time.time()
    cfs = cf_models(name)
    tables[name] = features.build(snaps[name], views, users[name], cfs, fallbacks[name], prog)
    print(f'{name}: {len(tables[name]):,} candidate rows for {tables[name].user_id.nunique()} users, {time.time() - t:.0f}s')
FEATS = features.feature_columns(tables['S16'])
print(len(FEATS), 'features:', FEATS)

# %%
# Candidate recall: share of target pairs that reach the ranker, overall and by source
rows = {}
for name in ['S16', 'S21']:
    df, s = tables[name], snaps[name]
    tp = s.target.explode().astype('int64').reset_index()
    m = tp.merge(df, on=['user_id', 'tv_show_id'], how='left')
    src = [c for c in df.columns if c.startswith('src_')]
    rows[name] = {'candidates per user, median': df.groupby('user_id').size().median(), 'recall of all sources': m.src_pop.notna().mean(),
                  **{f'recall {c[4:]}': (m[c] == 1).mean() for c in src}}
display(pd.DataFrame(rows).round(4))


# %%
# Train each objective on S16 and evaluate on S21
def map_s21(score, df=None, positive_only=False):
    df = tables['S21'] if df is None else df
    rank_score = ranker.user_rank_score(df, score)
    if positive_only:
        # a single heuristic leaves its empty places to the popularity fallback, as in the baselines notebook
        rank_score = np.where(np.asarray(score) > 0, rank_score, 0.0)
    sc = df[['user_id', 'tv_show_id']].assign(score=rank_score)
    r = recs.top5(sc, snaps['S21'].target.index, fallbacks['S21'], snaps['S21'].pool)
    return recs.evaluate(r, snaps['S21'].target, {'test users': test_users})


results, preds, best_iters, boosters = {}, {}, {}, {}
for obj in ranker.OBJECTIVES:
    t = time.time()
    b, it = ranker.fit(tables['S16'], obj)
    boosters[obj], best_iters[obj] = b, it
    preds[obj] = ranker.predict(b, tables['S21'])
    results[obj] = {**map_s21(preds[obj]), 'best iteration': it, 'seconds': round(time.time() - t)}
results['habit x schedule feature alone'] = map_s21(tables['S21'].hs.fillna(0).to_numpy(), positive_only=True)
display(pd.DataFrame(results).T.round(4))

# %%
# Ensembles: averaged reciprocal ranks of the objectives
rr = {obj: ranker.user_rank_score(tables['S21'], p) for obj, p in preds.items()}
ens = {
    'binary + lambdarank': rr['binary'] + rr['lambdarank'],
    'binary + poisson': rr['binary'] + rr['poisson'],
    'lambdarank + poisson': rr['lambdarank'] + rr['poisson'],
    'all three': rr['binary'] + rr['lambdarank'] + rr['poisson'],
}
ens_res = {k: map_s21(v) for k, v in ens.items()}
display(pd.DataFrame(ens_res).T.round(4))

# %%
# Feature importance (gain) of the binary model
imp = pd.Series(boosters['binary'].feature_importance('gain'), index=boosters['binary'].feature_name()).sort_values(ascending=False)
display((imp / imp.sum()).head(30).round(4).to_frame('gain share').T)

# %%
# Final models: refit every objective on S16 and S21 with 1.2 times the best iteration, predict the test window
final_train = pd.concat([tables['S16'], tables['S21']], ignore_index=True)
t31 = tables['S31']
p31 = {}
for obj in ranker.OBJECTIVES:
    b = ranker.refit(final_train, obj, int(best_iters[obj] * 1.2))
    p31[obj] = ranker.predict(b, t31)
s = snaps['S31']
out = {}
for name, score in {'ranker_binary': p31['binary'], 'ranker_lambdarank': ranker.user_rank_score(t31, p31['lambdarank']),
                    'ranker_poisson': p31['poisson'],
                    'ranker_ensemble': sum(ranker.user_rank_score(t31, p31[o]) for o in ranker.OBJECTIVES)}.items():
    sc = t31[['user_id', 'tv_show_id']].assign(score=ranker.user_rank_score(t31, score))
    r = recs.top5(sc, test_users, fallbacks['S31'], s.pool)
    recs.write_submission(r, test_users, os.path.join(SUB_DIR, f'{name}.csv'))
    out[name] = r
print(sorted(os.listdir(SUB_DIR)))
with open(os.path.join(ROOT, 'modelling', 'ranker_results.json'), 'w') as f:
    json.dump({'results': results, 'ensembles': ens_res, 'best_iterations': best_iters,
               'importance': (imp / imp.sum()).round(5).to_dict()}, f, indent=1, default=str)
print(f'total runtime {time.time() - T0:.0f}s')
