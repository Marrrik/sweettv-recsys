# %% [markdown]
# # Re-ranker, second round
#
# Against the first round (`ranker.ipynb`) the candidate lists grow to the top 60 of every collaborative source (top 20 for content) and the top 60 popular programmes. Three experiments, all trained on labels before week 21 and compared on S21: E1 trains on S16 alone, E2 on S14, S16 and S18, E3 is E2 with a more regularised parameter set. The best experiment is refitted for the test window on all snapshots whose labels end by week 30.

# %%
# Environment, data, snapshots, collaborative models and candidate tables
import gc
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
SNAPS = {'S14': (14, 20), 'S16': (16, 20), 'S18': (18, 20), 'S21': (21, 30), 'S26': (26, 30), 'S31': (31, 42)}
N_CF = {'als': 60, 'ppr': 60, 'rules': 60, 'content': 20, 'habit': 60}
N_POP = 60
snaps = {name: snapshot.build(views, sched, cut, end, with_target=(name != 'S31')) for name, (cut, end) in SNAPS.items()}
fallbacks = {name: baselines.popularity_ranking(s) for name, s in snaps.items()}


def cf_models(name):
    m = cf.Interactions.from_snapshot(snaps[name])
    als_kw = dict(SETTINGS['als'])
    als_kw['factors'] = int(als_kw['factors'])
    return {'als': cf.als(m, **als_kw), 'ppr': cf.ppr(m, **SETTINGS['ppr']), 'rules': cf.association_rules(m, **SETTINGS['rules']),
            'content': cf.content(m, prog, sorted(snaps[name].pool)), 'habit': features.habit_scores(snaps[name], views)}


tables = {}
for name, s in snaps.items():
    t = time.time()
    users = test_users if name == 'S31' else s.target.index
    tables[name] = features.build(s, views, users, cf_models(name), fallbacks[name], prog, n_cf=N_CF, n_pop=N_POP)
    gc.collect()
    print(f'{name}: history weeks {snapshot.FIRST_WEEK} to {s.cutoff_week - 1}, target weeks {s.cutoff_week} to {s.end_week}, '
          f'{len(tables[name]):,} rows for {tables[name].user_id.nunique()} users, {time.time() - t:.0f}s')

# %%
# Candidate recall per snapshot
rows = {}
for name in ['S14', 'S16', 'S18', 'S21', 'S26']:
    df, s = tables[name], snaps[name]
    m = s.target.explode().astype('int64').reset_index().merge(df, on=['user_id', 'tv_show_id'], how='left')
    rows[name] = {'candidates per user, median': df.groupby('user_id').size().median(), 'recall': m.src_pop.notna().mean(),
                  'positives': int(df.y.sum())}
display(pd.DataFrame(rows).round(4))


# %%
# Experiments: train on labels before week 21, evaluate on S21
def map_s21(score):
    df = tables['S21']
    sc = df[['user_id', 'tv_show_id']].assign(score=ranker.user_rank_score(df, score))
    r = recs.top5(sc, snaps['S21'].target.index, fallbacks['S21'], snaps['S21'].pool)
    return recs.evaluate(r, snaps['S21'].target, {'test users': test_users}), r


EXPERIMENTS = {'E1': (['S16'], 'base'), 'E2': (['S14', 'S16', 'S18'], 'base'), 'E3': (['S14', 'S16', 'S18'], 'regularised')}
OBJS = ['binary', 'lambdarank']
results, preds, iters = {}, {}, {}
for exp, (train_names, pset) in EXPERIMENTS.items():
    train_df = pd.concat([tables[n] for n in train_names], ignore_index=True)
    for obj in OBJS:
        t = time.time()
        b, it = ranker.fit(train_df, obj, params=ranker.PARAM_SETS[pset])
        preds[(exp, obj)] = ranker.predict(b, tables['S21'])
        iters[(exp, obj)] = it
        res, _ = map_s21(preds[(exp, obj)])
        results[f'{exp} {obj}'] = {**res, 'train rows': len(train_df), 'best iteration': it, 'seconds': round(time.time() - t)}
        print(exp, obj, results[f'{exp} {obj}'], flush=True)
    rr = sum(ranker.user_rank_score(tables['S21'], preds[(exp, o)]) for o in OBJS)
    res, _ = map_s21(rr)
    results[f'{exp} binary + lambdarank'] = res
    del train_df
    gc.collect()
display(pd.DataFrame(results).T.round(4))

# %%
# Selection on S21 (all target users) and the S21 lists of the selected model for the analysis notebook
best_name = max(results, key=lambda k: results[k]['all'])
best_exp = best_name.split()[0]
best_objs = OBJS if best_name.endswith('binary + lambdarank') else [best_name.split()[1]]
print('selected:', best_name, '| objectives', best_objs, '| S21 MAP@5', round(results[best_name]['all'], 4))
score = sum(ranker.user_rank_score(tables['S21'], preds[(best_exp, o)]) for o in best_objs)
_, lists_s21 = map_s21(score)
recs.write_submission(lists_s21, snaps['S21'].target.index, os.path.join(ROOT, 'modelling', 's21_ranker_v2_top5.csv'))

# %%
# Final: refit the selected experiment on every snapshot whose labels end by week 30 and predict the test window
train_names, pset = EXPERIMENTS[best_exp]
final_names = train_names + ['S21', 'S26'] if len(train_names) > 1 else train_names + ['S21']
for n in [n for n in tables if n not in final_names + ['S31']]:
    del tables[n]
final_df = pd.concat([tables[n] for n in final_names], ignore_index=True)
for n in final_names:
    del tables[n]
gc.collect()
print('final training snapshots', final_names, '| rows', len(final_df))
t31 = tables['S31']
p31 = {}
for obj in best_objs:
    t = time.time()
    b = ranker.refit(final_df, obj, int(iters[(best_exp, obj)] * 1.2), params=ranker.PARAM_SETS[pset])
    p31[obj] = ranker.predict(b, t31)
    print(obj, 'refitted in', round(time.time() - t), 's')
score31 = sum(ranker.user_rank_score(t31, p31[o]) for o in best_objs)
sc = t31[['user_id', 'tv_show_id']].assign(score=ranker.user_rank_score(t31, score31))
final_lists = recs.top5(sc, test_users, fallbacks['S31'], snaps['S31'].pool)
recs.write_submission(final_lists, test_users, os.path.join(SUB_DIR, 'ranker_v2.csv'))
imp = pd.Series(b.feature_importance('gain'), index=b.feature_name()).sort_values(ascending=False)
display((imp / imp.sum()).head(25).round(4).to_frame(f'gain share, final {best_objs[-1]} model').T)
with open(os.path.join(ROOT, 'modelling', 'ranker_v2_results.json'), 'w') as f:
    json.dump({'results': results, 'selected': best_name, 'final_snapshots': final_names, 'param_set': pset,
               'iterations': {f'{k[0]} {k[1]}': v for k, v in iters.items()}, 'n_cf': N_CF, 'n_pop': N_POP,
               'importance': (imp / imp.sum()).round(5).to_dict()}, f, indent=1, default=str)
print(f'total runtime {time.time() - T0:.0f}s')
