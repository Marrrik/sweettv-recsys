# %% [markdown]
# # Re-ranker, third round: habit coverage of whole airings and seed averaging
#
# Same candidates, snapshots and parameter set as experiment E3 of `ranker_v2.ipynb`. E3 is repeated with the old feature set, which also checks that the second round reproduces. E4 adds four coverage features: the user's usual viewing spread over every hour an airing covers, in the target window and in the history, the same over the last four history weeks, and fully watched airings per unit of history coverage. The better experiment is then averaged over three seeds. Selection is on S21 as before.

# %%
# Environment, data, snapshots, collaborative models and candidate tables with the coverage features
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
TRAIN = ['S14', 'S16', 'S18']
FINAL = ['S14', 'S16', 'S18', 'S21', 'S26']
PSET = 'regularised'
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
    tables[name] = features.build(s, views, users, cf_models(name), fallbacks[name], prog, n_cf=N_CF, n_pop=N_POP, coverage=True)
    gc.collect()
    print(f'{name}: {len(tables[name]):,} rows for {tables[name].user_id.nunique()} users, {time.time() - t:.0f}s')
ALL = features.feature_columns(tables['S16'])
OLD = [c for c in ALL if c not in features.COVERAGE_FEATURES]
print(len(OLD), 'old features,', len(ALL), 'with coverage')
display(tables['S21'][features.COVERAGE_FEATURES + ['habit_exposure', 'y']].corr(method='spearman').round(3))


# %%
# E3 (old features) and E4 (with coverage), binary and lambdarank, trained on S14, S16 and S18, evaluated on S21
def map_s21(score):
    df = tables['S21']
    sc = df[['user_id', 'tv_show_id']].assign(score=ranker.user_rank_score(df, score))
    r = recs.top5(sc, snaps['S21'].target.index, fallbacks['S21'], snaps['S21'].pool)
    return recs.evaluate(r, snaps['S21'].target, {'test users': test_users}), r


OBJS = ['binary', 'lambdarank']
train_df = pd.concat([tables[n] for n in TRAIN], ignore_index=True)
results, preds, iters = {}, {}, {}
for exp, cols in {'E3': OLD, 'E4': ALL}.items():
    for obj in OBJS:
        t = time.time()
        b, it = ranker.fit(train_df, obj, params=ranker.PARAM_SETS[PSET], cols=cols)
        preds[(exp, obj, 0)], iters[(exp, obj, 0)] = ranker.predict(b, tables['S21']), it
        res, _ = map_s21(preds[(exp, obj, 0)])
        results[f'{exp} {obj}'] = {**res, 'best iteration': it, 'seconds': round(time.time() - t)}
        print(exp, obj, results[f'{exp} {obj}'], flush=True)
    res, _ = map_s21(sum(ranker.user_rank_score(tables['S21'], preds[(exp, o, 0)]) for o in OBJS))
    results[f'{exp} binary + lambdarank'] = res
display(pd.DataFrame(results).T.round(4))

# %%
# Three seeds for the better experiment: seed changes the bagging, the feature subsampling and the early-stopping users
best_exp = 'E4' if results['E4 binary + lambdarank']['all'] > results['E3 binary + lambdarank']['all'] else 'E3'
cols_best = ALL if best_exp == 'E4' else OLD
SEEDS = [0, 1, 2]
for seed in SEEDS[1:]:
    for obj in OBJS:
        b, it = ranker.fit(train_df, obj, seed=seed, params=ranker.PARAM_SETS[PSET], cols=cols_best)
        preds[(best_exp, obj, seed)], iters[(best_exp, obj, seed)] = ranker.predict(b, tables['S21']), it
seed_score = sum(ranker.user_rank_score(tables['S21'], preds[(best_exp, o, sd)]) for o in OBJS for sd in SEEDS)
res, _ = map_s21(seed_score)
results[f'{best_exp} binary + lambdarank, 3 seeds'] = res
for sd in SEEDS:
    r_sd, _ = map_s21(sum(ranker.user_rank_score(tables['S21'], preds[(best_exp, o, sd)]) for o in OBJS))
    print(f'{best_exp} seed {sd} blend', {k: round(v, 4) for k, v in r_sd.items()})
display(pd.DataFrame(results).T.round(4))
del train_df
gc.collect()

# %%
# Selection on S21 among the blends, S21 lists of the selected model and its feature importance
blends = {k: v for k, v in results.items() if 'binary + lambdarank' in k}
selected = max(blends, key=lambda k: blends[k]['all'])
sel_exp = selected.split()[0]
sel_seeds = SEEDS if '3 seeds' in selected else [0]
sel_cols = ALL if sel_exp == 'E4' else OLD
print('selected:', selected, '| S21 MAP@5', round(blends[selected]['all'], 4))
score = sum(ranker.user_rank_score(tables['S21'], preds[(sel_exp, o, sd)]) for o in OBJS for sd in sel_seeds)
_, lists_s21 = map_s21(score)
recs.write_submission(lists_s21, snaps['S21'].target.index, os.path.join(ROOT, 'modelling', 's21_ranker_v3_top5.csv'))

# %%
# Final: refit the selected configuration on S14, S16, S18, S21 and S26 and predict the test window
for n in [n for n in tables if n not in FINAL + ['S21', 'S31']]:
    del tables[n]
final_df = pd.concat([tables[n] for n in FINAL], ignore_index=True)
for n in FINAL:
    del tables[n]
gc.collect()
print('final rows', len(final_df))
t31 = tables['S31']
score31 = np.zeros(len(t31))
for sd in sel_seeds:
    for obj in OBJS:
        t = time.time()
        b = ranker.refit(final_df, obj, int(iters[(sel_exp, obj, sd)] * 1.2), seed=sd, params=ranker.PARAM_SETS[PSET], cols=sel_cols)
        score31 += ranker.user_rank_score(t31, ranker.predict(b, t31))
        print(obj, 'seed', sd, 'refitted in', round(time.time() - t), 's', flush=True)
sc = t31[['user_id', 'tv_show_id']].assign(score=ranker.user_rank_score(t31, score31))
final_lists = recs.top5(sc, test_users, fallbacks['S31'], snaps['S31'].pool)
recs.write_submission(final_lists, test_users, os.path.join(SUB_DIR, 'ranker_v3.csv'))
imp = pd.Series(b.feature_importance('gain'), index=b.feature_name()).sort_values(ascending=False)
display((imp / imp.sum()).head(25).round(4).to_frame('gain share, last final model').T)
with open(os.path.join(ROOT, 'modelling', 'ranker_v3_results.json'), 'w') as f:
    json.dump({'results': results, 'selected': selected, 'seeds': sel_seeds, 'final_snapshots': FINAL, 'param_set': PSET,
               'iterations': {f'{k[0]} {k[1]} seed {k[2]}': v for k, v in iters.items()},
               'importance': (imp / imp.sum()).round(5).to_dict()}, f, indent=1, default=str)
print(f'total runtime {time.time() - T0:.0f}s')
