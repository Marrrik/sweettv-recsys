# %% [markdown]
# # Re-ranker, fourth round: full-watch features, parameter variants, five seeds
#
# Same candidates and snapshots as the third round. E4 repeats its selected configuration with the old 67 features as a check. E5 adds the extra features of `features.EXTRA_FEATURES` (expected fully watched airings from the minimum and the product of the habit over an airing's hours, weekday-pooled and recency-weighted habit, the rate of fully watched airings in the last four weeks, the share of window airings in its first four weeks) and is tried with three parameter sets. The best blend is averaged over five seeds and fused with reciprocal ranks at k = 0 and k = 60. Everything is selected on S21.

# %%
# Environment, data, snapshots, collaborative models and candidate tables with all features
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
    tables[name] = features.build(s, views, users, cf_models(name), fallbacks[name], prog, n_cf=N_CF, n_pop=N_POP,
                                  coverage=True, extra=True)
    gc.collect()
    print(f'{name}: {len(tables[name]):,} rows for {tables[name].user_id.nunique()} users, {time.time() - t:.0f}s')
ALL = features.feature_columns(tables['S16'])
OLD = [c for c in ALL if c not in features.EXTRA_FEATURES]
print(len(OLD), 'features of round 3,', len(ALL), 'with the extra ones')
display(tables['S21'][features.EXTRA_FEATURES + ['habit_cov', 'habit_exposure', 'hs', 'y']].corr(method='spearman')['y'].round(3).to_frame('Spearman with y on S21').T)


# %%
# Experiments trained on S14, S16 and S18 and evaluated on S21
def map_s21(score):
    df = tables['S21']
    sc = df[['user_id', 'tv_show_id']].assign(score=score)
    r = recs.top5(sc, snaps['S21'].target.index, fallbacks['S21'], snaps['S21'].pool)
    return recs.evaluate(r, snaps['S21'].target, {'test users': test_users}), r


def blend(preds_list, k=0):
    return sum(ranker.user_rank_score(tables['S21'], p, k) for p in preds_list)


OBJS = ['binary', 'lambdarank']
EXPERIMENTS = {'E4': (OLD, 'regularised'), 'E5': (ALL, 'regularised'), 'E5 small': (ALL, 'small'), 'E5 wide': (ALL, 'wide')}
train_df = pd.concat([tables[n] for n in TRAIN], ignore_index=True)
results, preds, iters = {}, {}, {}
for exp, (cols, pset) in EXPERIMENTS.items():
    for obj in OBJS:
        t = time.time()
        b, it = ranker.fit(train_df, obj, params=ranker.PARAM_SETS[pset], cols=cols)
        preds[(exp, obj, 0)], iters[(exp, obj, 0)] = ranker.predict(b, tables['S21']), it
        res, _ = map_s21(ranker.user_rank_score(tables['S21'], preds[(exp, obj, 0)]))
        results[f'{exp} {obj}'] = {**res, 'best iteration': it, 'seconds': round(time.time() - t)}
        print(exp, obj, {k: round(v, 4) if isinstance(v, float) else v for k, v in results[f'{exp} {obj}'].items()}, flush=True)
    res, _ = map_s21(blend([preds[(exp, o, 0)] for o in OBJS]))
    results[f'{exp} blend'] = res
display(pd.DataFrame(results).T.round(4))

# %%
# Five seeds for the best E5 variant, fused at k = 0 and k = 60
best_exp = max([e for e in EXPERIMENTS if e.startswith('E5')], key=lambda e: results[f'{e} blend']['all'])
cols_best, pset_best = EXPERIMENTS[best_exp]
SEEDS = [0, 1, 2, 3, 4]
for seed in SEEDS[1:]:
    for obj in OBJS:
        b, it = ranker.fit(train_df, obj, seed=seed, params=ranker.PARAM_SETS[pset_best], cols=cols_best)
        preds[(best_exp, obj, seed)], iters[(best_exp, obj, seed)] = ranker.predict(b, tables['S21']), it
    print('seed', seed, 'done', flush=True)
for n_seeds in [3, 5]:
    for k in [0, 60]:
        res, _ = map_s21(blend([preds[(best_exp, o, sd)] for o in OBJS for sd in SEEDS[:n_seeds]], k))
        results[f'{best_exp} blend, {n_seeds} seeds, k={k}'] = res
display(pd.DataFrame(results).T.round(4))
del train_df
gc.collect()

# %%
# Selection on S21 among all blends, and the S21 lists of the selected model
blends = {k_: v for k_, v in results.items() if 'blend' in k_}
selected = max(blends, key=lambda k_: blends[k_]['all'])
sel_exp = best_exp if 'seeds' in selected else selected.replace(' blend', '')
sel_seeds = SEEDS[:int(selected.split(' seeds')[0].split()[-1])] if 'seeds' in selected else [0]
sel_k = int(selected.split('k=')[1]) if 'k=' in selected else 0
sel_cols, sel_pset = EXPERIMENTS[sel_exp]
print('selected:', selected, '| experiment', sel_exp, '| seeds', sel_seeds, '| k', sel_k, '| S21 MAP@5', round(blends[selected]['all'], 4))
_, lists_s21 = map_s21(blend([preds[(sel_exp, o, sd)] for o in OBJS for sd in sel_seeds], sel_k))
recs.write_submission(lists_s21, snaps['S21'].target.index, os.path.join(ROOT, 'modelling', 's21_ranker_v4_top5.csv'))

# %%
# Final: refit the selected configuration on S14, S16, S18, S21 and S26 and predict the test window
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
        b = ranker.refit(final_df, obj, int(iters[(sel_exp, obj, sd)] * 1.2), seed=sd, params=ranker.PARAM_SETS[sel_pset], cols=sel_cols)
        score31 += ranker.user_rank_score(t31, ranker.predict(b, t31), sel_k)
        print(obj, 'seed', sd, 'refitted in', round(time.time() - t), 's', flush=True)
sc = t31[['user_id', 'tv_show_id']].assign(score=ranker.user_rank_score(t31, score31))
final_lists = recs.top5(sc, test_users, fallbacks['S31'], snaps['S31'].pool)
recs.write_submission(final_lists, test_users, os.path.join(SUB_DIR, 'ranker_v4.csv'))
imp = pd.Series(b.feature_importance('gain'), index=b.feature_name()).sort_values(ascending=False)
display((imp / imp.sum()).head(25).round(4).to_frame('gain share, last final model').T)
with open(os.path.join(ROOT, 'modelling', 'ranker_v4_results.json'), 'w') as f:
    json.dump({'results': results, 'selected': selected, 'experiment': sel_exp, 'seeds': sel_seeds, 'k': sel_k,
               'final_snapshots': FINAL, 'param_set': sel_pset, 'n_features': len(sel_cols),
               'iterations': {f'{k_[0]} {k_[1]} seed {k_[2]}': v for k_, v in iters.items()},
               'importance': (imp / imp.sum()).round(5).to_dict()}, f, indent=1, default=str)
print(f'total runtime {time.time() - T0:.0f}s')
