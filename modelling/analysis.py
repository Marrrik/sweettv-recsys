# %% [markdown]
# # Method comparison on the hold-out S21, two example users and list properties
#
# All lists are built from history weeks 11 to 20 and compared with the target of weeks 21 to 30. The two example users are fixed by a rule before any list is looked at: among test users with a target on S21, A is the user closest to the 90th percentile of labelled viewing hours in weeks 11 to 20 and B the user closest to the 10th percentile among those with at least one airing watched for 0.8 of its length.

# %%
# Environment, data, snapshot S21 and the lists of every method
import json
import os
import platform
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.abspath('..')
sys.path.insert(0, os.path.join(ROOT, 'src'))
from sweettv import baselines, cf, data, recs, snapshot, target  # noqa: E402

RAW, CACHE = os.path.join(ROOT, 'data', 'raw'), os.path.join(ROOT, 'data', 'cache')
SUB_DIR = os.path.join(ROOT, 'submissions')
pd.set_option('display.width', 250)
pd.set_option('display.max_columns', 40)
pd.set_option('display.max_colwidth', 60)
print(platform.platform(), '| python', platform.python_version(), '| numpy', np.__version__, '| pandas', pd.__version__)
T0 = time.time()
views = data.load_views(RAW, CACHE)
sched = data.load_schedule(RAW, CACHE)
test_users = pd.Index(data.load_submission(RAW).user_id)
SETTINGS = json.load(open(os.path.join(ROOT, 'modelling', 'cf_results.json')))['settings']
BASE = json.load(open(os.path.join(ROOT, 'modelling', 'baselines_results.json')))['selected']
lab = sched[sched.labelled]
prog = lab.groupby('tv_show_id').agg(title=('tv_show_title', 'first'), category=('tv_show_category', 'first'),
                                     genre_1=('tv_show_genre_1', 'first'), genre_2=('tv_show_genre_2', 'first'),
                                     genre_3=('tv_show_genre_3', 'first'), year=('year_of_production', 'first'),
                                     director=('director', 'first'), actors=('actors', 'first'))
s = snapshot.build(views, sched, 21, 30)
truth = s.target
users = truth.index
fb = baselines.popularity_ranking(s)
m = cf.Interactions.from_snapshot(s)
als_kw = dict(SETTINGS['als'])
als_kw['factors'] = int(als_kw['factors'])
empty = pd.DataFrame({'user_id': pd.Series(dtype='uint64'), 'tv_show_id': pd.Series(dtype='int64'), 'score': pd.Series(dtype=float)})
score_tables = {
    'Popularity': empty,
    'Habit x schedule': baselines.habit_schedule_scores(s, BASE['half_life_weeks'], BASE['prior']),
    'ALS': cf.als(m, **als_kw).long(s.pool, 50),
    'PPR': cf.ppr(m, **SETTINGS['ppr']).long(s.pool, 50),
    'Association rules': cf.association_rules(m, **SETTINGS['rules']).long(s.pool, 50),
    'Content-based': cf.content(m, prog, sorted(s.pool)).long(s.pool, 50),
}
lists = {name: recs.top5(sc, users, fb, s.pool) for name, sc in score_tables.items()}
lists['Hybrid (LightGBM re-ranker)'] = recs.read_submission(os.path.join(ROOT, 'modelling', 's21_ranker_v3_top5.csv'))
ap = pd.DataFrame({name: [target.apk(truth[u], lst[u]) for u in users] for name, lst in lists.items()}, index=users)
print('MAP@5 on S21:', ap.mean().round(4).to_dict())

# %%
# Paired bootstrap over users: difference of MAP@5 against the hybrid, 95% intervals
rng = np.random.default_rng(0)
idx = rng.integers(0, len(ap), size=(2000, len(ap)))
rows = {}
for name in ap.columns:
    d = (ap['Hybrid (LightGBM re-ranker)'] - ap[name]).to_numpy()
    boots = d[idx].mean(axis=1)
    rows[name] = {'MAP@5': ap[name].mean(), 'hybrid minus method': d.mean(), 'ci low': np.quantile(boots, 0.025),
                  'ci high': np.quantile(boots, 0.975), 'users where hybrid is higher': int((d > 0).sum()),
                  'users where hybrid is lower': int((d < 0).sum())}
display(pd.DataFrame(rows).T.round(4))

# %%
# MAP@5 by user activity tercile (labelled viewing hours in weeks 11 to 20) and for test users
hours = s.history.groupby('user_id').sec.sum().reindex(users).fillna(0) / 3600
terc = pd.qcut(hours.rank(method='first'), 3, labels=['light', 'medium', 'heavy'])
seg = ap.groupby(terc, observed=True).mean().T
seg['test users'] = ap[ap.index.isin(test_users)].mean()
display(seg.round(4))

# %%
# List properties: familiar against new programmes, popularity, coverage and overlap with the popularity list
def pairs_of(df):
    # to_numpy() on uint64 and int64 columns would upcast to float64 and corrupt the 20-digit user ids
    return set(zip(df.user_id.tolist(), df.tv_show_id.tolist()))


q_pairs = pairs_of(target.frequency(s.history))
t_pairs = pairs_of(s.history[['user_id', 'tv_show_id']].drop_duplicates())
pop_share = target.top_k(target.frequency(s.history), strict=False).explode().astype('int64').value_counts() / s.history.user_id.nunique()
pop5 = set(fb[:5])
props = {}
for name, lst in lists.items():
    flat = pd.DataFrame([(int(u), int(p)) for u in users for p in lst[u]], columns=['user_id', 'tv_show_id']).astype({'user_id': 'uint64'})
    pair = list(zip(flat.user_id.tolist(), flat.tv_show_id.tolist()))
    flat['watched'] = [x in q_pairs for x in pair]
    flat['touched'] = [x in t_pairs for x in pair]
    flat['hit'] = [p in set(truth[u]) for u, p in pair]
    new = ~flat.touched
    props[name] = {'share watched at 0.8 before': flat.watched.mean(), 'share only touched before': (flat.touched & ~flat.watched).mean(),
                   'share new to the user': new.mean(), 'precision on familiar': flat.hit[flat.touched].mean(),
                   'precision on new': flat.hit[new].mean() if new.any() else np.nan,
                   'distinct programmes': flat.tv_show_id.nunique(), 'catalogue coverage of the pool': flat.tv_show_id.nunique() / len(s.pool),
                   'mean popularity (share of users with it in their top 5)': flat.tv_show_id.map(pop_share).fillna(0).mean(),
                   'mean shared with the popularity top 5': flat.tv_show_id.isin(pop5).groupby(flat.user_id).sum().mean(),
                   'category share Инфо': flat.tv_show_id.map(prog.category).eq('Инфо').mean(),
                   'category share Фильмы': flat.tv_show_id.map(prog.category).eq('Фильмы').mean()}
props_df = pd.DataFrame(props).T
display(props_df.round(4))
tflat = truth.explode().astype('int64')
print('actual targets: share watched at 0.8 before', round(np.mean([(int(u), int(p)) in q_pairs for u, p in tflat.items()]), 4),
      '| share only touched before', round(np.mean([(int(u), int(p)) in t_pairs and (int(u), int(p)) not in q_pairs for u, p in tflat.items()]), 4),
      '| share new to the user', round(np.mean([(int(u), int(p)) not in t_pairs for u, p in tflat.items()]), 4))

# %%
# The two example users, chosen by the rule in the header
cand = users[users.isin(test_users)]
h_hours = hours.reindex(cand)
n_q = target.frequency(s.history).groupby('user_id').size().reindex(cand).fillna(0)


def closest(series, value):
    d = (series - value).abs()
    return d[d == d.min()].index.min()


user_a = closest(h_hours, h_hours.quantile(0.9))
light = h_hours[n_q > 0]
user_b = closest(light, light.quantile(0.1))
for label, u in [('A (heavy)', user_a), ('B (light)', user_b)]:
    hq = target.frequency(s.history)
    hq = hq[hq.user_id == u].sort_values('freq', ascending=False)
    print(f'user {label}: {u} | labelled hours in weeks 11 to 20: {hours[u]:.1f} | programmes watched at 0.8: {len(hq)} | '
          f'AP@5 per method: {ap.loc[u].round(3).to_dict()}')
    display(hq.head(8).assign(title=hq.tv_show_id.map(prog.title), category=hq.tv_show_id.map(prog.category))
            [['tv_show_id', 'title', 'category', 'freq']].rename(columns={'freq': 'airings watched at 0.8, weeks 11 to 20'}))
    display(pd.DataFrame({'tv_show_id': truth[u], 'title': [prog.title[p] for p in truth[u]],
                          'category': [prog.category[p] for p in truth[u]]}).rename_axis(f'actual top 5 of user {label}, weeks 21 to 30'))


def show_lists(u):
    rows_ = []
    for name, lst in lists.items():
        for rank, p in enumerate(lst[u], start=1):
            key = (int(u), int(p))
            rows_.append({'method': name, 'rank': rank, 'tv_show_id': p, 'title': prog.title.get(p), 'category': prog.category.get(p),
                          'seen before': 'watched' if key in q_pairs else ('touched' if key in t_pairs else 'new'),
                          'in actual top 5': p in set(truth[u])})
    return pd.DataFrame(rows_).set_index(['method', 'rank'])


for label, u in [('A', user_a), ('B', user_b)]:
    print(f'Top 5 of every method for user {label} on S21')
    display(show_lists(u))

# %%
# The same two users in the test window: each method's submitted five programmes (no ground truth is available)
sub_files = {'Popularity': 'popularity.csv', 'Habit x schedule': 'habit_schedule.csv', 'ALS': 'als.csv', 'PPR': 'ppr.csv',
             'Association rules': 'rules.csv', 'Content-based': 'content.csv', 'Hybrid (LightGBM re-ranker)': 'ranker_v3.csv'}
subs = {name: recs.read_submission(os.path.join(SUB_DIR, f)) for name, f in sub_files.items()}
for label, u in [('A', user_a), ('B', user_b)]:
    rows_ = [{'method': name, 'rank': k + 1, 'tv_show_id': p, 'title': prog.title.get(p), 'category': prog.category.get(p)}
             for name, sub in subs.items() for k, p in enumerate(sub[u])]
    print(f'Submitted five programmes for user {label} ({u}), weeks 31 to 42')
    display(pd.DataFrame(rows_).set_index(['method', 'rank']))


def example(u):
    hq = target.frequency(s.history)
    hq = hq[hq.user_id == u].sort_values('freq', ascending=False)
    seen = lambda p: 'watched' if (int(u), int(p)) in q_pairs else ('touched' if (int(u), int(p)) in t_pairs else 'new')  # noqa: E731
    return {'user_id': int(u), 'hours_w11_20': float(hours[u]), 'programmes_watched_w11_20': int(len(hq)),
            'history_top': [[int(p), int(c)] for p, c in zip(hq.tv_show_id.head(8), hq.freq.head(8))],
            'actual_top5_w21_30': [int(p) for p in truth[u]],
            'ap5': {k: float(v) for k, v in ap.loc[u].items()},
            # places of the five that the method filled itself, before the popularity fallback (the hybrid always has
            # at least 60 popular candidates of its own)
            'filled_by_method': {name: int(min(5, ((sc.user_id == u) & (sc.score > 0) & sc.tv_show_id.isin(s.pool)).sum()))
                                 for name, sc in score_tables.items()},
            's21_lists': {name: [[int(p), seen(p), bool(p in set(truth[u]))] for p in lst[u]] for name, lst in lists.items()},
            's31_lists': {name: [int(p) for p in sub[u]] for name, sub in subs.items()}}


examples = {'A': example(user_a), 'B': example(user_b)}
ids = {p for e in examples.values() for p in [x[0] for x in e['history_top']] + e['actual_top5_w21_30']
       + [x[0] for lst in e['s21_lists'].values() for x in lst] + [p for lst in e['s31_lists'].values() for p in lst]}
examples['programmes'] = {int(p): {'title': prog.title.get(p), 'category': prog.category.get(p)} for p in sorted(ids)}
with open(os.path.join(ROOT, 'modelling', 'analysis_results.json'), 'w', encoding='utf-8') as f:
    json.dump({'map5': ap.mean().to_dict(), 'bootstrap': rows, 'segments': seg.to_dict(), 'properties': props_df.to_dict(),
               'user_a': int(user_a), 'user_b': int(user_b), 'examples': examples}, f, indent=1, default=str, ensure_ascii=False)
print(f'total runtime {time.time() - T0:.0f}s')
