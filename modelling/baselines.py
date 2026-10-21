# %% [markdown]
# # Baselines: popularity and habit times schedule
#
# Snapshots: S16 has history weeks 11 to 15 and target weeks 16 to 20, S21 has history weeks 11 to 20 and target weeks 21 to 30 (the hold-out used for selection), S31 has history weeks 11 to 30 and the test window 31 to 42 without a target.

# %%
# Environment, data and snapshots
import json
import os
import platform
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.abspath('..')
sys.path.insert(0, os.path.join(ROOT, 'src'))
from sweettv import baselines, data, recs, snapshot, target  # noqa: E402

RAW, CACHE = os.path.join(ROOT, 'data', 'raw'), os.path.join(ROOT, 'data', 'cache')
SUB_DIR = os.path.join(ROOT, 'submissions')
os.makedirs(SUB_DIR, exist_ok=True)
pd.set_option('display.width', 250)
pd.set_option('display.max_columns', 40)
print(platform.platform(), '| python', platform.python_version(), '| numpy', np.__version__, '| pandas', pd.__version__)
T0 = time.time()
views = data.load_views(RAW, CACHE)
sched = data.load_schedule(RAW, CACHE)
test_users = pd.Index(data.load_submission(RAW).user_id)
snaps = {}
for name, (cut, end, with_t) in {'S16': (16, 20, True), 'S21': (21, 30, True), 'S31': (31, 42, False)}.items():
    t = time.time()
    snaps[name] = snapshot.build(views, sched, cut, end, with_target=with_t)
    print(f'{name} built in {time.time() - t:.1f}s')
prog = sched[sched.labelled].groupby('tv_show_id').agg(title=('tv_show_title', 'first'), category=('tv_show_category', 'first'))

# %%
# What each snapshot contains
rows = {}
for name, s in snaps.items():
    hist_q = s.history[s.history.frac >= target.SCREEN_TIME]
    rows[name] = {
        'history weeks': f'{snapshot.FIRST_WEEK} to {s.cutoff_week - 1}', 'target weeks': f'{s.cutoff_week} to {s.end_week}',
        'users with history events': s.history.user_id.nunique(), 'users with a qualifying airing': hist_q.user_id.nunique(),
        'programmes in the pool': len(s.pool), 'labelled airings in the window': len(s.window_slots),
        'users with a target': len(s.target) if s.target is not None else len(test_users),
        'target users with a qualifying airing in history': int((s.target.index if s.target is not None else test_users).isin(hist_q.user_id).sum()),
    }
pd.DataFrame(rows)


# %%
# Evaluation groups: all target users, users active in the last history week (like the test users) and, for S21, test users
def groups(s):
    last = s.history[s.history.week == s.cutoff_week - 1].user_id.unique()
    g = {'active in the last history week': pd.Index(last)}
    if s.cutoff_week == 21:
        g['test users'] = test_users
    return g


def run(s, scores, fallback):
    r = recs.top5(scores, s.target.index, fallback, s.pool)
    out = recs.evaluate(r, s.target, groups(s))
    out['places filled before fallback'] = recs.filled_share(scores, s.target.index, s.pool)
    return out


for name in ['S16', 'S21']:
    s = snaps[name]
    print(name, 'users per group', {k: int(s.target.index.isin(v).sum()) for k, v in groups(s).items()})

# %%
# Popularity baseline: the same five programmes for everyone, taken from the history top-five lists and restricted to the pool
results = {}
fallbacks = {name: baselines.popularity_ranking(s) for name, s in snaps.items()}
for name in ['S16', 'S21']:
    s = snaps[name]
    empty = pd.DataFrame({'user_id': pd.Series(dtype='uint64'), 'tv_show_id': pd.Series(dtype='int64'), 'score': pd.Series(dtype=float)})
    results[(name, 'popularity')] = run(s, empty, fallbacks[name])
display(pd.DataFrame({k: v for k, v in results.items()}).T.round(4))
for name in ['S21', 'S31']:
    print(name, 'popularity top 5:', [(p, prog.title[p]) for p in fallbacks[name][:5]])

# %%
# Habit times schedule on the hold-out S21 and on S16: half-life of the recency weight and the prior of the per-airing rate
grid = []
for hl in [None, 20, 10, 5, 2.5]:
    for use_schedule, priors in [(False, [0.0]), (True, [0.0, 1.0, 3.0, 10.0, 30.0, 100.0])]:
        for prior in priors:
            row = {'half-life, weeks': 'none' if hl is None else hl, 'schedule': use_schedule, 'prior': prior}
            for name in ['S16', 'S21']:
                s = snaps[name]
                sc = baselines.habit_schedule_scores(s, half_life_weeks=hl, prior=prior, use_schedule=use_schedule)
                m = run(s, sc, fallbacks[name])
                row[f'{name} all'] = m['all']
                row[f'{name} active last week'] = m['active in the last history week']
                if name == 'S21':
                    row['S21 test users'] = m['test users']
                    row['S21 filled'] = m['places filled before fallback']
            grid.append(row)
grid = pd.DataFrame(grid)
grid.sort_values('S21 all', ascending=False).round(4)

# %%
# Selected setting: the best MAP@5 on S21 over all target users
best = grid.sort_values('S21 all', ascending=False).iloc[0]
HL = None if best['half-life, weeks'] == 'none' else float(best['half-life, weeks'])
PRIOR = float(best['prior'])
print('selected half-life', HL, '| prior', PRIOR, '| schedule', bool(best['schedule']))
for name in ['S16', 'S21']:
    results[(name, 'habit x schedule')] = run(snaps[name], baselines.habit_schedule_scores(snaps[name], HL, PRIOR), fallbacks[name])
    results[(name, 'habit count only')] = run(snaps[name], baselines.habit_schedule_scores(snaps[name], HL, PRIOR, use_schedule=False), fallbacks[name])
display(pd.DataFrame(results).T.round(4))

# %%
# Test window: fit on S31 and write both submissions
s = snaps['S31']
sub_pop = recs.top5(pd.DataFrame(columns=['user_id', 'tv_show_id', 'score']), test_users, fallbacks['S31'], s.pool)
sc31 = baselines.habit_schedule_scores(s, HL, PRIOR)
sub_hs = recs.top5(sc31, test_users, fallbacks['S31'], s.pool)
recs.write_submission(sub_pop, test_users, os.path.join(SUB_DIR, 'popularity.csv'))
recs.write_submission(sub_hs, test_users, os.path.join(SUB_DIR, 'habit_schedule.csv'))
print('test users', len(test_users), '| places filled by habit x schedule before fallback',
      round(recs.filled_share(sc31, test_users, s.pool), 4))
check = recs.read_submission(os.path.join(SUB_DIR, 'habit_schedule.csv'))
print('rows', len(check), '| all ids in the pool', all(p in s.pool for v in check.values() for p in v))
with open(os.path.join(ROOT, 'modelling', 'baselines_results.json'), 'w') as f:
    json.dump({'selected': {'half_life_weeks': HL, 'prior': PRIOR}, 'grid': grid.to_dict(orient='records'),
               'results': {f'{k[0]} {k[1]}': v for k, v in results.items()}}, f, indent=1, default=str)
print(f'total runtime {time.time() - T0:.0f}s')
