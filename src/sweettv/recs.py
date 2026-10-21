"""Turn model scores into exactly five programme ids per user, evaluate them and write submissions.

Every method goes through `top5`: candidates outside the snapshot pool are dropped, repeats of already watched
programmes are allowed, and empty places are filled from the same popularity ranking.
"""
import pandas as pd

from . import target


def top5(scores, users, fallback, pool, k=target.K):
    """Dict user -> k ids. `scores` has user_id, tv_show_id, score. Only positive scores count."""
    s = scores[(scores.score > 0) & scores.tv_show_id.isin(pool)]
    s = s.sort_values(['user_id', 'score', 'tv_show_id'], ascending=[True, False, True])
    lists = s.groupby('user_id').head(k).groupby('user_id').tv_show_id.apply(list)
    out = {}
    for u in users:
        lst = list(lists.get(u, []))
        for p in fallback:
            if len(lst) == k:
                break
            if p not in lst:
                lst.append(p)
        out[u] = lst
    return out


def filled_share(scores, users, pool, k=target.K):
    """Share of the k places per user that the method fills itself, before the fallback."""
    s = scores[(scores.score > 0) & scores.tv_show_id.isin(pool)]
    n = s.groupby('user_id').size().clip(upper=k).reindex(users).fillna(0)
    return float(n.mean() / k)


def evaluate(recs, truth, groups=None):
    """MAP@5 overall and, optionally, per user group (dict name -> index of users)."""
    out = {'all': target.mapk(truth, recs)}
    for name, idx in (groups or {}).items():
        sub = truth[truth.index.isin(idx)]
        out[name] = target.mapk(sub, recs) if len(sub) else float('nan')
    return out


def write_submission(recs, users, path):
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        f.write('user_id,tv_show_id\n')
        for u in users:
            ids = recs[u]
            assert len(ids) == target.K and len(set(ids)) == target.K, u
            f.write(f'{u},{" ".join(str(int(p)) for p in ids)}\n')


def read_submission(path):
    df = pd.read_csv(path, dtype={'user_id': 'uint64', 'tv_show_id': str})
    return {u: [int(p) for p in s.split()] for u, s in zip(df.user_id, df.tv_show_id)}
