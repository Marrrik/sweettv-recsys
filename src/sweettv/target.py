"""The organisers' target rule and MAP@5.

From the competition data page: drop tv_show_id = 0, compute per user and airing the watched share of the airing,
keep airings watched for at least 0.8 of their length, count such airings per programme and take strictly the five
most frequent programmes per user. The page does not say how ties are broken. Here ties go to more covered seconds,
then to the smaller id.
"""
import numpy as np
import pandas as pd

SCREEN_TIME = 0.8
K = 5


def frequency(ua, threshold=SCREEN_TIME):
    """Per (user, programme): airings watched for at least `threshold` of their length, and their covered seconds."""
    q = ua[ua.frac >= threshold]
    return q.groupby(['user_id', 'tv_show_id']).agg(freq=('slot', 'size'), sec=('sec', 'sum')).reset_index()


def ranked(freq):
    """Programmes of every user in target order, with a 0-based `rank` column."""
    f = freq.sort_values(['user_id', 'freq', 'sec', 'tv_show_id'], ascending=[True, False, False, True])
    f['rank'] = f.groupby('user_id').cumcount()
    return f


def top_k(freq, k=K, strict=True):
    """Series user_id -> list of the top-k programmes. `strict` keeps only users with at least k programmes."""
    f = ranked(freq)
    if strict:
        n = f.groupby('user_id').size()
        f = f[f.user_id.isin(n.index[n >= k])]
    return f[f['rank'] < k].groupby('user_id').tv_show_id.apply(list)


def apk(actual, predicted, k=K):
    actual = set(actual)
    score, hits, seen = 0.0, 0, set()
    for i, p in enumerate(list(predicted)[:k]):
        if p in actual and p not in seen:
            hits += 1
            score += hits / (i + 1)
        seen.add(p)
    return score / min(len(actual), k) if actual else 0.0


def mapk(truth, recs, k=K):
    """Mean AP@k over the users in `truth` (Series user -> list). Users missing from `recs` score 0."""
    return float(np.mean([apk(truth[u], recs.get(u, []), k) for u in truth.index]))
