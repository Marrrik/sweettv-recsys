"""Two reference rankers: global popularity and the user's own habit scaled by the target-window schedule."""
import numpy as np
import pandas as pd

from . import target
from .snapshot import DAY, WEEK


def popularity_ranking(snap):
    """Pool programmes ordered by how many users have them in their history top five (non-strict).

    Programmes of the pool that are in nobody's top five follow, ordered by their airings in the target window.
    This ranking is the popularity baseline and the fallback of every method.
    """
    tops = target.top_k(target.frequency(snap.history), strict=False)
    counts = tops.explode().astype('int64').value_counts()
    pool = snap.pool
    ranked = [p for p in counts.index if p in pool]
    seen = set(ranked)
    by_airings = snap.window_slots.groupby('tv_show_id').size().sort_values(ascending=False, kind='stable')
    ranked += [p for p in by_airings.index if p not in seen]
    return ranked


def recency_weight(starts, t_cutoff, half_life_weeks):
    if half_life_weeks is None:
        return np.ones(len(starts))
    return 0.5 ** ((t_cutoff - np.asarray(starts)) / (half_life_weeks * WEEK))


def exposure(pairs, slots, active, t_cutoff, half_life_weeks):
    """Recency-weighted airings of each (programme, channel) that fell on a day the user was active.

    `pairs` holds user_id, tv_show_id, channel_id. The logging gap removes whole days for many users, and counting
    only active days keeps their per-airing rates comparable with users that were logged throughout.
    """
    pc = pairs[['tv_show_id', 'channel_id']].drop_duplicates().reset_index(drop=True)
    pc['pc'] = np.arange(len(pc))
    s = slots.merge(pc, on=['tv_show_id', 'channel_id'])
    day0 = int(min(s.start.min(), active.day.min() * DAY) // DAY)
    n_days = int(max(s.start.max() // DAY, active.day.max()) - day0 + 1)
    air = np.zeros((len(pc), n_days), dtype=np.float32)
    np.add.at(air, (s.pc.to_numpy(), (s.start.to_numpy() // DAY - day0).astype(int)),
              recency_weight(s.start.to_numpy(), t_cutoff, half_life_weeks).astype(np.float32))
    users = pd.Index(pairs.user_id.unique())
    act = np.zeros((len(users), n_days), dtype=np.float32)
    a = active[active.user_id.isin(users)]
    act[users.get_indexer(a.user_id), (a.day.to_numpy() - day0).astype(int)] = 1.0
    p = pairs.merge(pc, on=['tv_show_id', 'channel_id'])
    ui, pi = users.get_indexer(p.user_id), p.pc.to_numpy()
    out = np.empty(len(p), dtype=np.float64)
    for lo in range(0, len(p), 200_000):
        hi = lo + 200_000
        out[lo:hi] = (act[ui[lo:hi]] * air[pi[lo:hi]]).sum(axis=1)
    return p.drop(columns='pc').assign(exposure=out)


def habit_schedule_scores(snap, half_life_weeks=None, prior=0.0, use_schedule=True):
    """Expected number of fully watched airings of each programme in the target window.

    For every (user, programme, channel) the per-airing rate is the recency-weighted count of airings watched for at
    least 0.8 of their length, divided by the recency-weighted airings the user could have seen plus `prior`. The rate
    times the airings of that programme on that channel in the target window, summed over channels, is the score.
    With `use_schedule=False` the score is the weighted count itself, which ignores the window schedule.
    """
    q = snap.history[snap.history.frac >= target.SCREEN_TIME]
    q = q.assign(w=recency_weight(q.start.to_numpy(), snap.t_cutoff, half_life_weeks))
    qa = q.groupby(['user_id', 'tv_show_id', 'channel_id']).w.sum().reset_index()
    if not use_schedule:
        return qa.groupby(['user_id', 'tv_show_id']).w.sum().rename('score').reset_index()
    qa = exposure(qa, snap.history_slots, snap.active_days, snap.t_cutoff, half_life_weeks)
    win = snap.window_slots.groupby(['tv_show_id', 'channel_id']).size().rename('win_airings').reset_index()
    qa = qa.merge(win, on=['tv_show_id', 'channel_id'], how='left').fillna({'win_airings': 0})
    rate = (qa.w / (qa.exposure + prior).clip(lower=1e-9)).clip(upper=1.0)
    qa['score'] = rate * qa.win_airings
    return qa.groupby(['user_id', 'tv_show_id']).score.sum().reset_index()
