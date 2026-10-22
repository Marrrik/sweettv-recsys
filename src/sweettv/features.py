"""Candidate generation and features for the re-ranker of one snapshot.

Candidates are the union of every pool programme the user touched in the history, the top popular pool programmes
and the top-n of each collaborative model. Features use only the history before the cutoff and the public schedule
of the target window. Counts that depend on the window length are given per week.
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp

from . import baselines, target
from .snapshot import DAY, FIRST_WEEK, WEEK, week_start

# cells per channel: hours of the week, or hours of the day pooled over weekdays
PERIODS = {'week': 168, 'day': 24}


def hour_of_week(sec):
    hours = np.asarray(sec) // 3600
    # day 0 of the clock is 1970-01-01, a Thursday, so the weekday is shifted by 3 (Monday is 0)
    return ((hours // 24 + 3) % 7) * 24 + hours % 24


def cell_of(sec, period='week'):
    return hour_of_week(sec) if period == 'week' else (np.asarray(sec) // 3600) % 24


def habit_matrix(views, snap, channels, last_weeks=None, period='week', half_life=None):
    """Users x (channel, hour of week or of day): share of that hour the user watched, averaged over the history.

    `last_weeks` restricts the history to the weeks just before the cutoff. `half_life` (weeks) weights recent hours
    more, normalised so that a user who watches the hour every week still gets 1.
    """
    t1 = snap.t_cutoff
    weeks = snap.history_weeks if last_weeks is None else min(last_weeks, snap.history_weeks)
    t0 = t1 - weeks * WEEK
    v = views[(views.stop > t0) & (views.start < t1) & views.vsetv_id.isin(channels.index)]
    s, e = np.maximum(v.start.to_numpy(), t0), np.minimum(v.stop.to_numpy(), t1)
    h0, h1 = s // 3600, (e - 1) // 3600
    n = h1 - h0 + 1
    rep = np.repeat(np.arange(len(n)), n)
    hour = h0[rep] + np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
    sec = (np.minimum(e[rep], (hour + 1) * 3600) - np.maximum(s[rep], hour * 3600)).astype(np.float64)
    norm = float(weeks)
    if half_life is not None:
        sec *= 0.5 ** ((t1 - hour * 3600) / (half_life * WEEK))
        norm = float((0.5 ** ((np.arange(weeks) + 0.5) / half_life)).sum())
    if period == 'day':
        norm *= 7
    cells = PERIODS[period]
    cell = channels.reindex(v.vsetv_id.to_numpy()[rep]).to_numpy() * cells + cell_of(hour * 3600, period)
    users = pd.Index(np.sort(v.user_id.unique()))
    H = sp.csr_matrix((sec / (3600.0 * norm), (users.get_indexer(v.user_id.to_numpy()[rep]), cell)),
                      shape=(len(users), len(channels) * cells))
    H.sum_duplicates()
    return users, H


def slot_hours(slots, channels, period='week'):
    """Every (slot, hour) a slot overlaps: slot row, hour cell and the share of the slot inside that hour."""
    st, en = slots.start.to_numpy(), slots.end.to_numpy()
    h0, h1 = st // 3600, (en - 1) // 3600
    n = h1 - h0 + 1
    rep = np.repeat(np.arange(len(n)), n)
    hour = h0[rep] + np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
    share = (np.minimum(en[rep], (hour + 1) * 3600) - np.maximum(st[rep], hour * 3600)) / (en - st).clip(min=1)[rep]
    cell = channels.reindex(slots.channel_id.to_numpy()[rep]).to_numpy() * PERIODS[period] + cell_of(hour * 3600, period)
    return rep, cell, share


def full_watch(users, H, channels, slots, items, weeks, period='week', min_share=0.25, batch=128):
    """Expected fully watched airings per week, from the habit share of every hour an airing covers.

    Watching 0.8 of an airing needs the user at the screen for almost all of its hours. For every airing the habit
    shares of the hours holding at least `min_share` of it are combined in two ways: their minimum (the hours are
    watched together) and their product (the hours are independent). Airings with the same set of hour cells are
    grouped, so the work per user is one gather over the distinct sets. Returns two users x items arrays.
    """
    s = slots[slots.channel_id.isin(channels.index) & slots.tv_show_id.isin(items)].reset_index(drop=True)
    rep, cell, share = slot_hours(s, channels, period)
    keep = share >= min_share
    keep[pd.Series(share).groupby(rep).idxmax().to_numpy()] = True
    hours = pd.DataFrame({'slot': rep[keep], 'cell': cell[keep]}).sort_values(['slot', 'cell'])
    pattern = hours.groupby('slot').cell.apply(tuple)
    codes, uniq = pd.factorize(pattern)
    pad = H.shape[1]
    K = np.full((len(uniq), max(len(t) for t in uniq)), pad, dtype=np.int64)
    for i, t in enumerate(uniq):
        K[i, :len(t)] = t
    KP = sp.csr_matrix((np.full(len(codes), 1.0 / weeks), (codes, items.get_indexer(s.tv_show_id.to_numpy()[pattern.index]))),
                       shape=(len(uniq), len(items)))
    KP.sum_duplicates()
    KPT = KP.T.tocsr()
    out_min = np.zeros((len(users), len(items)), dtype=np.float32)
    out_prod = np.zeros((len(users), len(items)), dtype=np.float32)
    for b0 in range(0, len(users), batch):
        Hd = np.minimum(H[b0:b0 + batch].toarray().astype(np.float32), 1.0)
        # the padding column is 1, neutral for both the minimum and the product
        Hd = np.hstack([Hd, np.ones((Hd.shape[0], 1), dtype=np.float32)])
        G = Hd[:, K]
        out_min[b0:b0 + batch] = np.asarray(KPT @ G.min(axis=2).T).T
        out_prod[b0:b0 + batch] = np.asarray(KPT @ np.exp(np.log(np.maximum(G, 1e-4)).sum(axis=2)).T).T
    return out_min, out_prod


def coverage_matrix(slots, channels, items, weeks, period='week'):
    """(channel, hour of week) x programmes, with every airing spread over the hours it covers.

    An airing adds, to each hour cell it overlaps, the share of its own length that falls into that hour, divided by
    the number of weeks. Multiplied by a habit matrix this gives, per week, the summed expected share of each airing
    that the user's usual viewing covers, which is what the 0.8 rule asks for.
    """
    s = slots[slots.channel_id.isin(channels.index) & slots.tv_show_id.isin(items)]
    rep, cell, share = slot_hours(s, channels, period)
    G = sp.csr_matrix((share / weeks, (cell, items.get_indexer(s.tv_show_id.to_numpy()[rep]))),
                      shape=(len(channels) * PERIODS[period], len(items)))
    G.sum_duplicates()
    return G


def airing_matrix(slots, channels, items, weeks):
    """(channel, hour of week) x programmes: airings per week that start in the cell."""
    s = slots[slots.channel_id.isin(channels.index) & slots.tv_show_id.isin(items)]
    cell = channels.reindex(s.channel_id.to_numpy()).to_numpy() * 168 + hour_of_week(s.start.to_numpy())
    G = sp.csr_matrix((np.full(len(s), 1.0 / weeks), (cell, items.get_indexer(s.tv_show_id))),
                      shape=(len(channels) * 168, len(items)))
    G.sum_duplicates()
    return G


def habit_scores(snap, views):
    """Scores over the pool: airings per week of each programme in the channel-hour cells the user habitually watches."""
    from .cf import Scores
    channels = pd.Series(np.arange(views.vsetv_id.nunique()), index=np.sort(views.vsetv_id.unique()))
    users, H = habit_matrix(views, snap, channels)
    items = pd.Index(np.sort(list(snap.pool)))
    return Scores(users, items, (H @ airing_matrix(snap.window_slots, channels, items, snap.window_weeks)).toarray().astype(np.float32))


def candidates(snap, users, cf_scores, fallback, n_cf=30, n_pop=50):
    """`n_cf` is one top-n for every model or a dict model name -> top-n."""
    pool = snap.pool
    touched = snap.history[['user_id', 'tv_show_id']].drop_duplicates()
    parts = [touched[touched.user_id.isin(users) & touched.tv_show_id.isin(pool)].assign(src_touched=1)]
    pop = pd.MultiIndex.from_product([users, fallback[:n_pop]], names=['user_id', 'tv_show_id']).to_frame(index=False)
    parts.append(pop.assign(src_pop=1))
    for name, sc in cf_scores.items():
        top = sc.long(pool, n_cf[name] if isinstance(n_cf, dict) else n_cf)
        parts.append(top[top.user_id.isin(users)][['user_id', 'tv_show_id']].assign(**{f'src_{name}': 1}))
    cand = pd.concat(parts, ignore_index=True).fillna(0)
    flags = [c for c in cand.columns if c.startswith('src_')]
    return cand.groupby(['user_id', 'tv_show_id'], as_index=False)[flags].max()


def _user_rank(df, col):
    return df.groupby('user_id')[col].rank(ascending=False, method='first').astype(np.float32)


COVERAGE_FEATURES = ['habit_cov', 'habit_cov_hist', 'habit_recent_cov', 'q_per_cov']
EXTRA_FEATURES = ['fw_min', 'fw_prod', 'fwd_min', 'fwd_prod', 'fw_hist_min', 'q_per_fw', 'habit_day_cov', 'habit_rec_cov',
                  'exposure_last4', 'rate_last4', 'win_early_ratio', 'rank_fw_min', 'rank_habit_rec_cov']


def build(snap, views, users, cf_scores, fallback, prog, hs_half_life=10, hs_prior=30, n_cf=30, n_pop=50, coverage=False,
          extra=False):
    """Candidate table with features and, when the snapshot has a target, the labels `y` and `freq_pw`.

    `coverage=True` adds COVERAGE_FEATURES: habit scores with every airing spread over the hours it covers, over the
    whole history and over its last four weeks. `extra=True` adds EXTRA_FEATURES: expected fully watched airings from
    the minimum and the product of the habit over an airing's hours, weekday-pooled and recency-weighted habit, the
    rate of fully watched airings in the last four weeks and the share of window airings in its first four weeks.
    """
    t1, hw, ww = snap.t_cutoff, snap.history_weeks, snap.window_weeks
    df = candidates(snap, users, cf_scores, fallback, n_cf, n_pop)
    h = snap.history
    h = h.assign(q=(h.frac >= target.SCREEN_TIME).astype(np.float32), half=(h.frac >= 0.5).astype(np.float32),
                 w=baselines.recency_weight(h.start.to_numpy(), t1, hs_half_life).astype(np.float32),
                 age=(t1 - h.start.to_numpy()) / WEEK)
    h['qw'] = h.q * h.w
    for k in [1, 2, 4]:
        h[f'q_last{k}'] = h.q * (h.age <= k)
    h['q_start'] = np.where(h.q > 0, h.start, np.nan)
    up = h.groupby(['user_id', 'tv_show_id']).agg(
        n_touch=('sec', 'size'), sec_sum=('sec', 'sum'), frac_mean=('frac', 'mean'), frac_max=('frac', 'max'),
        n_q=('q', 'sum'), n_half=('half', 'sum'), n_qw=('qw', 'sum'), q_last1=('q_last1', 'sum'),
        q_last2=('q_last2', 'sum'), q_last4=('q_last4', 'sum'), last_start=('start', 'max'),
        first_start=('start', 'min'), last_q=('q_start', 'max'), n_channels_watched=('channel_id', 'nunique')).reset_index()
    up['weeks_since_last'] = (t1 - up.pop('last_start')) / WEEK
    up['weeks_since_first'] = (t1 - up.pop('first_start')) / WEEK
    up['weeks_since_last_q'] = (t1 - up.pop('last_q')) / WEEK
    for c in ['n_touch', 'n_q', 'n_half', 'n_qw', 'sec_sum']:
        up[c] = up[c] / hw
    df = df.merge(up, on=['user_id', 'tv_show_id'], how='left')

    # habit times schedule with the selected setting and two variants
    for name, kw in {'hs': dict(half_life_weeks=hs_half_life, prior=hs_prior), 'hs_p0': dict(half_life_weeks=hs_half_life, prior=0.0),
                     'hs_p3': dict(half_life_weeks=None, prior=3.0)}.items():
        sc = baselines.habit_schedule_scores(snap, **kw).rename(columns={'score': name})
        sc[name] = sc[name] / ww
        df = df.merge(sc, on=['user_id', 'tv_show_id'], how='left')

    # touches per airing the user could have seen, over all channels where the user touched the programme
    tc = h[['user_id', 'tv_show_id', 'channel_id']].drop_duplicates()
    tc = tc.merge(df[['user_id', 'tv_show_id']], on=['user_id', 'tv_show_id'])
    ex = baselines.exposure(tc, snap.history_slots, snap.active_days, t1, None)
    ex = ex.groupby(['user_id', 'tv_show_id']).exposure.sum().rename('exposure').reset_index()
    df = df.merge(ex, on=['user_id', 'tv_show_id'], how='left')
    df['touch_rate'] = df.n_touch * hw / df.exposure.clip(lower=1)
    df['q_rate'] = df.n_q * hw / df.exposure.clip(lower=1)
    df['exposure'] = df.exposure / hw

    # schedule of the window and of the history
    win = snap.window_slots
    wp = win.groupby('tv_show_id').agg(win_airings=('start', 'size'), win_channels=('channel_id', 'nunique'),
                                       win_slot_min=('duration', 'median'))
    wp['win_airings'] = wp.win_airings / ww
    wp['win_slot_min'] = wp.win_slot_min / 60
    hp = snap.history_slots.groupby('tv_show_id').size().rename('hist_airings') / hw
    df = df.merge(wp, left_on='tv_show_id', right_index=True, how='left').merge(hp, left_on='tv_show_id', right_index=True, how='left')
    df['airing_ratio'] = df.win_airings / df.hist_airings
    uc = tc.merge(win.groupby(['tv_show_id', 'channel_id']).size().rename('n').reset_index(), on=['tv_show_id', 'channel_id'])
    uc = uc.groupby(['user_id', 'tv_show_id']).n.sum().rename('win_airings_user_channels') / ww
    df = df.merge(uc, left_on=['user_id', 'tv_show_id'], right_index=True, how='left')

    # habitual channel and hour cells, from all airtime
    channels = pd.Series(np.arange(views.vsetv_id.nunique()), index=np.sort(views.vsetv_id.unique()))
    hu, H = habit_matrix(views, snap, channels)
    items = pd.Index(np.sort(df.tv_show_id.unique()))
    HG = (H @ airing_matrix(win, channels, items, ww)).toarray().astype(np.float32)
    ui, ii = hu.get_indexer(df.user_id), items.get_indexer(df.tv_show_id)
    df['habit_exposure'] = np.where(ui >= 0, HG[np.maximum(ui, 0), ii], np.nan)
    HGh = (H @ airing_matrix(snap.history_slots, channels, items, hw)).toarray().astype(np.float32)
    df['habit_exposure_hist'] = np.where(ui >= 0, HGh[np.maximum(ui, 0), ii], np.nan)
    df['q_per_habit'] = df.n_q / df.habit_exposure_hist.clip(lower=1e-3)
    del HG, HGh
    if coverage:
        hu4, H4 = habit_matrix(views, snap, channels, last_weeks=4)
        ui4 = hu4.get_indexer(df.user_id)
        # one product at a time: each is a dense users x programmes matrix of a few hundred megabytes
        for col, rows, Hx, slots_x, weeks_x in [('habit_cov', ui, H, win, ww), ('habit_cov_hist', ui, H, snap.history_slots, hw),
                                                ('habit_recent_cov', ui4, H4, win, ww)]:
            dense = (Hx @ coverage_matrix(slots_x, channels, items, weeks_x)).toarray().astype(np.float32)
            df[col] = np.where(rows >= 0, dense[np.maximum(rows, 0), ii], np.nan)
            del dense
        df['q_per_cov'] = df.n_q / df.habit_cov_hist.clip(lower=1e-3)
    if extra:
        # expected fully watched airings from the habit of every hour an airing covers, weekly and weekday-pooled cells
        hud, Hd = habit_matrix(views, snap, channels, period='day')
        uid = hud.get_indexer(df.user_id)
        for prefix, (uidx, rows, Hx, period) in {'fw': (hu, ui, H, 'week'), 'fwd': (hud, uid, Hd, 'day')}.items():
            mn, pr = full_watch(uidx, Hx, channels, win, items, ww, period)
            df[f'{prefix}_min'] = np.where(rows >= 0, mn[np.maximum(rows, 0), ii], np.nan)
            df[f'{prefix}_prod'] = np.where(rows >= 0, pr[np.maximum(rows, 0), ii], np.nan)
            del mn, pr
        mn, _ = full_watch(hu, H, channels, snap.history_slots, items, hw, 'week')
        df['fw_hist_min'] = np.where(ui >= 0, mn[np.maximum(ui, 0), ii], np.nan)
        del mn
        df['q_per_fw'] = df.n_q / df.fw_hist_min.clip(lower=1e-3)
        dense = (Hd @ coverage_matrix(win, channels, items, ww, 'day')).toarray().astype(np.float32)
        df['habit_day_cov'] = np.where(uid >= 0, dense[np.maximum(uid, 0), ii], np.nan)
        hur, Hr = habit_matrix(views, snap, channels, half_life=4)
        uir = hur.get_indexer(df.user_id)
        dense = (Hr @ coverage_matrix(win, channels, items, ww)).toarray().astype(np.float32)
        df['habit_rec_cov'] = np.where(uir >= 0, dense[np.maximum(uir, 0), ii], np.nan)
        del dense
        # fully watched airings over the airings the user could have seen, in the last four history weeks
        t4 = t1 - 4 * WEEK
        ex4 = baselines.exposure(tc, snap.history_slots[snap.history_slots.start >= t4],
                                 snap.active_days[snap.active_days.day >= t4 // DAY], t1, None)
        ex4 = ex4.groupby(['user_id', 'tv_show_id']).exposure.sum().rename('exposure_last4').reset_index()
        df = df.merge(ex4, on=['user_id', 'tv_show_id'], how='left')
        df['rate_last4'] = df.q_last4 / df.exposure_last4.clip(lower=1)
        early = win[win.start < t1 + 4 * WEEK].groupby('tv_show_id').size() / 4
        df['win_early_ratio'] = df.tv_show_id.map(early).fillna(0) / df.win_airings.clip(lower=1e-6)

    # programme popularity in the history and static attributes
    q = h[h.q > 0]
    tops = target.top_k(target.frequency(h), strict=False)
    n_hist_users = h.user_id.nunique()
    ip = pd.DataFrame({'pop_top5': tops.explode().astype('int64').value_counts() / n_hist_users,
                       'pop_q_users': q.groupby('tv_show_id').user_id.nunique() / n_hist_users,
                       'pop_q_per_user': q.groupby('tv_show_id').size() / q.groupby('tv_show_id').user_id.nunique() / hw,
                       'pop_q_last4': q[q.age <= 4].groupby('tv_show_id').user_id.nunique() / n_hist_users})
    df = df.merge(ip, left_on='tv_show_id', right_index=True, how='left')
    st = prog[['category', 'year']].copy()
    st['category'] = st.category.astype('category').cat.codes.astype(np.int16)
    st['year'] = pd.to_numeric(st.year.str[:4], errors='coerce')
    df = df.merge(st, left_on='tv_show_id', right_index=True, how='left')
    df['new_programme'] = (~df.tv_show_id.isin(snap.history_slots.tv_show_id.unique())).astype(np.int8)

    # user activity
    uq = q.groupby(['user_id', 'tv_show_id']).size()
    us = pd.DataFrame({'u_active_days': snap.active_days.groupby('user_id').size() / hw,
                       'u_lab_hours': h.groupby('user_id').sec.sum() / 3600 / hw,
                       'u_q_programmes': uq.groupby(level=0).size(),
                       'u_q_total': uq.groupby(level=0).sum() / hw,
                       'u_q_5th': uq.groupby(level=0).apply(lambda s: np.sort(s.to_numpy())[-5] if len(s) >= 5 else 0) / hw,
                       'u_active_last_week': h[h.age <= 1].groupby('user_id').size().clip(upper=1)})
    df = df.merge(us, left_on='user_id', right_index=True, how='left')

    # collaborative scores and within-user ranks
    for name, sc in cf_scores.items():
        df[f'cf_{name}'] = sc.lookup(df.user_id.to_numpy(), df.tv_show_id.to_numpy())
    for col in ['hs', 'hs_p0', 'n_qw', 'habit_exposure'] + [f'cf_{n}' for n in cf_scores] + (['fw_min', 'habit_rec_cov'] if extra else []):
        df[f'rank_{col}'] = _user_rank(df, col)

    if snap.target is not None:
        tset = snap.target.explode().astype('int64').reset_index().assign(y=1)
        df = df.merge(tset, on=['user_id', 'tv_show_id'], how='left').fillna({'y': 0})
        fq = snap.target_freq[['user_id', 'tv_show_id', 'freq']]
        df = df.merge(fq, on=['user_id', 'tv_show_id'], how='left').fillna({'freq': 0})
        df['freq_pw'] = df.pop('freq') / ww
    float_cols = df.columns.difference(['user_id', 'tv_show_id', 'category'])
    df[float_cols] = df[float_cols].astype(np.float32)
    df['cutoff'] = np.int16(snap.cutoff_week)
    return df.sort_values(['user_id', 'tv_show_id']).reset_index(drop=True)


FEATURES_EXCLUDE = ['user_id', 'tv_show_id', 'y', 'freq_pw', 'cutoff']


def feature_columns(df):
    return [c for c in df.columns if c not in FEATURES_EXCLUDE]
