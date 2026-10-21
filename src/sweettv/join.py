"""Join the view log to the broadcast schedule: same channel and overlapping time intervals."""
import numpy as np
import pandas as pd

BIG = np.int64(10**10)


def channel_codes(*channel_columns):
    ids = np.unique(np.concatenate([np.asarray(c) for c in channel_columns]))
    return pd.Series(np.arange(len(ids)), index=ids)


def encode(channels, codes):
    return pd.Series(channels).map(codes).fillna(-1).astype('int64').to_numpy()


def overlap_join(v_code, v_start, v_stop, s_code, s_start, s_end):
    """All (view, slot) pairs on the same channel with positive overlap, as (view index, slot index, overlap seconds).

    Slots are sorted by channel and start. A running maximum of slot ends makes the lower search bound valid even
    when slots inside one channel overlap.
    """
    order = np.lexsort((s_start, s_code))
    sc, ss, se = s_code[order], s_start[order], s_end[order]
    emax = pd.Series(se).groupby(sc).cummax().to_numpy()
    lo = np.searchsorted(sc * BIG + emax, v_code * BIG + v_start, side='right')
    hi = np.searchsorted(sc * BIG + ss, v_code * BIG + v_stop, side='left')
    n = np.where(v_code >= 0, np.maximum(hi - lo, 0), 0)
    vidx = np.repeat(np.arange(len(n)), n)
    sidx = lo[vidx] + np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
    ov = np.minimum(v_stop[vidx], se[sidx]) - np.maximum(v_start[vidx], ss[sidx])
    keep = ov > 0
    return vidx[keep], order[sidx[keep]], ov[keep]


SLOT_COLUMNS = ['tv_show_id', 'channel_id', 'start', 'end', 'duration', 'week', 'tv_show_category']


def user_airings(views, slots, window=None):
    """Covered seconds per (user, airing), summed over the user's views of that airing.

    `slots` should hold the labelled schedule rows. `window=(t0, t1)` clips every view to [t0, t1) first, so a
    split at t1 never lets later viewing into the history. `frac` is covered seconds over slot length, capped at 1.
    """
    v_start, v_stop = views.start.to_numpy(), views.stop.to_numpy()
    if window is not None:
        v_start, v_stop = np.maximum(v_start, window[0]), np.minimum(v_stop, window[1])
    codes = channel_codes(slots.channel_id)
    vi, si, ov = overlap_join(encode(views.vsetv_id, codes), v_start, v_stop, encode(slots.channel_id, codes),
                              slots.start.to_numpy(), slots.end.to_numpy())
    pairs = pd.DataFrame({'user_id': views.user_id.to_numpy()[vi], 'slot': si, 'sec': ov})
    ua = pairs.groupby(['user_id', 'slot'], sort=False).agg(sec=('sec', 'sum'), n_views=('sec', 'size')).reset_index()
    for col in SLOT_COLUMNS:
        ua[col] = slots[col].to_numpy()[ua.slot.to_numpy()]
    ua['frac'] = np.minimum(ua.sec / ua.duration, 1.0)
    return ua
