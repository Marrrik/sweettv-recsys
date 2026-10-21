"""Time snapshots: user-airing history before a cutoff week, the target after it and the schedule of the target window.

A snapshot with cutoff week T uses only views before the Monday of week T, clipped at that moment, for everything a
model may learn from. The target window starts at T. Its schedule is public in advance, so models may use it to know
which programme airs where and when, but never who watched it.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import data, join, target

DAY = 24 * 3600
WEEK = 7 * DAY
WEEK1_MONDAY = int((pd.Timestamp('2019-12-30') - pd.Timestamp('1970-01-01')) // pd.Timedelta(seconds=1))
FIRST_WEEK = 11


def week_start(week):
    """Monday 00:00 of an ISO week of 2020 on the data clock."""
    return WEEK1_MONDAY + (week - 1) * WEEK


@dataclass
class Snapshot:
    cutoff_week: int
    end_week: int
    history: pd.DataFrame
    history_slots: pd.DataFrame
    window_slots: pd.DataFrame
    active_days: pd.DataFrame
    target: pd.Series
    target_freq: pd.DataFrame = None

    @property
    def window_weeks(self):
        return self.end_week - self.cutoff_week + 1

    @property
    def history_weeks(self):
        return self.cutoff_week - FIRST_WEEK

    @property
    def t_cutoff(self):
        return week_start(self.cutoff_week)

    @property
    def pool(self):
        """Candidate pool: labelled programmes with at least one airing in the target window."""
        return set(self.window_slots.tv_show_id.unique())

    @property
    def history_users(self):
        return pd.Index(self.history.user_id.unique())


def active_days(views, t0, t1):
    """(user_id, day) pairs on which the user has any viewing inside [t0, t1)."""
    s = np.maximum(views.start.to_numpy(), t0)
    e = np.minimum(views.stop.to_numpy(), t1)
    keep = e > s
    s, e, users = s[keep], e[keep], views.user_id.to_numpy()[keep]
    d0, d1 = s // DAY, (e - 1) // DAY
    n = d1 - d0 + 1
    rep = np.repeat(np.arange(len(n)), n)
    days = d0[rep] + np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
    return pd.DataFrame({'user_id': users[rep], 'day': days}).drop_duplicates().reset_index(drop=True)


def build(views, schedule, cutoff_week, end_week, with_target=True):
    """Snapshot with history from week 11 up to `cutoff_week` and a target window from `cutoff_week` to `end_week`."""
    t0, t1, t2 = week_start(FIRST_WEEK), week_start(cutoff_week), week_start(end_week + 1)
    lab = data.unique_slots(schedule[schedule.labelled])
    history_slots = lab[(lab.end > t0) & (lab.start < t1)].reset_index(drop=True)
    window_slots = lab[(lab.start >= t1) & (lab.start < t2)].reset_index(drop=True)
    history = join.user_airings(views, history_slots, window=(t0, t1))
    tgt, freq = None, None
    if with_target:
        freq = target.frequency(join.user_airings(views, window_slots, window=(t1, t2)))
        tgt = target.top_k(freq)
    return Snapshot(cutoff_week, end_week, history, history_slots, window_slots, active_days(views, t0, t1), tgt, freq)
