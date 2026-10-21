"""Load the raw competition files into typed frames and cache them as parquet."""
import os

import pandas as pd

VIEWS_FILE = 'dataset11-30.csv'
SUBMISSION_FILE = 'submission.csv'
SCHEDULE_FILES = {
    'w11_20': 'export_arh_11-20-final.csv',
    'w21_30': 'export_arh_21-30-final.csv',
    'w31_42': 'export_arh_31-42-final.csv',
}
TEXT_COLUMNS = ['channel_title', 'tv_show_title', 'tv_show_category', 'tv_show_genre_1', 'tv_show_genre_2',
                'tv_show_genre_3', 'year_of_production', 'director', 'actors']


def to_seconds(col, fmt):
    """Parse naive timestamps into integer seconds on one common clock."""
    return pd.to_datetime(col, format=fmt).astype('datetime64[s]').astype('int64')


def as_datetime(sec):
    return pd.to_datetime(sec, unit='s')


def iso_week(sec):
    return pd.Series(as_datetime(pd.Series(sec))).dt.isocalendar().week.astype('int16').to_numpy()


def load_views(raw_dir, cache_dir):
    """View log with start/stop in seconds and the ISO week of the start."""
    path = os.path.join(cache_dir, 'views.parquet')
    if os.path.exists(path):
        return pd.read_parquet(path)
    df = pd.read_csv(os.path.join(raw_dir, VIEWS_FILE),
                     dtype={'user_id': 'uint64', 'vsetv_id': 'int64', 'duraton': 'int64'})
    df['start'] = to_seconds(df.pop('start_time'), '%Y-%m-%d %H:%M:%S')
    df['stop'] = to_seconds(df.pop('stop_time'), '%Y-%m-%d %H:%M:%S')
    df['week'] = iso_week(df.start)
    os.makedirs(cache_dir, exist_ok=True)
    df.to_parquet(path, index=False)
    return df


def load_schedule(raw_dir, cache_dir):
    """All three schedule files with the source file in `part`, airing interval in seconds and a labelled flag."""
    path = os.path.join(cache_dir, 'schedule.parquet')
    if os.path.exists(path):
        return pd.read_parquet(path)
    frames = []
    for part, name in SCHEDULE_FILES.items():
        df = pd.read_csv(os.path.join(raw_dir, name), dtype={c: str for c in TEXT_COLUMNS})
        df['part'] = part
        frames.append(df)
    s = pd.concat(frames, ignore_index=True)
    s['tv_show_id'] = s.tv_show_id.fillna(0).astype('int64')
    s['duration'] = s.duration.astype('int64')
    s['start'] = to_seconds(s.pop('start_time'), '%d.%m.%Y %H:%M:%S')
    s['end'] = s.start + s.duration
    s['labelled'] = s.tv_show_id != 0
    s['week'] = iso_week(s.start)
    os.makedirs(cache_dir, exist_ok=True)
    s.to_parquet(path, index=False)
    return s


def load_submission(raw_dir):
    return pd.read_csv(os.path.join(raw_dir, SUBMISSION_FILE), dtype={'user_id': 'uint64', 'tv_show_id': str})


def unique_slots(schedule):
    """Schedule rows with exact repeats removed (the files overlap for half a day at each boundary)."""
    return schedule.drop_duplicates(['channel_id', 'start', 'end', 'tv_show_id']).reset_index(drop=True)
