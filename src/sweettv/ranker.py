"""LightGBM re-ranker over the candidate tables of `features.build`.

Three objectives share the features: binary logloss on membership in the target five, LambdaRank with one group per
user, and Poisson regression on the weekly frequency of fully watched airings. Early stopping uses a fixed share of
the training users, never the validation snapshot.
"""
import lightgbm as lgb
import numpy as np
import pandas as pd

from .features import feature_columns

BASE_PARAMS = dict(learning_rate=0.05, num_leaves=63, min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.8,
                   bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=8)
OBJECTIVES = {
    'binary': dict(objective='binary', metric='binary_logloss'),
    'lambdarank': dict(objective='lambdarank', metric='ndcg', eval_at=[5], lambdarank_truncation_level=20),
    'poisson': dict(objective='poisson', metric='poisson'),
}


def _dataset(df, objective, cols):
    label = df.freq_pw if objective == 'poisson' else df.y.astype(int)
    # one group per user and snapshot, rows of a group are contiguous because each table is sorted by user
    group = df.groupby(['cutoff', 'user_id'], sort=False).size().to_numpy() if objective == 'lambdarank' else None
    return lgb.Dataset(df[cols], label=label, group=group, categorical_feature=['category'], free_raw_data=False)


def split_users(df, share=0.2, seed=0):
    users = df.user_id.unique()
    rng = np.random.default_rng(seed)
    held = set(rng.choice(users, int(len(users) * share), replace=False))
    mask = df.user_id.isin(held)
    return df[~mask], df[mask]


PARAM_SETS = {
    'base': {},
    'regularised': dict(learning_rate=0.03, num_leaves=31, min_data_in_leaf=300, feature_fraction=0.6, lambda_l2=10.0),
}


def fit(train_df, objective, seed=0, num_boost_round=3000, early_stopping=100, cols=None, params=None):
    """Fit with early stopping on held-out training users. Returns the booster and the best iteration."""
    cols = cols or feature_columns(train_df)
    tr, va = split_users(train_df, seed=seed)
    params = {**BASE_PARAMS, **(params or {}), **OBJECTIVES[objective], 'seed': seed}
    booster = lgb.train(params, _dataset(tr, objective, cols), num_boost_round=num_boost_round,
                        valid_sets=[_dataset(va, objective, cols)],
                        callbacks=[lgb.early_stopping(early_stopping, verbose=False)])
    return booster, booster.best_iteration


def refit(train_df, objective, num_boost_round, seed=0, cols=None, params=None):
    """Fit on all rows with a fixed number of rounds, for the final model."""
    cols = cols or feature_columns(train_df)
    params = {**BASE_PARAMS, **(params or {}), **OBJECTIVES[objective], 'seed': seed}
    return lgb.train(params, _dataset(train_df, objective, cols), num_boost_round=num_boost_round)


def predict(booster, df):
    return booster.predict(df[booster.feature_name()], num_iteration=booster.best_iteration or None)


def user_rank_score(df, score):
    """Turn a score into a within-user score in (0, 1], 1 for the best candidate, so objectives can be averaged."""
    r = pd.Series(score, index=df.index).groupby(df.user_id).rank(ascending=False, method='first')
    return 1.0 / r.to_numpy()
