"""Collaborative and content models fitted on the history of one snapshot.

All four share the user-programme matrix of airings watched for at least 0.8 of their length, weighted by recency.
Each returns a `Scores` object: a dense users x programmes matrix that `long` turns into a top-n table and `lookup`
reads for arbitrary pairs. Repeats are allowed, so a user's own programmes keep their scores.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits

from . import target
from .baselines import recency_weight


@dataclass
class Scores:
    users: pd.Index
    items: pd.Index
    values: np.ndarray

    def long(self, pool, topn):
        """Top-n pool programmes per user as user_id, tv_show_id, score."""
        cols = np.flatnonzero(self.items.isin(list(pool)))
        sub = self.values[:, cols]
        n = min(topn, len(cols))
        idx = np.argpartition(-sub, n - 1, axis=1)[:, :n]
        vals = np.take_along_axis(sub, idx, axis=1)
        return pd.DataFrame({'user_id': np.repeat(self.users.to_numpy(), n),
                             'tv_show_id': self.items.to_numpy()[cols[idx.ravel()]],
                             'score': vals.ravel()})

    def lookup(self, user_ids, item_ids):
        ui, ii = self.users.get_indexer(user_ids), self.items.get_indexer(item_ids)
        out = np.full(len(ui), np.nan, dtype=np.float32)
        ok = (ui >= 0) & (ii >= 0)
        out[ok] = self.values[ui[ok], ii[ok]]
        return out


@dataclass
class Interactions:
    users: pd.Index
    items: pd.Index
    X: sp.csr_matrix

    @classmethod
    def from_snapshot(cls, snap, half_life_weeks=10):
        q = snap.history[snap.history.frac >= target.SCREEN_TIME]
        w = recency_weight(q.start.to_numpy(), snap.t_cutoff, half_life_weeks)
        agg = q.assign(w=w).groupby(['user_id', 'tv_show_id']).w.sum().reset_index()
        users = pd.Index(np.sort(agg.user_id.unique()))
        items = pd.Index(np.sort(agg.tv_show_id.unique()))
        X = sp.csr_matrix((agg.w.to_numpy(), (users.get_indexer(agg.user_id), items.get_indexer(agg.tv_show_id))),
                          shape=(len(users), len(items)))
        return cls(users, items, X)

    @property
    def binary(self):
        B = self.X.copy()
        B.data = np.ones_like(B.data)
        return B


def _als_half(C, Y, reg):
    YtY = Y.T @ Y
    eye = reg * np.eye(Y.shape[1])
    out = np.zeros((C.shape[0], Y.shape[1]))
    for u in range(C.shape[0]):
        lo, hi = C.indptr[u], C.indptr[u + 1]
        if lo == hi:
            continue
        Yu, c = Y[C.indices[lo:hi]], C.data[lo:hi]
        out[u] = np.linalg.solve(YtY + (Yu.T * c) @ Yu + eye, Yu.T @ (1.0 + c))
    return out


def als(inter, factors=64, reg=1.0, alpha=10.0, iters=15, seed=0):
    """Implicit-feedback matrix factorisation (Hu, Koren and Volinsky 2008), confidence 1 + alpha * log(1 + w)."""
    C = inter.X.copy()
    C.data = alpha * np.log1p(C.data)
    Ct = C.T.tocsr()
    rng = np.random.default_rng(seed)
    U = rng.normal(0, 0.01, (C.shape[0], factors))
    V = rng.normal(0, 0.01, (C.shape[1], factors))
    # one small solve per row: a multi-threaded BLAS spends more time switching threads than computing here
    with threadpool_limits(limits=1, user_api='blas'):
        for _ in range(iters):
            U = _als_half(C, V, reg)
            V = _als_half(Ct, U, reg)
    return Scores(inter.users, inter.items, (U @ V.T).astype(np.float32))


def item_graph(inter, weighting='cosine'):
    """Programme co-view graph from users who fully watched both programmes, row-normalised into a transition matrix."""
    B = inter.binary
    W = (B.T @ B).tocsr().astype(np.float64)
    W.setdiag(0)
    W.eliminate_zeros()
    if weighting == 'cosine':
        d = np.sqrt(np.maximum(np.asarray(B.sum(axis=0)).ravel(), 1))
        W = sp.diags(1 / d) @ W @ sp.diags(1 / d)
    return normalize(W, norm='l1', axis=1).tocsr()


def ppr(inter, alpha=0.5, iters=30, weighting='cosine'):
    """Personalised PageRank on the programme graph, restarting at the user's own weighted history.

    `alpha` is the probability of following an edge, so 1 - alpha is the restart probability.
    """
    # the graph is dense enough (about a fifth of all pairs) that a dense BLAS product beats a sparse one
    P = item_graph(inter, weighting).toarray().astype(np.float32)
    R = normalize(inter.X.copy().astype(np.float64).log1p(), norm='l1', axis=1).toarray().astype(np.float32)
    S = R.copy()
    for _ in range(iters):
        S = (1 - alpha) * R + alpha * (S @ P)
    return Scores(inter.users, inter.items, S)


def association_rules(inter, min_support=10):
    """Score of a programme C for a user: the highest confidence P(C | A) over programmes A in the user's history.

    Rules come from pairs of programmes fully watched by at least `min_support` users. A programme that is only in
    the user's own history scores through rules from the user's other programmes.
    """
    B = inter.binary
    supp = np.asarray(B.sum(axis=0)).ravel()
    co = (B.T @ B).toarray().astype(np.float32)
    np.fill_diagonal(co, 0)
    conf = co / np.maximum(supp[:, None], 1)
    conf[co < min_support] = 0
    S = np.zeros(B.shape, dtype=np.float32)
    for u in range(B.shape[0]):
        idx = B.indices[B.indptr[u]:B.indptr[u + 1]]
        if len(idx):
            S[u] = conf[idx].max(axis=0)
    return Scores(inter.users, inter.items, S)


def programme_tokens(prog):
    """Content tokens of a programme: category, genres, decade of production, director and up to five actors."""
    def names(s, limit):
        return [] if not isinstance(s, str) else [x.strip() for x in s.split(',') if x.strip()][:limit]

    toks = []
    for r in prog.itertuples():
        t = [f'cat={r.category}'] + [f'genre={g}' for g in (r.genre_1, r.genre_2, r.genre_3) if isinstance(g, str)]
        if isinstance(r.year, str) and r.year[:4].isdigit():
            t.append(f'decade={int(r.year[:4]) // 10 * 10}')
        t += [f'director={x}' for x in names(r.director, 3)] + [f'actor={x}' for x in names(r.actors, 5)]
        toks.append(t)
    return toks


def content(inter, prog, items):
    """Cosine similarity between a TF-IDF profile of the user's history and every programme in `items`.

    `prog` holds programme attributes indexed by tv_show_id. It may include programmes that only air in the target
    window, since attributes come from the public schedule.
    """
    all_items = pd.Index(np.union1d(inter.items.to_numpy(), np.asarray(items)))
    vec = TfidfVectorizer(analyzer=lambda x: x, min_df=2, sublinear_tf=True)
    F = vec.fit_transform(programme_tokens(prog.reindex(all_items)))
    F = normalize(F)
    Fh = F[all_items.get_indexer(inter.items)]
    profile = normalize(inter.X.copy().astype(np.float64).log1p() @ Fh)
    Ft = F[all_items.get_indexer(items)]
    return Scores(inter.users, pd.Index(items), (profile @ Ft.T).toarray().astype(np.float32))
