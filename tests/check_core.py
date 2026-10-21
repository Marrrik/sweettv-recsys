"""Hand-made checks of the core pieces. Run: python tests/check_core.py (from the repository root)."""
import os
import sys

import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))
from sweettv import cf, join, recs, target  # noqa: E402


def check_apk():
    # hits at ranks 1 and 2 of five relevant: (1/1 + 2/2) / 5 = 0.4
    assert abs(target.apk([1, 2, 3, 4, 5], [1, 2, 9, 9, 9]) - 0.4) < 1e-12
    # hits at ranks 4 and 5: (1/4 + 2/5) / 5 = 0.13
    assert abs(target.apk([1, 2, 3, 4, 5], [7, 8, 9, 1, 2]) - 0.13) < 1e-12
    # a repeated id counts once
    assert abs(target.apk([1, 2, 3, 4, 5], [1, 1, 1, 1, 1]) - 0.2) < 1e-12
    assert target.apk([1, 2, 3, 4, 5], [6, 7, 8, 9, 10]) == 0.0
    print('apk: ok')


def check_overlap_join():
    # channel 0 has slots [0, 100) and [100, 250), channel 1 has [0, 50)
    s_code = np.array([0, 0, 1])
    s_start, s_end = np.array([0, 100, 0]), np.array([100, 250, 50])
    # view a on channel 0 covers [90, 120): 10 s of slot 0 and 20 s of slot 1; view b on channel 1 covers [60, 70): nothing
    vi, si, ov = join.overlap_join(np.array([0, 1]), np.array([90, 60]), np.array([120, 70]), s_code, s_start, s_end)
    got = sorted(zip(vi.tolist(), si.tolist(), ov.tolist()))
    assert got == [(0, 0, 10), (0, 1, 20)], got
    print('overlap_join: ok')


def check_target_rule():
    # user 1: programme 10 watched fully on 3 airings, programme 20 on 1 airing, programme 30 only at 0.5
    ua = pd.DataFrame({'user_id': [1] * 5, 'slot': [0, 1, 2, 3, 4], 'tv_show_id': [10, 10, 10, 20, 30],
                       'frac': [1.0, 0.85, 0.8, 0.9, 0.5], 'sec': [100, 90, 80, 300, 50]})
    f = target.frequency(ua)
    assert f.set_index('tv_show_id').freq.to_dict() == {10: 3, 20: 1}
    # strict top-5 needs five programmes, the relaxed one keeps the order by frequency
    assert len(target.top_k(f)) == 0
    assert target.top_k(f, strict=False)[1] == [10, 20]
    print('target rule: ok')


def check_als_half_step():
    rng = np.random.default_rng(0)
    C = sp.random(30, 20, density=0.2, random_state=0, format='csr')
    C.data = 5 * C.data
    Y = rng.normal(size=(20, 4))
    U = cf._als_half(C, Y, reg=0.5)
    for u in range(30):
        cu = C[u].toarray().ravel()
        p = (cu > 0).astype(float)
        # Hu, Koren and Volinsky (2008), eq. (4): x_u = (Y^T C^u Y + lambda I)^-1 Y^T C^u p(u), with C^u = diag(1 + c)
        Cu = np.diag(1 + cu)
        ref = np.linalg.solve(Y.T @ Cu @ Y + 0.5 * np.eye(4), Y.T @ Cu @ p) if p.any() else np.zeros(4)
        assert np.allclose(U[u], ref, atol=1e-8), u
    print('ALS half step against the direct formula: ok')


def check_submission(path):
    sub = pd.read_csv(path, dtype={'user_id': 'uint64', 'tv_show_id': str})
    tmpl = pd.read_csv(os.path.join(ROOT, 'data', 'raw', 'submission.csv'), dtype={'user_id': 'uint64', 'tv_show_id': str})
    assert list(sub.columns) == ['user_id', 'tv_show_id']
    assert sub.user_id.tolist() == tmpl.user_id.tolist(), 'users or their order differ from the template'
    lists = recs.read_submission(path)
    assert all(len(v) == 5 and len(set(v)) == 5 for v in lists.values())
    print(f'{os.path.basename(path)}: {len(sub)} rows, same users and order as the template, five distinct ids each: ok')


if __name__ == '__main__':
    check_apk()
    check_overlap_join()
    check_target_rule()
    check_als_half_step()
    sub_dir = os.path.join(ROOT, 'submissions')
    for name in sorted(os.listdir(sub_dir)):
        if name.endswith('.csv') and os.path.exists(os.path.join(ROOT, 'data', 'raw', 'submission.csv')):
            check_submission(os.path.join(sub_dir, name))
