"""Figure 1 of the report: daily active users with the logging gap, and the covered share of user-airing events.

Run from the repository root after eda/eda.ipynb has written data/cache/user_airings_w11_30.parquet.
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))
from sweettv import data  # noqa: E402

RAW, CACHE = os.path.join(ROOT, 'data', 'raw'), os.path.join(ROOT, 'data', 'cache')
views = data.load_views(RAW, CACHE)
test_users = set(data.load_submission(RAW).user_id)
ua = pd.read_parquet(os.path.join(CACHE, 'user_airings_w11_30.parquet'), columns=['frac'])

day = data.as_datetime(views.start).dt.normalize()
users_all = views.groupby(day).user_id.nunique()
is_test = views.user_id.isin(test_users)
users_test = views[is_test].groupby(day[is_test]).user_id.nunique()

plt.rcParams.update({'font.size': 7, 'axes.titlesize': 7, 'axes.labelsize': 7, 'legend.fontsize': 6})
fig, ax = plt.subplots(1, 2, figsize=(3.5, 1.45))
ax[0].plot(users_all.index, users_all.to_numpy(), lw=0.8, label='all users')
ax[0].plot(users_test.index, users_test.to_numpy(), lw=0.8, label='test users')
ax[0].axvspan(pd.Timestamp('2020-06-16'), pd.Timestamp('2020-07-05'), color='grey', alpha=0.25, lw=0)
ax[0].set_ylabel('users per day')
ax[0].set_xticks(pd.to_datetime(['2020-03-15', '2020-05-01', '2020-06-15', '2020-07-26']))
ax[0].set_xticklabels(['15 Mar', '1 May', '15 Jun', '26 Jul'])
ax[0].legend(frameon=False, loc='lower left')
ax[0].set_title('(a) logged users, weeks 11 to 30')
ax[1].hist(ua.frac.to_numpy(), bins=50, color='tab:blue')
ax[1].axvline(0.8, color='k', ls='--', lw=0.7)
ax[1].set_yscale('log')
ax[1].yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax[1].set_xlabel('covered share of an airing')
ax[1].set_title('(b) user-airing events')
fig.tight_layout(pad=0.3, w_pad=0.6)
out = os.path.join(ROOT, 'report', 'fig_data.pdf')
fig.savefig(out)
print('saved', out, '| events', len(ua), '| share at least 0.8', round(float((ua.frac >= 0.8).mean()), 4))
