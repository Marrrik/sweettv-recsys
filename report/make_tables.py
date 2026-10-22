"""Write the report tables and number macros from the saved results, so no number in the report is typed by hand.

Reads eda/eda_summary.json, modelling/*_results.json and report/kaggle_submissions.csv (the CLI export of the
submission list). Writes report/generated/*.tex. Run from anywhere: python report/make_tables.py
"""
import json
import os
import sys

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))
from sweettv.ranker import BASE_PARAMS, PARAM_SETS  # noqa: E402
OUT = os.path.join(ROOT, 'report', 'generated')
os.makedirs(OUT, exist_ok=True)


def load(path):
    with open(os.path.join(ROOT, path), encoding='utf-8') as f:
        return json.loads(f.read().replace('NaN', 'null'))


eda = load('eda/eda_summary.json')
base = load('modelling/baselines_results.json')
cfr = load('modelling/cf_results.json')
r1 = load('modelling/ranker_results.json')
r2 = load('modelling/ranker_v2_results.json')
r3 = load('modelling/ranker_v3_results.json')
an = load('modelling/analysis_results.json')
FINAL = an['final']  # the designated re-ranker round, set in modelling/analysis.py
rf = load(f'modelling/{FINAL}_results.json')
r4 = load('modelling/ranker_v4_results.json')
kag = pd.read_csv(os.path.join(ROOT, 'report', 'kaggle_submissions.csv'))
kag['date'] = pd.to_datetime(kag.date)
kag = kag.sort_values('date').groupby('fileName').tail(1).set_index('fileName')


def f3(x):
    return f'{x:.3f}'


def f4(x):
    return f'{x:.4f}'


def pct(x, d=1):
    return f'{100 * x:.{d}f}\\%'


def intc(x):
    return f'{int(x):,}'.replace(',', '{,}')


def lb(name, col='publicScore'):
    return f'{kag.loc[name, col]:.5f}' if name in kag.index else '--'


macros = {}


def m(name, value):
    macros[name] = value


# data and join
v, j, ts = eda['views'], eda['join'], eda['threshold_sensitivity']
m('nViews', intc(v['rows']))
m('nUsers', intc(v['users']))
m('nLogChannels', intc(v['channels (vsetv_id)']))
parts = eda['schedule_parts']
m('labRowsLo', pct(min(float(p['labelled share of rows']) for p in parts.values())))
m('labRowsHi', pct(max(float(p['labelled share of rows']) for p in parts.values())))
m('labAirLo', pct(min(float(p['labelled share of airtime']) for p in parts.values())))
m('labAirHi', pct(max(float(p['labelled share of airtime']) for p in parts.values())))
m('labRowsTrain', pct(float(parts['w11_20']['labelled share of rows'])))
m('inSlots', pct(j['viewing time inside any slot']))
m('inLabSlots', pct(j['viewing time inside labelled slots']))
m('nEvents', intc(ts['any overlap']['user-airing events']))
m('nEventsQ', intc(ts['>= 80% of slot (target rule)']['user-airing events']))
m('shareEventsQ', pct(ts['>= 80% of slot (target rule)']['user-airing events'] / ts['any overlap']['user-airing events']))
m('nPairsQ', intc(ts['>= 80% of slot (target rule)']['user-programme pairs']))
m('medProgAny', intc(ts['any overlap']['median programmes per user']))
m('medProgQ', intc(ts['>= 80% of slot (target rule)']['median programmes per user']))
tw = eda['target_windows']
m('nTargetWone', intc(tw['W1']['users with at least 5 (have a target)']))
m('nTargetWtwo', intc(tw['W2']['users with at least 5 (have a target)']))
m('nTestTargetWtwo', intc(tw['W2']['test users with at least 5']))
m('tieShare', pct(eda['target_ties']['share_5th_equals_6th']))
m('gapUsers', pct(eda['logging_gap']['users_missing_in_gap'] / eda['logging_gap']['users_before_and_after']))
m('prevTopMap', f4(eda['stability']['map5_prev_top5']))
cov = eda['candidate_coverage']
m('covQual', pct(cov['in_w1_qualifying']))
m('covTouched', pct(cov['in_w1_any_overlap']))
m('covAny', pct(cov['any of these']))
tu = eda['test_users_history']
m('testStillAir', pct(tu['share of their qualifying programmes still airing']))
fb = eda['fallback']
m('nFallbackNone', intc(fb['none']))
m('nFallbackFew', intc(fb['1 to 4']))

# baselines and single methods
sel = base['selected']
m('hsHalfLife', f"{sel['half_life_weeks']:g}")
m('hsPrior', f"{sel['prior']:g}")
br = base['results']
sm = cfr['summary']
st = cfr['settings']
m('alsFactors', str(int(st['als']['factors'])))
m('alsAlpha', f"{st['als']['alpha']:g}")
m('alsReg', f"{st['als']['reg']:g}")
m('pprAlpha', f"{st['ppr']['alpha']:g}")
m('pprWeighting', st['ppr']['weighting'])
m('rulesSupport', str(st['rules']['min_support']))

rows = [
    ('Popularity', br['S21 popularity']['all'], br['S21 popularity']['test users'], br['S16 popularity']['all'], 'popularity.csv', 'Popularity'),
    ('Habit $\\times$ schedule', br['S21 habit x schedule']['all'], br['S21 habit x schedule']['test users'], br['S16 habit x schedule']['all'],
     'habit_schedule.csv', 'Habit x schedule'),
    ('ALS', sm['als | S21 all'], sm['als | S21 test users'], sm['als | S16 all'], 'als.csv', 'ALS'),
    ('PPR', sm['ppr | S21 all'], sm['ppr | S21 test users'], sm['ppr | S16 all'], 'ppr.csv', 'PPR'),
    ('Association rules', sm['rules | S21 all'], sm['rules | S21 test users'], sm['rules | S16 all'], 'rules.csv', 'Association rules'),
    ('Content-based', sm['content | S21 all'], sm['content | S21 test users'], sm['content | S16 all'], 'content.csv', 'Content-based'),
]
hyb = rf['results'][rf['selected']]
boot = an['bootstrap']
lines = []
for name, a21, t21, a16, sub, key in rows:
    b = boot[key]
    lines.append(f"{name} & {f4(a21)} & {f4(t21)} & {f4(a16)} & {lb(sub)} & {lb(sub, 'privateScore')} & "
                 f"{b['hybrid minus method']:+.4f} [{b['ci low']:.4f}, {b['ci high']:.4f}] \\\\")
lines.append('\\midrule')
lines.append(f"Hybrid (final) & \\textbf{{{f4(hyb['all'])}}} & \\textbf{{{f4(hyb['test users'])}}} & (train) & "
             f"\\textbf{{{lb(FINAL + '.csv')}}} & {lb(FINAL + '.csv', 'privateScore')} & -- \\\\")
with open(os.path.join(OUT, 'tab_methods.tex'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines) + '\n')
m('mapHybrid', f4(hyb['all']))
m('mapHybridTest', f4(hyb['test users']))
m('mapHS', f4(br['S21 habit x schedule']['all']))
m('mapHScount', f4(br['S21 habit count only']['all']))
m('mapPop', f4(br['S21 popularity']['all']))
m('mapPPR', f4(sm['ppr | S21 all']))
m('mapALS', f4(sm['als | S21 all']))
m('mapRules', f4(sm['rules | S21 all']))
m('mapContent', f4(sm['content | S21 all']))
m('lbHybrid', lb(FINAL + '.csv'))
m('lbHybridPriv', lb(FINAL + '.csv', 'privateScore'))
m('finalFile', FINAL.replace('_', '\\_') + '.csv')
m('lbHS', lb('habit_schedule.csv'))
m('lbPop', lb('popularity.csv'))
m('lbPPR', lb('ppr.csv'))
for key, macro in [('Habit x schedule', 'HS'), ('PPR', 'PPR'), ('Popularity', 'Pop')]:
    b = boot[key]
    m(f'diff{macro}', f"{b['hybrid minus method']:.4f}")
    m(f'ci{macro}', f"[{b['ci low']:.4f}, {b['ci high']:.4f}]")
    m(f'higher{macro}', intc(b['users where hybrid is higher']))
    m(f'lower{macro}', intc(b['users where hybrid is lower']))
seg = an['segments']
for s_name, macro in [('light', 'Light'), ('heavy', 'Heavy')]:
    m(f'segHybrid{macro}', f4(seg[s_name]['Hybrid (LightGBM re-ranker)']))
    m(f'segHS{macro}', f4(seg[s_name]['Habit x schedule']))
    m(f'segPop{macro}', f4(seg[s_name]['Popularity']))

# re-ranker rounds
e1, e2 = r1['results'], r1['ensembles']
rr = r2['results']
r3r = r3['results']
rounds = [
    ('R1', 'S16', '30/50', '63', 'binary', e1['binary'], 'ranker_binary.csv'),
    ('R1', 'S16', '30/50', '63', 'lambdarank', e1['lambdarank'], 'ranker_lambdarank.csv'),
    ('R1', 'S16', '30/50', '63', 'poisson', e1['poisson'], 'ranker_poisson.csv'),
    ('R1', 'S16', '30/50', '63', 'RRF, all 3', e2['all three'], 'ranker_ensemble.csv'),
    ('R2 E1', 'S16', '60/60', '63', 'RRF b+l', rr['E1 binary + lambdarank'], None),
    ('R2 E2', 'S14--S18', '60/60', '63', 'RRF b+l', rr['E2 binary + lambdarank'], None),
    ('R2 E3', 'S14--S18', '60/60', '63', 'RRF b+l, reg', rr['E3 binary + lambdarank'], 'ranker_v2.csv'),
    ('R3 E4', 'S14--S18', '60/60', '67', 'RRF b+l, reg', r3r['E4 binary + lambdarank'], None),
    ('R3 E4', 'S14--S18', '60/60', '67', '3 seeds, reg', r3r['E4 binary + lambdarank, 3 seeds'], 'ranker_v3.csv'),
    ('R4 E5', 'S14--S18', '60/60', '80', 'RRF b+l, reg', r4['results']['E5 blend'], None),
    ('R4 E5', 'S14--S18', '60/60', '80', 'RRF b+l, wide', r4['results']['E5 wide blend'], None),
    ('R4 E5', 'S14--S18', '60/60', '80', '5 seeds, wide', r4['results']['E5 wide blend, 5 seeds, k=0'], 'ranker_v4.csv'),
]
lines = [f"{a} & {b} & {c} & {d} & {e} & {f4(res['all'])} & {f4(res['test users'])} & {lb(sub) if sub else '--'} \\\\"
         for a, b, c, d, e, res, sub in rounds]
with open(os.path.join(OUT, 'tab_ranker.tex'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines) + '\n')
m('mapRankerOneLambda', f4(e1['lambdarank']['all']))
m('mapEthree', f4(rr['E3 binary + lambdarank']['all']))
m('mapEfour', f4(r3r['E4 binary + lambdarank']['all']))
m('mapEfourSeeds', f4(r3r['E4 binary + lambdarank, 3 seeds']['all']))
m('mapHSfeature', f4(e1['habit x schedule feature alone']['all']))
imp = rf['importance']
top_imp = sorted(imp.items(), key=lambda kv: -kv[1])[:3]
words = {'rank_habit_exposure': 'the within-user rank of habit-slot exposure',
         'rank_cf_habit': 'the within-user rank of the habit-slot score',
         'habit_cov': 'habit coverage of whole airings in the window',
         'rank_habit_rec_cov': 'the within-user rank of recency-weighted habit coverage',
         'habit_day_cov': 'weekday-pooled habit coverage', 'rank_fw_min': 'the within-user rank of the full-watch minimum',
         'habit_exposure': 'habit-slot exposure', 'cf_habit': 'the habit-slot score', 'rank_cf_ppr': 'the within-user rank of PPR',
         'hs': 'the habit $\\times$ schedule score', 'rank_hs': 'the within-user rank of habit $\\times$ schedule'}
for i, (k, val) in enumerate(top_imp):
    m(f'impName{"ABC"[i]}', words.get(k, '\\texttt{' + k.replace('_', '\\_') + '}'))
    m(f'impShare{"ABC"[i]}', pct(val, 0))

# configuration of the final re-ranker
NUM = {1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five', 6: 'six', 7: 'seven', 8: 'eight', 9: 'nine', 10: 'ten'}
seeds = rf.get('seeds', [0])
m('nFeatures', str(rf.get('n_features', 67)))
m('nSeeds', NUM[len(seeds)])
m('nModels', NUM[2 * len(seeds)])
m('seedRange', f'{min(seeds)} to {max(seeds)}')
m('rrfK', str(rf.get('k', 0)))
p = {**BASE_PARAMS, **PARAM_SETS[rf.get('param_set', 'regularised')]}
m('lgbParams', f"learning rate {p['learning_rate']:g}, {p['num_leaves']} leaves, {p['min_data_in_leaf']} rows per leaf, "
               f"feature fraction {p['feature_fraction']:g}, bagging {p['bagging_fraction']:g}, L2 {p['lambda_l2']:g}")
nb = json.load(open(os.path.join(ROOT, 'modelling', f'{FINAL}.ipynb'), encoding='utf-8'))
runtime = None
for cell in nb['cells']:
    for out in cell.get('outputs', []):
        if out.get('output_type') == 'stream' and 'total runtime' in ''.join(out['text']):
            runtime = int(''.join(out['text']).split('total runtime')[1].split('s')[0])
m('finalRuntime', intc(runtime))
m('mapEfour', f4(r4['results']['E4 blend']['all']))
m('mapEfive', f4(r4['results']['E5 blend']['all']))
m('mapEfiveWide', f4(r4['results']['E5 wide blend']['all']))
m('mapFinal', f4(hyb['all']))
m('nSubmissions', str(len([n for n in os.listdir(os.path.join(ROOT, 'submissions')) if n.endswith('.csv')])))

# list properties
pr = an['properties']
order = ['Popularity', 'Habit x schedule', 'ALS', 'PPR', 'Association rules', 'Content-based', 'Hybrid (LightGBM re-ranker)']
short = {'Popularity': 'Popularity', 'Habit x schedule': 'Habit $\\times$ sched.', 'ALS': 'ALS', 'PPR': 'PPR',
         'Association rules': 'Assoc. rules', 'Content-based': 'Content', 'Hybrid (LightGBM re-ranker)': 'Hybrid'}
lines = []
for k in order:
    lines.append(f"{short[k]} & {pct(pr['share watched at 0.8 before'][k], 0)} & {pct(pr['share only touched before'][k], 0)} & "
                 f"{pct(pr['share new to the user'][k], 0)} & {pr['precision on familiar'][k]:.3f} & {pr['precision on new'][k]:.3f} & "
                 f"{intc(pr['distinct programmes'][k])} & {pr['mean shared with the popularity top 5'][k]:.2f} & "
                 f"{pct(pr['category share Инфо'][k], 0)} \\\\")
with open(os.path.join(OUT, 'tab_props.tex'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines) + '\n')
H_ = 'Hybrid (LightGBM re-ranker)'
m('hybWatched', pct(pr['share watched at 0.8 before'][H_], 0))
m('hybNew', pct(pr['share new to the user'][H_], 0))
m('hybPrecFam', f"{pr['precision on familiar'][H_]:.3f}")
m('hsPrecFam', f"{pr['precision on familiar']['Habit x schedule']:.3f}")
m('maxPrecNew', f"{max(pr['precision on new'].values()):.3f}")
m('pprWatched', pct(pr['share watched at 0.8 before']['PPR'], 0))
m('pprDistinct', intc(pr['distinct programmes']['PPR']))
m('alsDistinct', intc(pr['distinct programmes']['ALS']))
m('rulesShared', f"{pr['mean shared with the popularity top 5']['Association rules']:.2f}")
m('rulesNews', pct(pr['category share Инфо']['Association rules'], 0))
act = an['actual_shares']
m('actWatched', pct(act['watched']))
m('actTouched', pct(act['touched']))
m('actNew', pct(act['new']))
m('actNewRound', pct(act['new'], 0))

# example users on S21
ex = an['examples']
progs = ex['programmes']
for u in ['A', 'B']:
    e = ex[u]
    m(f'user{u}', str(e['user_id']))
    m(f'hours{u}', f"{e['hours_w11_20']:.0f}")
    m(f'nProg{u}', str(e['programmes_watched_w11_20']))
    m(f'apHybrid{u}', f"{e['ap5']['Hybrid (LightGBM re-ranker)']:.2f}")
    m(f'apHS{u}', f"{e['ap5']['Habit x schedule']:.2f}")
    m(f'apPPR{u}', f"{e['ap5']['PPR']:.2f}")
    hl = e['s21_lists']['Hybrid (LightGBM re-ranker)']
    m(f'hybList{u}', ', '.join(str(p_) for p_, _, _ in hl))
    m(f'hybHits{u}', NUM.get(sum(int(h_) for _, _, h_ in hl), 'no'))
    hits_at = [i + 1 for i, (_, _, h_) in enumerate(hl) if h_]
    m(f'hybHitRanks{u}', ', '.join(str(r_) for r_ in hits_at[:-1]) + (' and ' if len(hits_at) > 1 else '') + str(hits_at[-1]) if hits_at else 'none')
    m(f'hybApFormula{u}', '(' + '+'.join(f'{j + 1}/{r_}' for j, r_ in enumerate(hits_at)) + ')/5' if hits_at else '0')
    short_fill = [k for k, n in e['filled_by_method'].items() if k != 'Popularity' and n < 5]
    m(f'fill{u}', 'every method filled all five places itself' if not short_fill else
      'the popularity fallback filled ' + ', '.join(f"{5 - e['filled_by_method'][k]} of the places of {k}" for k in short_fill))


def cell(p, seen, hit):
    s = str(p)
    if hit:
        s = f'\\textbf{{{s}}}'
    if seen == 'new':
        s += '$^{n}$'
    elif seen == 'touched':
        s += '$^{t}$'
    return s


lines = []
lines.append('Actual top 5 & ' + ' & '.join(f'\\textbf{{{p}}}' for p in ex['A']['actual_top5_w21_30']) + ' & & '
             + ' & '.join(f'\\textbf{{{p}}}' for p in ex['B']['actual_top5_w21_30']) + ' & \\\\')
lines.append('\\midrule')
for k in order:
    ca = ' & '.join(cell(p, s, h) for p, s, h in ex['A']['s21_lists'][k])
    cb = ' & '.join(cell(p, s, h) for p, s, h in ex['B']['s21_lists'][k])
    lines.append(f"{short[k]} & {ca} & {ex['A']['ap5'][k]:.2f} & {cb} & {ex['B']['ap5'][k]:.2f} \\\\")
with open(os.path.join(OUT, 'tab_users.tex'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines) + '\n')

used = []
for u in ['A', 'B']:
    used += ex[u]['actual_top5_w21_30'] + [p for lst in ex[u]['s21_lists'].values() for p, _, _ in lst]
used = sorted(set(used))
def tex(s):
    for a, b in [('\\', '\\textbackslash{}'), ('&', '\\&'), ('%', '\\%'), ('$', '\\$'), ('#', '\\#'), ('_', '\\_'),
                 ('{', '\\{'), ('}', '\\}'), ('~', '\\textasciitilde{}'), ('^', '\\textasciicircum{}')]:
        s = s.replace(a, b)
    return s


cat_en = {'Инфо': 'news', 'Детям': 'kids', 'Фильмы': 'film', 'Сериалы': 'series', 'Развлечения': 'entertainment',
          'Познавательное': 'factual', 'Другое': 'other', 'Спорт': 'sport'}
with open(os.path.join(OUT, 'legend_users.tex'), 'w', encoding='utf-8') as f:
    f.write('; '.join(f"{p} \\foreignlanguage{{russian}}{{{tex(progs[str(p)]['title'].strip().rstrip('.'))}}} "
                      f"({cat_en.get(progs[str(p)]['category'], progs[str(p)]['category'])})" for p in used) + '.\n')

with open(os.path.join(OUT, 'numbers.tex'), 'w', encoding='utf-8') as f:
    for k, val in macros.items():
        f.write(f'\\newcommand{{\\{k}}}{{{val}}}\n')
print('wrote', sorted(os.listdir(OUT)), '|', len(macros), 'macros')
