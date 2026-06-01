import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from formulas.kelly_criterion import fractional_kelly
from formulas.ev_gap import compute_ev
from formulas.kl_divergence import binary_kl
from formulas.lmsr_pricing import price_impact
from formulas.bregman_projection import detect_arb
from formulas.bayesian_update import sequential_update, evidence_strength

EV_THRESHOLD = 0.05
KL_THRESHOLD = 0.20
LMSR_B_THIN  = 50
KELLY_FRAC   = 0.50
FEE          = 0.02


def run_lmsr_scan(df):
    if 'b' not in df.columns:
        return pd.DataFrame()
    thin = df[df['b'] < LMSR_B_THIN].copy()
    thin['impact_10'] = thin.apply(lambda r: price_impact(0, 10, r['b']), axis=1)
    return thin.sort_values('impact_10', ascending=False).reset_index(drop=True)


def run_ev_scan(df):
    df = df.copy()
    df['ev']      = df.apply(lambda r: compute_ev(r['model_p'], r['price'], FEE), axis=1)
    df['kelly_f'] = df.apply(lambda r: fractional_kelly(r['model_p'], r['price'], KELLY_FRAC), axis=1)
    return df[df['ev'] > EV_THRESHOLD].sort_values('ev', ascending=False).reset_index(drop=True)


def run_kl_scan(df, pairs):
    price_map = dict(zip(df['market_id'], df['price']))
    rows = []
    for a, b in pairs:
        if a not in price_map or b not in price_map:
            continue
        pa, pb = price_map[a], price_map[b]
        kl = binary_kl(pa, pb)
        rows.append({
            'market_a': a,
            'market_b': b,
            'price_a':  pa,
            'price_b':  pb,
            'kl_score': round(kl, 4),
            'arb':      kl > KL_THRESHOLD,
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values('kl_score', ascending=False)


def run_bregman_scan(multi_outcome_markets):
    results = []
    for market in multi_outcome_markets:
        result = detect_arb(market['prices'], market.get('names'))
        if result['arb_signal']:
            results.append({
                'market_id':  market.get('market_id', 'unknown'),
                'price_sum':  result['price_sum'],
                'arb_type':   result['arb_type'],
                'deviation':  result['deviation'],
            })
    return pd.DataFrame(results)


def run_bayesian_scan(df, evidence_stream):
    df = df.copy()
    updated = []
    for _, row in df.iterrows():
        posteriors = sequential_update(row['model_p'], evidence_stream)
        updated.append(posteriors[-1])
    df['bayesian_p'] = updated
    df['bayesian_ev'] = df.apply(lambda r: compute_ev(r['bayesian_p'], r['price'], FEE), axis=1)
    return df[df['bayesian_ev'] > EV_THRESHOLD].sort_values('bayesian_ev', ascending=False).reset_index(drop=True)


def run_full_scan(df, pairs=[], multi_outcome_markets=[], evidence_stream=[]):
    lmsr    = run_lmsr_scan(df)
    ev      = run_ev_scan(df)
    kl      = run_kl_scan(df, pairs)
    bregman = run_bregman_scan(multi_outcome_markets)
    bayes   = run_bayesian_scan(df, evidence_stream) if evidence_stream else pd.DataFrame()

    kl_arbs = kl[kl['arb']] if not kl.empty else pd.DataFrame()

    return {
        'thin_pools':        lmsr.to_dict('records'),
        'ev_opportunities':  ev.to_dict('records'),
        'kl_arbs':           kl_arbs.to_dict('records'),
        'bregman_arbs':      bregman.to_dict('records') if not bregman.empty else [],
        'bayesian_signals':  bayes.to_dict('records') if not bayes.empty else [],
        'n_thin_pools':      len(lmsr),
        'n_ev_bets':         len(ev),
        'n_kl_arbs':         len(kl_arbs),
        'n_bregman_arbs':    len(bregman),
        'n_bayesian_signals': len(bayes),
    }


if __name__ == '__main__':
    np.random.seed(0)
    n  = 30
    df = pd.DataFrame({
        'market_id': [f'mkt_{i}' for i in range(n)],
        'price':     np.random.beta(2, 3, n).clip(0.05, 0.95),
        'model_p':   np.random.beta(2, 3, n).clip(0.05, 0.95),
        'b':         np.random.uniform(20, 200, n),
    })
    pairs = [(f'mkt_{i}', f'mkt_{i+1}') for i in range(0, 10, 2)]
    evidence_stream = [
        evidence_strength('tweet_spike', 1.5),
        evidence_strength('news_report', 1.0),
    ]
    report = run_full_scan(df, pairs, evidence_stream=evidence_stream)

    print(f"Thin pools       : {report['n_thin_pools']}")
    print(f"EV opportunities : {report['n_ev_bets']}")
    print(f"KL arbs          : {report['n_kl_arbs']}")
    print(f"Bayesian signals : {report['n_bayesian_signals']}")
