import time
import argparse
import logging
import numpy as np
import pandas as pd

from dotenv import load_dotenv
load_dotenv()

from signal_engine import run_full_scan
from signal_engine import evidence_strength
from alert import send, format_ev, format_kl, format_thin_pool, format_bregman

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger(__name__)

BANKROLL = 1000.0

CORRELATED_PAIRS = [
]

MULTI_OUTCOME_MARKETS = [
]

EVIDENCE_STREAM = [
    evidence_strength('tweet_spike', 1.0),
    evidence_strength('news_report', 1.0),
]

DRAWDOWN_STOP = 0.20


def load_markets():
    try:
        from data_fetcher import PolymarketFetcher
        fetcher  = PolymarketFetcher()
        markets  = fetcher.get_markets(limit=100)
        df       = fetcher.to_dataframe(markets)
        df['model_p'] = (df['price'] + np.random.normal(0, 0.03, len(df))).clip(0.01, 0.99)
        df['b']       = np.random.uniform(30, 300, len(df))
        log.info(f'Fetched {len(df)} live markets.')
        return df
    except Exception as e:
        log.warning(f'Live fetch failed ({e}). Using sample data.')
        df = pd.read_csv('../data/sample_markets.csv')
        df['b'] = np.random.uniform(30, 300, len(df))
        return df


def check_drawdown(current_bankroll, peak_bankroll):
    if peak_bankroll == 0:
        return False
    drawdown = (peak_bankroll - current_bankroll) / peak_bankroll
    return drawdown >= DRAWDOWN_STOP


def run_once(bankroll_state):
    log.info('Scan cycle start')

    if check_drawdown(bankroll_state['current'], bankroll_state['peak']):
        log.warning(f'Drawdown stop hit. Current: ${bankroll_state["current"]:.2f}, Peak: ${bankroll_state["peak"]:.2f}')
        return

    df     = load_markets()
    report = run_full_scan(df, CORRELATED_PAIRS, MULTI_OUTCOME_MARKETS, EVIDENCE_STREAM)

    log.info(f'Thin pools       : {report["n_thin_pools"]}')
    log.info(f'EV opportunities : {report["n_ev_bets"]}')
    log.info(f'KL arbs          : {report["n_kl_arbs"]}')
    log.info(f'Bregman arbs     : {report["n_bregman_arbs"]}')
    log.info(f'Bayesian signals : {report["n_bayesian_signals"]}')

    for pool in report['thin_pools'][:2]:
        send(format_thin_pool(pool))

    for opp in report['ev_opportunities'][:3]:
        send(format_ev(opp, BANKROLL))

    for pair in report['kl_arbs'][:2]:
        send(format_kl(pair))

    for arb in report['bregman_arbs'][:2]:
        send(format_bregman(arb))

    log.info('Scan cycle complete')


def main():
    parser = argparse.ArgumentParser(description='Polymarket Quant Bot')
    parser.add_argument('--loop', type=int, default=0, help='Scan interval in seconds (0 = run once)')
    args = parser.parse_args()

    bankroll_state = {'current': BANKROLL, 'peak': BANKROLL}

    if args.loop > 0:
        log.info(f'Loop mode — scanning every {args.loop}s. Ctrl+C to stop.')
        while True:
            try:
                run_once(bankroll_state)
            except Exception as e:
                log.error(f'Scan error: {e}')
            time.sleep(args.loop)
    else:
        run_once(bankroll_state)


if __name__ == '__main__':
    main()
