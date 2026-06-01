import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def compute_ev(p_true, market_price, fee=0.02):
    payout = 1 / market_price
    return (p_true - market_price) * payout - fee


def scan_markets(df, threshold=0.05, fee=0.02):
    df = df.copy()
    df['payout'] = 1 / df['market_price']
    df['ev'] = (df['model_p'] - df['market_price']) * df['payout'] - fee
    df['ev_gap'] = df['model_p'] - df['market_price']
    df['edge_pct'] = df['ev_gap'] / df['market_price'] * 100
    return df[df['ev'] > threshold].sort_values('ev', ascending=False).reset_index(drop=True)


def plot_ev_distribution(df, threshold=0.05, fee=0.02):
    if 'ev' not in df.columns:
        df = df.copy()
        df['ev'] = (df['model_p'] - df['market_price']) * (1 / df['market_price']) - fee

    n_opps = (df['ev'] > threshold).sum()
    plt.figure(figsize=(10, 5))
    plt.hist(df['ev'], bins=50, color='#4fa8d5', alpha=0.8, edgecolor='white', linewidth=0.4)
    plt.axvline(threshold, color='crimson', linestyle='--', linewidth=2, label=f'Threshold = {threshold}')
    plt.axvline(0, color='gray', linestyle='-', alpha=0.4)
    plt.xlabel('Expected Value (net of fees)')
    plt.ylabel('Number of Markets')
    plt.title(f'EV Distribution  |  {n_opps} opportunities above threshold')
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    market_price = 0.47
    p_model = 0.52

    ev = compute_ev(p_model, market_price)
    print(f'Market price  : {market_price:.0%}')
    print(f'Model p       : {p_model:.0%}')
    print(f'Gap           : +{p_model - market_price:.0%}')
    print(f'Net EV        : {ev:.4f}')
    print(f'Signal        : {"BET" if ev > 0.05 else "SKIP"}')

    np.random.seed(42)
    n = 500
    df = pd.DataFrame({
        'market_id': [f'mkt_{i}' for i in range(n)],
        'market_price': np.random.beta(2, 3, n).clip(0.05, 0.95),
        'model_p': np.random.beta(2, 3, n).clip(0.05, 0.95),
        'volume': np.random.lognormal(10, 2, n),
    })

    opps = scan_markets(df)
    print(f'\nMarkets scanned    : {len(df)}')
    print(f'+EV opportunities  : {len(opps)}')
    print(opps[['market_id', 'market_price', 'model_p', 'ev', 'edge_pct']].head().to_string(index=False))

    plot_ev_distribution(df)
