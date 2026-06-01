import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import entropy
from itertools import combinations


def kl_divergence(p, q):
    p = np.clip(np.array(p, dtype=float), 1e-10, 1)
    q = np.clip(np.array(q, dtype=float), 1e-10, 1)
    p /= p.sum()
    q /= q.sum()
    return float(entropy(p, q))


def symmetric_kl(p, q):
    return (kl_divergence(p, q) + kl_divergence(q, p)) / 2


def binary_kl(price_a, price_b):
    p = [price_a, 1 - price_a]
    q = [price_b, 1 - price_b]
    return symmetric_kl(p, q)


def scan_pairs(markets, pairs, threshold=0.2):
    price_map = dict(zip(markets['market_id'], markets['price']))
    rows = []
    for a, b in pairs:
        if a not in price_map or b not in price_map:
            continue
        pa, pb = price_map[a], price_map[b]
        kl = binary_kl(pa, pb)
        rows.append({
            'market_a': a,
            'market_b': b,
            'price_a': pa,
            'price_b': pb,
            'kl_score': round(kl, 4),
            'arb': kl > threshold,
        })
    return pd.DataFrame(rows).sort_values('kl_score', ascending=False)


def plot_kl_heatmap(markets):
    ids = markets['market_id'].tolist()
    prices = markets['price'].tolist()
    n = len(ids)
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                matrix[i, j] = binary_kl(prices[i], prices[j])

    plt.figure(figsize=(9, 7))
    im = plt.imshow(matrix, cmap='hot_r', aspect='auto')
    plt.colorbar(im, label='KL Divergence')
    plt.xticks(range(n), ids, rotation=45, ha='right', fontsize=8)
    plt.yticks(range(n), ids, fontsize=8)
    plt.title('KL Divergence Heatmap')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    vance = 0.21
    newsom = 0.17
    kl = binary_kl(vance, newsom)

    print(f'Vance 2028  : {vance:.0%}')
    print(f'Newsom 2028 : {newsom:.0%}')
    print(f'KL score    : {kl:.4f}')
    print(f'Signal      : {"ARB" if kl > 0.2 else "Monitor"}')

    markets = pd.DataFrame({
        'market_id': ['vance', 'newsom', 'harris', 'desantis', 'trump'],
        'price': [0.21, 0.17, 0.19, 0.12, 0.31],
    })
    pairs = list(combinations(markets['market_id'], 2))
    results = scan_pairs(markets, pairs)

    print('\nPairwise KL scan:')
    print(results.to_string(index=False))
    print(f'\nArb signals: {results["arb"].sum()}')

    plot_kl_heatmap(markets)
