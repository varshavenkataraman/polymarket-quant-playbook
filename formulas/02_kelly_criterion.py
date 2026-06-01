import numpy as np
import matplotlib.pyplot as plt


def kelly(p, market_price):
    odds = (1 / market_price) - 1
    return (p * odds - (1 - p)) / odds


def fractional_kelly(p, market_price, fraction=0.5):
    return max(0.0, kelly(p, market_price) * fraction)


def simulate(p, market_price, fraction=0.5, trials=200, n_paths=20, seed=42):
    np.random.seed(seed)
    f = fractional_kelly(p, market_price, fraction)
    odds = (1 / market_price) - 1
    paths = []
    for _ in range(n_paths):
        br = [1.0]
        for _ in range(trials):
            win = np.random.rand() < p
            br.append(br[-1] * (1 + f * odds if win else 1 - f))
        paths.append(br)
    return paths


def plot_growth_curve(p, market_price):
    odds = (1 / market_price) - 1
    f_range = np.linspace(0.001, 0.999, 400)
    growth = p * np.log(1 + f_range * odds) + (1 - p) * np.log(np.maximum(1 - f_range, 1e-9))
    f_star = kelly(p, market_price)

    plt.figure(figsize=(9, 5))
    plt.plot(f_range, growth, color='#00c896', linewidth=2)
    plt.axvline(f_star, color='crimson', linestyle='--', label=f'Full Kelly f*={f_star:.2f}')
    plt.axvline(f_star * 0.5, color='orange', linestyle='--', label=f'Half Kelly={f_star*0.5:.2f}')
    plt.axhline(0, color='gray', alpha=0.4)
    plt.xlabel('Bet Fraction (f)')
    plt.ylabel('Expected Log Growth per Bet')
    plt.title(f'Kelly Growth Curve  |  p={p}, market={market_price:.0%}')
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.show()


def plot_paths(p, market_price, fraction=0.5):
    paths = simulate(p, market_price, fraction)
    plt.figure(figsize=(10, 5))
    for path in paths:
        plt.plot(path, alpha=0.3, linewidth=1, color='#4fa8d5')
    plt.plot(paths[0], linewidth=2, color='#00c896')
    plt.yscale('log')
    plt.xlabel('Bet Number')
    plt.ylabel('Bankroll (log scale)')
    plt.title(f'Bankroll Simulation  |  {fraction}x Kelly')
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    market_price = 0.21
    p_model = 0.25

    f_full = kelly(p_model, market_price)
    f_half = fractional_kelly(p_model, market_price, 0.5)

    print(f'Market price  : {market_price:.0%}')
    print(f'Model p       : {p_model:.0%}')
    print(f'Edge          : +{p_model - market_price:.0%}')
    print(f'Full Kelly    : {f_full:.3f}')
    print(f'Half Kelly    : {f_half:.3f}')
    print(f'On $1,000     : bet ${f_half * 1000:.0f}')

    plot_growth_curve(p_model, market_price)
    plot_paths(p_model, market_price)
