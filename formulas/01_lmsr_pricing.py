import numpy as np
import sympy as sp
import matplotlib.pyplot as plt


def lmsr_price(quantities, b):
    exp_q = np.exp(quantities / b)
    return exp_q / exp_q.sum()


def price_impact(q_initial, q_buy, b):
    before = lmsr_price(np.array([q_initial, 0.0]), b)[0]
    after = lmsr_price(np.array([q_initial + q_buy, 0.0]), b)[0]
    return after - before


def plot_pricing_curve(b=100, max_q=1000):
    q_yes = sp.symbols('q_yes')
    expr = sp.exp(q_yes / b) / (sp.exp(q_yes / b) + 1)
    f = sp.lambdify(q_yes, expr)
    qs = np.linspace(0, max_q, 300)
    prices = [f(q) for q in qs]

    plt.figure(figsize=(9, 5))
    plt.plot(qs, prices, color='#00c896', linewidth=2.5)
    plt.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    plt.xlabel('YES Shares Bought')
    plt.ylabel('Implied Probability')
    plt.title(f'LMSR Pricing Curve  |  b = {b}')
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.show()


def plot_impact_comparison():
    b_values = [20, 50, 100, 500]
    q_range = np.linspace(0, 200, 200)

    plt.figure(figsize=(10, 5))
    for b in b_values:
        impacts = [price_impact(0, q, b) for q in q_range]
        plt.plot(q_range, impacts, linewidth=2, label=f'b = {b}')

    plt.xlabel('Shares Purchased')
    plt.ylabel('Price Impact')
    plt.title('Price Impact vs Liquidity Depth')
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    b = 100
    q_buy = 10
    impact = price_impact(0, q_buy, b)

    print(f'Liquidity depth (b) : {b}')
    print(f'Shares purchased    : {q_buy}')
    print(f'Price impact        : +{impact:.2%}')

    thin = price_impact(0, q_buy, b=30)
    print(f'Thin pool (b=30)    : +{thin:.2%}')

    plot_pricing_curve(b=100)
    plot_impact_comparison()
