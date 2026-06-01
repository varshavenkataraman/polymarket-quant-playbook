import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt


def project_simplex(v):
    v = np.asarray(v, dtype=float)
    n = len(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.where(u * np.arange(1, n + 1) > cssv - 1)[0][-1]
    theta = (cssv[rho] - 1) / (rho + 1)
    return np.maximum(v - theta, 0)


def project_kl(theta):
    n = len(theta)
    theta = np.array(theta, dtype=float)
    mu = cp.Variable(n)
    obj = sum(cp.kl_div(mu[i], theta[i]) for i in range(n))
    prob = cp.Problem(cp.Minimize(obj), [cp.sum(mu) == 1, mu >= 0])
    prob.solve(solver=cp.SCS)
    return mu.value


def detect_arb(prices, names=None, tolerance=0.03):
    prices = np.array(prices)
    total = prices.sum()
    names = names or [f'outcome_{i}' for i in range(len(prices))]
    fair = project_simplex(prices)
    gaps = prices - fair

    return {
        'prices': dict(zip(names, prices.round(4))),
        'fair': dict(zip(names, fair.round(4))),
        'gaps': dict(zip(names, gaps.round(4))),
        'price_sum': round(total, 4),
        'deviation': round(abs(total - 1.0), 4),
        'arb_signal': abs(total - 1.0) > tolerance,
        'arb_type': 'buy_all' if total < 1 - tolerance else ('sell_all' if total > 1 + tolerance else 'none'),
    }


def plot_arb(result):
    names = list(result['prices'].keys())
    market = list(result['prices'].values())
    fair = list(result['fair'].values())
    x = np.arange(len(names))

    plt.figure(figsize=(10, 5))
    plt.bar(x - 0.2, market, 0.35, label='Market price', color='#4fa8d5', alpha=0.85)
    plt.bar(x + 0.2, fair, 0.35, label='Fair prob', color='#00c896', alpha=0.85)
    plt.xticks(x, names, rotation=20, ha='right')
    plt.ylabel('Probability')
    plt.title(f'Market vs Fair  |  Sum={result["price_sum"]}  |  {result["arb_type"]}')
    plt.legend()
    plt.grid(axis='y', alpha=0.2)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    names = ['Sinners', 'Conclave', 'Emilia Perez', 'Anora', 'Wicked', 'The Brutalist']
    prices = [0.15, 0.22, 0.18, 0.19, 0.12, 0.16]

    result = detect_arb(prices, names)

    print(f'Price sum   : {result["price_sum"]}')
    print(f'Arb signal  : {result["arb_signal"]}')
    print(f'Type        : {result["arb_type"]}')
    print(f'Magnitude   : {result["deviation"]}')
    print()
    for name in names:
        gap = result['gaps'][name]
        tag = 'overpriced' if gap > 0 else 'underpriced'
        print(f'  {name:18s}: market={result["prices"][name]:.3f}  fair={result["fair"][name]:.3f}  {tag}')

    plot_arb(result)
