import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


EVIDENCE_LIKELIHOODS = {
    'tweet_spike':        (0.70, 0.40),
    'poll_update':        (0.75, 0.35),
    'news_report':        (0.80, 0.30),
    'official_statement': (0.90, 0.15),
    'rumor':              (0.60, 0.45),
}


def bayesian_update(prior, lh_true, lh_false):
    p_e = lh_true * prior + lh_false * (1 - prior)
    if p_e == 0:
        return prior
    return float(np.clip((lh_true * prior) / p_e, 0, 1))


def sequential_update(prior, evidence_stream):
    posteriors = [prior]
    p = prior
    for lh_true, lh_false in evidence_stream:
        p = bayesian_update(p, lh_true, lh_false)
        posteriors.append(p)
    return posteriors


def evidence_strength(signal, magnitude=1.0):
    if signal not in EVIDENCE_LIKELIHOODS:
        raise ValueError(f'Unknown signal. Options: {list(EVIDENCE_LIKELIHOODS)}')
    lh_t, lh_f = EVIDENCE_LIKELIHOODS[signal]
    lh_t = min(0.99, lh_t + (magnitude - 1) * 0.05)
    lh_f = max(0.01, lh_f - (magnitude - 1) * 0.05)
    return lh_t, lh_f


class BetaBayesian:
    def __init__(self, alpha=1.0, beta=1.0):
        self.alpha = alpha
        self.beta = beta

    @property
    def mean(self):
        return self.alpha / (self.alpha + self.beta)

    @property
    def std(self):
        a, b = self.alpha, self.beta
        return np.sqrt(a * b / ((a + b) ** 2 * (a + b + 1)))

    def update(self, yes, no):
        self.alpha += yes
        self.beta += no
        return self

    def credible_interval(self, level=0.95):
        lo = (1 - level) / 2
        d = stats.beta(self.alpha, self.beta)
        return d.ppf(lo), d.ppf(1 - lo)


def plot_belief_evolution(posteriors, labels=None):
    x = list(range(len(posteriors)))
    labels = ['Prior'] + (labels or [f'E{i}' for i in range(1, len(posteriors))])

    plt.figure(figsize=(10, 5))
    plt.plot(x, posteriors, 'o-', color='#00c896', linewidth=2, markersize=8)
    plt.axhline(0.5, color='gray', linestyle='--', alpha=0.4)
    for i, (p, lbl) in enumerate(zip(posteriors, labels)):
        plt.annotate(f'{p:.2f}', (i, p), xytext=(0, 10),
                     textcoords='offset points', ha='center', fontsize=9)
    plt.xticks(x, labels, rotation=15)
    plt.ylim(0, 1)
    plt.ylabel('P(YES)')
    plt.title('Bayesian Belief Evolution')
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    prior = 0.50
    evidence = [
        ('tweet_spike', 1.5),
        ('news_report', 1.0),
        ('tweet_spike', 2.0),
        ('official_statement', 1.0),
    ]

    ev_stream = []
    labels = []
    for sig, mag in evidence:
        lh_t, lh_f = evidence_strength(sig, mag)
        ev_stream.append((lh_t, lh_f))
        labels.append(f'{sig} x{mag}')

    posteriors = sequential_update(prior, ev_stream)
    for label, p in zip(labels, posteriors[1:]):
        print(f'{label:30s}: {p:.3f}')

    final = posteriors[-1]
    print(f'\nFinal posterior : {final:.3f}')
    print(f'Edge vs market  : +{final - prior:.0%}')

    plot_belief_evolution(posteriors, labels)

    b = BetaBayesian(1, 1)
    for yes, no in [(3, 1), (5, 2), (2, 3), (8, 1)]:
        b.update(yes, no)
    lo, hi = b.credible_interval()
    print(f'\nBeta estimate : {b.mean:.3f}  (+/-{b.std:.3f})')
    print(f'95% CI        : [{lo:.3f}, {hi:.3f}]')
