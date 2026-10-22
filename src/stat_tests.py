# stat tests (protocol 7), no API

from scipy.stats import binomtest, fisher_exact


def mcnemar_exact(only_a, only_b):
    # exact mcnemar = binomial test on discordant pairs
    # no discordant pairs -> p = 1
    n = only_a + only_b
    if n == 0:
        return 1.0
    result = binomtest(only_a, n, 0.5, alternative="two-sided")
    return float(result.pvalue)


def fisher_rates(k1, n1, k2, n2):
    # fisher exact test, k1/n1 vs k2/n2
    table = [[k1, n1 - k1], [k2, n2 - k2]]
    odds, p = fisher_exact(table, alternative="two-sided")

    if n1:
        rate_1 = k1 / n1
    else:
        rate_1 = None
    if n2:
        rate_2 = k2 / n2
    else:
        rate_2 = None

    return {"rate_1": rate_1, "rate_2": rate_2,
            "k_1": k1, "n_1": n1, "k_2": k2, "n_2": n2, "odds_ratio": float(odds), "p_value": float(p)}
