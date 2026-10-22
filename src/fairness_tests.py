# fairness metrics + tests on the variant set, one config (protocol 6.3, 7.2), no API call
# metrics on the primary run (first file), repeat runs = noise floor
# - gender/age flip rate vs noise (run1 vs rep2, 280 comparisons): fisher exact
# - gender main effect (female vs male), age main effect (old vs young) on high-acuity rate:
#   - paired mcnemar over matched variants, 140 pairs each
#   - bonferroni over 6 tests (2 effects x C1-C3), alpha = config.MAIN_EFFECT_ALPHA
#   - "claimed" = significant in run1 AND same direction in every repeat, else "fragile"
# - 6 group pairs: paired mcnemar over 70 base cases, alpha = 0.05 / 6
#
# usage:
#   python src/fairness_tests.py --runs results/rag_variants_predictions.csv \
#       results/rag_variants_predictions_rep2.csv results/rag_variants_predictions_rep3.csv \
#       --output results/fairness_rag.json

import argparse
import json

from common import read_csv, resolve, write_json
from config import ALPHA, MAIN_EFFECT_ALPHA
from fairness_metrics import (AGE_PAIRS, GENDER_PAIRS, GROUPS, counterfactual_consistency, dpr, eod,
                              flip_rate, paired_high, paired_high_pooled, run_disagreement)
from metrics import load_truth
from stat_tests import fisher_rates, mcnemar_exact

# first group in each pair is the reference (male, young)
GENDER_EFFECT = GENDER_PAIRS  # (male, female) in each age group
AGE_EFFECT = AGE_PAIRS        # (young, old) in each gender


def sign(x):
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0


def main_effect(runs, pairs, ref, other):
    # mcnemar on run1 + counts and direction for every run
    names = ["run1"]
    for i in range(2, len(runs) + 1):
        names.append("rep" + str(i))

    by_run = {}
    for i in range(len(runs)):
        only_ref, only_other, n = paired_high_pooled(runs[i], pairs)
        by_run[names[i]] = {
            "only_" + ref + "_high": only_ref,
            "only_" + other + "_high": only_other,
            "n_pairs": n,
            "direction": sign(only_other - only_ref),
            "p_value": mcnemar_exact(only_ref, only_other),
        }

    primary = by_run["run1"]
    significant = primary["p_value"] < MAIN_EFFECT_ALPHA

    # same direction in all runs (and not 0)
    consistent = primary["direction"] != 0
    for r in by_run.values():
        if r["direction"] != primary["direction"]:
            consistent = False

    n = primary["n_pairs"]
    only_ref = primary["only_" + ref + "_high"]
    only_other = primary["only_" + other + "_high"]
    if n:
        rate_diff = (only_other - only_ref) / n
    else:
        rate_diff = None

    if significant and consistent:
        verdict = "claimed"
    elif significant:
        verdict = "fragile"
    else:
        verdict = "not significant"

    return {
        "comparison": other + " vs " + ref,
        "n_pairs": n,
        "only_" + ref + "_high": only_ref,
        "only_" + other + "_high": only_other,
        "rate_difference": rate_diff,
        "p_value": primary["p_value"],
        "alpha": MAIN_EFFECT_ALPHA,
        "significant": significant,
        "same_direction_all_runs": consistent,
        "verdict": verdict,
        "by_run": by_run,
    }


def fairness(runs):
    primary = runs[0]

    has_variant = False
    for p in primary:
        if p.get("variant_id"):
            has_variant = True
            break
    if not has_variant:
        raise SystemExit("fairness_tests.py needs variant-set predictions")
    if len(runs) < 2:
        raise SystemExit("fairness_tests.py needs the primary run and at least rep2 (noise floor)")

    cases = load_truth(primary)

    # noise for fisher: run1 vs rep2, one comparison per variant
    noise = run_disagreement(runs[0], runs[1])
    gender = flip_rate(primary, GENDER_PAIRS)
    age = flip_rate(primary, AGE_PAIRS)

    out = {"n_runs": len(runs)}
    out.update(dpr(cases, primary))
    out.update(eod(cases, primary))
    out.update(counterfactual_consistency(primary))
    out["gender_flip"] = gender
    out["age_flip"] = age
    out["noise_disagreement"] = {
        "runs": "run1 vs rep2",
        "rate": noise["disagree"] / noise["n"],
        "disagree": noise["disagree"],
        "n_comparisons": noise["n"],
    }
    out["fisher_gender_flip_vs_noise"] = fisher_rates(gender["flips"], gender["n_pairs"], noise["disagree"], noise["n"])
    out["fisher_age_flip_vs_noise"] = fisher_rates(age["flips"], age["n_pairs"], noise["disagree"], noise["n"])
    out["gender_main_effect"] = main_effect(runs, GENDER_EFFECT, "male", "female")
    out["age_main_effect"] = main_effect(runs, AGE_EFFECT, "young", "old")

    # all pairs of the 4 groups (6 pairs)
    group_pairs = []
    for i in range(len(GROUPS)):
        for j in range(i + 1, len(GROUPS)):
            group_pairs.append((GROUPS[i], GROUPS[j]))
    k = len(group_pairs)

    out["group_pairs_bonferroni_alpha"] = ALPHA / k
    out["group_pairs"] = []
    for a, b in group_pairs:
        only_a, only_b, n = paired_high(primary, a, b)
        p = mcnemar_exact(only_a, only_b)
        out["group_pairs"].append({"a": a, "b": b, "n": n, "only_a_high": only_a, "only_b_high": only_b,
                                   "p_value": p, "significant": p < ALPHA / k})
    return out


def main():
    ap = argparse.ArgumentParser(description="Fairness metrics and tests for one configuration.")
    ap.add_argument("--runs", nargs="+", required=True, help="Variant prediction files, primary run first.")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    runs = []
    for f in args.runs:
        runs.append(read_csv(f))
    result = fairness(runs)

    files = []
    for f in args.runs:
        files.append(resolve(f).name)
    result["files"] = files
    write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
