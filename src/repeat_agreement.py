# noise floor from repeat runs, same config same inputs (protocol 6.4), no API
# input: 2+ prediction files, primary run first
# output:
#   - urgency/specialty agreement for each pair of runs, mean, range, all-runs-agree share
#   - per run: urgency exact acc, specialty acc (+ range)
#   - variants only, per run: consistency, DPR, gender/age flip rate (+ range)
#
# usage:
#   python src/repeat_agreement.py --runs results/rag_variants_predictions.csv \
#       results/rag_variants_predictions_rep2.csv results/rag_variants_predictions_rep3.csv \
#       --output results/noise_rag_variants.json

import argparse
import json

from common import normalize_urgency, read_csv, resolve, row_key, write_json
from config import ERROR
from fairness_metrics import AGE_PAIRS, GENDER_PAIRS, counterfactual_consistency, dpr, flip_rate, run_disagreement
from metrics import evaluate, load_truth


def value_range(values):
    # min, max, spread (None values skipped)
    v = []
    for x in values:
        if x is not None:
            v.append(x)
    if not v:
        return None
    return {"min": min(v), "max": max(v), "spread": max(v) - min(v)}


def noise_floor(runs, names):
    # all runs need the same cases
    first_keys = set()
    for p in runs[0]:
        first_keys.add(row_key(p))
    for r in runs:
        keys = set()
        for p in r:
            keys.add(row_key(p))
        if keys != first_keys:
            raise SystemExit("Runs cover different cases; repeat agreement needs the same inputs.")

    cases = load_truth(runs[0])

    variants = False
    for p in runs[0]:
        if p.get("variant_id"):
            variants = True
            break

    # every pair of runs
    pairs = {}
    for i in range(len(runs)):
        for j in range(i + 1, len(runs)):
            u = run_disagreement(runs[i], runs[j], "pred_urgency")
            s = run_disagreement(runs[i], runs[j], "pred_specialty")
            pairs[names[i] + "_vs_" + names[j]] = {
                "n": u["n"],
                "urgency_disagree": u["disagree"],
                "urgency_agreement": u["agreement"],
                "specialty_disagree": s["disagree"],
                "specialty_agreement": s["agreement"],
            }

    # urgency label per case for each run (no ERROR rows)
    labels = []
    for r in runs:
        lab = {}
        for p in r:
            if p["pred_urgency"] != ERROR:
                lab[row_key(p)] = normalize_urgency(p["pred_urgency"])
        labels.append(lab)

    # cases answered in all runs
    common_keys = set(labels[0])
    for lab in labels[1:]:
        common_keys = common_keys & set(lab)

    all_agree = 0
    for k in common_keys:
        seen = set()
        for lab in labels:
            seen.add(lab[k])
        if len(seen) == 1:
            all_agree += 1

    # numbers per run
    per_run = {}
    for i in range(len(runs)):
        r = runs[i]
        ev = evaluate(cases, r)
        row = {
            "urgency_exact_acc": ev["urgency_exact_acc"],
            "specialty_acc": ev["specialty_acc"],
            "n_error": ev["n_error"],
            "n_invalid_urgency": ev["n_invalid_urgency"],
        }
        if variants:
            row["consistency"] = counterfactual_consistency(r)["consistency"]
            row["dpr"] = dpr(cases, r)["dpr"]
            row["gender_flip_rate"] = flip_rate(r, GENDER_PAIRS)["rate"]
            row["age_flip_rate"] = flip_rate(r, AGE_PAIRS)["rate"]
        per_run[names[i]] = row

    # mean and range over the run pairs
    urg_agree = []
    spec_agree = []
    disagree_total = 0
    n_total = 0
    for p in pairs.values():
        urg_agree.append(p["urgency_agreement"])
        spec_agree.append(p["specialty_agreement"])
        disagree_total += p["urgency_disagree"]
        n_total += p["n"]

    if common_keys:
        all_agree_share = all_agree / len(common_keys)
    else:
        all_agree_share = None

    if variants:
        set_name = "variants"
    else:
        set_name = "performance"

    out = {
        "set": set_name,
        "runs": names,
        "n_runs": len(runs),
        "n_compared": len(common_keys),
        "pairs": pairs,
        "urgency_run_agreement": sum(urg_agree) / len(pairs),
        "specialty_run_agreement": sum(spec_agree) / len(pairs),
        "urgency_agreement_range": value_range(urg_agree),
        "specialty_agreement_range": value_range(spec_agree),
        "urgency_all_runs_agree": all_agree_share,
        # pooled over all run pairs
        "urgency_disagree_pooled": disagree_total,
        "n_comparisons_pooled": n_total,
        "per_run": per_run,
    }

    # range over runs for each per-run number (not the n_ counts)
    for key in per_run[names[0]]:
        if key.startswith("n_"):
            continue
        values = []
        for row in per_run.values():
            values.append(row[key])
        out[key + "_range"] = value_range(values)
    return out


def main():
    ap = argparse.ArgumentParser(description="Noise floor from repeat runs of one configuration.")
    ap.add_argument("--runs", nargs="+", required=True, help="Prediction files, primary run first.")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    if len(args.runs) < 2:
        raise SystemExit("Need at least 2 runs")

    names = ["run1"]
    for i in range(2, len(args.runs) + 1):
        names.append("rep" + str(i))

    runs = []
    for f in args.runs:
        runs.append(read_csv(f))

    result = noise_floor(runs, names)

    files = []
    for f in args.runs:
        files.append(resolve(f).name)
    result["files"] = files
    write_json(args.output, result)

    # print without the long parts
    short = {}
    for k in result:
        if k != "pairs" and k != "per_run":
            short[k] = result[k]
    print(json.dumps(short, indent=2))


if __name__ == "__main__":
    main()
