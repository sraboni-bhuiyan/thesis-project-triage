# paired exact mcnemar between two prediction files (protocol 7.1), no API
# urgency = exact colour, specialty = strict (first listed)
# --mapped = sub-specialty mapping (sensitivity 6.2.1)
# both files must have the same cases
#
# usage:
#   python src/mcnemar.py -a results/baseline_predictions.csv -b results/rag_predictions.csv \
#       --metric urgency --output results/mcnemar_baseline_vs_rag_urgency.json [--mapped]

import argparse
import json

from common import read_csv, resolve, row_key, write_json
from metrics import correct_by_case, load_truth
from stat_tests import mcnemar_exact


def compare(preds_a, preds_b, metric, mapped=False):
    # check same cases
    keys_a = set()
    for p in preds_a:
        keys_a.add(row_key(p))
    keys_b = set()
    for p in preds_b:
        keys_b.add(row_key(p))
    if keys_a != keys_b:
        raise SystemExit("The two prediction files cover different cases; McNemar needs paired data.")

    cases = load_truth(preds_a)
    ca = correct_by_case(cases, preds_a, metric, mapped)
    cb = correct_by_case(cases, preds_b, metric, mapped)

    # cases answered in both
    keys = []
    for k in ca:
        if k in cb:
            keys.append(k)
    keys.sort()

    correct_a = 0
    correct_b = 0
    both = 0
    neither = 0
    only_a = 0
    only_b = 0
    for k in keys:
        if ca[k]:
            correct_a += 1
        if cb[k]:
            correct_b += 1
        if ca[k] and cb[k]:
            both += 1
        elif ca[k]:
            only_a += 1
        elif cb[k]:
            only_b += 1
        else:
            neither += 1

    return {
        "metric": metric,
        "mapped": mapped,
        "n_paired": len(keys),
        "acc_a": correct_a / len(keys),
        "acc_b": correct_b / len(keys),
        "both_correct": both,
        "neither_correct": neither,
        "only_a_correct": only_a,
        "only_b_correct": only_b,
        "p_value": mcnemar_exact(only_a, only_b),
    }


def main():
    ap = argparse.ArgumentParser(description="Paired exact McNemar between two prediction files.")
    ap.add_argument("-a", required=True)
    ap.add_argument("-b", required=True)
    ap.add_argument("--metric", choices=["urgency", "specialty"], required=True)
    ap.add_argument("--mapped", action="store_true", help="Specialty with the sub-specialty mapping.")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    if args.mapped and args.metric != "specialty":
        raise SystemExit("--mapped applies to --metric specialty only")

    result = compare(read_csv(args.a), read_csv(args.b), args.metric, args.mapped)
    result["file_a"] = resolve(args.a).name
    result["file_b"] = resolve(args.b).name
    write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
