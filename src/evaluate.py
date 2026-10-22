# metrics for one prediction file, no API
# truth = fairness_variants.csv if preds have variant_id, else main.csv
# n_error and n_invalid_urgency are used by the smoke gate in run_all.py
#
# usage:
#   python src/evaluate.py -p results/rag_predictions.csv --output results/eval_rag.json

import argparse
import json

from common import read_csv, resolve, row_key, write_json
from config import CASES_PATH, FAIRNESS_PATH
from metrics import evaluate


def main():
    ap = argparse.ArgumentParser(description="Evaluate one prediction file.")
    ap.add_argument("-p", "--pred-file", required=True)
    ap.add_argument("--cases-file", default=None, help="Ground truth (default: chosen from the prediction ids).")
    ap.add_argument("--output", default=None, help="Optional JSON output path.")
    args = ap.parse_args()

    preds = read_csv(args.pred_file)

    # pick truth file
    if args.cases_file:
        cases_path = resolve(args.cases_file)
    else:
        has_variant = False
        for p in preds:
            if p.get("variant_id"):
                has_variant = True
                break
        if has_variant:
            cases_path = FAIRNESS_PATH
        else:
            cases_path = CASES_PATH

    cases = {}
    for c in read_csv(cases_path):
        cases[row_key(c)] = c

    result = evaluate(cases, preds)

    # which models answered
    models = set()
    for p in preds:
        if p.get("response_model"):
            models.add(p["response_model"])

    result["pred_file"] = resolve(args.pred_file).name
    result["cases_file"] = cases_path.name
    result["models"] = sorted(models)

    print(json.dumps(result, indent=2))
    if args.output:
        write_json(args.output, result)
        print(f"Saved {resolve(args.output)}")


if __name__ == "__main__":
    main()
