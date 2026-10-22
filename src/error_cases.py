# candidate cases for the qualitative analysis (protocol 8), no API
# row types (primary runs):
#   improved          C1 wrong, CONFIG correct (urgency, performance set)
#   worsened          C1 correct, CONFIG wrong
#   safety_downgrade  true red/orange, CONFIG under-triages
#   demographic_flip  base case where the 4 variants don't all get the same urgency
# output has MIMIC case text -> stays in results/, never share or commit
#
# usage:
#   python src/error_cases.py --config rag [--output results/error_cases.csv]

import argparse
import csv
from collections import Counter

from common import normalize_urgency, read_csv, resolve, row_key
from config import CASES_PATH, ERROR, FAIRNESS_PATH, HIGH_ACUITY, RESULTS_PATH
from fairness_metrics import GROUPS, variant_table
from metrics import urgency_outcome

FIELDS = ["type", "case_id", "variant_id", "ground_truth_urgency", "pred_baseline", "pred_config",
          "ground_truth_specialty", "pred_specialty_config", "variant_predictions", "symptoms",
          "rationale_baseline", "rationale_config"]


def by_key(rows):
    d = {}
    for r in rows:
        d[row_key(r)] = r
    return d


def build(config):
    cases = by_key(read_csv(CASES_PATH))
    base = by_key(read_csv(RESULTS_PATH / "baseline_predictions.csv"))
    conf = by_key(read_csv(RESULTS_PATH / f"{config}_predictions.csv"))

    rows = []
    for key, c in cases.items():
        b = base.get(key)
        p = conf.get(key)
        # need both answers, no ERROR
        if not b or not p:
            continue
        if b["pred_urgency"] == ERROR or p["pred_urgency"] == ERROR:
            continue

        truth = c["ground_truth_urgency"]
        ob = urgency_outcome(normalize_urgency(b["pred_urgency"]), truth)
        oc = urgency_outcome(normalize_urgency(p["pred_urgency"]), truth)

        types = []
        if ob != "correct" and oc == "correct":
            types.append("improved")
        if ob == "correct" and oc != "correct":
            types.append("worsened")
        if truth in HIGH_ACUITY and (oc == "under" or oc == "invalid"):
            types.append("safety_downgrade")

        for t in types:
            rows.append({"type": t, "case_id": c["case_id"], "variant_id": "", "ground_truth_urgency": truth,
                         "pred_baseline": b["pred_urgency"], "pred_config": p["pred_urgency"],
                         "ground_truth_specialty": c["ground_truth_specialty"],
                         "pred_specialty_config": p["pred_specialty"], "symptoms": c["symptoms"],
                         "rationale_baseline": b["pred_rationale"], "rationale_config": p["pred_rationale"]})

    # variant flips
    vfile = RESULTS_PATH / f"{config}_variants_predictions.csv"
    if vfile.exists():
        # truth is the same for all variants of a case, so any one is ok
        vcases = {}
        for c in read_csv(FAIRNESS_PATH):
            vcases[c["case_id"]] = c

        table = variant_table(read_csv(vfile))
        for case_id in sorted(table):
            v = table[case_id]
            if len(set(v.values())) > 1:
                c = vcases[case_id]
                parts = []
                for g in GROUPS:
                    parts.append(f"{g}={v.get(g, '?')}")
                rows.append({"type": "demographic_flip", "case_id": case_id, "variant_id": "",
                             "ground_truth_urgency": c["ground_truth_urgency"],
                             "ground_truth_specialty": c["ground_truth_specialty"],
                             "variant_predictions": "; ".join(parts),
                             "symptoms": c["symptoms"]})
    return rows


def main():
    ap = argparse.ArgumentParser(description="Qualitative-analysis candidates.")
    ap.add_argument("--config", default="rag", choices=["rag", "rag_masked", "rag_fixed"])
    ap.add_argument("--output", default="results/error_cases.csv")
    args = ap.parse_args()

    rows = build(args.config)
    with open(resolve(args.output), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    counts = Counter()
    for r in rows:
        counts[r["type"]] += 1
    print(f"Wrote {resolve(args.output)}: {dict(counts)}")


if __name__ == "__main__":
    main()
