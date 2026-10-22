# phase 12: pick cases for the qualitative analysis from results/error_cases.csv (protocol 8), no API
# rule (fixed before reading any case text):
#   1. eligible: C1 and C2 urgency the same in all 3 runs (so it's not run noise)
#      demographic flips: base case inconsistent over its 4 C2 variants in every C2 variant run
#   2. categories (9 cases, protocol says 5-10):
#        c2_safety_downgrade  2  true red/orange, C1 correct, C2 under-triages
#        shared_downgrade     1  true red/orange, C1 and C2 both under-triage (model problem, not RAG)
#        improved             2  C1 wrong, C2 correct
#        worsened_over        2  C1 correct, C2 over-triages
#        demographic_flip     2  C2 variants of one base case disagree
#   3. random pick in each category (config.SEED), each case used once
#   4. plus PURPOSIVE cases, added after the random picks (so the random picks don't change).
#      case 10 = 38195943: the C2 age effect (young yellow / old orange in all 3 C2 variant runs,
#      C1 and C3 stable). none of the random flips shows it. chosen from predictions only,
#      label it as purposive in the thesis.
# outputs:
#   results/qualitative_selection.csv   ids, labels, predictions only (no case text)
#   results/qualitative/case_<id>.md    reading sheet per case: full case text, C1/C2/C4 rationales,
#                                       C2 chunks. MIMIC + copyrighted text: local only, gitignored,
#                                       never share. write about the cases in own words.
#
# usage:
#   python src/select_qualitative.py

import csv
import json
import random

from common import normalize_urgency, read_csv, row_key
from config import CASES_PATH, FAIRNESS_PATH, RESULTS_PATH, SEED
from fairness_metrics import GROUPS, variant_table

N_RUNS = 3
QUOTA = [("c2_safety_downgrade", 2), ("shared_downgrade", 1), ("improved", 2), ("worsened_over", 2),
         ("demographic_flip", 2)]
PURPOSIVE = [("age_effect_purposive", "38195943")]  # variant cases, see rule 4
OUT_DIR = RESULTS_PATH / "qualitative"


def run_files(cfg, tag=""):
    files = [RESULTS_PATH / f"{cfg}{tag}_predictions.csv"]
    for i in range(2, N_RUNS + 1):
        files.append(RESULTS_PATH / f"{cfg}{tag}_predictions_rep{i}.csv")
    return files


def by_key(path):
    d = {}
    for p in read_csv(path):
        d[row_key(p)] = p
    return d


def stable(runs, k):
    # same urgency in every run
    seen = set()
    for r in runs:
        seen.add(normalize_urgency(r[k]["pred_urgency"]))
    return len(seen) == 1


def variants_text(v):
    parts = []
    for g in GROUPS:
        parts.append(f"{g}={v.get(g, '?')}")
    return "; ".join(parts)


def categorise(error_rows, c1_runs, c2_runs, variant_runs):
    # category -> sorted list of eligible case ids
    types = {}
    for r in error_rows:
        if r["case_id"] not in types:
            types[r["case_id"]] = set()
        types[r["case_id"]].add(r["type"])

    cats = {}
    for name, _ in QUOTA:
        cats[name] = []

    for cid in sorted(types):
        t = types[cid]
        if "demographic_flip" in t:
            # flip in every variant run
            flip_all = True
            for v in variant_runs:
                if not (cid in v and len(set(v[cid].values())) > 1):
                    flip_all = False
                    break
            if flip_all:
                cats["demographic_flip"].append(cid)
            t = t - {"demographic_flip"}

        if not t:
            continue
        if not (stable(c1_runs, cid) and stable(c2_runs, cid)):
            continue

        if "safety_downgrade" in t:
            if "worsened" in t:
                cats["c2_safety_downgrade"].append(cid)
            else:
                cats["shared_downgrade"].append(cid)
        elif "improved" in t:
            cats["improved"].append(cid)
        elif "worsened" in t:
            cats["worsened_over"].append(cid)
    return cats


def select(cats):
    rng = random.Random(SEED)
    picked = []
    used = set()
    for name, n in QUOTA:
        pool = []
        for c in cats[name]:
            if c not in used:
                pool.append(c)
        for cid in rng.sample(pool, min(n, len(pool))):
            picked.append((name, cid))
            used.add(cid)
    return picked


def reading_sheet(cid, category, case, preds, chunks, variants):
    lines = [f"# Case {cid} ({category})", "",
             "LOCAL ONLY: MIMIC-derived and copyrighted text. Do not share or upload; paraphrase in the thesis.", "",
             f"- Ground truth: urgency **{case['ground_truth_urgency']}**, specialty {case['ground_truth_specialty']}"
             f" (all listed: {case.get('ground_truth_specialty_all', '')})"]

    for label, rows in preds.items():
        answers = []
        for r in rows:
            answers.append(f"{r['pred_urgency']} / {r['pred_specialty']}")
        lines.append(f"- {label}: " + ", ".join(answers) + "  (run1, rep2, rep3)")

    if variants:
        for i in range(len(variants)):
            lines.append(f"- C2 variants run {i + 1}: " + variants_text(variants[i]))

    lines += ["", "## Case text", "", f"**Age / gender:** {case['age']} / {case['gender']}", "",
              f"**Vitals:** {case['vitals']}", "", "**Symptoms:**", "", case["symptoms"], ""]

    for label, rows in preds.items():
        lines += [f"## {label} rationale (run1)", "", rows[0]["pred_rationale"], ""]

    if chunks:
        lines += ["## C2 retrieved chunks (run1)", ""]
        for chunk_id, score, text in chunks:
            lines += [f"### {chunk_id} (similarity {score:.3f})", "", text, ""]
    return "\n".join(lines)


def main():
    errors = read_csv(RESULTS_PATH / "error_cases.csv")

    c1 = []
    for f in run_files("baseline"):
        c1.append(by_key(f))
    c2 = []
    for f in run_files("rag"):
        c2.append(by_key(f))
    c4 = []
    for f in run_files("rag_fixed"):
        c4.append(by_key(f))
    vruns = []
    for f in run_files("rag", "_variants"):
        vruns.append(variant_table(read_csv(f)))

    cats = categorise(errors, c1, c2, vruns)
    picked = select(cats)
    for category, cid in PURPOSIVE:
        picked.append((category, cid))

    cases = {}
    for c in read_csv(CASES_PATH):
        cases[row_key(c)] = c
    # one variant per base case is enough for the text
    vcases = {}
    for c in read_csv(FAIRNESS_PATH):
        if c["variant_id"].endswith("_young_male"):
            vcases[c["case_id"]] = c

    logs = {}
    with open(RESULTS_PATH / "retrieval_logs.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                e = json.loads(line)
                logs[e["case_id"]] = e

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_rows = []
    for category, cid in picked:
        flip = category in ("demographic_flip", "age_effect_purposive")
        if flip:
            case = vcases[cid]
            preds = {}
            chunks = None
            variants = []
            for v in vruns:
                variants.append(v[cid])
        else:
            case = cases[cid]
            preds = {"C1": [], "C2": [], "C4": []}
            for r in c1:
                preds["C1"].append(r[cid])
            for r in c2:
                preds["C2"].append(r[cid])
            for r in c4:
                preds["C4"].append(r[cid])
            e = logs[cid]
            chunks = list(zip(e["retrieved_chunk_ids"], e["retrieved_scores"], e["retrieved_texts"]))
            variants = None

        sheet = reading_sheet(cid, category, case, preds, chunks, variants)
        (OUT_DIR / f"case_{cid}.md").write_text(sheet, encoding="utf-8")

        row = {"category": category, "case_id": cid, "ground_truth_urgency": case["ground_truth_urgency"],
               "ground_truth_specialty": case["ground_truth_specialty"]}
        if flip:
            run_texts = []
            for v in variants:
                run_texts.append(variants_text(v))
            row["c2_variant_predictions"] = " | ".join(run_texts)
        else:
            for label, rows in preds.items():
                urg = []
                for r in rows:
                    urg.append(r["pred_urgency"])
                row[f"{label.lower()}_urgency_runs"] = "/".join(urg)
            row["c2_specialty"] = preds["C2"][0]["pred_specialty"]
            ids = []
            for c in chunks:
                ids.append(c[0])
            row["c2_chunk_ids"] = ";".join(ids)
        out_rows.append(row)

    # columns = all keys, in order seen
    fields = []
    for r in out_rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with open(RESULTS_PATH / "qualitative_selection.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out_rows)

    counts = []
    for k, v in cats.items():
        counts.append(f"{k} {len(v)}")
    print("Eligible per category: " + ", ".join(counts))
    for category, cid in picked:
        print(f"  {category:20s} {cid}")
    print(f"Wrote results/qualitative_selection.csv and {len(picked)} reading sheets in {OUT_DIR}")


if __name__ == "__main__":
    main()
