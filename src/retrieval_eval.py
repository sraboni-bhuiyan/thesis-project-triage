# manual retrieval relevance check of C2 on the performance set (protocol 6.5), no API
#   sample  pick N cases at random (config.SEED) from a retrieval log, write a rating sheet
#           (one row per chunk, empty "relevant" column, fill with 1/0 using the rule in PROTOCOL.md)
#           gold labels and predictions are not shown so they can't affect the rating
#   score   read the rated sheet, compute P@3, MRR, nDCG@3, precision per rank
# sheet is UTF-8 with BOM so Excel opens it ok
# existing sheet is never overwritten without --force (don't lose ratings)
#
# usage:
#   python src/retrieval_eval.py sample [--log-file F] [--n 25] [--output F] [--force]
#   python src/retrieval_eval.py score  [--sheet F] [--output F]

import argparse
import csv
import json
import math
import random

from common import resolve, write_json
from config import SEED, TOP_K

SHEET_COLUMNS = ["case_id", "rank", "chunk_id", "similarity", "query", "chunk_text", "relevant"]


# ---- metrics (tested in tests/test_retrieval_eval.py)

def precision_at_k(rels, k):
    return sum(rels[:k]) / k


def reciprocal_rank(rels):
    # 1 / rank of first relevant chunk
    for i in range(len(rels)):
        if rels[i]:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(rels, k):
    # binary nDCG, ideal = judged chunks sorted (only top k judged)
    # no relevant chunk -> 0
    # note: use sum() for floats, not +=, sum() is more exact
    gains = []
    top = rels[:k]
    for i in range(len(top)):
        gains.append(top[i] / math.log2(i + 2))
    dcg = sum(gains)

    ideal_gains = []
    ideal = sorted(rels, reverse=True)[:k]
    for i in range(len(ideal)):
        ideal_gains.append(ideal[i] / math.log2(i + 2))
    idcg = sum(ideal_gains)

    if idcg:
        return dcg / idcg
    return 0.0


def score_ratings(ratings, k=TOP_K):
    # ratings: case_id -> list of 0/1 in rank order (length k)
    n = len(ratings)

    per_rank = []
    for i in range(k):
        total = 0
        for r in ratings.values():
            total += r[i]
        per_rank.append(total / n)

    n_relevant = 0
    no_relevant = 0
    p_list = []
    rr_list = []
    ndcg_list = []
    for r in ratings.values():
        n_relevant += sum(r)
        p_list.append(precision_at_k(r, k))
        rr_list.append(reciprocal_rank(r))
        ndcg_list.append(ndcg_at_k(r, k))
        if not any(r):
            no_relevant += 1
    # sum() for floats (more exact than +=)
    p_sum = sum(p_list)
    rr_sum = sum(rr_list)
    ndcg_sum = sum(ndcg_list)

    by_rank = {}
    for i in range(len(per_rank)):
        by_rank[str(i + 1)] = per_rank[i]

    return {
        "n_cases": n,
        "n_judgments": n * k,
        "n_relevant": n_relevant,
        f"p_at_{k}": p_sum / n,
        "mrr": rr_sum / n,
        f"ndcg_at_{k}": ndcg_sum / n,
        "precision_by_rank": by_rank,
        "cases_with_no_relevant": no_relevant,
    }


# ---- sheet read / write

def read_log(path):
    entries = []
    with open(resolve(path), encoding="utf-8") as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))
    return entries


def sample(args):
    out = resolve(args.output)
    if out.exists() and not args.force:
        raise SystemExit(f"{out} exists; it may hold ratings. Use --force to overwrite.")

    # performance cases only
    entries = []
    for e in read_log(args.log_file):
        if not e.get("variant_id"):
            entries.append(e)

    ids = set()
    for e in entries:
        ids.add(e["case_id"])
    if len(ids) != len(entries):
        raise SystemExit("Duplicate case_id in the log; expected one entry per case.")

    # log order = case order
    picked = random.Random(SEED).sample(entries, args.n)

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SHEET_COLUMNS)
        w.writeheader()
        for e in picked:
            chunk_ids = e["retrieved_chunk_ids"]
            scores = e["retrieved_scores"]
            texts = e["retrieved_texts"]
            n_chunks = min(len(chunk_ids), len(scores), len(texts))
            for i in range(n_chunks):
                w.writerow({"case_id": e["case_id"], "rank": i + 1, "chunk_id": chunk_ids[i],
                            "similarity": f"{scores[i]:.4f}", "query": e["query"], "chunk_text": texts[i],
                            "relevant": ""})

    print(f"Wrote {out}: {args.n} cases x {TOP_K} chunks = {args.n * TOP_K} rows to rate (seed {SEED}).")
    print("Fill `relevant` with 1 or 0 using the rule in PROTOCOL.md section 6.5, then save as "
          "results/retrieval_rating_sheet_v2_rated.csv")


def read_ratings(path):
    ratings = {}  # case_id -> {rank: 0/1}
    problems = []
    with open(resolve(path), encoding="utf-8-sig", newline="") as f:
        line_no = 1
        for row in csv.DictReader(f):
            line_no += 1
            raw = (row.get("relevant") or "").strip()
            try:
                val = float(raw)
            except ValueError:
                val = None
            if val != 0.0 and val != 1.0:
                problems.append(f"line {line_no} ({row['case_id']} rank {row['rank']}): relevant = {raw!r}")
                continue
            if row["case_id"] not in ratings:
                ratings[row["case_id"]] = {}
            ratings[row["case_id"]][int(row["rank"])] = int(val)

    # every case needs ranks 1..k
    for cid, ranks in ratings.items():
        if sorted(ranks) != list(range(1, TOP_K + 1)):
            problems.append(f"{cid}: ranks {sorted(ranks)}, expected 1..{TOP_K}")
    if problems:
        raise SystemExit("Rating sheet is incomplete or invalid:\n  " + "\n  ".join(problems))

    result = {}
    for cid, ranks in ratings.items():
        values = []
        for r in range(1, TOP_K + 1):
            values.append(ranks[r])
        result[cid] = values
    return result


def score(args):
    result = score_ratings(read_ratings(args.sheet))
    # general names, build_report.py reads these
    result["n_cases_rated"] = result["n_cases"]
    result["precision_at_k"] = result[f"p_at_{TOP_K}"]
    result["ndcg_at_k"] = result[f"ndcg_at_{TOP_K}"]
    result["sheet"] = str(args.sheet)
    write_json(args.output, result)
    print(json.dumps(result, indent=2))


def main():
    ap = argparse.ArgumentParser(description="Retrieval relevance: rating sheet and scores.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--log-file", default="results/retrieval_logs.jsonl")
    s.add_argument("--n", type=int, default=25)
    s.add_argument("--output", default="results/retrieval_rating_sheet_v2.csv")
    s.add_argument("--force", action="store_true")
    c = sub.add_parser("score")
    c.add_argument("--sheet", default="results/retrieval_rating_sheet_v2_rated.csv")
    c.add_argument("--output", default="results/retrieval_eval.json")
    args = ap.parse_args()

    if args.cmd == "sample":
        sample(args)
    else:
        score(args)


if __name__ == "__main__":
    main()
