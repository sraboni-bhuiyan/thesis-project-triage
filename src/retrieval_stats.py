# retrieval stats (protocol 6.5) + C4 overlap, no API
# for each main retrieval log (C2/C3, performance + variants):
#   share per source document, distinct chunks, distinct chunks at rank 1, mean top-1 similarity
# also: how many of C4's fixed chunks are in C2's top-k for each performance case (0..k)
#
# usage:
#   python src/retrieval_stats.py [--output results/retrieval_stats.json]

import argparse
import json
from collections import Counter

from common import resolve, write_json
from config import FIXED_CONTEXT_FILE, RESULTS_PATH, TOP_K

LOGS = {"C2 performance": "retrieval_logs.jsonl", "C3 performance": "retrieval_logs_masked.jsonl",
        "C2 variants": "retrieval_logs_variants.jsonl", "C3 variants": "retrieval_logs_masked_variants.jsonl"}


def read_log(path):
    entries = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))
    return entries


def document_of(chunk_id):
    # '06_ctas_participant_manual_023' -> '06_ctas_participant_manual'
    return chunk_id.rsplit("_", 1)[0]


def describe(entries):
    ids = []
    for e in entries:
        for c in e["retrieved_chunk_ids"]:
            ids.append(c)

    docs = Counter()
    for c in ids:
        docs[document_of(c)] += 1

    top1 = []
    rank1 = set()
    for e in entries:
        top1.append(e["retrieved_scores"][0])
        rank1.add(e["retrieved_chunk_ids"][0])

    share = {}
    count = {}
    for d, n in docs.most_common():
        share[d] = n / len(ids)
        count[d] = n

    return {
        "n_queries": len(entries),
        "n_retrievals": len(ids),
        "share_by_document": share,
        "count_by_document": count,
        "distinct_chunks": len(set(ids)),
        "distinct_chunks_rank1": len(rank1),
        "mean_top1_similarity": sum(top1) / len(top1),
        "most_retrieved": Counter(ids).most_common(5),
    }


def c4_overlap(c2_entries):
    data = json.loads(FIXED_CONTEXT_FILE.read_text(encoding="utf-8"))
    fixed = set()
    for c in data["chunks"]:
        fixed.add(c["chunk_id"])

    # how many fixed chunks in each case's top-k
    hits = Counter()
    for e in c2_entries:
        hits[len(fixed & set(e["retrieved_chunk_ids"]))] += 1
    n = len(c2_entries)

    by_overlap = {}
    for k in range(TOP_K + 1):
        by_overlap[str(k)] = hits.get(k, 0)
    total = 0
    for k, v in hits.items():
        total += k * v

    return {"fixed_chunk_ids": sorted(fixed), "n_cases": n, "cases_by_overlap": by_overlap,
            "mean_overlap": total / n}


def main():
    ap = argparse.ArgumentParser(description="Descriptive retrieval statistics.")
    ap.add_argument("--output", default="results/retrieval_stats.json")
    args = ap.parse_args()

    out = {}
    for name, fname in LOGS.items():
        path = RESULTS_PATH / fname
        if path.exists():
            out[name] = describe(read_log(path))

    c2_log = RESULTS_PATH / LOGS["C2 performance"]
    if FIXED_CONTEXT_FILE.exists() and c2_log.exists():
        out["c4_overlap_with_c2"] = c4_overlap(read_log(c2_log))

    write_json(args.output, out)
    print(json.dumps(out, indent=2))
    print(f"Saved {resolve(args.output)}")


if __name__ == "__main__":
    main()
