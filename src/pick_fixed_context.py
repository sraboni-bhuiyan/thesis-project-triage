# C4 fixed guideline context from the retrieval logs (protocol 4), no API
# the TOP_K chunks retrieved most often (any rank) in the 4 main logs
# (C2 and C3, performance + variant sets), ties broken by chunk_id
# only those 4 logs are read, by name: C4's own log and repeat logs (_rep2, _rep3, ...)
# are never counted, any other retrieval log found is listed as ignored
#
# output: data/guidelines_clean/fixed_context.json (frozen once C4 has run)
#
# usage:
#   python src/pick_fixed_context.py

import json
from collections import Counter

from common import write_json
from config import CORPUS_FILE, FIXED_CONTEXT_FILE, RESULTS_PATH, TOP_K

MAIN_LOGS = ["retrieval_logs.jsonl", "retrieval_logs_masked.jsonl",
             "retrieval_logs_variants.jsonl", "retrieval_logs_masked_variants.jsonl"]


def count_retrievals(paths):
    # (Counter chunk_id -> times retrieved, {log name: number of queries})
    counts = Counter()
    per_log = {}
    for path in paths:
        entries = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    entries.append(json.loads(line))
        for e in entries:
            counts.update(e["retrieved_chunk_ids"])
        per_log[path.name] = len(entries)
    return counts, per_log


def count_order(item):
    # most retrieved first, then chunk_id (so ties always come out the same)
    chunk_id, n = item
    return (-n, chunk_id)


def top_chunks(counts, n):
    # the n most retrieved chunk ids
    ranked = sorted(counts.items(), key=count_order)
    ids = []
    for cid, _ in ranked[:n]:
        ids.append(cid)
    return ids


def main():
    paths = []
    for name in MAIN_LOGS:
        paths.append(RESULTS_PATH / name)

    missing = []
    for p in paths:
        if not p.exists():
            missing.append(p.name)
    if missing:
        raise SystemExit(f"Missing main retrieval logs (run Phase 7 first): {missing}")

    # other retrieval logs (C4, repeats) -> not counted
    ignored = []
    for p in RESULTS_PATH.glob("retrieval_logs*.jsonl"):
        if p.name not in MAIN_LOGS:
            ignored.append(p.name)
    ignored.sort()

    counts, per_log = count_retrievals(paths)
    n_queries = sum(per_log.values())
    chosen = top_chunks(counts, TOP_K)

    corpus = {}
    with open(CORPUS_FILE, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            corpus[rec["chunk_id"]] = rec

    log_text = []
    for k, v in per_log.items():
        log_text.append(f"{k} {v}")
    print(f"Counted {n_queries} queries: " + ", ".join(log_text))
    if ignored:
        print(f"Ignored (not main logs): {', '.join(ignored)}")
    print(f"{len(counts)} distinct chunks retrieved. Top {TOP_K + 3}:")
    for cid in top_chunks(counts, TOP_K + 3):
        if cid in chosen:
            mark = "*"
        else:
            mark = " "
        print(f"  {mark} {cid:<34} {counts[cid]:>5} / {n_queries} ({counts[cid] / n_queries:.1%})")

    chunks = []
    for cid in chosen:
        chunks.append({"chunk_id": cid, "doc_id": corpus[cid]["doc_id"], "retrieval_count": counts[cid],
                       "text": corpus[cid]["text"]})

    write_json(FIXED_CONTEXT_FILE, {
        "description": "C4 fixed context: the TOP_K chunks most often retrieved in the four main logs.",
        "n_queries": n_queries,
        "log_files": per_log,
        "ignored_log_files": ignored,
        "n_chunks": len(chosen),
        "chunk_ids": chosen,
        "chunks": chunks,
        "all_retrieval_counts": dict(sorted(counts.items(), key=count_order)),
    })
    print(f"Wrote {FIXED_CONTEXT_FILE}")


if __name__ == "__main__":
    main()
