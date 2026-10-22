# C2 RAG, C3 RAG + demographic masking, C4 fixed-context control (protocol 4). calls the API
#   C2  (default)            top-k chunks for this case's case block
#   C3  --mask-demographics  masked case block, retrieval also with the masked block
#   C4  --fixed-context      same chunks for every case (fixed_context.json), no retrieval
#
# all three use rag.txt and join chunks with config.CHUNK_SEPARATOR,
# so C2 and C4 prompts are only different inside <guidelines>
# one JSON line per case in the retrieval log (query, chunk ids, scores, texts)
# on resume, cases already in the log are not logged again
#
# usage:
#   python src/run_rag.py [--mask-demographics | --fixed-context] [--cases-file F] [--output F]
#                         [--log-file F] [--limit N] [--no-resume] [--sleep S]

import argparse
import json

from common import LLM, build_case_block, load_prompt, read_csv, resolve, row_key, run_predictions
from config import (CASES_PATH, CHUNK_SEPARATOR, CORPUS_FILE, EMBEDDING_MODEL, FAISS_INDEX, FIXED_CONTEXT_FILE,
                    ID_MAP_FILE, TOP_K)


class Retriever:
    # exact cosine search over the frozen FAISS index (C2, C3)

    def __init__(self, top_k):
        import faiss
        from sentence_transformers import SentenceTransformer

        self.top_k = top_k
        self.index = faiss.read_index(str(FAISS_INDEX))
        self.ids = json.loads(ID_MAP_FILE.read_text(encoding="utf-8"))

        texts = {}
        with open(CORPUS_FILE, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                texts[rec["chunk_id"]] = rec["text"]

        # index, id map and corpus must fit together
        if len(self.ids) != self.index.ntotal or set(self.ids) != set(texts):
            raise SystemExit("Index, id_map and corpus disagree; rebuild with build_index.py")
        self.texts = texts
        self.model = SentenceTransformer(EMBEDDING_MODEL)

    def __call__(self, query):
        from build_index import embed

        # one query at a time
        scores, idx = self.index.search(embed([query], self.model), self.top_k)
        ids = []
        for i in idx[0]:
            ids.append(self.ids[i])
        score_list = []
        for s in scores[0]:
            score_list.append(float(s))
        texts = []
        for i in ids:
            texts.append(self.texts[i])
        return ids, score_list, texts


def fixed_context():
    # C4: (chunk ids, texts) in saved order, file made by pick_fixed_context.py
    if not FIXED_CONTEXT_FILE.exists():
        raise SystemExit(f"{FIXED_CONTEXT_FILE} not found; run python src/pick_fixed_context.py first")
    chunks = json.loads(FIXED_CONTEXT_FILE.read_text(encoding="utf-8"))["chunks"]
    if len(chunks) != TOP_K:
        raise SystemExit(f"fixed_context.json has {len(chunks)} chunks; C4 needs TOP_K = {TOP_K}")
    ids = []
    texts = []
    for c in chunks:
        ids.append(c["chunk_id"])
        texts.append(c["text"])
    return ids, texts


def main():
    ap = argparse.ArgumentParser(description="C2/C3/C4 retrieval-augmented triage predictions.")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--mask-demographics", action="store_true", help="C3: mask age/gender.")
    mode.add_argument("--fixed-context", action="store_true", help="C4: same chunks for every case.")
    ap.add_argument("--cases-file", default=str(CASES_PATH))
    ap.add_argument("--output", default=None, help="Default: results/rag[_masked|_fixed]_predictions.csv")
    ap.add_argument("--log-file", default=None, help="Default: results/retrieval_logs[_masked|_fixed].jsonl")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--sleep", type=float, default=0.0)
    args = ap.parse_args()

    if args.mask_demographics:
        suffix = "_masked"
        label = "C3 RAG+mask"
    elif args.fixed_context:
        suffix = "_fixed"
        label = "C4 fixed"
    else:
        suffix = ""
        label = "C2 RAG"
    output = resolve(args.output or f"results/rag{suffix}_predictions.csv")
    log_path = resolve(args.log_file or f"results/retrieval_logs{suffix}.jsonl")
    label += f" ({resolve(args.cases_file).stem})"  # e.g. "(fairness_variants)"

    template = load_prompt("rag.txt")
    cases = read_csv(args.cases_file)[:args.limit]
    if args.fixed_context:
        fixed_ids, fixed_texts = fixed_context()
        print(f"{label}: fixed chunks {', '.join(fixed_ids)}")
    else:
        retrieve = Retriever(TOP_K)
    llm = LLM()

    # one log entry per case: keep old entries on resume, never log a case twice
    logged = {}
    if log_path.exists() and not args.no_resume:
        with open(log_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    key = entry["variant_id"] or entry["case_id"]
                    if key not in logged:
                        logged[key] = entry

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as log:
        for entry in logged.values():
            log.write(json.dumps(entry, ensure_ascii=False) + "\n")
        log.flush()

        def predict(case):
            case_block = build_case_block(case, mask=args.mask_demographics)
            if args.fixed_context:
                ids = fixed_ids
                scores = None
                texts = fixed_texts
            else:
                ids, scores, texts = retrieve(case_block)

            if row_key(case) not in logged:
                entry = {"case_id": case.get("case_id"), "variant_id": case.get("variant_id", ""),
                         "masked": args.mask_demographics, "fixed_context": args.fixed_context,
                         "query": case_block, "retrieved_chunk_ids": ids, "retrieved_scores": scores,
                         "retrieved_texts": texts}
                log.write(json.dumps(entry, ensure_ascii=False) + "\n")
                log.flush()
                logged[row_key(case)] = entry

            prompt = template.format(retrieved_context=CHUNK_SEPARATOR.join(texts), case_block=case_block)
            return llm.ask(prompt)

        run_predictions(cases, output, predict, desc=label, resume=not args.no_resume, sleep=args.sleep)
    print(f"Retrieval log: {log_path} ({len(logged)} entries)")


if __name__ == "__main__":
    main()
