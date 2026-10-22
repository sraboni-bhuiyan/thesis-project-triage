# build the retrieval corpus: data/guidelines_raw/*.pdf -> data/guidelines_clean/corpus.jsonl
# (protocol 2.4)
#
# for each pdf, in this order:
#   1. text of every page (pypdf, pinned version, see requirements.txt)
#   2. remove running heads/footers: lines on more than half of the pages
#   3. join pages, fix extraction artifacts, join words hyphenated over line breaks,
#      remove web navigation text, normalise whitespace
#   4. cut out training-exercise vignettes (unanswered numbered patient scenarios)
#   5. 250-word chunks with 50-word overlap
#   6. drop chunks that are only course material (no clinical or acuity content)
#
# gives the same corpus as September 2026 (207 chunks)
# --verify rebuilds in memory and compares content hashes with the existing file
#
# CTAS and ESI manuals are copyrighted: local research use only, never commit
#
# usage:
#   python src/build_corpus.py             # build
#   python src/build_corpus.py --verify    # rebuild + compare, writes nothing
#   python src/build_corpus.py --verify --against PATH   # compare with another corpus file

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import pypdf

from config import (CHUNK_WORDS, CORPUS_FILE, GUIDELINES_PATH, HEADER_FOOTER_RATIO, MIN_TAIL_WORDS,
                    OVERLAP_WORDS)

# ---- step 3: cleaning
# characters pypdf gives for symbols in these pdfs (e.g. ENA(R) -> U+FFFD)
# written as codes so you can see them: replacement char, curly quotes, dashes, no-break space
ARTIFACTS = {"\ufffd": "", "\u2019": "'", "\u2018": "'", "\u201c": '"', "\u201d": '"',
             "\u2013": "-", "\u2014": "-", "\u00a0": " "}
# web navigation text in the saved NHS / Charite pages
BOILERPLATE = [r"skip navigation", r"sidebar menu", r"sidebar contact person", r"sidebar location",
               r"focus area and clinic finder", r"construction cku \d+"]

# ---- step 4: training vignettes
# vignette list = at least VIGNETTE_MIN_RUN numbered items in a row that start like a patient
# ("3. A 45-year-old ..."), each max VIGNETTE_MAX_GAP characters after the one before
# single numbered sentences (e.g. GCS criteria) are real content and stay
VIGNETTE_ITEM = re.compile(r"\b(\d{1,2})\.\s+(?:A|An|The|Patient|Pt)\b")
VIGNETTE_MIN_RUN = 4
VIGNETTE_MAX_GAP = 350
# the instruction sentence before ("For the following patients, identify ...") goes too
EXERCISE_CUE = re.compile(r"\b(identify|following patient|complete the|match the|for the following)\b", re.I)
EXERCISE_CUE_LOOKBACK = 300
# run ends at the next numbered heading ("4.2 Chest pain") if it's within SECTION_HEAD_MAX
# characters of the last item, otherwise LAST_ITEM_SPAN characters after it
SECTION_HEAD = re.compile(r"\s\d{0,3}\s*\d+\.\d+\s+[A-Z]")
SECTION_HEAD_MAX = 600
LAST_ITEM_SPAN = 400

# ---- step 6: course material filter
# dropped only if it looks like course material AND no acuity term AND fewer than
# MIN_CLINICAL_HITS clinical words -> clinical text never dropped
SCAFFOLDING = re.compile(
    r"\b(exercise|scenario \d|practice (case|question)|answer key|quiz|workshop|small group|"
    r"course (goals|organization|overview)|participant.s manual|learning objectives|module \d|"
    r"table of contents|acknowledg|acronyms)\b", re.I)
CLINICAL = re.compile(
    r"\b(pain|bleed|breath|airway|chest|abdom|fever|trauma|vital sign|pulse|blood pressure|"
    r"conscious|sepsis|stroke|seizure|injury|wound|vomit|dyspnea|hypoten|tachycard|oxygen|"
    r"glasgow|triage level|undertriag|symptom)\b", re.I)
MIN_CLINICAL_HITS = 3

# acuity words saved per chunk (metadata for analysis, not used for retrieval)
URGENCY_VOCAB = ["ctas 1", "ctas 2", "ctas 3", "ctas 4", "ctas 5", "level 1", "level 2", "level 3",
                 "level 4", "level 5", "resuscitation", "emergent", "urgent", "less urgent", "non-urgent",
                 "red", "orange", "yellow", "green", "blue", "immediate", "life-threatening", "acuity"]


def page_texts(pdf):
    # step 1: one string per page, empty pages kept so the header ratio counts all pages
    pages = []
    for page in pypdf.PdfReader(str(pdf)).pages:
        pages.append(page.extract_text() or "")
    return pages


def drop_running_lines(pages):
    # step 2: remove lines on more than HEADER_FOOTER_RATIO of the pages (needs >= 3 pages)
    if len(pages) < 3:
        return pages

    # on how many pages each line appears
    seen_on = Counter()
    for page in pages:
        lines = set()
        for ln in page.splitlines():
            if ln.strip():
                lines.add(ln.strip())
        for line in lines:
            seen_on[line] += 1

    running = set()
    for line, n in seen_on.items():
        if n > len(pages) * HEADER_FOOTER_RATIO:
            running.add(line)

    out = []
    for page in pages:
        keep = []
        for ln in page.splitlines():
            if ln.strip() not in running:
                keep.append(ln)
        out.append("\n".join(keep))
    return out


def clean(text):
    # step 3
    for bad, good in ARTIFACTS.items():
        text = text.replace(bad, good)
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)  # "assess-\nment" -> "assessment"
    text = re.sub(r"\s+", " ", text)
    for pattern in BOILERPLATE:
        text = re.sub(pattern, " ", text, flags=re.I)
    return re.sub(r"\s+", " ", text).strip()


def vignette_runs(text):
    # step 4 helper: list of runs, each run = list of (offset, number)
    runs = []
    run = []
    for m in VIGNETTE_ITEM.finditer(text):
        pos = m.start()
        num = int(m.group(1))
        # next number and close enough -> same run
        if run and num == run[-1][1] + 1 and pos - run[-1][0] <= VIGNETTE_MAX_GAP:
            run.append((pos, num))
            continue
        if len(run) >= VIGNETTE_MIN_RUN:
            runs.append(run)
        run = [(pos, num)]
    if len(run) >= VIGNETTE_MIN_RUN:
        runs.append(run)
    return runs


def excise_vignettes(text):
    # step 4: cut every vignette run (+ instruction sentence) out of the text
    # go from the end so earlier offsets still work
    for run in reversed(vignette_runs(text)):
        start = run[0][0]
        last = run[-1][0]

        # instruction sentence before the run
        lead = text.rfind(". ", max(0, start - EXERCISE_CUE_LOOKBACK), start)
        if lead != -1 and EXERCISE_CUE.search(text[lead:start]):
            start = lead + 2

        # end at next section heading, or LAST_ITEM_SPAN after the last item
        head = SECTION_HEAD.search(text, last)
        if head and head.start() - last < SECTION_HEAD_MAX:
            end = head.start()
        else:
            end = min(len(text), last + LAST_ITEM_SPAN)
        text = text[:start] + " " + text[end:]
    return re.sub(r"\s+", " ", text).strip()


def chunk_words(text):
    # step 5: CHUNK_WORDS-word windows, moving CHUNK_WORDS - OVERLAP_WORDS each time
    words = text.split()
    step = CHUNK_WORDS - OVERLAP_WORDS
    chunks = []
    for start in range(0, len(words), step):
        piece = words[start:start + CHUNK_WORDS]
        # short tail -> drop it (unless it's the only chunk)
        if len(piece) < MIN_TAIL_WORDS and chunks:
            break
        chunks.append(" ".join(piece))
        if start + CHUNK_WORDS >= len(words):
            break
    return chunks


def urgency_terms(text):
    low = text.lower()
    found = set()
    for t in URGENCY_VOCAB:
        if re.search(rf"\b{re.escape(t)}\b", low):
            found.add(t)
    return sorted(found)


def is_scaffolding(chunk):
    # step 6
    if not SCAFFOLDING.search(chunk):
        return False
    if urgency_terms(chunk):
        return False
    return len(CLINICAL.findall(chunk)) < MIN_CLINICAL_HITS


def build_records(verbose=True):
    pdfs = sorted(GUIDELINES_PATH.glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"No PDFs in {GUIDELINES_PATH}")

    records = []
    for pdf in pdfs:
        doc_id = pdf.stem.lower()
        pages = drop_running_lines(page_texts(pdf))
        body = excise_vignettes(clean(" ".join(pages)))

        chunks = []
        for c in chunk_words(body):
            if not is_scaffolding(c):
                chunks.append(c)

        for i in range(len(chunks)):
            chunk = chunks[i]
            records.append({"doc_id": doc_id, "chunk_id": f"{doc_id}_{i + 1:03d}", "text": chunk,
                            "source": pdf.stem, "section": "unknown", "urgency_terms": urgency_terms(chunk)})
        if verbose:
            print(f"  {pdf.name:<40} {len(pages):>3} pages {len(body.split()):>6} words -> {len(chunks):>3} chunks")
    return records


def serialise(records):
    # one json per line
    text = ""
    for r in records:
        text += json.dumps(r, ensure_ascii=False) + "\n"
    return text.encode("utf-8")


def corpus_content_sha256(data):
    # hash of the records themselves (key order and line endings don't matter)
    recs = []
    for line in data.decode("utf-8").splitlines():
        if line.strip():
            recs.append(json.loads(line))
    canon = json.dumps(recs, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


def main():
    ap = argparse.ArgumentParser(description="Build the guideline retrieval corpus from the PDFs.")
    ap.add_argument("--verify", action="store_true", help="Rebuild in memory and compare; write nothing.")
    ap.add_argument("--against", default=None, help=f"Corpus to compare with (default {CORPUS_FILE}).")
    args = ap.parse_args()

    print(f"pypdf {pypdf.__version__} | {CHUNK_WORDS}-word chunks, {OVERLAP_WORDS}-word overlap | {GUIDELINES_PATH}")
    records = build_records()
    data = serialise(records)

    words = 0
    docs = set()
    for r in records:
        words += len(r["text"].split())
        docs.add(r["doc_id"])
    print(f"{len(records)} chunks, {words} words, {len(docs)} documents")

    if args.verify:
        if args.against:
            target = Path(args.against)
        else:
            target = CORPUS_FILE
        if not target.exists():
            raise SystemExit(f"--verify: {target} does not exist")
        existing = target.read_bytes()
        same_content = corpus_content_sha256(existing) == corpus_content_sha256(data)
        print(f"existing {target}: file {hashlib.sha256(existing).hexdigest()[:16]} "
              f"content {corpus_content_sha256(existing)[:16]}")
        print(f"rebuilt: file {hashlib.sha256(data).hexdigest()[:16]} content {corpus_content_sha256(data)[:16]}")
        if same_content:
            print("MATCH")
            raise SystemExit(0)
        print("DIFFER")
        raise SystemExit(1)

    CORPUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    CORPUS_FILE.write_bytes(data)
    print(f"Wrote {CORPUS_FILE}: file sha256 {hashlib.sha256(data).hexdigest()[:16]}, "
          f"content sha256 {corpus_content_sha256(data)[:16]}")
    print("The FAISS index is now stale: run python src/build_index.py")


if __name__ == "__main__":
    main()
