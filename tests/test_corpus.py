"""Tests for the corpus cleaning and chunking steps (build_corpus.py). Pure functions, no PDFs."""
import json

import build_corpus as bc
from config import CHUNK_WORDS, MIN_TAIL_WORDS, OVERLAP_WORDS


def words(n, start=0):
    return " ".join(f"w{i}" for i in range(start, start + n))


# ---------------------------------------------------------------- chunking
def test_chunks_have_fixed_size_and_overlap():
    chunks = bc.chunk_words(words(600))
    step = CHUNK_WORDS - OVERLAP_WORDS
    assert chunks[0] == words(CHUNK_WORDS)
    assert chunks[1].split()[0] == f"w{step}"                      # 50-word overlap
    assert all(len(c.split()) == CHUNK_WORDS for c in chunks[:-1])


def test_short_tail_is_dropped_but_short_document_is_kept():
    # 450 words: windows at 0 and 200 cover 0-449, so no tail chunk is needed.
    assert len(bc.chunk_words(words(450))) == 2
    # A document shorter than MIN_TAIL_WORDS is still one chunk.
    assert bc.chunk_words(words(MIN_TAIL_WORDS - 1)) == [words(MIN_TAIL_WORDS - 1)]
    assert bc.chunk_words("") == []


def test_tail_of_at_least_min_words_is_kept():
    n = CHUNK_WORDS + (CHUNK_WORDS - OVERLAP_WORDS) + MIN_TAIL_WORDS - OVERLAP_WORDS + 10
    chunks = bc.chunk_words(words(n))
    assert len(chunks[-1].split()) >= MIN_TAIL_WORDS


# ---------------------------------------------------------------- cleaning
def test_clean_fixes_artifacts_hyphenation_and_boilerplate():
    raw = "ENA� triage – assess-\nment of “pain” level. Skip navigation  end"
    assert bc.clean(raw) == 'ENA triage - assessment of "pain" level. end'


def test_running_heads_are_dropped_only_with_three_or_more_pages():
    pages = ["Header\nalpha\nFooter 1", "Header\nbeta", "Header\ngamma"]
    assert bc.drop_running_lines(pages) == ["alpha\nFooter 1", "beta", "gamma"]
    assert bc.drop_running_lines(pages[:2]) == pages[:2]


# ---------------------------------------------------------------- vignette excision
ITEMS = ("1. A 45-year-old with chest pain. 2. An infant with fever. 3. The patient fell. "
         "4. A man with a cut. 5. Patient with headache. 4.2 Chest Pain Real guidance here.")


def test_vignette_run_and_colon_instruction_are_removed():
    text = "Chapter text before. For the following patients, identify the level: " + ITEMS
    assert bc.excise_vignettes(text) == "Chapter text before. 4.2 Chest Pain Real guidance here."


def test_instruction_ending_in_a_period_is_kept():
    # Frozen behaviour (matches the September corpus): the look-back stops at the instruction's own
    # final period, so only the numbered run is removed.
    text = "Chapter text before. For the following patients, identify the level. " + ITEMS
    assert bc.excise_vignettes(text) == ("Chapter text before. For the following patients, identify the level. "
                                         "4.2 Chest Pain Real guidance here.")


def test_short_numbered_lists_are_kept():
    text = "GCS: 1. A score of 3. 2. The best response. 3. A pain stimulus."
    assert bc.excise_vignettes(text) == text


# ---------------------------------------------------------------- scaffolding filter and metadata
def test_scaffolding_dropped_only_without_clinical_content():
    assert bc.is_scaffolding("Learning objectives for this module and the small group workshop.")
    assert not bc.is_scaffolding("Learning objectives: assess chest pain, airway, and pulse in triage.")
    assert not bc.is_scaffolding("Course overview: CTAS level 2 is emergent.")      # acuity term keeps it


def test_urgency_terms_are_whole_words():
    assert bc.urgency_terms("Level 2 emergent, not urgent; reddish") == ["emergent", "level 2", "urgent"]


def test_content_hash_ignores_key_order_and_line_endings():
    recs = [{"chunk_id": "a_001", "text": "x"}, {"chunk_id": "a_002", "text": "y"}]
    lf = bc.serialise(recs)
    crlf = "".join(json.dumps({"text": r["text"], "chunk_id": r["chunk_id"]}) + "\r\n" for r in recs).encode()
    assert bc.corpus_content_sha256(lf) == bc.corpus_content_sha256(crlf)
