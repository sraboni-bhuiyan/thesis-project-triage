"""
Tests for the pure functions every metric depends on: label constants, urgency and specialty
normalisation, output parsing, the case block, and the resumable runner (with a fake model).
No API calls.
"""
import csv

import pytest

import config
from common import (ModelReply, PRED_FIELDS, build_case_block, extract_json_object, normalize_specialty,
                    normalize_urgency, parse_model_output, read_csv, row_key, run_predictions)
from config import ERROR, INVALID


# ---------------------------------------------------------------- label constants
def test_urgency_map_covers_the_scale():
    assert set(config.URGENCY_MAP.values()) == set(config.URGENCY_ORDER)
    assert config.URGENCY_MAP[1] == "red" and config.URGENCY_MAP[5] == "blue"


def test_urgency_order_runs_least_to_most_severe():
    assert config.URGENCY_ORDER == ["blue", "green", "yellow", "orange", "red"]


def test_high_acuity_is_esi_1_and_2():
    assert config.HIGH_ACUITY == {config.URGENCY_MAP[1], config.URGENCY_MAP[2]}


def test_synonym_targets_are_valid_colours():
    assert {colour for _, colour in config.URGENCY_SYNONYMS} <= set(config.URGENCY_ORDER)


def test_specialty_synonyms_point_at_listed_specialties():
    listed = {s.lower() for s in config.SPECIALTIES}
    assert set(config.SPECIALTY_SYNONYMS) <= listed


def test_specialty_parent_map_is_consistent():
    listed = {s.lower() for s in config.SPECIALTIES}
    synonyms = {a for aliases in config.SPECIALTY_SYNONYMS.values() for a in aliases}
    assert len(config.SPECIALTY_PARENT) == 13
    for label, parents in config.SPECIALTY_PARENT.items():
        assert label not in listed and label not in synonyms   # only unreachable labels are mapped
        assert parents and set(parents) <= listed               # every target is a listed specialty
        assert normalize_specialty(label) == label              # primary analysis keeps it distinct


def test_variant_id_format_parts_are_unambiguous():
    # variant_id = {case_id}_{young|old}_{male|female}; parts must not contain "_".
    assert all("_" not in p for p in config.AGE_GROUPS + config.GENDERS)


# ---------------------------------------------------------------- urgency normalisation
@pytest.mark.parametrize("raw, expected", [
    ("orange", "orange"), ("RED", "red"), ("  Yellow ", "yellow"), ("Green.", "green"),
    ("orange (very urgent)", "orange"),
    ("1", "red"), ("5", "blue"), (3, "yellow"),
    ("ESI 2", "orange"), ("level 4", "green"), ("CTAS level 1", "red"), ("triage 3", "yellow"),
    ("non-urgent", "blue"), ("Non urgent", "blue"), ("not urgent", "blue"),
    ("less urgent", "green"), ("semi-urgent", "green"), ("standard", "green"),
    ("very urgent", "orange"), ("emergent", "orange"),
    ("immediate", "red"), ("resuscitation", "red"),
    ("urgent", "yellow"),
])
def test_normalize_urgency_valid(raw, expected):
    assert normalize_urgency(raw) == expected


@pytest.mark.parametrize("raw", [
    "", None, "purple", "orange or red", "ESI 2 or 3", "emergency", "6", "level 7", "high",
])
def test_normalize_urgency_invalid(raw):
    assert normalize_urgency(raw) == INVALID


def test_non_urgent_is_checked_before_urgent():
    # Regression guard for the synonym order in config.URGENCY_SYNONYMS.
    assert normalize_urgency("non-urgent") == "blue"
    assert normalize_urgency("less urgent") == "green"
    assert normalize_urgency("very urgent") == "orange"


def test_error_passes_through_urgency():
    assert normalize_urgency(ERROR) == ERROR


# ---------------------------------------------------------------- specialty normalisation
def test_every_listed_specialty_maps_to_itself():
    for name in config.SPECIALTIES:
        assert normalize_specialty(name) == name.lower()


@pytest.mark.parametrize("raw, expected", [
    ("Orthopedic Surgery", "orthopedics"), ("orthopaedics", "orthopedics"),
    ("Obstetrics and Gynecology", "gynecology"), ("ENT", "otolaryngology"),
    ("Ear, Nose and Throat", "otolaryngology"),
    ("Cardiology or Neurology", "cardiology"), ("Cardiology OR Neurology", "cardiology"),
    ("Cardiology; Neurology", "cardiology"), ("Cardiology / Neurology", "cardiology"),
    ("['Gastroenterology', 'Cardiology']", "gastroenterology"),
    ("General Surgery (acute abdomen)", "general surgery"),
    ("  Neurosurgery. ", "neurosurgery"),
    ("Oral and Maxillofacial Surgery", "oral and maxillofacial surgery"),
])
def test_normalize_specialty(raw, expected):
    assert normalize_specialty(raw) == expected


def test_unknown_specialty_is_kept_not_guessed():
    # Sub-specialties outside the list stay distinct (see config.SPECIALTY_SYNONYMS).
    assert normalize_specialty("Rheumatology") == "rheumatology"
    assert normalize_specialty("Hepatology") == "hepatology"


def test_specialty_edge_values():
    assert normalize_specialty("") == INVALID
    assert normalize_specialty(None) == INVALID
    assert normalize_specialty(ERROR) == ERROR


# ---------------------------------------------------------------- output parsing
def test_parse_plain_json():
    out = '{"urgency": "orange", "specialty": "Cardiology", "rationale": "Chest pain."}'
    assert parse_model_output(out) == ("orange", "Cardiology", "Chest pain.")


def test_parse_strips_think_block_and_fences():
    out = '<think>maybe {"urgency": "red"}</think>\n```json\n{"urgency": "Yellow", "specialty": "Urology"}\n```'
    assert parse_model_output(out) == ("yellow", "Urology", "")


def test_parse_unclosed_think_block_is_invalid():
    assert parse_model_output('<think>still reasoning {"urgency": "red"}')[0] == INVALID


def test_parse_json_with_surrounding_text():
    out = 'Here is my answer: {"urgency": "green", "specialty": "Dermatology", "rationale": "Rash {mild}."} Done.'
    assert parse_model_output(out) == ("green", "Dermatology", "Rash {mild}.")


@pytest.mark.parametrize("out", ["", "orange", "{not json}", "[1, 2]"])
def test_parse_garbage_is_invalid(out):
    assert parse_model_output(out)[:2] == (INVALID, INVALID)


def test_parse_missing_specialty_is_invalid_specialty_only():
    assert parse_model_output('{"urgency": "red"}')[:2] == ("red", INVALID)


def test_extract_json_object_skips_non_objects():
    assert extract_json_object('[{"a": 1}]') == {"a": 1}


# ---------------------------------------------------------------- case block
CASE = {"case_id": "1", "symptoms": "Pt is a 57 y/o M\nwith flank pain.", "vitals": "HR 102",
        "age": "57", "gender": "Male"}


def test_case_block_matches_september_layout():
    case = {**CASE, "symptoms": "  Pt is a 57 y/o M\nwith flank pain.\n"}
    assert build_case_block(case) == ("Patient case:\n- Symptoms: Pt is a 57 y/o M\nwith flank pain.\n"
                                      "- Vitals: HR 102\n- Age: 57\n- Gender: Male")


def test_content_hash_ignores_line_endings_but_not_content(tmp_path):
    from common import content_sha256
    rows = [["case_id", "symptoms"], ["1", " pain\nin chest "], ["2", "fever"]]
    lf, crlf, other = tmp_path / "lf.csv", tmp_path / "crlf.csv", tmp_path / "other.csv"
    for path, term in ((lf, "\n"), (crlf, "\r\n")):
        with open(path, "w", encoding="utf-8", newline="") as f:
            csv.writer(f, lineterminator=term).writerows(
                [[v.replace("\n", term) for v in r] for r in rows])
    with open(other, "w", encoding="utf-8", newline="") as f:
        csv.writer(f, lineterminator="\n").writerows([rows[0], ["1", "pain\nin chest"], rows[2]])
    assert content_sha256(lf) == content_sha256(crlf)
    assert content_sha256(lf) != content_sha256(other)          # whitespace is content


def test_row_key_prefers_variant_id():
    assert row_key({"case_id": "1", "variant_id": "1_young_male"}) == "1_young_male"
    assert row_key({"case_id": "1", "variant_id": ""}) == "1"


# ---------------------------------------------------------------- metrics
def _truth(cid, urgency, spec, spec_all=None):
    return {"case_id": cid, "ground_truth_urgency": urgency, "ground_truth_specialty": spec,
            "ground_truth_specialty_all": spec_all or spec}


def _pred(cid, urgency, spec):
    return {"case_id": cid, "variant_id": "", "pred_urgency": urgency, "pred_specialty": spec}


TRUTH = {t["case_id"]: t for t in [
    _truth("1", "yellow", "Cardiology"),
    _truth("2", "orange", "Orthopedic Surgery", "Orthopedic Surgery|Neurosurgery"),
    _truth("3", "red", "Hepatology", "Hepatology|Gastroenterology"),
    _truth("4", "yellow", "Urology"),
    _truth("5", "green", "Dermatology"),
]}
PREDS = [
    _pred("1", "orange", "Cardiology"),          # over-triage, specialty right
    _pred("2", "orange", "Neurosurgery"),        # correct; specialty only in the list
    _pred("3", "orange", "Gastroenterology"),    # under-triage; strict wrong, any-listed right, mapped right
    _pred("4", INVALID, INVALID),                # unparseable: counts as wrong
    _pred("5", ERROR, ERROR),                    # infrastructure failure: excluded
    _pred("9", "red", "Urology"),                # not in ground truth: ignored
]


def test_evaluate_counts_and_denominators():
    from metrics import evaluate
    m = evaluate(TRUTH, PREDS)
    assert (m["n_cases"], m["n_error"], m["n_answered"], m["n_invalid_urgency"]) == (5, 1, 4, 1)
    assert m["urgency_exact_acc"] == 1 / 4
    assert m["over_triage_rate"] == 1 / 4 and m["under_triage_rate"] == 1 / 4
    assert m["urgency_adjacent_acc"] == 3 / 4          # INVALID is never adjacent


def test_evaluate_balanced_accuracy_and_red_recall():
    from metrics import evaluate
    m = evaluate(TRUTH, PREDS)
    # Classes present among answered rows: yellow (0/2), orange (1/1), red (0/1). Green row was ERROR.
    assert m["urgency_recall_by_class"] == {"yellow": 0.0, "orange": 1.0, "red": 0.0}
    assert m["urgency_balanced_acc"] == 1 / 3
    assert (m["red_recall"], m["n_red"]) == (0.0, 1)


def test_evaluate_specialty_strict_any_mapped_and_ceiling():
    from metrics import evaluate
    m = evaluate(TRUTH, PREDS)
    assert m["specialty_acc"] == 1 / 4                 # only case 1
    assert m["specialty_acc_any_listed"] == 3 / 4      # cases 1, 2, 3
    assert m["specialty_acc_mapped"] == 2 / 4          # + case 3 (hepatology -> gastroenterology)
    assert m["specialty_ceiling"] == 3 / 4             # hepatology is not on the list
    assert m["n_invalid_specialty"] == 1


def test_evaluate_all_errors_gives_none_rates():
    from metrics import evaluate
    m = evaluate(TRUTH, [_pred("1", ERROR, ERROR)])
    assert m["n_answered"] == 0 and m["urgency_exact_acc"] is None


def test_c2_and_c4_prompts_differ_only_inside_guidelines():
    from common import load_prompt
    from config import CHUNK_SEPARATOR
    template = load_prompt("rag.txt")
    block = build_case_block(CASE)
    c2 = template.format(retrieved_context=CHUNK_SEPARATOR.join(["a", "b", "c"]), case_block=block)
    c4 = template.format(retrieved_context=CHUNK_SEPARATOR.join(["x", "y", "z"]), case_block=block)
    outside = lambda p: p.split("<guidelines>")[0] + p.split("</guidelines>")[1]
    assert c2 != c4 and outside(c2) == outside(c4)


def test_fixed_context_picks_most_retrieved_with_id_tiebreak():
    from collections import Counter
    from pick_fixed_context import MAIN_LOGS, top_chunks
    counts = Counter({"b_002": 5, "a_001": 5, "c_003": 9, "d_004": 1})
    assert top_chunks(counts, 3) == ["c_003", "a_001", "b_002"]
    assert not any("_fixed" in name or "_rep" in name for name in MAIN_LOGS)
    assert len(MAIN_LOGS) == 4


# ---------------------------------------------------------------- resumable runner (fake model)
CASES = [{"case_id": str(i), "symptoms": "x", "vitals": "y", "age": "50", "gender": "Female"} for i in range(4)]


def _reply(text):
    return ModelReply(text=text, model="fake/model", json_mode=True)


def test_runner_writes_ok_invalid_and_error_rows(tmp_path):
    out = tmp_path / "preds.csv"

    def predict(case):
        if case["case_id"] == "1":
            return _reply("no json here")
        if case["case_id"] == "2":
            raise RuntimeError("network down")
        return _reply('{"urgency": "orange", "specialty": "Cardiology", "rationale": "r"}')

    counts = run_predictions(CASES, out, predict, desc="test")
    rows = {r["case_id"]: r for r in read_csv(out)}
    assert counts == {"ok": 2, INVALID: 1, ERROR: 1}
    assert list(read_csv(out)[0].keys()) == PRED_FIELDS
    assert rows["1"]["pred_urgency"] == INVALID and rows["1"]["raw_model_output"] == "no json here"
    assert rows["2"]["pred_urgency"] == ERROR and "network down" in rows["2"]["pred_rationale"]
    assert rows["0"]["response_model"] == "fake/model"


def test_runner_resume_retries_only_error_rows(tmp_path):
    out = tmp_path / "preds.csv"
    calls = []

    def flaky(case):
        calls.append(case["case_id"])
        if case["case_id"] == "2":
            raise RuntimeError("timeout")
        return _reply('{"urgency": "red", "specialty": "Neurology"}')

    run_predictions(CASES, out, flaky, desc="first")
    calls.clear()

    def fixed(case):
        calls.append(case["case_id"])
        return _reply('{"urgency": "yellow", "specialty": "Neurology"}')

    run_predictions(CASES, out, fixed, desc="resume")
    rows = read_csv(out)
    assert calls == ["2"]                                   # only the ERROR row was re-run
    assert len(rows) == 4 and not any(r["pred_urgency"] == ERROR for r in rows)
    assert {r["case_id"]: r["pred_urgency"] for r in rows}["0"] == "red"   # finished rows untouched


def test_runner_rejects_duplicate_keys(tmp_path):
    with pytest.raises(ValueError):
        run_predictions(CASES + CASES[:1], tmp_path / "p.csv", lambda c: _reply("{}"), desc="dup")


def test_runner_refuses_file_with_other_columns(tmp_path):
    out = tmp_path / "preds.csv"
    with open(out, "w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerows([["case_id", "pred_urgency"], ["0", "red"]])
    with pytest.raises(SystemExit):
        run_predictions(CASES, out, lambda c: _reply("{}"), desc="cols")
