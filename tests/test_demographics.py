"""Tests for the demographic swap (fairness variants), masking (C3) and the sex-specific audit."""
import re

import pytest

from common import build_case_block
from demographics import mask_text, sex_specific_mentions, swap_text

FEMALE_HPI = ("Ms. ___ is a ___ year old woman with HTN. She reports that her pain began after she "
              "fell; her husband drove her to the ED. ___ yo F, denies fever. Temp 101 F.")
MALE_HPI = "Mr. ___ is a ___ yo M with HE and MS. He says his chest hurts; the pain woke him. 45 y/o male."

# Pronouns and sex words in any case; titles case-sensitive, so "MS" (multiple sclerosis) is not a title.
FEMALE_CUES = re.compile(r"(?i:\b(she|her|hers|herself|woman|female|lady)\b)|\bMs\.|\bMrs\.|yo F\b")
MALE_CUES = re.compile(r"\b(he|him|his|himself|man|male|gentleman)\b|\bMr\.|yo M\b")


# ---------------------------------------------------------------- swap
def test_swap_to_male_removes_female_cues():
    out = swap_text(FEMALE_HPI, "male", 70)
    assert not FEMALE_CUES.search(out.replace("Temp 101 F", ""))
    assert out.startswith("Mr. ___ is a ___ year old man with HTN. He reports that his pain")
    assert "drove him to the ED" in out
    assert "___ yo M" in out


def test_swap_to_female_removes_male_cues():
    out = swap_text(MALE_HPI, "female", 30)
    assert not MALE_CUES.search(out)
    assert "She says her chest hurts; the pain woke her." in out
    assert "30 y/o female" in out


def test_swap_leaves_relationships_abbreviations_and_temperatures():
    out = swap_text(FEMALE_HPI, "male", 70)
    assert "husband" in out                       # describes someone else
    assert "Temp 101 F" in out                    # a temperature, not a sex letter
    assert "HE and MS" in swap_text(MALE_HPI, "female", 30)   # abbreviations untouched


def test_swap_is_idempotent_for_same_target():
    once = swap_text(FEMALE_HPI, "male", 70)
    assert swap_text(once, "male", 70) == once


def test_swap_changes_explicit_ages_but_not_durations():
    out = swap_text("A 45-year-old with a 5 year history of DM, aged 45, 45yo.", "male", 70)
    assert out == "A 70-year-old with a 5 year history of DM, aged 70, 70yo."


def test_male_and_female_swaps_differ_only_in_gender_cues():
    m, f = swap_text(FEMALE_HPI, "male", 30), swap_text(FEMALE_HPI, "female", 30)
    assert len(m.split()) == len(f.split())


def test_swap_rejects_unknown_gender():
    with pytest.raises(ValueError):
        swap_text("text", "other", 30)


# ---------------------------------------------------------------- mask
@pytest.mark.parametrize("text", [FEMALE_HPI, MALE_HPI])
def test_mask_removes_all_gender_cues(text):
    out = mask_text(text)
    assert not FEMALE_CUES.search(out.replace("Temp 101 F", ""))
    assert not MALE_CUES.search(out)
    assert "HE and MS" in mask_text(MALE_HPI)


def test_mask_makes_swapped_variants_identical():
    # The point of C3: all four variants of a case give the same masked text.
    variants = {swap_text(FEMALE_HPI, g, a) for g in ("male", "female") for a in (30, 70)}
    assert len({mask_text(v) for v in variants}) == 1


def test_her_before_line_break_is_still_possessive():
    assert swap_text("pain in her\nback", "male", 30) == "pain in his\nback"
    assert swap_text("pain in her back", "male", 30) == "pain in his back"


# Real phrases from the 16 variant texts that differed from September (2026-10-14 check).
@pytest.mark.parametrize("female, male", [
    ("helped her up and walked her home.", "helped him up and walked him home."),
    ("she got in a car with her SO and then", "he got in a car with his SO and then"),
    ("She was dropped off at her home at 2pm", "He was dropped off at his home at 2pm"),
    ("a new skin tear on\nher\nanterior shin", "a new skin tear on\nhis\nanterior shin"),
    ("risperidal in her\nsoup every day", "risperidal in his\nsoup every day"),
    ("she began shaking in her\nupper\nextremities", "he began shaking in his\nupper\nextremities"),
    ("pain that radiates to her\nback, identical", "pain that radiates to his\nback, identical"),
    ("on the back of her\nscalp/occiput", "on the back of his\nscalp/occiput"),
    ("continues her home medications", "continues his home medications"),
    ("brought her back to the ED", "brought him back to the ED"),
])
def test_her_resolution_on_real_phrases(female, male):
    assert swap_text(female, "male", 30) == male


def test_mask_removes_explicit_ages():
    # September spacing kept: the removed sex letter leaves its space ("y/o , aged").
    assert mask_text("A 45 yo F, aged 45.") == "A ___ y/o , aged ___."
    assert mask_text("45M with chest pain") == "___ with chest pain"


def test_mask_uses_september_wording():
    assert mask_text("She says her pain began; it woke her.") == "The patient says their pain began; it woke them."
    assert mask_text("Ms. ___ is a ___ year old woman") == "Pt ___ is a ___ year old person"
    assert mask_text("He hurt himself.") == "The patient hurt themself."


def test_mask_keeps_whitespace():
    assert mask_text("HTN, Afib,  presenting") == "HTN, Afib,  presenting"


def test_abbreviations_are_not_gender_cues():
    text = "hx of AR/MR/TR, HE and MS. An MR was ordered. Did not miss a dose."
    assert mask_text(text) == text
    assert swap_text(text, "female", 30) == text


def test_all_caps_he_at_sentence_start_is_the_pronoun():
    # Real case (2026-10-14): "No seizure like activity.  HE reports that he tries ..."
    text = "No seizures.  HE reports that he drinks water. Hx of HE."
    assert mask_text(text) == "No seizures.  THE PATIENT reports that the patient drinks water. Hx of HE."
    assert swap_text(text, "female", 30) == "No seizures.  SHE reports that she drinks water. Hx of HE."


def test_masked_case_block_has_no_demographic_fields():
    block = build_case_block({"symptoms": FEMALE_HPI, "vitals": "HR 90", "age": "64", "gender": "Female"}, mask=True)
    assert "Age:" not in block and "Gender:" not in block and "64" not in block
    assert not FEMALE_CUES.search(block.replace("Temp 101 F", ""))


# ---------------------------------------------------------------- sex-specific audit
def test_sex_specific_affirmed():
    assert sex_specific_mentions("___ yo G4P4 with ovarian cyst")["affirmed"] == ["g4p4", "ovarian"]
    assert sex_specific_mentions("PSA elevated, prostate enlarged")["affirmed"] == ["prostate"]
    assert sex_specific_mentions("bleeding from rectum and vagina")["affirmed"] == ["vagina"]


def test_sex_specific_negated_is_separate():
    m = sex_specific_mentions("Abd pain. No vaginal bleeding or discharge. She reports chills.")
    assert m == {"affirmed": [], "negated": ["vaginal"]}
    m = sex_specific_mentions("She denies fever, melena, dysuria, vaginal\ndischarge. Pain worse.")
    assert m == {"affirmed": [], "negated": ["vaginal"]}


def test_fertility_treatment_is_sex_specific():
    # Real case (2026-10-14): only "vaginal" was matched, and only negated; the IVF treatment counts.
    m = sex_specific_mentions("Received Lupron/Gonal F for stimulation protocol.\n\nNo vaginal bleeding.")
    assert m == {"affirmed": ["gonal f", "stimulation protocol"], "negated": ["vaginal"]}


def test_sex_specific_affirmed_wins_over_negated():
    m = sex_specific_mentions("Vaginal bleeding for 3 days. No vaginal discharge.")
    assert m == {"affirmed": ["vaginal"], "negated": []}


def test_sex_specific_excludes_breast_and_cervical_spine():
    assert sex_specific_mentions("hx breast cancer, cervical spine fracture") == {"affirmed": [], "negated": []}
