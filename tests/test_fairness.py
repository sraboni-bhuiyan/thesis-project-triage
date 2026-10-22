import pytest

from fairness_metrics import (AGE_PAIRS, GENDER_PAIRS, GROUPS, age_group, counterfactual_consistency, dpr, eod,
                              flip_rate, group_of, paired_high, paired_high_pooled, run_disagreement)
from metrics import correct_by_case
from stat_tests import fisher_rates, mcnemar_exact


def variant_preds(table):
    """{base: {group: urgency}} -> prediction rows."""
    return [{"case_id": b, "variant_id": f"{b}_{g}", "pred_urgency": u, "pred_specialty": "Cardiology"}
            for b, groups in table.items() for g, u in groups.items()]


def variant_truth(table, truth="orange"):
    return {f"{b}_{g}": {"case_id": b, "variant_id": f"{b}_{g}", "age_group": g.split("_")[0],
                         "gender": g.split("_")[1].title(), "age": "30", "ground_truth_urgency": truth}
            for b, groups in table.items() for g in groups}


ALL_SAME = {g: "orange" for g in GROUPS}
ONE_OFF = {"young_male": "orange", "young_female": "yellow", "old_male": "orange", "old_female": "orange"}


def test_age_group_boundaries():
    assert age_group(39) == "young" and age_group(40) is None
    assert age_group(64) is None and age_group(65) == "old"


def test_group_of_real_and_variant():
    assert group_of({"age": "70", "gender": "Female"}) == "old_female"
    assert group_of({"age": "50", "gender": "Male"}) is None
    assert group_of({"age_group": "young", "age": "30", "gender": "Male"}) == "young_male"


def test_dpr():
    t = {"a": ALL_SAME, "b": ONE_OFF}
    d = dpr(variant_truth(t), variant_preds(t))
    assert d["high_rate_by_group"]["young_female"] == 0.5
    assert d["high_rate_by_group"]["young_male"] == 1.0
    assert d["dpr"] == 0.5 and d["n_by_group"]["old_male"] == 2


def test_eod_tpr_and_fpr():
    t = {"a": ALL_SAME, "b": ONE_OFF}
    truth = variant_truth({"a": ALL_SAME}, "orange") | variant_truth({"b": ONE_OFF}, "yellow")
    e = eod(truth, variant_preds(t))
    assert e["tpr_gap"] == 0.0                     # all positives (base a) predicted high
    assert e["fpr_gap"] == 1.0                     # base b negatives: young_female not high, others high
    assert e["eod"] == 1.0


def test_error_rows_excluded():
    t = {"a": ALL_SAME}
    preds = variant_preds(t)
    preds[0]["pred_urgency"] = "ERROR"
    assert dpr(variant_truth(t), preds)["n_by_group"]["young_male"] == 0
    assert counterfactual_consistency(preds)["n_base_cases"] == 0   # incomplete base case dropped


def test_consistency_and_flips():
    preds = variant_preds({"a": ALL_SAME, "b": ONE_OFF})
    c = counterfactual_consistency(preds)
    assert c["consistency"] == 0.5 and c["n_base_cases"] == 2
    g, a = flip_rate(preds, GENDER_PAIRS), flip_rate(preds, AGE_PAIRS)
    assert (g["flips"], g["n_pairs"]) == (1, 4)   # young_male vs young_female in b
    assert (a["flips"], a["n_pairs"]) == (1, 4)   # young_female vs old_female in b


def test_paired_high():
    preds = variant_preds({"a": ALL_SAME, "b": ONE_OFF})
    assert paired_high(preds, "young_male", "young_female") == (1, 0, 2)
    assert paired_high_pooled(preds, GENDER_PAIRS) == (1, 0, 4)


def test_run_disagreement():
    a = [{"case_id": "1", "pred_urgency": "red", "pred_specialty": "Cardiology"},
         {"case_id": "2", "pred_urgency": "yellow", "pred_specialty": "Neurology"}]
    b = [{"case_id": "1", "pred_urgency": "Red", "pred_specialty": "Cardiology"},
         {"case_id": "2", "pred_urgency": "orange", "pred_specialty": "Neurology"}]
    assert run_disagreement(a, b)["disagree"] == 1
    assert run_disagreement(a, b, "pred_specialty")["agreement"] == 1.0


def test_correct_by_case_mapped():
    cases = {"1": {"case_id": "1", "ground_truth_urgency": "red", "ground_truth_specialty": "Hepatology",
                   "ground_truth_specialty_all": "Hepatology"}}
    preds = [{"case_id": "1", "pred_urgency": "red", "pred_specialty": "Gastroenterology"}]
    assert correct_by_case(cases, preds, "urgency") == {"1": True}
    assert correct_by_case(cases, preds, "specialty") == {"1": False}
    assert correct_by_case(cases, preds, "specialty", mapped=True) == {"1": True}


def test_mcnemar_exact():
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(5, 5) == 1.0
    assert mcnemar_exact(10, 0) == pytest.approx(2 * 0.5 ** 10)


def test_fisher_rates():
    r = fisher_rates(5, 10, 5, 10)
    assert r["p_value"] == 1.0 and r["rate_1"] == 0.5
