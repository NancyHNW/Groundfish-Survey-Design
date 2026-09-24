"""Tests for the paired comparison harness.

Solve and evaluator are both stubbed. The claim here is about arithmetic --
that differencing inside a block recovers an effect instance spread would
otherwise bury -- and that is only checkable when the effect is known upfront.
"""

import math

import pytest

from experiments import common
from experiments.common import (EVAL_KEYS, PAIRED_COLUMNS, _t_critical,
                                paired_compare, parse_instances, print_paired)


class StubEvaluator:
    """Scores a solution by a rule the test supplies."""

    def __init__(self, score):
        self.score = score
        self.calls = []

    def evaluate(self, trips, instance, **eval_kw):
        self.calls.append(eval_kw)
        mean = self.score(sum(t["total_time"] for t in trips), eval_kw)
        return {
            "n_simulations": 100,
            "p_capacity_exceedance": 0.5,
            "expected_unscheduled_returns": 2.0,
            "total_time_distribution": {"mean": mean, "std": 1.0,
                                        "p5": mean, "p50": mean, "p95": mean},
        }


@pytest.fixture
def stub_solve(monkeypatch):
    """Fake solve() recording its calls. Planned time tracks the instance."""
    calls = []

    def fake_solve(instance=1, seed=42, capacity_buffer=1.0, method="tabu_move",
                   **kwargs):
        calls.append({"instance": instance, "seed": seed,
                      "capacity_buffer": capacity_buffer, "method": method})
        planned = 100.0 * instance + (1.0 - capacity_buffer) * 50.0
        return {
            "trips": [{"total_time": planned, "boat_id": 0, "nodes": [],
                       "fish_time": 0.0, "catch": 0.0}],
            "instance": object(),
            "planned_time": planned,
            "feasible": True,
        }

    monkeypatch.setattr(common, "solve", fake_solve)
    return calls


STRATEGIES = [
    ("backtrack", {"strategy": "backtrack"}),
    ("forward", {"strategy": "forward"}),
    ("preemptive", {"strategy": "preemptive", "preemptive_threshold": 0.8}),
]


def _flat(planned, eval_kw):
    return planned


def _forward_saves_3h(planned, eval_kw):
    return planned + (-3.0 if eval_kw["strategy"] == "forward" else 0.0)


def _forward_saves_3h_noisy(planned, eval_kw):
    """-3h with +/-1h of block-to-block wobble, so sd is nonzero."""
    if eval_kw["strategy"] != "forward":
        return planned
    return planned - 3.0 + (1.0 if int(planned // 100) % 2 else -1.0)


# ---- the claim the harness rests on ---------------------------------------

def test_recovers_a_constant_effect_exactly(stub_solve):
    """Instances 100h apart, a constant -3h effect: recover -3.0, sd zero."""
    summary, blocks = paired_compare(
        STRATEGIES, "backtrack", instances=range(1, 9),
        evaluator=StubEvaluator(_forward_saves_3h), progress=False)

    forward = next(r for r in summary if r["setting"] == "forward")
    assert forward["n"] == 8
    assert forward["diff"] == pytest.approx(-3.0)
    assert forward["sd"] == pytest.approx(0.0)
    assert forward["ci_lo"] == pytest.approx(-3.0)
    assert forward["ci_hi"] == pytest.approx(-3.0)
    assert forward["significant"] is True

    # 700h of instance spread hiding a 3h effect, so this is not vacuous
    means = [b["mean"] for b in blocks if b["setting"] == "forward"]
    assert max(means) - min(means) > 100 * abs(forward["diff"])


def test_baseline_differences_against_itself_are_zero(stub_solve):
    summary, _ = paired_compare(STRATEGIES, "backtrack", instances=range(1, 6),
                                evaluator=StubEvaluator(_flat), progress=False)
    base = next(r for r in summary if r["setting"] == "backtrack")
    assert base["diff"] == 0.0
    assert base["sd"] == 0.0
    assert base["significant"] is False


def test_effect_that_flips_sign_is_not_significant(stub_solve):
    def score(planned, eval_kw):
        if eval_kw["strategy"] != "forward":
            return planned
        return planned + (50.0 if int(planned // 100) % 2 else -50.0)

    summary, _ = paired_compare(STRATEGIES, "backtrack", instances=range(1, 9),
                                evaluator=StubEvaluator(score), progress=False)
    forward = next(r for r in summary if r["setting"] == "forward")
    assert forward["significant"] is False
    assert forward["ci_lo"] < 0 < forward["ci_hi"]


# ---- solve sharing --------------------------------------------------------

def test_scoring_only_settings_share_one_solve(stub_solve):
    """Three strategies over five instances costs five solves, not fifteen."""
    paired_compare(STRATEGIES, "backtrack", instances=range(1, 6),
                   evaluator=StubEvaluator(_flat), progress=False)
    assert len(stub_solve) == 5


def test_settings_changing_the_plan_get_their_own_solve(stub_solve):
    buffers = [("b1.0", {"capacity_buffer": 1.0}),
               ("b0.8", {"capacity_buffer": 0.8})]
    paired_compare(buffers, "b1.0", instances=range(1, 4),
                   evaluator=StubEvaluator(_flat), progress=False)
    assert len(stub_solve) == 6
    assert {c["capacity_buffer"] for c in stub_solve} == {1.0, 0.8}


def test_every_block_uses_the_same_seed_across_settings(stub_solve):
    """Shared restarts are half of what the pairing cancels."""
    paired_compare(STRATEGIES, "backtrack", instances=[1, 2],
                   solver_seeds=[7, 9], evaluator=StubEvaluator(_flat),
                   progress=False)
    seen = {(c["instance"], c["seed"]) for c in stub_solve}
    assert seen == {(1, 7), (1, 9), (2, 7), (2, 9)}


def test_only_eval_keys_reach_the_evaluator(stub_solve):
    evaluator = StubEvaluator(_flat)
    paired_compare(STRATEGIES, "backtrack", instances=[1],
                   evaluator=evaluator, progress=False)
    for call in evaluator.calls:
        assert set(call) <= set(EVAL_KEYS)
    assert {c.get("strategy") for c in evaluator.calls} == {
        "backtrack", "forward", "preemptive"}


def test_blocks_are_instances_crossed_with_seeds(stub_solve):
    summary, blocks = paired_compare(
        STRATEGIES, "backtrack", instances=[1, 2, 3], solver_seeds=[1, 2],
        evaluator=StubEvaluator(_flat), progress=False)
    assert len(blocks) == 3 * 2 * len(STRATEGIES)
    assert all(r["n"] == 6 for r in summary)


# ---- guards ---------------------------------------------------------------

def test_unknown_baseline_is_rejected(stub_solve):
    with pytest.raises(ValueError, match="baseline"):
        paired_compare(STRATEGIES, "nonesuch", instances=[1],
                       evaluator=StubEvaluator(_flat), progress=False)


@pytest.mark.parametrize("clash", ["instance", "seed"])
def test_instance_and_seed_cannot_come_through_fixed(stub_solve, clash):
    """They drive the blocks, so pinning them would collapse them."""
    with pytest.raises(ValueError, match=clash):
        paired_compare(STRATEGIES, "backtrack", instances=[1],
                       evaluator=StubEvaluator(_flat), progress=False,
                       **{clash: 3})


def test_infeasible_blocks_are_counted(stub_solve, monkeypatch):
    def fake_solve(instance=1, seed=42, **kwargs):
        return {"trips": [{"total_time": 100.0 * instance}],
                "instance": object(), "planned_time": 100.0 * instance,
                "feasible": instance != 2}

    monkeypatch.setattr(common, "solve", fake_solve)
    summary, _ = paired_compare(STRATEGIES, "backtrack", instances=[1, 2, 3],
                                evaluator=StubEvaluator(_flat), progress=False)
    assert all(r["infeasible_blocks"] == 1 for r in summary)


def test_single_block_gives_a_difference_but_no_interval(stub_solve):
    summary, _ = paired_compare(
        STRATEGIES, "backtrack", instances=[1],
        evaluator=StubEvaluator(_forward_saves_3h), progress=False)
    forward = next(r for r in summary if r["setting"] == "forward")
    assert forward["n"] == 1
    assert forward["diff"] == pytest.approx(-3.0)
    assert forward["ci_lo"] == forward["ci_hi"] == pytest.approx(-3.0)
    assert forward["significant"] is False


def test_significant_is_a_plain_bool(stub_solve):
    """A numpy bool leaks into the CSV and fails an `is False` check."""
    summary, _ = paired_compare(STRATEGIES, "backtrack", instances=range(1, 5),
                                evaluator=StubEvaluator(_flat), progress=False)
    assert all(type(r["significant"]) is bool for r in summary)


# ---- statistics -----------------------------------------------------------

def test_interval_uses_t_not_the_normal(stub_solve):
    """At n=8 t is 2.365; 1.96 would overstate significance by a fifth."""
    summary, _ = paired_compare(
        STRATEGIES, "backtrack", instances=range(1, 9),
        evaluator=StubEvaluator(_forward_saves_3h_noisy), progress=False)
    forward = next(r for r in summary if r["setting"] == "forward")
    half = forward["ci_hi"] - forward["diff"]
    assert half == pytest.approx(2.365 * forward["se"], rel=1e-3)
    assert half > 1.96 * forward["se"]


def test_se_is_sd_over_root_n(stub_solve):
    summary, _ = paired_compare(
        STRATEGIES, "backtrack", instances=range(1, 7),
        evaluator=StubEvaluator(_forward_saves_3h_noisy), progress=False)
    forward = next(r for r in summary if r["setting"] == "forward")
    assert forward["se"] == pytest.approx(forward["sd"] / math.sqrt(forward["n"]))


@pytest.mark.parametrize("df,expected", [
    (1, 12.706), (7, 2.365), (8, 2.306), (30, 2.042),
    (35, 2.042),  # between entries, falls back to the wider one
    (40, 2.021), (200, 1.980),
])
def test_t_critical_table(df, expected):
    assert _t_critical(df) == pytest.approx(expected)


def test_t_critical_never_narrower_than_the_normal():
    assert all(_t_critical(df) >= 1.96 for df in range(1, 500))


# ---- instance parsing -----------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("1-5", [1, 2, 3, 4, 5]),
    ("1,4,7", [1, 4, 7]),
    ("1-3,9", [1, 2, 3, 9]),
    ("7", [7]),
    (" 2 , 3 ", [2, 3]),
    ("1-30", list(range(1, 31))),
])
def test_parse_instances(text, expected):
    assert parse_instances(text) == expected


# ---- reporting ------------------------------------------------------------

def test_verdict_separates_better_from_worse(stub_solve, capsys):
    """A setting can separate by being reliably worse, which is not a win."""
    def score(planned, eval_kw):
        return planned + {"backtrack": 0.0, "forward": -3.0,
                          "preemptive": +9.0}[eval_kw["strategy"]]

    summary, _ = paired_compare(STRATEGIES, "backtrack", instances=range(1, 9),
                                evaluator=StubEvaluator(score), progress=False)
    print_paired(summary, "backtrack")

    out = capsys.readouterr().out
    better = out.index("Beats backtrack:")
    worse = out.index("Reliably worse than backtrack:")
    assert out.index("forward", better) < worse
    assert "preemptive" in out[worse:]


def test_paired_columns_show_n_and_the_interval():
    keyed = {c[0] for c in PAIRED_COLUMNS}
    assert {"setting", "n", "diff", "ci_lo", "ci_hi", "significant"} <= keyed
