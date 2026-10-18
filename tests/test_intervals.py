"""Tests for the intervals the report quotes.

Every headline number in the method comparison carries an error bar, and the
whole point of the exercise is that those bars are trustworthy. Two of them are
easy to get subtly wrong:

  * P(exceed) saturates -- it sits at 98-99% on the full problem, which is the
    exact regime where the normal approximation goes past 1.0 and where it
    collapses to zero width at p = 1;
  * the mean time penalty is a second mean over the same scenarios, so it needs
    the same sqrt(n-1) treatment as the mean realised time and not a fresh one.
"""

import math

import pytest

from experiments.common import (CORE_COLUMNS, INTERVAL_COLUMNS, _mc_se,
                                _wilson_ci, summarise)


def _result(n=1000, p_exceed=0.5, penalty_mean=20.0, penalty_std=10.0,
            returns=(2.0, 2.0), mean=400.0, std=13.0):
    """A minimal evaluator result, shaped like StochasticEvaluator.evaluate."""
    return {
        "n_simulations": n,
        "p_capacity_exceedance": p_exceed,
        "expected_unscheduled_returns": sum(returns) / len(returns),
        "total_time_distribution": {"mean": mean, "std": std,
                                    "p5": mean - 20, "p50": mean,
                                    "p95": mean + 20},
        "time_penalty_distribution": {"mean": penalty_mean, "std": penalty_std,
                                      "p5": 0.0, "p50": penalty_mean,
                                      "p95": penalty_mean * 2},
        "all_results": [{"n_unscheduled_returns": r} for r in returns],
    }


# ---- Wilson interval on a proportion --------------------------------------

@pytest.mark.parametrize("k,n", [(0, 100), (1, 100), (50, 100), (99, 100),
                                 (100, 100), (990, 1000), (1000, 1000)])
def test_wilson_stays_inside_the_unit_interval(k, n):
    lo, hi = _wilson_ci(k, n)
    assert 0.0 <= lo <= hi <= 1.0


@pytest.mark.parametrize("k,n", [(100, 100), (0, 100), (1000, 1000)])
def test_wilson_has_width_at_the_boundaries(k, n):
    """The normal approximation gives zero width at p=0 and p=1. That is the bug."""
    lo, hi = _wilson_ci(k, n)
    assert hi - lo > 0.0


def test_wilson_brackets_a_saturated_proportion_below_one():
    """At 99% of 1000 the upper end must not claim certainty."""
    lo, hi = _wilson_ci(990, 1000)
    assert lo < 0.99 < hi < 1.0


def test_wilson_narrows_with_more_scenarios():
    widths = [_wilson_ci(n // 2, n)[1] - _wilson_ci(n // 2, n)[0]
              for n in (100, 1000, 10000)]
    assert widths[0] > widths[1] > widths[2]


def test_wilson_matches_the_published_value():
    """p = 0.5, n = 100, z = 1.96 -> [0.4038, 0.5962].

    Mid-range is where the two agree -- the normal approximation gives
    [0.4020, 0.5980] here. The reason for preferring Wilson is the tails, which
    the boundary tests above cover.
    """
    lo, hi = _wilson_ci(50, 100)
    assert lo == pytest.approx(0.4038, abs=1e-4)
    assert hi == pytest.approx(0.5962, abs=1e-4)


def test_wilson_is_symmetric_about_a_half():
    lo, hi = _wilson_ci(50, 100)
    assert lo == pytest.approx(1.0 - hi)


def test_wilson_on_no_scenarios_is_not_a_division_error():
    assert _wilson_ci(0, 0) == (0.0, 0.0)


# ---- the penalty, and its interval ----------------------------------------

def test_penalty_is_carried_through_from_the_evaluator():
    """It was computed all along and thrown away before the row was built."""
    row = summarise(_result(penalty_mean=21.4), planned=375.3, n_trips=10)
    assert row["penalty"] == pytest.approx(21.4)


def test_penalty_interval_uses_the_same_rule_as_the_mean():
    row = summarise(_result(n=1000, penalty_std=10.0), planned=1.0, n_trips=1)
    assert row["penalty_ci95"] == pytest.approx(1.96 * 10.0 / math.sqrt(999))
    assert row["penalty_ci95"] == pytest.approx(1.96 * _mc_se(10.0, 1000))


def test_penalty_interval_scales_with_the_scenario_count():
    few = summarise(_result(n=100), planned=1.0, n_trips=1)["penalty_ci95"]
    many = summarise(_result(n=10000), planned=1.0, n_trips=1)["penalty_ci95"]
    assert few > many > 0.0


def test_a_result_without_a_penalty_distribution_still_summarises():
    """The paired tests stub the evaluator, and the stub does not have one."""
    stub = _result()
    del stub["time_penalty_distribution"]
    row = summarise(stub, planned=100.0, n_trips=2)
    assert row["penalty"] == 0.0
    assert row["penalty_ci95"] == 0.0


# ---- the row the table reads ----------------------------------------------

def test_exceedance_interval_brackets_the_point_estimate():
    row = summarise(_result(n=1000, p_exceed=0.98), planned=1.0, n_trips=1)
    assert row["p_exceed_lo"] < row["p_exceed"] < row["p_exceed_hi"]


def test_every_interval_column_is_actually_produced():
    """A column keyed on a name summarise never sets prints as blank, silently."""
    row = summarise(_result(), planned=1.0, n_trips=1)
    for key, _, _, _ in CORE_COLUMNS + INTERVAL_COLUMNS:
        assert key in row, f"{key} is a column but not a row key"


# ---- the columns the Results tables need ----------------------------------

def _scenario(boats, times):
    """One per-scenario result with given per-boat trip times."""
    return {"n_unscheduled_returns": 0,
            "trip_details": [{"boat_id": b, "adjusted_time": t}
                             for b, t in zip(boats, times)]}


def test_delay_p95_and_u_share_are_carried_through():
    """Table 4.1 quotes both; only the mean penalty was being kept."""
    row = summarise(_result(penalty_mean=20.0), planned=200.0, n_trips=4)
    assert row["penalty_p95"] == pytest.approx(40.0)   # 2x the mean, per _result
    assert row["u_pct"] == pytest.approx(0.1)          # 20 / 200


def test_u_share_is_zero_rather_than_undefined_at_zero_planned():
    row = summarise(_result(), planned=0.0, n_trips=1)
    assert row["u_pct"] == 0.0


def test_makespan_is_the_busiest_vessel_not_the_sum():
    res = _result()
    res["all_results"] = [_scenario([1, 1, 2], [10.0, 20.0, 5.0])]
    row = summarise(res, planned=35.0, n_trips=3)
    assert row["makespan"] == pytest.approx(30.0)      # boat 1: 10 + 20
    assert row["makespan"] < 35.0                      # strictly under the sum


def test_makespan_is_bounded_by_the_sum_and_its_fair_share():
    res = _result()
    res["all_results"] = [_scenario([1, 2, 3, 4], [10.0, 20.0, 30.0, 40.0])]
    row = summarise(res, planned=100.0, n_trips=4)
    assert 100.0 / 4 <= row["makespan"] <= 100.0


def test_a_result_with_no_trip_details_still_summarises():
    """Stubbed results carry neither key; summarise must not reach into them."""
    row = summarise(_result(), planned=100.0, n_trips=1)
    assert "makespan" not in row          # absent, not crashed
    assert row["mean"] == pytest.approx(400.0)


def test_plan_balance_reports_the_cap_check():
    """Which planners may be ranked depends on this, so it has to be a column."""
    from experiments.common import _plan_balance
    from unified.problem import N_PORTS
    trips = [
        {"boat_id": 1, "nodes": [0, N_PORTS, N_PORTS + 1, 0], "total_time": 10.0},
        {"boat_id": 1, "nodes": [0, N_PORTS + 2, N_PORTS + 3, 0],
         "total_time": 5.0},
        {"boat_id": 2, "nodes": [0, N_PORTS + 4, N_PORTS + 5, 0],
         "total_time": 3.0},
    ]
    b = _plan_balance(trips)
    assert b["max_stations"] == 2 and b["min_stations"] == 1
    assert b["makespan_planned"] == pytest.approx(15.0)
    assert b["vessel_spread"] == pytest.approx(12.0)


def test_plan_balance_of_nothing_is_nothing():
    from experiments.common import _plan_balance
    assert _plan_balance([]) == {}


# ---- paired across buffers ------------------------------------------------

def test_paired_across_buffers_recovers_a_known_difference():
    """The buffer question rests on this table, so pin what it computes."""
    from experiments.full_sweep import _paired_across_buffers
    base = [100.0, 102.0, 98.0, 101.0, 99.0]
    by_buffer = {1.0: {"forward": base},
                 0.8: {"forward": [t - 7.0 for t in base]}}
    rows = _paired_across_buffers(by_buffer, [("forward", {})], "forward", 5)

    ref = next(r for r in rows if r["setting"] == "buffer_100%")
    other = next(r for r in rows if r["setting"] == "buffer_80%")
    assert ref["diff"] == pytest.approx(0.0)
    assert ref["sd"] == pytest.approx(0.0)
    assert other["diff"] == pytest.approx(-7.0)
    assert other["sd"] == pytest.approx(0.0)     # constant shift, no spread
    assert other["significant"] is True


def test_paired_across_buffers_covers_every_policy_not_just_the_printed_one():
    from experiments.full_sweep import _paired_across_buffers
    by_buffer = {1.0: {"a": [10.0, 11.0], "b": [20.0, 21.0]},
                 0.9: {"a": [12.0, 13.0], "b": [19.0, 20.0]}}
    rows = _paired_across_buffers(by_buffer, [("a", {}), ("b", {})], "a", 2)
    assert {r["case"] for r in rows} == {"a", "b"}
    assert len(rows) == 4                        # 2 policies x 2 buffers


def test_paired_across_buffers_needs_a_100_percent_reference():
    """Without the unbuffered plan there is nothing to difference against."""
    from experiments.full_sweep import _paired_across_buffers
    assert _paired_across_buffers({0.8: {"a": [1.0]}, 0.9: {"a": [2.0]}},
                                  [("a", {})], "a", 1) == []
    assert _paired_across_buffers({1.0: {"a": [1.0]}}, [("a", {})], "a", 1) == []
