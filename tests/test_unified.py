"""Tests for the unified adapter layer.

Organised into three groups:
  1. Loader fidelity   — ProblemInstance matches raw file contents
  2. Adapter correctness — conversion to solver formats preserves semantics
  3. Output consistency — deterministic seeds → reproducible results (golden baselines)

Run with:
    cd <repo-root>
    python -m pytest tests/test_unified.py -v

First run generates golden_baselines.json.  Subsequent runs compare against it.

To run only fast tests (no solver execution):
    python -m pytest tests/test_unified.py -v -m "not slow"
"""

import gc
import math
import os
import sys
import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, ROOT)

from unified.problem import ProblemInstance, N_PORTS
from unified.loaders import (
    load_gfsp_problem,
    load_gfsp_full_problem,
    load_heuristic_problem,
    list_gfsp_problems,
    list_heuristic_problems,
)
from unified.adapters import heuristic_context, to_heuristic_problem
from unified.evaluate import (
    evaluate_heuristic_solution,
    solution_to_trips,
    compute_total_time,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
GFSP_DATA = os.path.join(ROOT, "gfsp_code", "data")
HEURISTIC_DIR = os.path.join(ROOT, "final_code")


# ===================================================================
# GROUP 1 — Loader fidelity
# ===================================================================

class TestGfspLoader:
    """Verify load_gfsp_problem matches raw file contents."""

    @pytest.mark.parametrize("ns,nv,cf,instance", [
        (10, 2, 62.5, 1),
        (20, 2, 62.5, 1),
        (20, 3, 125, 5),
        (40, 4, 250, 10),
    ])
    def test_fields_match_raw_file(self, ns, nv, cf, instance):
        """Parse the file manually and compare every field."""
        base_cap = int(cf * ns)
        path = os.path.join(
            GFSP_DATA, "test_problems",
            str(ns), str(nv), str(base_cap),
            f"ins_{ns}_{nv}_{base_cap}_{instance}.txt",
        )
        if not os.path.isfile(path):
            pytest.skip(f"Test problem file not found: {path}")

        with open(path) as f:
            lines = f.readlines()

        expected_ports = np.array(lines[1].strip().split(","), dtype=int)
        expected_stations = np.array(lines[3].strip().split(","), dtype=int)
        expected_home = int(lines[5].strip())
        expected_caps = np.array(lines[7].strip().split(","), dtype=float)

        inst = load_gfsp_problem(ns, nv, cf, instance)

        np.testing.assert_array_equal(inst.port_ids, expected_ports)
        np.testing.assert_array_equal(inst.station_ids, expected_stations)
        assert inst.home_port_index == expected_home
        np.testing.assert_array_equal(inst.capacities, expected_caps)
        assert inst.n_boats == nv
        assert inst.ns == ns
        assert inst.source == "gfsp"

    def test_fish_time_limit_formula(self):
        """fish_time_limit should be ns * 7.5 + 450."""
        inst = load_gfsp_problem(20, 2, 62.5, 1)
        assert inst.fish_time_limit == 20 * 7.5 + 450

    def test_work_limit_formula(self):
        """work_limit should match ceil(ns * (1/nv + 1/(3*nv^2))) * 2."""
        ns, nv = 20, 2
        inst = load_gfsp_problem(ns, nv, 62.5, 1)
        expected = int(math.ceil(ns * ((1 / nv) + (1 / (3 * nv ** 2)))) * 2)
        assert inst.work_limit == expected

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_gfsp_problem(999, 2, 62.5, 1)


class TestHeuristicLoader:
    """Verify load_heuristic_problem matches raw file contents."""

    @pytest.mark.parametrize("prob_id", [0, 6, 12, 24, 34])
    def test_fields_match_raw_file(self, prob_id):
        """Parse the semicolon file manually and compare."""
        path = os.path.join(
            HEURISTIC_DIR, "better_test_problems",
            f"test_problem{prob_id:02d}.txt",
        )
        if not os.path.isfile(path):
            pytest.skip(f"Test problem file not found: {path}")

        with open(path) as f:
            raw = f.read()
        parts = raw.split("; ")

        expected_stations = np.array(eval(parts[0]))
        expected_ports = np.array(eval(parts[1]))
        expected_nboats = int(parts[2])
        expected_caps = np.array(eval(parts[3]))
        expected_fish = int(parts[4])
        expected_work = int(parts[5])

        inst = load_heuristic_problem(prob_id)

        np.testing.assert_array_equal(inst.station_ids, expected_stations)
        np.testing.assert_array_equal(inst.port_ids, expected_ports)
        assert inst.n_boats == expected_nboats
        np.testing.assert_array_equal(inst.capacities, expected_caps)
        assert inst.fish_time_limit == expected_fish
        assert inst.work_limit == expected_work
        assert inst.home_port_index == 0
        assert inst.source == "heuristic"

    def test_upper_bound_parsed(self):
        """Upper bound (7th field) should be parsed."""
        inst = load_heuristic_problem(0)
        assert np.isfinite(inst.upper_bound)
        assert inst.upper_bound == 2167.0

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_heuristic_problem(999)


class TestFullProblem:
    """Verify the full 581-station gfsp problem loads."""

    def test_full_problem_station_count(self):
        inst = load_gfsp_full_problem()
        assert inst.ns == 581

    def test_full_problem_source(self):
        inst = load_gfsp_full_problem()
        assert inst.source == "gfsp"


class TestDiscovery:
    """Verify discovery helpers find problems."""

    def test_list_gfsp_problems_nonempty(self):
        problems = list_gfsp_problems()
        assert len(problems) > 0
        # Each entry is (ns, nv, cap, instance)
        assert len(problems[0]) == 4

    def test_list_heuristic_problems_nonempty(self):
        problems = list_heuristic_problems()
        assert len(problems) > 0

    def test_gfsp_problems_count(self):
        """Should be 1620 = 6 sizes * 3 vessels * 3 capacities * 30 instances."""
        problems = list_gfsp_problems()
        assert len(problems) == 1620


# ===================================================================
# GROUP 2 — Adapter correctness
# ===================================================================

class TestNodeConversion:
    """Verify station_id <-> node_index helpers."""

    def test_roundtrip(self):
        """station_ids -> node_indices -> station_ids should round-trip."""
        ids = np.array([0, 5, 100, 580])
        inst = ProblemInstance(
            station_ids=ids,
            port_ids=np.array([0, 1]),
            n_boats=2,
            capacities=np.array([1000.0, 1000.0]),
            home_port_index=0,
            fish_time_limit=120,
            work_limit=50,
        )
        nodes = inst.station_ids_to_node_indices()
        recovered = ProblemInstance.node_indices_to_station_ids(nodes)
        np.testing.assert_array_equal(recovered, ids)

    def test_node_indices_are_pairs(self):
        """Each station ordinal s should map to nodes 2s+13 and 2s+14."""
        ids = np.array([0, 10, 222])
        inst = ProblemInstance(
            station_ids=ids,
            port_ids=np.array([0]),
            n_boats=1,
            capacities=np.array([1000.0]),
            home_port_index=0,
            fish_time_limit=120,
            work_limit=50,
        )
        nodes = inst.station_ids_to_node_indices()
        expected = []
        for s in ids:
            expected.extend([s * 2 + N_PORTS, s * 2 + 1 + N_PORTS])
        np.testing.assert_array_equal(nodes, np.array(expected))

    def test_home_port_property(self):
        """home_port should return port_ids[home_port_index]."""
        inst = ProblemInstance(
            station_ids=np.array([0, 1]),
            port_ids=np.array([5, 11]),
            n_boats=1,
            capacities=np.array([1000.0]),
            home_port_index=1,
            fish_time_limit=120,
            work_limit=50,
        )
        assert inst.home_port == 11


class TestToHeuristicProblem:
    """Verify conversion to heuristic Problem preserves data."""

    def test_station_nodes_match(self):
        """Heuristic Problem.stations should match node_indices from ProblemInstance."""
        inst = load_gfsp_problem(20, 2, 62.5, 1)
        with heuristic_context():
            prob = to_heuristic_problem(inst, catch_source="heuristic")
            expected_nodes = list(inst.station_ids_to_node_indices())
            assert prob.stations == expected_nodes

    def test_boat_count(self):
        inst = load_gfsp_problem(20, 2, 62.5, 1)
        with heuristic_context():
            prob = to_heuristic_problem(inst)
            assert prob.n_boats == inst.n_boats
            assert len(prob.boats) == inst.n_boats

    def test_boat_capacities(self):
        inst = load_gfsp_problem(20, 2, 62.5, 1)
        with heuristic_context():
            prob = to_heuristic_problem(inst)
            for i, boat in enumerate(prob.boats):
                assert boat.capacity == inst.capacities[i]

    def test_home_ports(self):
        """All boats should have home_port == instance.home_port."""
        inst = load_gfsp_problem(20, 2, 62.5, 1)
        with heuristic_context():
            prob = to_heuristic_problem(inst)
            for boat in prob.boats:
                assert boat.home_port == inst.home_port

    def test_port_ids(self):
        inst = load_gfsp_problem(20, 2, 62.5, 1)
        with heuristic_context():
            prob = to_heuristic_problem(inst)
            assert prob.ports == list(inst.port_ids)

    def test_fish_time_limit(self):
        inst = load_gfsp_problem(20, 2, 62.5, 1)
        with heuristic_context():
            prob = to_heuristic_problem(inst)
            assert prob.fish_time_limit == inst.fish_time_limit

    def test_station_count(self):
        """n_stations should be 2 * ns (pairs)."""
        inst = load_gfsp_problem(20, 2, 62.5, 1)
        with heuristic_context():
            prob = to_heuristic_problem(inst)
            assert prob.n_stations == inst.ns * 2


class TestCatchSourceSwitching:
    """Verify catch_source flag loads different data."""

    def test_catch_differs_by_source(self):
        """gfsp and heuristic catch data should produce different arrays.

        Note: LimitedCatch.csv has 582 rows while RandomMatchingCatches.dat
        has 581.  We verify that the actual values loaded differ (checking
        the first 581 entries which both have).
        """
        inst = load_gfsp_problem(20, 2, 62.5, 1)
        with heuristic_context():
            prob_heur = to_heuristic_problem(inst, catch_source="heuristic")
            catch_heur = prob_heur.catch.copy()

        with heuristic_context():
            prob_gfsp = to_heuristic_problem(inst, catch_source="gfsp")
            catch_gfsp = prob_gfsp.catch.copy()

        # Either the lengths differ (582 vs 581) or the values differ
        # Both indicate different catch sources are being used
        if len(catch_heur) == len(catch_gfsp):
            assert not np.array_equal(catch_heur, catch_gfsp), \
                "Catch arrays should differ between gfsp and heuristic sources"
        else:
            # Different lengths already proves different sources
            min_len = min(len(catch_heur), len(catch_gfsp))
            assert not np.array_equal(catch_heur[:min_len], catch_gfsp[:min_len]), \
                "Catch arrays should have different values even in overlapping range"

    def test_gfsp_catch_propagated_to_boats(self):
        """When catch_source='gfsp', boats and trips should also get gfsp catch."""
        inst = load_gfsp_problem(20, 2, 62.5, 1)
        with heuristic_context():
            prob = to_heuristic_problem(inst, catch_source="gfsp")
            for boat in prob.boats:
                np.testing.assert_array_equal(boat.catch, prob.catch)


class TestCrossSourceLoading:
    """Test loading heuristic problems through the unified layer."""

    def test_heuristic_problem_to_solver(self):
        """A heuristic-source problem should convert to a solver Problem cleanly."""
        inst = load_heuristic_problem(6)
        with heuristic_context():
            prob = to_heuristic_problem(inst, catch_source="heuristic")
            assert prob.n_boats == inst.n_boats
            assert prob.n_stations == inst.ns * 2


# ===================================================================
# GROUP 3 — Output consistency (determinism + golden baselines)
# ===================================================================

class TestDeterminism:
    """Same seed must produce identical results across two runs.

    These tests verify determinism by running twice within the SAME
    heuristic_context() block, using prob.reset() + re-seed to ensure
    the monkey-patching state is identical for both runs.
    """

    @pytest.mark.slow
    def test_greedy_deterministic(self):
        """Two runs of generate_initial_solution with same seed must match."""
        inst = load_gfsp_problem(10, 2, 62.5, 1)

        with heuristic_context():
            prob = to_heuristic_problem(inst, catch_source="gfsp")

            # Run 1
            prob.generate_initial_solution(seed=42)
            sol1 = prob.save_solution_as_list()
            prob.restore_solution_from_list(sol1)
            k_catch = max(1.0, (400000 - np.mean(inst.capacities)) * 0.001)
            k_fish = np.mean(inst.capacities) / inst.fish_time_limit * k_catch
            obj1, pen1 = prob.evaluate_solution(k_catch, k_fish, 10000)

            # Run 2 (reset and re-run with same seed)
            prob.reset()
            prob.generate_initial_solution(seed=42)
            sol2 = prob.save_solution_as_list()
            prob.restore_solution_from_list(sol2)
            obj2, pen2 = prob.evaluate_solution(k_catch, k_fish, 10000)

        assert obj1 == obj2, f"Objectives differ: {obj1} vs {obj2}"
        assert pen1 == pen2, f"Penalties differ: {pen1} vs {pen2}"
        assert sol1 == sol2, "Solutions differ despite same seed"

    @pytest.mark.slow
    def test_grasp_deterministic(self):
        """Two runs of GRASP with same seed must match."""
        inst = load_gfsp_problem(10, 2, 62.5, 1)

        with heuristic_context():
            prob = to_heuristic_problem(inst, catch_source="gfsp")

            # Run 1
            prob.GRASP(rcl_size=2, seed=42)
            sol1 = prob.save_solution_as_list()
            prob.restore_solution_from_list(sol1)
            k_catch = max(1.0, (400000 - np.mean(inst.capacities)) * 0.001)
            k_fish = np.mean(inst.capacities) / inst.fish_time_limit * k_catch
            obj1, pen1 = prob.evaluate_solution(k_catch, k_fish, 10000)

            # Run 2
            prob.reset()
            prob.GRASP(rcl_size=2, seed=42)
            sol2 = prob.save_solution_as_list()
            prob.restore_solution_from_list(sol2)
            obj2, pen2 = prob.evaluate_solution(k_catch, k_fish, 10000)

        assert obj1 == obj2, f"Objectives differ: {obj1} vs {obj2}"
        assert pen1 == pen2, f"Penalties differ: {pen1} vs {pen2}"
        assert sol1 == sol2, "Solutions differ despite same seed"


class TestEvaluation:
    """Verify evaluate_heuristic_solution returns correct structure."""

    @pytest.mark.slow
    def test_result_keys(self):
        inst = load_gfsp_problem(10, 2, 62.5, 1)
        with heuristic_context():
            prob = to_heuristic_problem(inst, catch_source="gfsp")
            prob.generate_initial_solution(seed=42)
            sol = prob.save_solution_as_list()
            prob.restore_solution_from_list(sol)
            result = evaluate_heuristic_solution(prob, inst)

        expected_keys = {"objective", "penalty", "feasible", "total_time",
                         "trips", "boats_summary"}
        assert set(result.keys()) == expected_keys

    @pytest.mark.slow
    def test_boats_summary_structure(self):
        inst = load_gfsp_problem(10, 2, 62.5, 1)
        with heuristic_context():
            prob = to_heuristic_problem(inst, catch_source="gfsp")
            prob.generate_initial_solution(seed=42)
            sol = prob.save_solution_as_list()
            prob.restore_solution_from_list(sol)
            result = evaluate_heuristic_solution(prob, inst)

        assert len(result["boats_summary"]) == 2
        for b in result["boats_summary"]:
            assert "id" in b
            assert "total_time" in b
            assert "n_trips" in b
            assert "n_stations" in b

    @pytest.mark.slow
    def test_objective_positive(self):
        inst = load_gfsp_problem(10, 2, 62.5, 1)
        with heuristic_context():
            prob = to_heuristic_problem(inst, catch_source="gfsp")
            prob.generate_initial_solution(seed=42)
            sol = prob.save_solution_as_list()
            prob.restore_solution_from_list(sol)
            result = evaluate_heuristic_solution(prob, inst)

        assert result["objective"] > 0

    @pytest.mark.slow
    def test_all_stations_visited(self):
        """After initial solution, every station pair node should appear in some trip."""
        inst = load_gfsp_problem(10, 2, 62.5, 1)
        with heuristic_context():
            prob = to_heuristic_problem(inst, catch_source="gfsp")
            prob.generate_initial_solution(seed=42)
            sol = prob.save_solution_as_list()
            prob.restore_solution_from_list(sol)
            result = evaluate_heuristic_solution(prob, inst)

        visited_nodes = set()
        for trip in result["trips"]:
            for n in trip["nodes"]:
                if n >= N_PORTS:
                    visited_nodes.add(n)

        expected_nodes = set(inst.station_ids_to_node_indices())
        assert expected_nodes == visited_nodes, (
            f"Missing nodes: {expected_nodes - visited_nodes}, "
            f"Extra nodes: {visited_nodes - expected_nodes}"
        )


class TestGoldenBaselines:
    """Record and compare golden baselines for regression detection.

    On first run, baselines are saved to tests/golden_baselines.json.
    On subsequent runs, results are compared against saved values.
    Tolerance of 1e-6 is used for floating-point comparison.
    """

    CASES = [
        ("greedy_10_2_625_1", 10, 2, 62.5, 1, "greedy", 42),
        ("grasp_10_2_625_1",  10, 2, 62.5, 1, "grasp",  42),
    ]

    @pytest.mark.slow
    @pytest.mark.parametrize("key,ns,nv,cf,instance,method,seed", CASES)
    def test_baseline(self, baselines, key, ns, nv, cf, instance, method, seed):
        inst = load_gfsp_problem(ns, nv, cf, instance)

        with heuristic_context():
            prob = to_heuristic_problem(inst, catch_source="gfsp")

            if method == "greedy":
                prob.generate_initial_solution(seed=seed)
            else:
                prob.GRASP(rcl_size=2, seed=seed)

            sol = prob.save_solution_as_list()
            prob.restore_solution_from_list(sol)

            k_catch = max(1.0, (400000 - np.mean(inst.capacities)) * 0.001)
            k_fish = np.mean(inst.capacities) / inst.fish_time_limit * k_catch
            obj, pen = prob.evaluate_solution(k_catch, k_fish, 10000)

        result = {"objective": obj, "penalty": pen, "solution": sol}

        if key not in baselines:
            # First run: record the baseline
            baselines[key] = {
                "objective": result["objective"],
                "penalty": result["penalty"],
                "solution": result["solution"],
            }
            pytest.skip(f"Baseline recorded for {key} — re-run to verify")
        else:
            saved = baselines[key]
            assert abs(result["objective"] - saved["objective"]) < 1e-6, (
                f"Objective changed: was {saved['objective']}, now {result['objective']}"
            )
            assert abs(result["penalty"] - saved["penalty"]) < 1e-6, (
                f"Penalty changed: was {saved['penalty']}, now {result['penalty']}"
            )
            assert result["solution"] == saved["solution"], (
                f"Solution structure changed for {key}"
            )


class TestDirectVsUnified:
    """Compare running the heuristic code directly vs through the unified layer.

    This is the most important correctness check: the unified layer should
    produce *exactly* the same Problem state as calling the original code
    with the same parameters.
    """

    @pytest.mark.slow
    def test_greedy_direct_vs_unified(self):
        """Load a gfsp problem, run greedy via both paths, compare solutions."""
        inst = load_gfsp_problem(10, 2, 62.5, 1)

        with heuristic_context():
            from classes import Problem
            from unified.adapters import _load_gfsp_catch_array, override_catch_data

            # --- Unified path ---
            prob_unified = to_heuristic_problem(inst, catch_source="gfsp")
            prob_unified.generate_initial_solution(seed=42)
            sol_unified = prob_unified.save_solution_as_list()
            prob_unified.restore_solution_from_list(sol_unified)

            k_catch = max(1.0, (400000 - np.mean(inst.capacities)) * 0.001)
            k_fish = np.mean(inst.capacities) / inst.fish_time_limit * k_catch
            obj_unified, pen_unified = prob_unified.evaluate_solution(
                k_catch, k_fish, 10000)

            # --- Direct path (replicate what the unified layer does) ---
            node_indices = list(inst.station_ids_to_node_indices())
            home_ports_list = [inst.home_port] * inst.n_boats

            prob_direct = Problem(
                stations=node_indices,
                ports=list(inst.port_ids),
                fish_time_limit=inst.fish_time_limit,
                n_boats=inst.n_boats,
                boat_capacities=list(inst.capacities),
                home_ports=home_ports_list,
            )

            # Apply gfsp catch data (same as adapter), including the ns == 10
            # halving and trip-limit bump that gfsp_models.py:48-50 applies.
            # Without it this instance has no feasible solution and greedy
            # construction never terminates.
            from unified.adapters import _patch_catch_loader_gfsp

            gfsp_catch = _load_gfsp_catch_array().to_numpy().reshape(-1)
            if inst.ns == 10:
                gfsp_catch = gfsp_catch / 2
                prob_direct.fish_time_limit = inst.fish_time_limit * 1.2
            _patch_catch_loader_gfsp(gfsp_catch)
            override_catch_data(prob_direct, catch_array=gfsp_catch)

            prob_direct.generate_initial_solution(seed=42)
            sol_direct = prob_direct.save_solution_as_list()
            prob_direct.restore_solution_from_list(sol_direct)

            obj_direct, pen_direct = prob_direct.evaluate_solution(
                k_catch, k_fish, 10000)

        assert obj_unified == obj_direct, (
            f"Objective mismatch: unified={obj_unified}, direct={obj_direct}"
        )
        assert pen_unified == pen_direct, (
            f"Penalty mismatch: unified={pen_unified}, direct={pen_direct}"
        )
        assert sol_unified == sol_direct, \
            "Solution routes differ between unified and direct"

    @pytest.mark.slow
    def test_heuristic_source_direct_vs_unified(self):
        """Load a heuristic test problem, compare unified vs direct construction."""
        inst = load_heuristic_problem(6)

        with heuristic_context():
            from classes import Problem

            # --- Unified path ---
            prob_unified = to_heuristic_problem(inst, catch_source="heuristic")
            prob_unified.generate_initial_solution(seed=42)
            sol_unified = prob_unified.save_solution_as_list()
            prob_unified.restore_solution_from_list(sol_unified)

            k_catch = max(1.0, (400000 - np.mean(inst.capacities)) * 0.001)
            k_fish = np.mean(inst.capacities) / inst.fish_time_limit * k_catch
            upper_bound = inst.upper_bound if np.isfinite(inst.upper_bound) else 10000
            obj_unified, pen_unified = prob_unified.evaluate_solution(
                k_catch, k_fish, upper_bound
            )

            # --- Direct path ---
            node_indices = list(inst.station_ids_to_node_indices())
            home_ports_list = [inst.home_port] * inst.n_boats

            prob_direct = Problem(
                stations=node_indices,
                ports=list(inst.port_ids),
                fish_time_limit=inst.fish_time_limit,
                n_boats=inst.n_boats,
                boat_capacities=list(inst.capacities),
                home_ports=home_ports_list,
            )
            prob_direct.generate_initial_solution(seed=42)
            sol_direct = prob_direct.save_solution_as_list()
            prob_direct.restore_solution_from_list(sol_direct)

            obj_direct, pen_direct = prob_direct.evaluate_solution(
                k_catch, k_fish, upper_bound
            )

        assert obj_unified == obj_direct
        assert pen_unified == pen_direct
        assert sol_unified == sol_direct
