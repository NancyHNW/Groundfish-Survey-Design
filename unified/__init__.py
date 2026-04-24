"""Unified adapter layer for the Ground Fish Survey Problem.

Lets you run heuristic solvers (final_code) and Kun Er's MIP
formulations (gfsp_code) on the same test problems.
"""

from .problem import ProblemInstance
from .loaders import (
    load_gfsp_problem,
    load_gfsp_full_problem,
    load_heuristic_problem,
    list_gfsp_problems,
    list_heuristic_problems,
)
from .adapters import (
    heuristic_context,
    gfsp_context,
    to_heuristic_problem,
    to_gfsp_model,
    override_catch_data,
)
from .evaluate import (
    evaluate_heuristic_solution,
    solution_to_trips,
    compute_total_distance,
    compute_total_time,
    compare_results,
)
