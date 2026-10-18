# Research decisions

This record summarises the main methodological and implementation decisions
that shaped the project. Each entry explains the context, alternatives
considered, decision taken, and resulting trade-offs.

## Decision 2: Strict separation of planning and execution

**Context:** Plans had to be evaluated when catch varies, while real captains
may change routes during a survey.

**Options:** Build a continuously updating dynamic simulation, or enforce a
two-stage planning and execution barrier.

**Decision:** Plans enter the simulator only as standardised trip data. Future
catch information does not flow back to the planner unless the selected
policy is explicitly **Repair**.

**Justification and trade-off:** This mirrors the deterministic inherited
planners and isolates the effect of the recourse policy. It provides a
controlled baseline for measuring the cost of uncertainty, although it is
simpler than current operational practice.

**Affected code:** `unified/solve.py`, `unified/stochastic_eval.py`,
`experiments/common.py`, and the stochastic evaluator tests.

## Decision 3: Choosing the default planner

**Context:** The pipeline needed a default planner for baseline routes.

**Options:** Use one of the six inherited heuristic planners, including GRASP,
Tabu Search, and Simulated Annealing.

**Decision:** Use Tabu Search with the move neighbourhood as the default
planner.

**Justification and trade-off:** The choice followed Patel's prior
benchmarking. Tabu Search is slower and completed only four restarts in one
300-second run, whereas GRASP alone completed 308. The project accepted fewer
restarts in exchange for higher-quality individual routes and mitigated the
resulting variability with paired comparisons across seeds on test problems.

**Affected code:** `experiments/common.py`, `unified/solve.py`, and
`final_code/`.

## Decision 4: Choosing the reference baseline

**Context:** Recourse policies needed a baseline for comparison.

**Options:** Use Backtrack, the classic VRPSD baseline, or Forward, which
completes the tow and continues.

**Decision:** Use Forward as the reference arm.

**Justification and trade-off:** Backtrack requires discarding catch and
repeating a tow, which is not legal in the Icelandic survey. Forward is
operationally realistic, so reported intervals measure the gap to doing
nothing special rather than the gap to a textbook but unrealistic baseline.
This makes the comparisons more useful for the survey operator.

**Affected code:** `unified/stochastic_eval.py`, `experiments/strategies.py`,
and the strategy-comparison tests.

## Decision 8: Selecting the recourse policies to investigate

**Context:** The stochastic-routing literature identifies detour-to-depot,
preventive restocking, and re-optimisation as three main responses to a
capacity event. Survey vessels differ from standard delivery vehicles: they
make multiple trips, can land catch at any of 13 ports, and cannot discard
catch after a tow is hauled.

**Options:** Test textbook definitions strictly, or adapt them to the
operational reality of the survey.

**Decision:** Implement four policies mapped to the three strategy types:

1. **Forward:** adapted detour-to-depot; complete the tow and continue.
2. **Backtrack:** classic detour-to-depot; discard the catch and repeat the
   tow.
3. **Pre-emptive:** rule-based preventive restocking using fill threshold
   alpha.
4. **Repair:** re-optimisation.

**Justification and trade-off:** Backtrack provides a literature baseline,
although it is operationally unrealistic for this survey. Forward represents
the actual operational baseline: doing nothing special after an overflow.
The other policies therefore have to improve on a realistic baseline rather
than on an artificial one.

**Affected code:** `unified/stochastic_eval.py`,
`experiments/strategies.py`, and the recourse-policy tests.

## Decision 9: Testing trip-scope and vessel-scope re-optimisation

**Context:** Repair can re-plan only the remaining stations in the current
trip, or all remaining work assigned to the vessel for the rest of the
survey.

**Options:** Test only the simpler remaining-trip scope, or compare both trip
and vessel scope.

**Decision:** Test both scopes using five routing methods for each:
keep order, nearest neighbour, 2-opt, exact ordering, and tabu search.

**Justification and trade-off:** The hypothesis was that vessel scope would
perform better because it gives the re-planner more freedom to reassign work.
The results showed the opposite: more freedom made performance worse. Vessel
scope inherits approximately 150 stations and re-solves against stale
information, producing a large increase in unplanned returns. Testing both
scopes was therefore necessary to expose this structural result.

**Affected code:** `unified/stochastic_eval.py`,
`experiments/strategies.py`, and repair-policy tests.

## Decision 10: Omitting the rollout safety net for Repair

**Context:** The literature describes a policy-improvement guarantee for
rollout algorithms: a re-planner should adopt a new route only when it is
demonstrably cheaper than retaining the existing route. The Repair policy
does not include that safeguard.

**Options:** Add a rollout safeguard that evaluates the expected cost of the
remaining route before committing, or use a simpler greedy re-plan.

**Decision:** Deliberately omit the rollout safeguard.

**Justification and trade-off:** Evaluating thousands of downstream scenarios
at every overflow was computationally infeasible within the Monte Carlo
experiments. Omitting the safeguard also made it possible to measure the cost
of that omission. The results showed that vessel-scope Repair could be up to
70 hours worse than Forward. This supports the literature's rationale for a
rollout guarantee: re-planning without a safety net can fail structurally,
not merely because one routing heuristic is poor.

**Affected code:** `unified/stochastic_eval.py`,
`experiments/strategies.py`, and repair-policy tests.

## Cross-cutting implementation decisions

These decisions support the architecture and experimental comparisons
described above.

| Decision | Rationale |
|---|---|
| Use a unified problem representation | Allows inherited planners and the stochastic evaluator to share a consistent interface. |
| Preserve saved plans | Makes route-scoring comparisons faster and avoids relying only on wall-clock-dependent solver restarts. |
| Keep active travel time separate from inherited distance data | The heuristic planner and GFSP solver use different matrix conventions and objectives. |
| Test overflow and repair policies on shared scenarios where appropriate | Allows differences between policies to be attributed to the policy rather than unrelated random variation. |

## Related records

The final experiment settings, report outputs, and known limitations are
documented in:

- [`03_experiments_and_outputs/report_output_register.csv`](../03_experiments_and_outputs/report_output_register.csv)
- [`04_reproduction_and_verification/known_limitations.md`](../04_reproduction_and_verification/known_limitations.md)
- [`02_methods_and_data/experimental_design.md`](../02_methods_and_data/experimental_design.md)
