# Research method

This section explains the research independently of the implementation.

The supporting system schematics are in
[`diagrams/`](diagrams/), including the pipeline architecture, project
structure, and recourse-policy diagrams.

## Research question

How much additional vessel time is caused by uncertain catch, and how can
planning buffers or route-repair policies reduce the realised cost of
capacity overflows?

## System

The survey contains 581 stations and four vessels. A station is represented by
two tow-end nodes. A planned route is evaluated under catch scenarios sampled
from historical spring-survey data. If accumulated catch exceeds a vessel's
capacity, the vessel makes an unscheduled return to the nearest port, incurs a
round-trip detour, offloads, and continues according to the selected recourse
policy.

## Method components

- Deterministic route planning uses inherited heuristic planners and saved
  plans.
- Historical catch distributions are fitted per station where data exists;
  stations without records inherit the nearest station's fitted distribution.
- The stochastic evaluator scores a fixed route against repeated catch
  scenarios.
- Planning buffers reduce the capacity used during route construction.
- Recourse strategies include Forward, Backtrack, Preemptive, and Repair
  variants.
- Paired experiments use shared blocks or scenarios so comparisons are not
  dominated by instance or catch noise.

## Main measures

- planned time;
- realised time;
- cost of uncertainty, realised time minus planned time;
- probability of at least one true capacity overflow;
- Wilson 95% interval for overflow probability;
- expected unscheduled returns;
- trip-level overflow position;
- repair and diversion penalties.

## Scope and limitations

The report's results were generated on one Windows 11 machine with Gurobi
limited to one thread. Heuristic restart loops stop on wall-clock time, so
planned routes can differ between machines even with the same seed. Scenario
sets with different sizes are separate random draws and must not be
differenced as if one were nested inside the other.
