"""Where do overflows happen, and where do they cost?

The evaluator already records every overflow -- station, boat, trip, the port
it diverted to, what the diversion cost -- and until now nothing read it. This
aggregates those events over a scenario set and asks three questions:

    where on the map does time get lost
    where in a trip does the hold fill
    is the station that overflows most the one that hurts most

    python -m experiments.overflow_map --ns 100 --nv 2 --cf 125 --time-limit 60
    python -m experiments.overflow_map --full --time-limit 300 \
        --n-scenarios 1000 --home-ports 0 6 10 12

One solve, so the answer is conditional on one set of routes. Use
--solver-seeds to check the pattern holds across several.
"""

from experiments.common import (base_parser, describe_run, make_evaluator,
                                output_path, solve_cached, strategy_tag)


def run(ns=100, nv=2, cf=125, instance=1, method="tabu_move", time_limit=60,
        catch_source="historical", capacity_buffer=1.0, strategy="forward",
        repair_scope="trip", repair_planner="nn", n_scenarios=500,
        scenario_seed=123, full=False, home_ports=None, solver_seed=42,
        top_trips=None, plan=None):
    """Solve once, score a scenario set, and map where the overflows fell."""
    from unified.stochastic_eval import (overflow_profile, plot_overflow_map,
                                         plot_overflow_profile,
                                         plot_trip_fragility)

    det = solve_cached(plan, ns=ns, nv=nv, cf=cf, instance=instance,
                       method=method, time_limit=time_limit,
                       catch_source=catch_source,
                       capacity_buffer=capacity_buffer, full=full,
                       home_ports=home_ports, seed=solver_seed,
                       verbose=False)
    trips, inst = det["trips"], det["instance"]
    print(f"Planned {det['planned_time']:.1f}h over {len(trips)} trips "
          f"(feasible={det['feasible']})")

    evaluator = make_evaluator(scenario_seed, n_scenarios)
    result = evaluator.evaluate(trips, inst, strategy=strategy,
                                repair_scope=repair_scope,
                                repair_planner=repair_planner,
                                planned_catch=det.get("planned_catch"))

    profile = overflow_profile(result, trips)
    _report(profile, det, strategy)

    tag = f"{method}-{strategy_tag(strategy, repair_scope, repair_planner)}"
    plot_overflow_map(profile, inst,
                      save_path=output_path("overflow-map", tag, inst,
                                            home_ports=home_ports),
                      title_suffix=f"{strategy}, {n_scenarios} scenarios")
    plot_overflow_profile(profile,
                          save_path=output_path("overflow-profile", tag, inst,
                                                home_ports=home_ports),
                          title_suffix=f"{strategy}, {n_scenarios} scenarios")
    plot_trip_fragility(profile, top=top_trips,
                        save_path=output_path("trip-fragility", tag, inst,
                                              home_ports=home_ports),
                        title_suffix=f"{strategy}, {n_scenarios} scenarios")
    return profile


def _port_proximity(profile, instance):
    """What could re-ordering trips to finish near port actually save?

    An overflow costs roughly the distance to the nearest port, so where it
    lands sets the bill. The benchmark is NOT the survey average -- that is
    not a floor and says nothing about headroom. The floor for re-ordering
    alone is the port-nearest station the trip already visits: reaching it
    needs no reassignment, only a different order.

    Event-weighted throughout. A station that overflows ten times a season
    sets ten times as much of the cost as one that overflows once.
    """
    import numpy as np

    from unified.stochastic_eval import station_port_times

    st, tstns = profile["stations"], profile["trip_stations"]
    if not st:
        return

    allp = station_port_times(instance)
    survey = np.array(list(allp.values()))
    over = np.array([st[s]["port_time"] for s in st])
    wts = np.array([st[s]["rate"] for s in st])
    actual = float(np.average(over, weights=wts))

    print("\n  Time to nearest port (h) -- the cost driver:")
    print(f"    {'':<26} {'mean':>6} {'p10':>6} {'p50':>6} {'p90':>6}")
    print(f"    {'all ' + str(len(survey)) + ' stations':<26} "
          f"{survey.mean():>6.2f} {np.percentile(survey, 10):>6.2f} "
          f"{np.percentile(survey, 50):>6.2f} "
          f"{np.percentile(survey, 90):>6.2f}")
    print(f"    {'the ' + str(len(over)) + ' that overflow':<26} "
          f"{over.mean():>6.2f} {np.percentile(over, 10):>6.2f} "
          f"{np.percentile(over, 50):>6.2f} {np.percentile(over, 90):>6.2f}")
    print(f"    {'where overflow lands (ev-wt)':<26} {actual:>6.2f}")

    # Best each trip could do by re-ordering only: the nearest-to-port station
    # it already holds. Weighted by how often that trip overflows.
    floors, fw = [], []
    for t, stns in zip(profile["trips"], tstns):
        if t["count"] and stns:
            floors.append(min(allp[s] for s in stns if s in allp))
            fw.append(t["count"])
    if not floors:
        return
    floor = float(np.average(floors, weights=fw))
    print(f"    {'best by re-ordering alone':<26} {floor:>6.2f}")

    rate = sum(d["rate"] for d in st.values())
    gain = actual - floor
    print(f"    headroom: {gain:.2f} h per event, {rate:.1f} events a "
          f"season -> up to {rate * gain:.1f} h")
    print("    (upper bound: ignores the travel time a distorted order adds)")


def _report(profile, det, strategy):
    st, boats = profile["stations"], profile["boats"]
    total = sum(s["hours"] for s in st.values())
    print(f"\n--- Overflows under {strategy}, {profile['n_scenarios']} "
          f"scenarios ---")
    print(f"  stations that ever overflow: {len(st)} of {det['instance'].ns}")
    print(f"  expected hours lost a season: {total:.1f} "
          f"({total / det['planned_time']:.1%} of planned)")

    pos = profile["positions"]
    if pos:
        late = sum(1 for p in pos if p >= 0.8) / len(pos)
        print(f"  mean position in trip: {sum(pos) / len(pos):.2f}  "
              f"({late:.0%} in the last fifth)")

    print("\n  Worst stations by expected hours lost:")
    print(f"    {'stn':>5} {'per season':>11} {'hours':>8} {'h each':>8} "
          f"{'to port':>9}")
    for s in sorted(st, key=lambda s: -st[s]["hours"])[:8]:
        d = st[s]
        print(f"    {s:>5} {d['rate']:>11.2f} {d['hours']:>8.2f} "
              f"{d['mean_cost']:>8.2f} {d['port_time']:>9.2f}")

    _port_proximity(profile, det["instance"])

    # The interesting disagreement: most frequent is usually not most costly.
    if st:
        often = max(st, key=lambda s: st[s]["rate"])
        dearest = max(st, key=lambda s: st[s]["hours"])
        if often != dearest:
            print(f"\n  Most frequent is station {often} "
                  f"({st[often]['rate']:.2f}/season, "
                  f"{st[often]['hours']:.2f} h); costliest is {dearest} "
                  f"({st[dearest]['rate']:.2f}/season, "
                  f"{st[dearest]['hours']:.2f} h).")

    print("\n  Worst trips by expected hours lost:")
    print(f"    {'trip':>5} {'boat':>5} {'stns':>5} {'P(over)':>9} "
          f"{'hours':>8} {'mean pos':>9}")
    ranked = sorted(enumerate(profile["trips"]), key=lambda kv: -kv[1]["hours"])
    for i, t in ranked[:8]:
        # nan when every overflow on this trip came from a repair re-plan,
        # which has no position in any planned trip
        pos = t["mean_position"]
        shown = "-" if pos != pos else f"{pos:.2f}"
        print(f"    {i:>5} {t['boat']:>5} {t['n_stations']:>5} "
              f"{t['p_overflow']:>9.1%} {t['hours']:>8.2f} {shown:>9}")

    print("\n  Per boat:")
    for b in sorted(boats):
        print(f"    boat {b}: {boats[b]['rate']:.2f} overflows, "
              f"{boats[b]['hours']:.2f} h a season")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__.split("\n")[0], parents=[base_parser()])
    parser.add_argument("--top-trips", type=int, default=None,
                        help="Show only the N worst trips in the fragility "
                             "chart (default: all)")
    args = parser.parse_args()

    describe_run("OVERFLOW MAP", args, strategy=args.strategy)
    run(ns=args.ns, nv=args.nv, cf=args.cf, instance=args.instance,
        method=args.method, time_limit=args.time_limit,
        catch_source=args.catch_source,
        capacity_buffer=args.capacity_buffer, strategy=args.strategy,
        repair_scope=args.repair_scope, repair_planner=args.repair_planner,
        n_scenarios=args.n_scenarios, scenario_seed=args.scenario_seed,
        full=args.full, home_ports=args.home_ports,
        solver_seed=args.solver_seeds[0], top_trips=args.top_trips,
        plan=args.plan)
