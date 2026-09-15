
import copy
import time

from Solvers.mornpdec_sat_solver import MORNPDECSATSolver


def _solve_at_t(instance, t, time_limit_sec=None):
    """
    Build and solve a fresh MORNPDECSATSolver at cardinality bound t.
    Returns a list (SAT), None (UNSAT), or the string "TIMEOUT".
    """
    inst_t = copy.copy(instance)
    inst_t.k = t

    solver = MORNPDECSATSolver(inst_t)
    solver.build_formula()
    result = solver.solve(time_limit_sec=time_limit_sec)

    if result.get("timed_out"):
        return "TIMEOUT"
    if result["sat"]:
        return result["selected_objectives"]
    return None


def solve_mornpdec_with_halving_t(instance, *, verbose=True, time_limit_sec=None):
    """
    Iteratively solve MORNP-DEC by halving t each iteration:
      t0 = n, t1 = floor(n/2), t2 = floor(t1/2), ...
    Stop when the solver first returns NO after a YES, OR when a solve
    times out (in which case the search stops immediately, since a timeout
    means we don't actually know whether that t is SAT or UNSAT).

    Mirrors moppdec_sat_solver.solve_with_halving_k, adapted to
    MORNPDECSATSolver's dict-based solve() interface.
    """
    n = len(instance.objectives)
    if n == 0:
        raise ValueError("Instance has zero objectives.")

    t_values = []
    t = n
    while t >= 1:
        t_values.append(t)
        next_t = t // 2
        if next_t == t:
            break
        t = next_t

    iteration_log = []
    last_yes = None  # {"t": ..., "solution": [...], "solve_time_sec": ...}
    first_no = None
    timed_out_entry = None

    for it, t in enumerate(t_values, start=1):
        if verbose:
            print("\n" + "=" * 60)
            print(f"[MORNP Halving-t Iteration {it}] Trying t = {t} (n = {n})")
            print("=" * 60)

        t_solve_start = time.perf_counter()
        solution = _solve_at_t(instance, t, time_limit_sec=time_limit_sec)
        t_solve_end = time.perf_counter()
        solve_time = t_solve_end - t_solve_start

        if solution == "TIMEOUT":
            entry = {
                "iteration": it, "t": t, "is_sat": None, "timed_out": True,
                "solution": None, "solve_time_sec": solve_time,
            }
            iteration_log.append(entry)
            timed_out_entry = entry
            if verbose:
                print(f"TIMEOUT at t={t} (limit={time_limit_sec}s) -- stopping halving search here.")
            break

        entry = {
            "iteration": it, "t": t, "is_sat": solution is not None, "timed_out": False,
            "solution": solution, "solve_time_sec": solve_time,
        }
        iteration_log.append(entry)

        if verbose:
            print(f"SAT solver time: {solve_time:.6f} sec")
            if solution is None:
                print("NO: No objective subset satisfies the sample.")
            else:
                print("YES: Found consistent objective subset Ω")
                print("Selected objectives:", solution)

        if solution is None and last_yes is not None:
            first_no = {"t": t, "solve_time_sec": solve_time, "iteration": it}
            if verbose:
                print("\n--- Transition detected: YES -> NO ---")
                print(f"Sub-optimal (last YES) t = {last_yes['t']}")
            break

        if solution is not None:
            last_yes = {
                "t": t,
                "solution": solution,
                "solve_time_sec": solve_time,
                "iteration": it,
            }

    return {
        "n": n,
        "iterations": iteration_log,
        "last_yes": last_yes,
        "first_no": first_no,
        "t_schedule": t_values,
        "timed_out_entry": timed_out_entry,
    }


def solve_mornpdec_with_optimal_t(instance, *, verbose=True, time_limit_sec=None):
    """
    Find the *minimal* t that yields SAT (optimal t) for MORNP-DEC, using:
      1) halving schedule to find a YES/NO bracket
      2) binary search between first NO and last YES

    Stops when: (t_no + 1 == t_yes), i.e. consecutive NO then YES -- OR
    when any solve times out, in which case the search stops immediately
    and the returned "optimal" reflects the best CONFIRMED YES found so
    far, with "certified_minimal": False (since a timeout means we cannot
    rule out that some smaller t is also feasible).

    Returns a dict with full logs and the optimal solution -- structurally
    parallel to moppdec_sat_solver.solve_with_optimal_k's return shape
    (using "t" instead of "k", matching the paper's notation), plus the
    extra "certified_minimal" / "timed_out" bookkeeping MOPP's version
    doesn't currently expose.
    """
    halving_res = solve_mornpdec_with_halving_t(
        instance, verbose=verbose, time_limit_sec=time_limit_sec
    )

    last_yes = halving_res["last_yes"]
    first_no = halving_res["first_no"]
    halving_timed_out = halving_res["timed_out_entry"] is not None

    if last_yes is None:
        if verbose:
            if halving_timed_out:
                print("\nNo confirmed SAT solution found before timing out.")
            else:
                print("\nNo SAT solution found even with t = n.")
        return {
            "mode": "halving+binary",
            "halving": halving_res,
            "binary_iterations": [],
            "optimal": None,
            "certified_minimal": False,
            "timed_out": halving_timed_out,
        }

    # If halving itself timed out (rather than cleanly finding a NO), we
    # cannot safely binary-search further -- report the last confirmed YES
    # as-is, uncertified.
    if halving_timed_out:
        if verbose:
            print(f"\nHalving timed out after confirming YES at t={last_yes['t']}; "
                  f"reporting that as a sub-optimal (uncertified) result.")
        return {
            "mode": "halving+binary",
            "halving": halving_res,
            "binary_iterations": [],
            "optimal": {
                "t": last_yes["t"],
                "solution": last_yes["solution"],
                "solve_time_sec": last_yes["solve_time_sec"],
            },
            "certified_minimal": False,
            "timed_out": True,
        }

    if first_no is None:
        if verbose:
            print("\nHalving never produced NO. Refining by directly testing t=1.")
        t0 = time.perf_counter()
        sol1 = _solve_at_t(instance, 1, time_limit_sec=time_limit_sec)
        t1 = time.perf_counter()
        solve_time_1 = t1 - t0

        if sol1 == "TIMEOUT":
            if verbose:
                print(f"TIMEOUT at t=1; reporting last confirmed YES (t={last_yes['t']}) as uncertified.")
            return {
                "mode": "halving+binary",
                "halving": halving_res,
                "binary_iterations": [{
                    "t": 1, "is_sat": None, "timed_out": True,
                    "solution": None, "solve_time_sec": solve_time_1,
                }],
                "optimal": {
                    "t": last_yes["t"], "solution": last_yes["solution"],
                    "solve_time_sec": last_yes["solve_time_sec"],
                },
                "certified_minimal": False,
                "timed_out": True,
            }

        if sol1 is not None:
            if verbose:
                print("YES at t=1 => optimal t = 1")
            return {
                "mode": "halving+binary",
                "halving": halving_res,
                "binary_iterations": [{
                    "t": 1, "is_sat": True, "timed_out": False, "solution": sol1,
                    "solve_time_sec": solve_time_1,
                }],
                "optimal": {"t": 1, "solution": sol1, "solve_time_sec": solve_time_1},
                "certified_minimal": True,
                "timed_out": False,
            }
        else:
            if verbose:
                print("NO at t=1. Using bracket [t_no=1, t_yes=last_yes.t] for binary search.")
            t_no = 1
            t_yes = last_yes["t"]
            best_sol = last_yes["solution"]
            best_time = last_yes["solve_time_sec"]
            binary_log = []
    else:
        t_no = first_no["t"]
        t_yes = last_yes["t"]
        best_sol = last_yes["solution"]
        best_time = last_yes["solve_time_sec"]
        binary_log = []

    if verbose:
        print("\n" + "=" * 60)
        print(f"[MORNP Binary Search Refinement] Bracket: NO at t={t_no}, YES at t={t_yes}")
        print("=" * 60)

    certified_minimal = True
    binary_timed_out = False

    while (t_yes - t_no) > 1:
        mid = (t_yes + t_no) // 2

        if verbose:
            print("\n" + "-" * 60)
            print(f"Trying mid t = {mid} (current bracket: NO={t_no}, YES={t_yes})")
            print("-" * 60)

        t0 = time.perf_counter()
        sol_mid = _solve_at_t(instance, mid, time_limit_sec=time_limit_sec)
        t1 = time.perf_counter()
        solve_time = t1 - t0

        if sol_mid == "TIMEOUT":
            entry = {
                "t": mid, "is_sat": None, "timed_out": True,
                "solution": None, "solve_time_sec": solve_time,
            }
            binary_log.append(entry)
            certified_minimal = False
            binary_timed_out = True
            if verbose:
                print(f"TIMEOUT at t={mid} (limit={time_limit_sec}s) -- stopping binary search here.")
            break

        entry = {
            "t": mid, "is_sat": sol_mid is not None, "timed_out": False,
            "solution": sol_mid, "solve_time_sec": solve_time,
        }
        binary_log.append(entry)

        if verbose:
            print(f"SAT solver time: {solve_time:.6f} sec")
            if sol_mid is None:
                print("NO at t =", mid)
            else:
                print("YES at t =", mid)
                print("Selected objectives:", sol_mid)

        if sol_mid is not None:
            t_yes = mid
            best_sol = sol_mid
            best_time = solve_time
        else:
            t_no = mid

    optimal = {"t": t_yes, "solution": best_sol, "solve_time_sec": best_time}

    if verbose:
        print("\n" + "=" * 60)
        if binary_timed_out:
            print("[MORNP Search Stopped Early Due to Timeout]")
        else:
            print("[MORNP Optimal t Found]")
        print(f"NO at t = {t_no}")
        print(f"{'Best confirmed' if binary_timed_out else 'YES (optimal)'} at t = {t_yes}")
        print("Selected objectives:", best_sol)
        print("=" * 60)

    return {
        "mode": "halving+binary",
        "halving": halving_res,
        "binary_iterations": binary_log,
        "optimal": optimal,
        "certified_minimal": certified_minimal,
        "timed_out": binary_timed_out,
    }
