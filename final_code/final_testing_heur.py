import numpy as np
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import GRB
from util.distance import *
from data_loader import *
from itertools import permutations, combinations, product
from classes import Problem, Boat, Trip
from better_tsp import solve_tsp
import random
import math
import time
import glob
from neighbourhood_rules import *
from ant_colony_optimisation import *

# testing script for heuristic algorithms on test problems

# next_descent_swap, next_descent_swap_improved, next_descent_move, steepest_descent_move, next_descent_combined
for method in range(7, 10):
    probs = [f'{i}_0' for i in [34, 24, 12, '06']]
    for prob_id in probs:
    # for method in [5, 6, 7]: #[0, 1, 2, 3, 4]:
        # if method == 0 and prob_id in [35, 33, 30, 28, 27, 24, 23, 21, 18, 16, 15, 12, 10, 8, 7]:
        #     continue
        # 35 - 1 or 2
        filename = glob.glob(f"final_test_problems/test_problem{prob_id}*")[0]
        with open(filename, "r") as f:
            result = f.read()
        split = result.split('; ')
        s = np.array(eval(split[0]))
        p = np.array(eval(split[1]))
        b = int(split[2])
        c = np.array(eval(split[3]))
        t = int(split[4])
        w = int(split[5])
        # note that w is not used for the swap algorithms as swaps do not change the number of stations visited by each boat
        filename = glob.glob(f"better_test_problems/test_problem{prob_id[:2]}*")[0]
        with open(filename, "r") as f:
            result = f.read()
        split = result.split('; ')
        u = int(split[6])
        stations_idx = np.stack([s * 2, s * 2 + 1]).T.reshape(-1) + 13

        Prob = Problem(list(stations_idx), list(p), t, b, c, [p[0]]*b)
        # print(sum(Prob.catch[s]))
        k_catch= (400000-np.mean(c)) * 0.001
        k_fish = np.mean(c) / t * k_catch
        upper_bound = u
        min_obj = 10000
        count = 0
        time_list = []
        obj_list = []
        best_obj_list = []
        t0 = time.time()
        it_since_improvement = 0
        time_lim = 1
        while time.time() - t0 < time_lim and it_since_improvement < 5000:
            it_since_improvement += 1
            count += 1
            Prob.reset()
            # Prob.generate_initial_solution(seed=i)
            best_initial_cost = np.inf
            for k in range(1):
                Prob.reset()
                Prob.GRASP(rcl_size=2, seed=count*50+k)
                obj, pen = Prob.evaluate_solution(k_catch, k_fish, upper_bound)
                cost = obj if pen < 1e-8 else obj + pen + upper_bound
                if cost < best_initial_cost:
                    if k == 49 or max([len(boat.stations) for boat in Prob.boats]) / 2 <= w:
                        best_initial_cost = cost
                        best_sol = Prob.save_solution_as_list()

            if count == 1:
                best_overall_sol = best_sol

            Prob.restore_solution_from_list(best_sol)

            for boat in Prob.boats:
                boat.improve_route(Prob)
            for boat in Prob.boats:
                if len(boat.route) == 1:
                    boat.route.append(Trip(boat.home_port, [], boat.home_port))

            # k_catch/k_fish are the catch and fish time penalty coefficients, respectively
            # w is the work limit which is used to make sure that no boat visits too many of the stations (close to equal distribution)
            if method == 0:
                pass
                # old_obj, old_pen, old_cost = next_descent_swap(Prob, k_catch, k_fish, upper_bound, seed=count, test=False, tsp=0.02)
            elif method == 1:
                old_obj, old_pen, old_cost = next_descent_swap_improved(Prob, k_catch, k_fish, upper_bound, seed=None, test=False, tsp=0.02)
            elif method == 2:
                old_obj, old_pen, old_cost = next_descent_move(Prob, k_catch, k_fish, upper_bound, w, seed=count, tsp=0.02)
            elif method == 3:
                old_obj, old_pen, old_cost = steepest_descent_move(Prob, k_catch, k_fish, w, upper_bound, tsp=0.02)
            elif method == 4:
                old_obj, old_pen, old_cost = next_descent_combined(Prob, k_catch, k_fish, w, upper_bound, seed=count, tsp=0.02)
            elif method == 5:
                old_obj, old_pen, old_cost = steepest_descent_swap(Prob, k_catch, k_fish, upper_bound, tsp=0.02)
            elif method == 6:
                old_obj, old_pen, old_cost = tabu_search_move(Prob, k_catch, k_fish, upper_bound, w, history_size=len(Prob.stations)//10, max_iter=10, max_time=300, tsp=0.02)
            elif method == 7:
                old_obj, old_pen, old_cost = tabu_search_swap(Prob, k_catch, k_fish, upper_bound, w, history_size=len(Prob.stations)//10, max_iter=10, max_time=300, tsp=0.02)
            elif method == 8:
                old_obj, old_pen, old_cost = simulated_annealing(Prob, k_catch, k_fish, upper_bound, temp_init=50, alpha=0.99, stopping_temp=1e-10, max_iter=10, tsp=0.02, seed=count)
            elif method == 9:
                old_obj, old_pen, old_cost = tabu_search_combined(Prob, k_catch, k_fish, upper_bound, w, history_size=len(Prob.stations)//10, max_iter=10, max_time=300, tsp=0.02)
            elif method == 10:
                old_obj, old_pen, old_cost = ant_colony_optimization2(Prob, k_catch, k_fish, upper_bound, w, initial_iterations=1, num_ants=1, num_iterations=1, alpha=1, beta=2, rho=0.5, Q=100, seed=count)

            # actual_obj, actual_pen = Prob.evaluate_solution(k_catch, k_fish, upper_bound)
            # if abs(old_obj - actual_obj) > 1e-6 or abs(old_pen - actual_pen) > 1e-6:
            #     print(actual_obj, old_obj, actual_pen, old_pen)
            #     print(f'WARNING: ERROR WITH METHOD {method}')

            if old_pen > 1e-8:
                obj = old_cost
            else:
                    
                for boat in Prob.boats:
                    boat.improve_route(Prob)

                obj = sum([boat.total_time for boat in Prob.boats])
            
                if obj < min_obj:
                    valid = True
                    for boat in Prob.boats:
                        if len(boat.stations) / 2 > w:
                            valid = False
                            print(f'rejected because too many stations')
                            break
                    if valid:
                        it_since_improvement = 0
                        min_obj = obj
                        best_overall_sol = Prob.save_solution_as_list()
            
            time_list.append(time.time()-t0)
            best_obj_list.append(min_obj)
            obj_list.append(obj)
            
            print(f'finished iteration {count} for problem {prob_id} with method {method} giving obj {obj} fter {math.floor((time.time()-t0)/60)} minutes and {time.time()-t0-(math.floor((time.time()-t0)/60))*60:.00f} seconds, {it_since_improvement} iterations since last improvement, (best obj so far = {min_obj:.02f})')

        result = {
            "problem_id": prob_id,
            "method": method,
            "num_stations": len(stations_idx),
            "num_boats": b,
            "num_iter": count,
            "best_obj": min_obj,
            "best_solution": best_overall_sol,
            "solution_log": [(time_list[i], obj_list[i], best_obj_list[i]) for i in range(len(time_list))]
        }

        # with open('my_code/output_final9-15.txt', 'a') as f:
        #     f.write(str(result) + '\n')
        
