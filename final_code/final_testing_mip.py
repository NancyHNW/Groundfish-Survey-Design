from formulation_tester_functions import *
from classes import *
import matplotlib.pyplot as plt
import glob

# same as mip testing but trying to focus on just finding feasible solutions

prob_id = 35
filename = glob.glob(f"final_code/better_test_problems/test_problem{prob_id:02d}*")[0]
with open(filename, "r") as f:
    result = f.read()
split = result.split('; ')
s = np.array(eval(split[0]))
p = np.array(eval(split[1]))
b = int(split[2])
c = np.array(eval(split[3]))
t = int(split[4])
w = int(split[5])

stations_ids, port_idx, n_boats, catch_capacity, dist_limit, max_work = s, p, b, c, t, w
print(catch_capacity)
stations_ids, port_idx, n_boats, catch_capacity, dist_limit, max_work = get_test_problem(prob_id)
n_dummy_port = 1
work_limit = 600
island_data, nodes, n_ports, n_stations, dist, dummy_start, catch, index = get_problem_data()
print(dummy_start)
dist_orig = dist.copy()
node_compress = True
# percentage of arcs to keep from the network
perc_to_keep = 10
n_closest_ports = 2
dist, nodes_idx, ports_idx, stations_idx, dummy_idx, station_idx, node_idx = get_distance_matrix(dist_orig, port_idx, stations_ids, n_ports, n_dummy_port, dummy_start)
valid_pairs = get_valid_pairs(dist, perc_to_keep, stations_idx, ports_idx, dummy_idx, n_closest_ports, n_dummy_port)

m = gp.Model()
m, Xs = create_model3(m, valid_pairs, dist, nodes_idx, catch, ports_idx, stations_idx, n_boats, dummy_idx, station_idx, node_idx, max_work, n_dummy_port, node_compress, catch_capacity, dist_limit)    


# def my_callback(m, where):
#     if where == GRB.Callback.MIPSOL:
#         obj = m.cbGet(GRB.Callback.MIPSOL_OBJ)
#         bound = m.cbGet(GRB.Callback.MIPSOL_OBJBND)
#         time = m.cbGet(GRB.Callback.RUNTIME)
#         solution_log.append((time, obj, bound))

def my_callback(m, where):
    # when a solution is found, store the current values
    if where == GRB.Callback.MIPSOL:
        obj = m.cbGet(GRB.Callback.MIPSOL_OBJ)
        bound = m.cbGet(GRB.Callback.MIPSOL_OBJBND)
        run_time = m.cbGet(GRB.Callback.RUNTIME)
        solution_log.append((run_time, obj, bound))
    # even if no new solution is found, store the current values
    elif where == GRB.Callback.MIP:
        obj = m.cbGet(GRB.Callback.MIP_OBJBST)
        bound = m.cbGet(GRB.Callback.MIP_OBJBND)
        run_time = m.cbGet(GRB.Callback.RUNTIME)

        # store previous best bound and only store new info if bound has been improved
        if not hasattr(m, "_last_bound"):
            m._last_bound = None
        if m._last_bound != bound:
            solution_log.append((run_time, obj, bound))
            m._last_bound = bound
    if where == GRB.Callback.MIPSOL:
        # A new incumbent was found
        last_incumbent_time[0] = time.time()
        m._found_sol = True

    elif where == GRB.Callback.MIPNODE:
        # Check how long since last improvement
        elapsed = time.time() - last_incumbent_time[0]
        if elapsed > max_no_improve_time and m._found_sol is True:
            print(f"Stopping early: no new incumbent in {elapsed:.1f} seconds.")
            m.terminate()

# problems = [15, 31, 29, 28, 14, 13, 12, 11, 10, 9, 8, 7, 6]
# problems = [0]

results = []

max_no_improve_time = 20*60  # stop if 5 minutes pass without a new incumbent

for node_compress in [True]:
    # probs = [35, 33, 30, 28, 27, 24, 23, 21, 18, 16, 15, 12, 10, 8, 7, 2, 0]
    # probs = [16, 15, 12, 10, 8, 7, 2, 0]
    # # for prob_id in problems:
    # for prob_id in probs:
    probs = []
    for pr in ['12']: #, '06', '00']:
        probs += [pr + "_" + str(i) for i in range(10)]
    for prob_id in probs:
        print(f'solving problem {prob_id}')
        filename = glob.glob(f"my_code/final_test_problems/test_problem{prob_id}*")[0]
        with open(filename, "r") as f:
            result = f.read()
        split = result.split('; ')
        s = np.array(eval(split[0]))
        p = np.array(eval(split[1]))
        b = int(split[2])
        c = np.array(eval(split[3]))
        t = int(split[4])
        w = int(split[5])
        filename = glob.glob(f"my_code/better_test_problems/test_problem{prob_id[:2]}*")[0]
        with open(filename, "r") as f:
            result = f.read()
        split = result.split('; ')
        u = int(split[6])
        stations_idx = np.stack([s * 2, s * 2 + 1]).T.reshape(-1) + 13
        # stations_ids, port_idx, n_boats, catch_capacity, dist_limit, max_work, upper_bound = get_test_problem(prob_id)
        stations_ids, port_idx, n_boats, catch_capacity, dist_limit, max_work, upper_bound = s, p, b, c, t, w, u
        # work_limit = 600
        island_data, nodes, n_ports, n_stations, dist, dummy_start, catch, index = get_problem_data()
        dist_orig = dist.copy()
        # if len(stations_ids) > 200:
        #     n_closest_ports = 100
        # else:
        #     n_closest_ports = 2
        n_closest_ports = 2

        n_dummy_port = int(math.ceil(sum(catch[stations_ids]) / np.mean(catch_capacity) / len(port_idx)))+1

        # perc_list = [100, 75, 50, 25, 10, 5, 1]
        perc_list = [20]
        # if prob_id == 15:
        #     perc_list = [1]
        for perc_to_keep in perc_list:
            dist, nodes_idx, ports_idx, stations_idx, dummy_idx, station_idx, node_idx = get_distance_matrix(dist_orig, port_idx, stations_ids, n_ports, n_dummy_port, dummy_start)
            valid_pairs = get_valid_pairs(dist, perc_to_keep, stations_idx, ports_idx, dummy_idx, n_closest_ports, n_dummy_port)

            # Store last improvement time
            last_incumbent_time = [time.time()]
            
            m = gp.Model()
            m._found_sol = False
            # m, Xs = create_model2(m, valid_pairs, dist, nodes_idx, catch, ports_idx, stations_idx, n_boats, dummy_idx, station_idx, node_idx, max_work, n_dummy_port, node_compress, catch_capacity, dist_limit)    
            m, Xs = create_model3(m, valid_pairs, dist, nodes_idx, catch, ports_idx, stations_idx, n_boats, dummy_idx, station_idx, node_idx, max_work, n_dummy_port, node_compress, catch_capacity, dist_limit)    
            # m.Params.WorkLimit = work_limit
            # m.setParam('TimeLimit', 20*60)
            m.setParam('MIPFocus', 1) # focus on finding incumbent sols
            m.setParam('MIPGap', 0.05)
            # m.setParam('TimeLimit', 1200)
            m.setParam('SolutionLimit', 1)
        
            infeasible = True
            
            while infeasible:
                solution_log = []

                m.reset()
                presolved = m.presolve()
                num_bin_vars = presolved.NumBinVars
                num_int_vars = presolved.NumIntVars - presolved.NumBinVars
                num_cont_vars = presolved.NumVars - presolved.NumIntVars
                num_constraints = presolved.NumConstrs
                print(f' problem_id: {prob_id}, num_stations: {len(stations_ids)}, num_boats": {n_boats}, percentage_kep": {perc_to_keep}, num_dummy_ports: {n_dummy_port}')
                # m.optimize(my_callback)
                m.optimize()
                # if model infeasible, need more dummy ports
                if m.Status == gp.GRB.INFEASIBLE:
                    n_dummy_port += 1
                    dist, nodes_idx, ports_idx, stations_idx, dummy_idx, station_idx, node_idx = get_distance_matrix(dist_orig, port_idx, stations_ids, n_ports, n_dummy_port, dummy_start)

                    # Store last improvement time
                    last_incumbent_time = [time.time()]
                    
                    m = gp.Model()
                    m._found_sol = False
                    # valid_pairs = get_valid_pairs(dist, perc_to_keep, stations_idx, ports_idx, dummy_idx)
                    valid_pairs = get_valid_pairs(dist, perc_to_keep, stations_idx, ports_idx, dummy_idx, n_closest_ports, n_dummy_port) 
                    # m, Xs = create_model2(m, valid_pairs, dist, nodes_idx, catch, ports_idx, stations_idx, n_boats, dummy_idx, station_idx, node_idx, max_work, n_dummy_port, node_compress, catch_capacity, dist_limit)   
                    m, Xs = create_model3(m, valid_pairs, dist, nodes_idx, catch, ports_idx, stations_idx, n_boats, dummy_idx, station_idx, node_idx, max_work, n_dummy_port, node_compress, catch_capacity, dist_limit)     
                    # m, Xs, Ys, Zs, Bs, Cs, Ds, Es, Ps, Ts, dist_dict = create_model(m, dist, nodes_idx, catch, ports_idx, stations_idx, n_boats, dummy_idx, station_idx, node_idx, max_work, n_dummy_port, node_compress, catch_capacity, dist_limit)    
                    # m.Params.WorkLimit = work_limit
                    # m.setParam('TimeLimit', 20*60)
                    m.setParam('MIPFocus', 1)
                    m.setParam('MIPGap', 0.05)
                    # m.setParam('TimeLimit', 1200)
                    m.setParam('SolutionLimit', 1)
                    # if n_closest is not None:
                    #     m = keep_closest_arcs(m, Xs, Ys, Zs, Bs, Cs, Ds, Es, Ps, Ts, dist, n_closest, stations_idx)
                    continue
                infeasible = False

            best_obj = m.ObjVal if m.Status == gp.GRB.OPTIMAL or m.SolCount > 0 else None
            best_bound = m.ObjBound
            work_units = m.Work
            run_time = m.RunTime
            # num_bin_vars = sum(1 for v in m.getVars() if v.VType == gp.GRB.BINARY)
            # num_cont_vars = sum(1 for v in m.getVars() if v.VType == gp.GRB.CONTINUOUS)
            # num_bin_vars = m.numBinVars
            # num_cont_vars = m.numVars - m.numBinVars
            # num_constraints = m.NumConstrs

            if m.SolCount > 0:
                vals = m.getAttr('X', Xs)
                vals = [k for (k,v) in vals.items() if v > .5]
                vals = [(int(a), int(b)) for a, b in vals]
            else:
                vals = None

            result = {
                "problem_id": prob_id,
                "best_obj": best_obj,
                "best_bound": best_bound,
                "num_stations": len(stations_ids),
                "max_work": max_work,
                "num_boats": n_boats,
                # "n_closest": n_closest,
                "percentage_kept": perc_to_keep,
                "num_dummy_ports": n_dummy_port,
                "node_compression": node_compress,
                "num_continuous_vars": num_cont_vars,
                "num_integer_vars": num_int_vars,
                "num_binary_vars": num_bin_vars,
                "num_constraints": num_constraints,
                "work_units_used": work_units,
                "runtime": run_time,
                "best_solution": vals,
                # "solution_log": solution_log
            }

            with open('my_code/output_final_mip2.txt', 'a') as f:
                f.write(str(result) + '\n')

            
            