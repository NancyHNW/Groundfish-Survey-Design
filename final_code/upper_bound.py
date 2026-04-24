from formulation_tester_functions import get_test_problem
from classes import *
import glob

# generate upper bounds for test problems by running initialisation algos many times

for i in [32, 33]:
    # 33, 32
    # s, p, b, c, t, w = get_test_problem(i)
    filename = glob.glob(f"my_code/better_test_problems/test_problem{i:02d}*")[0]
    with open(filename, "r") as f:
        result = f.read()

    # print(result)
    split = result.split('; ')
    s = np.array(eval(split[0]))
    p = np.array(eval(split[1]))
    b = int(split[2])
    c = np.array(eval(split[3]))
    t = int(split[4])
    w = int(split[5])
    stations_idx = list(np.stack([s * 2, s * 2 + 1]).T.reshape(-1) + 13)
    Prob = Problem(stations_idx, p, t, b, c, [p[0]]*b)
    
    iterations = 1000

    if b == 1:
        # Prob.plot_all_routes()
        # plt.show()
        Prob.generate_initial_solution(seed=0)
        max_obj = sum([boat.total_time for boat in Prob.boats])
        counter = 0
        for j in range(iterations):
            if j % 100 == 0:
                print(f'{i}: {j}')
            Prob.reset()
            Prob.GRASP(rcl_size=1, seed=j)
            obj, pen = Prob.evaluate_solution(100, 100, 1000)
            if pen < 1e-6:
                counter += 1
                max_obj = max(max_obj, obj)
        
    else:
        # Prob.plot_all_routes()
        # plt.show()
        max_obj = 0
        for j in range(iterations):
            if j % 100 == 0:
                print(f'{i}: {j}')
            Prob.reset()
            
            Prob.generate_initial_solution(seed=j)
            # Prob.plot_all_routes()
            # plt.show()
            obj = sum([boat.total_time for boat in Prob.boats])
            max_obj = max(max_obj, obj)

    print(max_obj)
    filename = glob.glob(f"my_code/better_test_problems/test_problem{i:02d}*")[0]
    with open(filename, "a") as f:
        f.write(f'; {max_obj:.0f}')

# [141, 139, 10, 12, 60, 11, 61, 13, 72, 68, 142, 74, 65, 71, 67, 63, 140, 144, 62, 9]; [12]; 1; [10000]; 30; 20