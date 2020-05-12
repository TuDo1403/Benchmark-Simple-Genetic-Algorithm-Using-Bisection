import POPOP as pp
import numpy as np
import fitness_function as ff
import pandas as pd


def simple_GA_10_runs(user_options, func_inf, seeds):
    results = [pp.POPOP(user_options, func_inf, seed) for seed in seeds]
    return np.array(results, dtype=np.int)
    

def find_upper_bound(user_options, func_inf, seeds):
    user_options.POP_SIZE = user_options.TOURNAMENT_SIZE   # define initial upper bound
    success = False
    while not success:
        user_options.POP_SIZE = user_options.POP_SIZE * 2
        results = simple_GA_10_runs(user_options, func_inf, seeds)
        success = np.sum(results[:, 0], axis=0) == len(seeds)
        if not success and user_options.POP_SIZE >= 8192:
            print("Upper bound not found!\n")
            return 0
        
    return user_options.POP_SIZE

def find_MRPS(upper_bound, user_options, func_inf, seeds):
    lower_bound = upper_bound // 2
    avg_number_fitness_eval = 0
    while (upper_bound - lower_bound) / upper_bound > 0.1:
        user_options.POP_SIZE = (upper_bound + lower_bound) // 2
        results = simple_GA_10_runs(user_options, func_inf, seeds)
        success = np.sum(results[:, 0], axis=0) == len(seeds)
        if success:
            upper_bound = user_options.POP_SIZE
        else:
            lower_bound = user_options.POP_SIZE
        if upper_bound - lower_bound <= 2:
            break
    
    user_options.POP_SIZE = upper_bound
    results = simple_GA_10_runs(user_options, func_inf, seeds)
    avg_number_fitness_eval = np.mean(results[:, 1], axis=0)

    return (upper_bound, avg_number_fitness_eval)

def bisection_10_runs(id, user_options, func_inf):
    results = []
    for i in range(10):
        seeds = [id + j for j in range(10 * i, 10 * (i+1))]
        mode = "1X" if user_options.CROSSOVER_MODE == pp.Crossover.ONEPOINT else "UX"
        with open('../report/problem_size={}_mode={}_func={}.txt'.format(user_options.PROBLEM_SIZE, mode, func_inf.NAME), 'a+') as f:
            upper_bound = find_upper_bound(user_options, func_inf, seeds)
            if upper_bound != 0:
                result = find_MRPS(upper_bound, user_options, func_inf, seeds)
                results.append(result)
                f.write("{}\t{}\t{}\n".format(result[0], result[1], seeds))
    
    return np.array(results)

def report_df(id, user_options, func_inf):
    problem_sizes = [10 * 2**i for i in range(5)]
    results = []
    for size in problem_sizes:
        user_options.PROBLEM_SIZE = size
        result = bisection_10_runs(id, user_options, func_inf)
        if len(result) != 0:
            results.append(np.mean(result, axis=0))
        else:
            print("Solution not found!\n")
            results.append(['-', '-'])

    table = pd.DataFrame(results, columns=['MRPS', 'Evaluations'], index=problem_sizes)
    return table

# ##
# id = 18521578
# user_options = pp.POPOPConfig
# user_options.TOURNAMENT_SIZE = 4
# user_options.CROSSOVER_MODE = pp.Crossover.UX
# func_inf = ff.FuncInf("One Max", ff.onemax)

# df = report_df(id, user_options, func_inf)
# df_name = 'sGA-UX-OneMax'
# df.to_csv('../report/{}.csv'.format(df_name))
# ##

# ##
# user_options.CROSSOVER_MODE = pp.Crossover.ONEPOINT
# func_inf = ff.FuncInf("One Max", ff.onemax)

# df = report_df(id, user_options, func_inf)
# df_name = 'sGA-1X-OneMax'
# df.to_csv('../report/{}.csv'.format(df_name))
# ##

# ##
# user_options.CROSSOVER_MODE = pp.Crossover.UX
# func_inf = ff.FuncInf("Trap Five", ff.trap_five)

# df = report_df(id, user_options, func_inf)
# df_name = 'sGA-UX-TrapFive'
# df.to_csv('../report/{}.csv'.format(df_name))
# ##


# ##
# user_options.CROSSOVER_MODE = pp.Crossover.ONEPOINT
# func_inf = ff.FuncInf("Trap Five", ff.trap_five)

# df = report_df(id, user_options, func_inf)
# df_name = 'sGA-1X-TrapFive'
# df.to_csv('../report/{}.csv'.format(df_name))
# ##


