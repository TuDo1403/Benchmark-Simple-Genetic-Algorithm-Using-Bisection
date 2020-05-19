import POPOP as pp
import numpy as np
import fitness_function as ff
import pandas as pd
import ECGA as ec


def simple_GA_10_runs(opt_method, user_options, func_inf, seeds):
    results = []
    for seed in seeds:
        result = opt_method(user_options, func_inf, seed)
        if result[0] == False:
            return np.zeros((10, 2))
        results.append(result)
    return np.array(results, dtype=np.int)
    

def find_upper_bound(opt_method, user_options, func_inf, seeds):
    user_options.POP_SIZE = user_options.TOURNAMENT_SIZE   # define initial upper bound
    success = False
    while not success:
        user_options.POP_SIZE = user_options.POP_SIZE * 2
        results = simple_GA_10_runs(opt_method, user_options, func_inf, seeds)
        success = np.sum(results[:, 0], axis=0) == len(seeds)
        if not success and user_options.POP_SIZE >= 8192:
            return 0
        
    return user_options.POP_SIZE

def find_MRPS(upper_bound, opt_method, user_options, func_inf, seeds):
    lower_bound = upper_bound // 2
    avg_number_fitness_eval = 0
    while (upper_bound - lower_bound) / upper_bound > 0.1:
        user_options.POP_SIZE = (upper_bound + lower_bound) // 2
        results = simple_GA_10_runs(opt_method, user_options, func_inf, seeds)
        success = np.sum(results[:, 0], axis=0) == len(seeds)
        if success:
            upper_bound = user_options.POP_SIZE
            avg_number_fitness_eval = np.mean(results[:, 1], axis=0)
        else:
            lower_bound = user_options.POP_SIZE
        if upper_bound - lower_bound <= 2:
            break
    
    if avg_number_fitness_eval == 0:
        user_options.POP_SIZE = upper_bound
        results = simple_GA_10_runs(opt_method, user_options, func_inf, seeds)
        avg_number_fitness_eval = np.mean(results[:, 1], axis=0)

    return (upper_bound, avg_number_fitness_eval)

def bisection_10_runs(id, opt_method, user_options, func_inf):
    results = []
    for i in range(10):
        seeds = [id + j for j in range(10 * i, 10 * (i+1))]
        filename = ''
        if user_options.NAME == 'POPOP':
            mode = "1X" if user_options.CROSSOVER_MODE == pp.Crossover.ONEPOINT else "UX"
            filename = '../report/problem_size={}_mode{}_func={}.txt'.format(user_options.PROBLEM_SIZE, mode, func_inf.NAME)
        else:
            filename = '../report/problem_size={}_func={}.txt'.format(user_options.PROBLEM_SIZE, func_inf.NAME)
        with open(filename, 'a+') as f:
            upper_bound = find_upper_bound(opt_method, user_options, func_inf, seeds)
            if upper_bound != 0:
                result = find_MRPS(upper_bound, opt_method, user_options, func_inf, seeds)
                results.append(result)
                f.write("{}\t{}\t{}\n".format(result[0], result[1], seeds))
    
    return np.array(results)

def report_df(id, opt_method, user_options, func_inf):
    problem_sizes = [10 * 2**i for i in range(0, 5)]
    results = []
    for size in problem_sizes:
        user_options.PROBLEM_SIZE = size
        result = bisection_10_runs(id, opt_method, user_options, func_inf)
        if len(result) != 0:
            result = np.hstack((np.mean(result, axis=0), np.std(result, axis=0)))
            results.append(result)
        else:
            results.append(['-', '-', '-', '-'])

    table = pd.DataFrame(results, columns=['MRPS', 'Evaluations', 'MRPS std', 'Evaluations std'], index=problem_sizes)
    return table

##
id = 18521578
user_options = pp.POPOPConfig
user_options.TOURNAMENT_SIZE = 4
# user_options.CROSSOVER_MODE = pp.Crossover.UX
# func_inf = ff.FuncInf("One Max", ff.onemax)

# df = report_df(id, pp.POPOP, user_options, func_inf)
# df_name = 'sGA-UX-OneMax'
# df.to_csv('../report/{}.csv'.format(df_name))
##

##
user_options.CROSSOVER_MODE = pp.Crossover.ONEPOINT
func_inf = ff.FuncInf("One Max", ff.onemax)

df = report_df(id, pp.POPOP, user_options, func_inf)
df_name = 'sGA-1X-OneMax'
df.to_csv('../report/{}.csv'.format(df_name))
##

##
# user_options.CROSSOVER_MODE = pp.Crossover.UX
# func_inf = ff.FuncInf("Trap Five", ff.trap_five)

# df = report_df(id, pp.POPOP, user_options, func_inf)
# df_name = 'sGA-UX-TrapFive'
# df.to_csv('../report/{}.csv'.format(df_name))
##


##
user_options.CROSSOVER_MODE = pp.Crossover.ONEPOINT
func_inf = ff.FuncInf("Trap Five", ff.trap_five)

df = report_df(id, pp.POPOP, user_options, func_inf)
df_name = 'sGA-1X-TrapFive'
df.to_csv('../report/{}.csv'.format(df_name))
##

# ##
# id = 18521578
# user_options = ec.ECGAConfig
# user_options.TOURNAMENT_SIZE = 4
# func_inf = ff.FuncInf('One Max', ff.onemax)

# df = report_df(id, ec.ECGA, user_options, func_inf)
# df_name = 'ECGA-OneMax'
# df.to_csv('../report/{}.csv'.format(df_name))
# ##

# ##
# func_inf = ff.FuncInf('Trap Five', ff.trap_five)

# df = report_df(id, ec.ECGA, user_options, func_inf)
# df_name = 'ECGA-TrapFive'
# df.to_csv('../report/{}.csv'.format(df_name))
# ##
