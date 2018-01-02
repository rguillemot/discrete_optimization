#!/usr/bin/python
# -*- coding: utf-8 -*-

import math
from collections import namedtuple

import pandas as pd
import numpy as np
from datetime import datetime
from collections import deque

def load_data(filename):
    data = pd.read_csv(filename,sep=' ').reset_index()
    data = data.rename(columns={data.columns[0]:'x', data.columns[1]:'y'})
    return data

def dist2(p1,p2):
    return (p1['x']-p2['x'])**2 + (p1['y']-p2['y'])**2

def dist(p1,p2):
    return np.sqrt(dist2(p1,p2))

def closest_point(visited_point, data):
    p0 = data.loc[visited_point[-1]]
    return data.loc[~data.index.isin(visited_point)].apply(
        lambda p: dist2(p,p0),
        axis=1
    ).argmin()

def tsp_simple(data):
    sol = [data.index[0]]
    while len(sol) < len(data):
        sol += [closest_point(sol,data)]
    return sol

def tsp_simple_fast(data,closest_matrix):
    sol = [data.index[0]]
    while len(sol) < len(data):
        last = sol[-1]
        sol += [closest_matrix[last][~closest_matrix[last].isin(sol)].values[0]]
    return sol

def objective(sol, data):
    return sum([dist(data.loc[p1],data.loc[p2]) for p1, p2 in zip(sol,sol[1:]+[sol[0]])])

def objective2_vec(sol, data):
    return [dist2(data.loc[p1],data.loc[p2]) for p1, p2 in zip(sol,sol[1:]+[sol[0]])]

def objective2(sol, data):
    return sum(objective2_vec(sol, data))

def save_sol(sol, data, filename):
    obj = objective(sol,data)
    output_data = str(obj) + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, sol))
    f = open(filename.replace('data','out'),'w')
    f.write(output_data)
    f.close()

def shift(seq, n):
    return seq[n:]+seq[:n]

def swap_tsp_2(sol, p1, p2):
    pos1 = sol.index(p1)
    shift_sol = shift(sol,pos1)
    pos2 = shift_sol.index(p2)
    res_sol = shift_sol[:pos2]+shift_sol[pos2:][::-1]
    return shift(res_sol,len(sol)-pos1)

def objective_swap(sol, obj, p1 ,p2, data):
    pos1 = sol.index(p1)
    pos2 = sol.index(p2)
    p1_next = sol[pos1-1]
    p2_next = sol[pos2-1]
    dp1 = data.loc[p1]
    dp2 = data.loc[p2]
    dp1_next = data.loc[p1_next]
    dp2_next = data.loc[p2_next]
    return obj - dist2(dp1,dp1_next) - dist2(dp2,dp2_next) + dist2(dp1,dp2) + dist2(dp1_next,dp2_next)

def objective_vec_swap(sol, obj_vec, p1 ,p2, data):
    pos1 = sol.index(p1)
    shift_sol = shift(sol,pos1)
    shift_obj_vec = shift(obj_vec,pos1)
    pos2 = shift_sol.index(p2)
    res_obj_vec = shift_obj_vec[:pos2]+ (shift_obj_vec[pos2:][::-1][1:]+[0])
    p1_next = sol[pos1-1]
    p2_next = shift_sol[pos2-1]
    dp1 = data.loc[p1]
    dp2 = data.loc[p2]
    dp1_next = data.loc[p1_next]
    dp2_next = data.loc[p2_next]
    res_obj_vec[pos2-1]=dist2(dp1_next,dp2_next)
    res_obj_vec[-1]=dist2(dp1,dp2)
    return shift(res_obj_vec,len(sol)-pos1)

def tabu_print(cur_sol,
        cur_obj_vec,
        data,
        p1,
        p2,
        tabu_print_size):
    cur_obj_vec = objective_vec_swap(cur_sol,cur_obj_vec,p1,p2,data)
    return pd.Series(cur_obj_vec).sort_values(ascending=False).values[:tabu_print_size].tolist()

def tabu_check_func(
        cur_sol,
        cur_obj_vec,
        data,
        p1,
        p2,
        tabu_list,
        tabu_flag,
        tabu_print_size):

    tabu_print_check = tabu_print(cur_sol,cur_obj_vec,data,p1,p2,tabu_print_size)
    return (tabu_flag and tabu_print_check not in tabu_list)  or not tabu_flag

def local_search(
    data,
    max_trials,
    start_sol,
    start_obj_vec,
    closest_matrix,
    first_closest=None,
    tabu_flag = False,
    tabu_print_frac=1.0,
    tabu_size=1000):

    size = len(start_sol)
    tabu_print_size = int(tabu_print_frac*size)
    cur_sol = start_sol
    cur_obj_vec = start_obj_vec
    cur_obj = sum(cur_obj_vec)
    opt_sol = start_sol
    opt_obj_vec = cur_obj_vec
    opt_obj = cur_obj

    tabu_list = deque([])
    it = 0
    print 'it 0 obj: %f'%(sum(map(np.sqrt,cur_obj_vec)))
    while (it < max_trials):
        order_points = pd.Series(cur_obj_vec).sort_values(ascending=False).index

        neighbourhood = { (p1,p2) : objective_swap(cur_sol, cur_obj,p1,p2,data) \
                          for p1 in order_points[:first_closest] \
                          for p2 in closest_matrix[p1][1:first_closest]}

        it += len(neighbourhood)

        neighbourhood = { k : neighbourhood[k] for k in neighbourhood.keys()
                          if neighbourhood[k] <> cur_obj
                          if tabu_check_func(cur_sol, cur_obj_vec, data,
                                             k[0], k[1],
                                             tabu_list,tabu_flag,tabu_print_size)}

        p1,p2 = min(neighbourhood, key=lambda k: neighbourhood[k])

        tabu_list.appendleft(tabu_print(cur_sol,cur_obj_vec,data,p1,p2,tabu_print_size))
        while (len(tabu_list)>tabu_size):
            #print 'empty tabu list'
            tabu_list.pop()

        cur_obj_vec = objective_vec_swap(cur_sol,cur_obj_vec,p1,p2,data)
        cur_sol = swap_tsp_2(cur_sol,p1,p2)
        cur_obj = sum(cur_obj_vec)
        print 'it %d swap (%d %d) obj: %f'%(it,
                                            p1,p2,
                                            sum(map(np.sqrt,cur_obj_vec)))

        if cur_obj < opt_obj:
            opt_sol[:] = cur_sol
            opt_obj_vec[:] = cur_obj_vec
            opt_obj = cur_obj
            print 'it opt %d obj: %f'%(it,sum(map(np.sqrt,opt_obj_vec)))
    return sum(map(np.sqrt,opt_obj_vec)), opt_sol

def dist_matrix_calc(data):
    data_x = np.repeat(data['x'].values,len(data)).reshape((len(data),len(data)))
    data_y = np.repeat(data['y'].values,len(data)).reshape((len(data),len(data)))
    return pd.DataFrame((data_x-data_x.T)*(data_x-data_x.T)+(data_y-data_y.T)*(data_y-data_y.T))


def closest_matrix_calc_step(data, start_idx, end_idx, first_closest):
    data_x = data.x.values
    data_y = data.y.values
    data_x_1 = np.repeat(data_x,end_idx-start_idx).reshape(len(data),end_idx-start_idx)
    data_x_2 = np.repeat(data_x[start_idx:end_idx],len(data)).reshape((end_idx-start_idx,len(data))).T
    data_y_1 = np.repeat(data_y,end_idx-start_idx).reshape(len(data),end_idx-start_idx)
    data_y_2 = np.repeat(data_y[start_idx:end_idx],len(data)).reshape((end_idx-start_idx,len(data))).T
    dist_matrix = pd.DataFrame((data_x_2-data_x_1)*(data_x_2-data_x_1)
                               +(data_y_2-data_y_1)*(data_y_2-data_y_1),
                              columns=range(start_idx,end_idx))
    return pd.DataFrame([dist_matrix[c].nsmallest(first_closest).index for c in dist_matrix.columns],
                        index=range(start_idx, end_idx)).T

def closest_matrix_calc(data, first_closest, step=100):
    steps = range(0,len(data),step)+[len(data)]
    return pd.concat([closest_matrix_calc_step(data,start_idx,end_idx,first_closest)
               for (start_idx, end_idx) in zip(steps[:-1],steps[1:])],axis=1)


def solve_tsp(filename,max_trials,
              first_closest,
              tabu_flag = False,
              tabu_print_frac = 1.0,
              tabu_size = 1000,
              closest_matrix=None,
              calc_first_closest=1000,
              step=1000,
              simple_fast_flag = True):
    data = load_data(filename)
    start_time = datetime.now()
    if closest_matrix is None:
        closest_matrix = closest_matrix_calc(data,calc_first_closest,step)
    closest_matrix_time = datetime.now()
    print 'closest_matrix time : %d'%((closest_matrix_time-start_time).seconds)
    if simple_fast_flag:
        sol_step0 = tsp_simple_fast(data,closest_matrix)
    else:
        sol_step0 = tsp_simple(data)
    start_sol_time = datetime.now()
    print 'start sol time : %d'%((start_sol_time-closest_matrix_time).seconds)
    obj_vec = objective2_vec(sol_step0,data)
    opt_obj, opt_sol = local_search(data,max_trials,sol_step0,obj_vec,closest_matrix,first_closest,
                                    tabu_flag,
                                    tabu_print_frac,
                                    tabu_size)
    start_local_search_time = datetime.now()
    print 'local_search time : %d'%((start_local_search_time-start_sol_time).seconds)
    return opt_obj, opt_sol



def solve_it(input_file):
    # Modify this code to run your optimization algorithm

    f = open(input_file.replace('data','out'))

    return f.read()


import sys

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/tsp_51_1)')

