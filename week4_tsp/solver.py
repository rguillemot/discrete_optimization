#!/usr/bin/python
# -*- coding: utf-8 -*-

import math
from collections import namedtuple

import pandas as pd
import numpy as np

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

def local_search(
    data,
    max_trials,
    start_sol,
    start_obj_vec,
    closest_matrix,
    first_closest=None):

    size = len(start_sol)
    cur_sol = start_sol
    cur_obj_vec = start_obj_vec
    cur_obj = sum(cur_obj_vec)
    opt_sol = start_sol
    opt_obj_vec = cur_obj_vec
    opt_obj = cur_obj

    it = 0
    while (it < max_trials):
        order_points = pd.Series(cur_obj_vec).sort_values(ascending=False).index
        found = False
        max_trials_reached = False
        for p1 in order_points:
            for idx2 in closest_matrix[p1].index[1:first_closest if first_closest is not None else size]:
                if it >= max_trials:
                    max_trials_reached = True
                    break
                else:
                    it+=1
                p2 = closest_matrix[p1][idx2]
                #print 'try (%d %d)'%(p1,p2)
                new_obj = objective_swap(cur_sol, cur_obj, p1, p2, data)
                if new_obj < opt_obj:
                    cur_obj_vec = objective_vec_swap(cur_sol,cur_obj_vec,p1,p2,data)
                    cur_obj = sum(cur_obj_vec)
                    cur_sol = swap_tsp_2(cur_sol,p1,p2)
                    print 'order %d closest rank %d swap (%d %d) obj: %f'%(order_points.get_loc(p1),idx2, p1,p2,sum(map(np.sqrt,cur_obj_vec)))
                    opt_sol = cur_sol
                    opt_obj_vec = cur_obj_vec
                    opt_obj = sum(cur_obj_vec)
                    found = True
                    break
            if found or max_trials_reached:
                break
    return sum(map(np.sqrt,opt_obj_vec)), opt_sol

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

