#!/usr/bin/python
# -*- coding: utf-8 -*-

import math
from collections import namedtuple

import pandas as pd
import numpy as np
from scipy.spatial import KDTree
from datetime import datetime
from collections import deque
from tqdm import tqdm_notebook, tqdm
import matplotlib.pyplot as plt

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

def tsp_simple_fast_kd(data, step=10):
    sol = [data.index[0]]
    remainings = set(range(1,len(data)))
    kd = KDTree(data.loc[1:].values)
    kd_index = np.array(list(remainings))
    bar = tqdm_notebook(total=len(data)-1,desc='first sol')
    idx = 0
    while len(sol) < len(data):
        last = sol[-1]
        not_found = True
        while not_found:
            closest = kd_index[kd.query(data.loc[last].values,min(step, len(kd_index)))[1]]
            closest_remaining = filter(lambda e: e in remainings, closest)
            if len(closest_remaining) > 0:
                not_found = False
            else:
                kd = KDTree(data.loc[list(remainings)].values)
                kd_index = np.array(list(remainings))

        bar.update(1)
        idx = idx + 1
        sol += [closest_remaining[0]]

        remainings -= set([sol[-1]])
    return sol

def plot_sol(data,sol):
    plt.plot(data.loc[sol].x,data.loc[sol].y)
    plt.scatter(data.loc[sol].x,data.loc[sol].y,color='r',s=5)

def objective(sol, data):
    return sum([dist(data.loc[p1],data.loc[p2]) for p1, p2 in zip(sol,sol[1:]+[sol[0]])])

def objective2_vec(sol, data):
    return [dist2(data.loc[p1],data.loc[p2]) for p1, p2 in zip(sol,sol[1:]+[sol[0]])]

def objective2(sol, data):
    return sum(objective2_vec(sol, data))

def objective2_vec_fast(sol, data):
    return map(lambda x: x[0]**2+x[1]**2,data.loc[sol].values-data.loc[shift(sol,1)].values)

def objective_fast(sol, data):
    return sum(np.sqrt(objective2_vec_fast(sol, data)))

def save_sol(sol, data, filename,suffix):
    obj = objective(sol,data)
    output_data = str(obj) + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, sol))
    f = open(filename.replace('data','out')+suffix,'w')
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

def local_search_greedy(
    data,
    max_trials,
    start_sol,
    start_obj_vec,
    closest_matrix,
    first_points=None,
    first_closest=None,
    tabu_flag = False,
    tabu_print_size=None,
    tabu_size=None):

    size = len(start_sol)
    cur_sol = start_sol
    cur_obj_vec = start_obj_vec
    cur_obj = sum(cur_obj_vec)
    opt_sol = start_sol
    opt_obj_vec = cur_obj_vec
    opt_obj = cur_obj

    bar = tqdm_notebook(total=max_trials,desc='ls greedy')

    tabu_list = deque([])
    it = 0
    #print 'it 0 obj: %f'%(sum(map(np.sqrt,cur_obj_vec)))
    while (it < max_trials):
        order_points = pd.Series(cur_obj_vec).sort_values(ascending=False).index
        max_trials_reached = False
        local_opt_sol = None
        local_opt_obj_vec = None
        local_opt_obj = None
        for p1 in order_points[:first_points if first_points is not None else size]:
            for idx2 in closest_matrix[p1].index[1:first_closest if first_closest is not None else size]:
                if it >= max_trials:
                    max_trials_reached = True
                    break
                else:
                    it+=1
                    bar.update(1)
                p2 = closest_matrix[p1][idx2]
                #print 'try (%d %d)'%(p1,p2)
                new_obj = objective_swap(cur_sol, cur_obj, p1, p2, data)
                if (local_opt_sol is None) or (new_obj < local_opt_obj):
                    tmp_obj_vec = objective_vec_swap(cur_sol,cur_obj_vec,p1,p2,data)
                    tabu_check = pd.Series(tmp_obj_vec).sort_values(ascending=False).values[:tabu_print_size if tabu_print_size is not None else size].tolist()
                    if (tabu_flag and tabu_check not in tabu_list)  or \
                    not tabu_flag:
                        #if local_opt_sol is None:
                        #    print 'it %d init order %d closest rank %d swap (%d %d) obj: %f'%(it, order_points.get_loc(p1),idx2, p1,p2,sum(map(np.sqrt,tmp_obj_vec)))
                        #else:
                        #    print 'it %d order %d closest rank %d swap (%d %d) obj: %f'%(it, order_points.get_loc(p1),idx2, p1,p2,sum(map(np.sqrt,tmp_obj_vec)))
                        local_opt_sol = swap_tsp_2(cur_sol,p1,p2)
                        local_opt_obj_vec = tmp_obj_vec
                        local_opt_obj = new_obj
                        tabu_list.appendleft([pd.Series(local_opt_obj_vec).sort_values(ascending=False).values[:tabu_size].tolist()])
            if max_trials_reached:
                break
        while (len(tabu_list)>tabu_size if tabu_size is not None else 100):
            #print 'empty tabu list'
            tabu_list.pop()
        cur_sol = local_opt_sol
        cur_obj_vec = local_opt_obj_vec
        cur_obj = local_opt_obj
        if local_opt_obj < opt_obj:
            opt_sol[:] = local_opt_sol
            opt_obj_vec[:] = local_opt_obj_vec
            opt_obj = local_opt_obj
            bar.set_description('obj: %f'%(sum(map(np.sqrt,opt_obj_vec))))
            #print 'it %d obj: %f'%(it,sum(map(np.sqrt,opt_obj_vec)))
    return sum(map(np.sqrt,opt_obj_vec)), opt_sol

def local_search_tabu(
    data,
    max_trials,
    start_sol,
    start_obj_vec,
    closest_matrix,
    first_points=None,
    first_closest=None,
    tabu_flag = False,
    tabu_print_size=100,
    tabu_size=1000):

    size = len(start_sol)
    cur_sol = start_sol
    cur_obj_vec = start_obj_vec
    cur_obj = sum(cur_obj_vec)
    opt_sol = start_sol
    opt_obj_vec = cur_obj_vec
    opt_obj = cur_obj

    bar = tqdm_notebook(total=max_trials,desc='ls tabu')

    tabu_list = deque([])
    it = 0
    #print 'it 0 obj: %f'%(sum(map(np.sqrt,cur_obj_vec)))
    while (it < max_trials):
        order_points = pd.Series(cur_obj_vec).sort_values(ascending=False).index

        neighbourhood = { (p1,p2) : objective_swap(cur_sol, cur_obj,p1,p2,data) \
                          for p1 in order_points[:first_points if first_points is not None else size] \
                          for p2 in closest_matrix[p1][1:first_closest if first_closest is not None else size]}

        it += len(neighbourhood)

        bar.set_description('obj: %f'%(sum(map(np.sqrt,cur_obj_vec))))
        bar.update(len(neighbourhood))


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
        #print 'it %d swap (%d %d) obj: %f'%(it,
        #                                    p1,p2,
        #                                    sum(map(np.sqrt,cur_obj_vec)))

        if cur_obj < opt_obj:
            opt_sol[:] = cur_sol
            opt_obj_vec[:] = cur_obj_vec
            opt_obj = cur_obj
            #print 'it opt %d obj: %f'%(it,sum(map(np.sqrt,opt_obj_vec)))
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

def closest_matrix_calc_kd(data,first_closest):
    kd = KDTree(data.values)
    return pd.DataFrame([kd.query(data.loc[idx].values,first_closest)[1] for idx in tqdm_notebook(xrange(len(data)),
                                                                                                  desc='closest matrix')]).T

def solve_tsp(filename,
              max_trials,
              first_points,
              first_closest,
              local_search_type,
              tabu_flag = False,
              tabu_print_size = 100,
              tabu_size = 1000):
    data = load_data(filename)
    size = len(data)
    closest_matrix = closest_matrix_calc_kd(data,first_closest)
    sol_step0 = tsp_simple_fast_kd(data,10)
    save_sol(sol_step0,data,filename,'_step0')
    obj_vec = objective2_vec_fast(sol_step0,data)
    #tqdm.write('sol step 0 : %f'%(np.sum(np.sqrt(obj_vec))))
    if local_search_type == 'greedy':
        opt_obj, opt_sol = local_search_greedy(
            data,
            max_trials,
            sol_step0[:],
            obj_vec,
            closest_matrix,
            first_points,
            first_closest,
            tabu_flag,
            tabu_print_size,
            tabu_size)
    else:
        opt_obj, opt_sol = local_search_tabu(
            data,
            max_trials,
            sol_step0[:],
            obj_vec,
            closest_matrix,
            first_points,
            first_closest,
            tabu_flag,
            tabu_print_size,
            tabu_size)

    save_sol(sol_step0,data,filename,'_opt')
    return data, sol_step0, opt_obj, opt_sol

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

