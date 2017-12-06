#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import namedtuple
from operator import attrgetter
Item = namedtuple("Item", ['index', 'value', 'weight','value_weight'])

import numpy as np
import pandas as pd
import math
import sys

def check_knapsack(capacity,
                  items,
                  taken,
                   nb_points):
    check = 0
    value = 0
    delta = float(capacity)/nb_points
    for item in items:
        if item.index in taken:
            check += math.ceil(item.weight/delta)*delta
            value += item.value

    return check < capacity,check,value

def dyn_prog_solve(
       capacity,
        items
):
    o_k_j = np.zeros((capacity+1,len(items)+1))
    for item, j in zip(items,range(1,len(items)+1)):
        for k in range(capacity+1):
            if item.weight <= k:
                o_k_j[k,j] = max(o_k_j[k,j-1],item.value+o_k_j[k-item.weight,j-1])
            else:
                o_k_j[k,j] = o_k_j[k,j-1]

    value = o_k_j[-1,-1]
    weight = 0
    taken = [0]*len(items)

    k = capacity
    for j in reversed(range(1,len(items)+1)):
        item = items[j-1]
        if o_k_j[k,j]<>o_k_j[k,j-1]:
            taken[j-1]=1
            weight+=item.weight
            k-=item.weight
        else:
            taken[j-1]=0

    return value,weight,taken

def dyn_prog_solve_v2(
    capacity,
    items,
    nb_points = -1,
    lb_flag=True):

    weights = [0]*len(items)
    values = [0]*len(items)

    if nb_points<=0:
        nb_points = capacity

    nb_points = min(nb_points,capacity)
    delta = float(capacity)/nb_points
    for item, idx in zip(items,range(len(items))):
        if lb_flag:
            weights[idx] = int(math.ceil(item.weight/delta))
        else:
            weights[idx] = int(math.floor(item.weight/delta))
        values[idx] = item.value

    o_k_j = np.zeros((nb_points+1,len(items)+1))
    for j in range(1,len(items)+1):
        item = items[j-1]
        for k in range(nb_points+1):
            if weights[j-1] <= k:
                o_k_j[k,j] = max(o_k_j[k,j-1],values[j-1]+o_k_j[k-weights[j-1],j-1])
            else:
                o_k_j[k,j] = o_k_j[k,j-1]

    value = o_k_j[-1,-1]
    weight = 0

    k = nb_points
    taken = []
    for j in reversed(range(1,len(items)+1)):
        item = items[j-1]
        if o_k_j[k,j]<>o_k_j[k,j-1]:
            taken+=[item.index]
            weight+=item.weight
            k-=weights[j-1]

    return int(value),int(weight),taken

def heuristic(capacity, items):
    check = 0
    res = 0
    for item in items:
        check += item.weight
        if check < capacity:
            res += item.value
        else:
            res = float(item.weight+check)/item.weight*item.value
            break
    return res

def banch_bound_depth_first(capacity, items, max_error, left_opt=0, current_val=0, error = 0, taken=[]):
    #sys.stdout.write(("\r left_opt : %7d current_val : %7d heuristic : %7d taken: %s"%(left_opt,
    #                                                                 current_val,
    #                                                                 current_val+heuristic(capacity,items),
    #                                                                 str(taken))).ljust(160))
    #sys.stdout.flush()
    if items<>[]:
        item = items[0]

        if (current_val+heuristic(capacity,items)<=left_opt) | (error > max_error):
            return 0, []

        if item.weight <= capacity:
            res_left, taken_left =  banch_bound_depth_first(capacity-item.weight,
                                                            items[1:],
                                                            max_error,
                                                            left_opt,
                                                            current_val+item.value,
                                                            error,
                                                            taken+[item.index])
            res_left+=item.value
            taken_left=[item.index]+taken_left
            if current_val+res_left < left_opt:
                error+=1
        else:
            res_left = 0
            taken_left = []

        res_right, taken_right =  banch_bound_depth_first(capacity,
                                                          items[1:],
                                                          max_error,
                                                          max(left_opt,current_val+res_left),
                                                          current_val,
                                                          error,
                                                          taken)

        if res_left>res_right:
            return res_left, taken_left
        else:
            return res_right, taken_right
    else:
        return 0, []

def branch_bound_depth_first_v2(capacity, items, max_error, error_index= []):
    if max_error < 0:
        return 0, 0, []

    total_weight = 0
    total_value = 0
    selected_index = []
    for item in items:
        if item.index not in error_index:
            if total_weight + item.weight < capacity:
                selected_index += [item.index]
                total_weight += item.weight
                total_value += item.value

    for idx in selected_index:
        total_value_2, total_weight_2, selected_index_2 = branch_bound_depth_first_v2(capacity,
                                                                      items,
                                                                      max_error-1,
                                                                      error_index+[idx])
        if total_value_2>total_value:
            total_value = total_value_2
            total_weight = total_weight_2
            selected_index = selected_index_2

    return total_value, total_weight, selected_index

def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    firstLine = lines[0].split()
    item_count = int(firstLine[0])
    capacity = int(firstLine[1])

    items = []

    for i in range(1, item_count+1):
        line = lines[i]
        parts = line.split()
        items.append(Item(i-1, int(parts[0]), int(parts[1]),float(parts[0])/float(parts[1])))

    # a trivial greedy algorithm for filling the knapsack
    # it takes items in-order until the knapsack is full
    #value = 0
    #weight = 0
    #taken = [0]*len(items)

    #for item in items:
    #    if weight + item.weight <= capacity:
    #        taken[item.index] = 1
    #        value += item.value
    #        weight += item.weight

    if len(items) in (30,50):
        value,weight,takens=dyn_prog_solve_v2(capacity,items,20000)
    elif len(items) == 200:
        value,weight,takens=dyn_prog_solve_v2(capacity,items,10000)
    else:
        value,weight,takens=branch_bound_depth_first_v2(
            capacity,
            sorted(items, key=attrgetter('value_weight'), reverse=True),
            1)

    taken = [1 if item.index in takens else 0 for item in items]
    
    # prepare the solution in the specified output format
    output_data = str(value) + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, taken))
    return output_data


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/ks_4_0)')

