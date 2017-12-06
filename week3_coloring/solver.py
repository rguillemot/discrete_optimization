#!/usr/bin/python
# -*- coding: utf-8 -*-

def graph_coloring_v1(nodes, edges, visited_nodes = [], rank = 0):
    if nodes == []:
        return visited_nodes
    if visited_nodes == []:
        visited_nodes = [-1]*len(nodes)
    adj_nodes = []
    for n in nodes:
        if visited_nodes[n] == -1:
            visited_nodes[n]=rank
        for e in edges:
            if (e[0] == n) and  (visited_nodes[e[1]] == -1):
                visited_nodes[e[1]]=rank+1
                adj_nodes += [e[1]]
        visited_nodes = graph_coloring_v1(adj_nodes, edges, visited_nodes, rank + 1)

    return visited_nodes





def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    first_line = lines[0].split()
    node_count = int(first_line[0])
    edge_count = int(first_line[1])

    edges = []
    for i in range(1, edge_count + 1):
        line = lines[i]
        parts = line.split()
        edges.append((int(parts[0]), int(parts[1])))

    visited_nodes = graph_coloring_v1(range(node_count),edges)

    # build a trivial solution
    # every node has its own color
    solution = range(0, node_count)

    # prepare the solution in the specified output format
    output_data = str(node_count) + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, solution))

    return output_data


import sys

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/gc_4_1)')

