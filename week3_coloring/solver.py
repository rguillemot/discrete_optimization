#!/usr/bin/python
# -*- coding: utf-8 -*-

import pandas as pd

def graph_order(voisins,
                nodes = None,
                all_nodes = None,
                rank_nodes = None,
                visited_nodes = None,
                rank = 0):

    if nodes is None:
        nodes = voisins.index.tolist()

    if all_nodes is None:
        all_nodes = nodes

    if rank_nodes is None:
        rank_nodes = [-1]*len(all_nodes)

    if visited_nodes is None:
        visited_nodes = set()

    if nodes == set():
        return visited_nodes, rank_nodes
    for n in nodes:
        if n not in visited_nodes:
            visited_nodes |= set([n])
            #print 'node %d rank %d pos %d'%(n,rank,len(visited_nodes)-1)
            rank_nodes[len(visited_nodes)-1]=(rank,n)
        adj_nodes = voisins[n]-visited_nodes
        adj_nodes = filter(lambda x: x in adj_nodes, all_nodes)
        for n2 in adj_nodes:
            visited_nodes |= set([n2])
            #print 'node %d rank %d pos %d'%(n2,rank+1,len(visited_nodes)-1)
            rank_nodes[len(visited_nodes)-1]=(rank+1,n2)
        visited_nodes, rank_nodes = graph_order(voisins,
                                                adj_nodes,
                                                all_nodes,
                                                rank_nodes,
                                                visited_nodes,
                                                rank + 1)

    return visited_nodes, rank_nodes

def graph_color(node_count,voisins):
    visited_nodes, rank_nodes = graph_order(voisins)
    #print rank_nodes
    colors = dict(zip(voisins.index.tolist(),[None]*node_count))
    color_set=set()
    for (rank,n) in rank_nodes:
        #print 'rank %d node %d'%(r, n)
        neighbours_already_colored = set([colors[n2] for n2 in voisins[n] if colors[n2] is not None])
        remaining_colors = color_set-neighbours_already_colored
        #print 'neighbours_already_colored : ' + str(neighbours_already_colored)
        #print 'remaining_colors : ' + str(remaining_colors)
        if remaining_colors == set([]):
            new_color = len(color_set)
            color_set|=set([new_color])
            colors[n]=new_color
            #print 'remaining_colors is empty add new color : node %d color %d'%(n, colors[n])
        else:
            colors[n] = min(remaining_colors)
            #print 'using an existing color : node %d color %d'%(n, colors[n])

    return colors, len(color_set)

def compute_voisins(node_count, data):
    voisins1 = data.groupby('from').apply(lambda x: set(x['to'].values))
    voisins1 = voisins1.loc[range(node_count)].apply(lambda x:  set() if type(x) is not set else x)
    voisins2 = data.groupby('to').apply(lambda x: set(x['from'].values))
    voisins2 = voisins2.loc[range(node_count)].apply(lambda x:  set() if type(x) is not set else x)
    voisins = pd.DataFrame([voisins1,voisins2],index=['c1','c2']).apply(lambda x: x['c1']|x['c2'])
    return voisins

def swap(idx,pos1,pos2):
    tmp = idx[pos2]
    idx[pos2]=idx[pos1]
    idx[pos1]=tmp

def graph_color_backtrack(node_count, voisins, max_level = None, deepness = None, idx = None, level = 0):
    if max_level is None:
        max_level = node_count
    if deepness is None:
        deepness = node_count
    if idx is None:
        voisins_count=voisins.map(len).sort_values(ascending=False)
        idx = voisins_count.index.tolist()
    min_colors, min_nb_colors = graph_color(node_count,voisins.loc[idx])
    print level, min_nb_colors
    for pos in range(level+1, deepness):
        swap(idx,level, pos)
        colors, nb_colors = graph_color(node_count,voisins.loc[idx])
        if (nb_colors < min_nb_colors) & (level+1<max_level):
            print level, nb_colors, pos
            min_colors, min_nb_colors = graph_color_backtrack(node_count, voisins, max_level, deepness, idx, level+1)

    return min_colors, min_nb_colors

def solve_it(file_location):
    # Modify this code to run your optimization algorithm

    # parse the input
    data = pd.read_csv(file_location,sep=' ')
    node_count = int(data.columns[0])
    edge_count = int(data.columns[1])
    data = data.rename(columns={data.columns[0]:'from',data.columns[1]:'to'})

    voisins=compute_voisins(node_count, data)

    if node_count in (50,70):
        colors, nb_colors = graph_color_backtrack(node_count,voisins)
    elif node_count == 100:
        colors, nb_colors = graph_color_backtrack(node_count,voisins,5,50)
    elif node_count == 250:
        colors, nb_colors = graph_color_backtrack(node_count,voisins,5,20)
    elif node_count in (500,1000):
        colors, nb_colors = graph_color_backtrack(node_count,voisins,2,10)


    # prepare the solution in the specified output format
    output_data = str(nb_colors) + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, colors.values()))

    return output_data


import sys

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        print(solve_it(file_location))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/gc_4_1)')

