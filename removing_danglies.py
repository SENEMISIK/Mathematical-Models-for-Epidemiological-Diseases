#undirected graph

import heapq as hpq
import random

from configuration_model import tuples_to_dict
from percolation import find_connected_nodes

def isPath(node1, node2, graph):
    return True

def isPathExceptEdge(edge, graph_dict):
    node1 = edge[0]
    node2 = edge[1]
    N = len(graph_dict)
    temp_graph_dict = graph_dict.copy()
    # print (graph_dict[node1])
    temp_graph_dict[node1].remove(node2)
    # print (graph_dict[node1])
    visited =[False]*N
    queue=[]
    queue.append(node1)
    visited[node1] = True
    while queue:
        curr_node = queue.pop()
        if curr_node == node2:
        return True
        
        for neighbor in temp_graph_dict[curr_node]:
        if visited[neighbor] == False:
            queue.append(neighbor)
            visited[neighbor] = True
    graph_dict[node1].append(node2)
  return False

def findDangly(node, graph_dict, dangly):
    find_connected_nodes(node, graph_dict, dangly)
    return len(dangly)

def cut_edge(cut, graph):
    new_graph = []
    for edge in graph:
        if edge != cut:
            new_graph += edge
    return new_graph

def new_edge_removal_startegy(graph, budget):
    edge_dictionary = {}
    dangly_size_heap = []
    for edge in graph:
        new_graph = cut_edge(edge, graph)
        if not isPath(edge[0], edge[1], new_graph):
            new_dict = tuples_to_dict(new_graph)
            dangly1 = []
            dangly2 = []
            size1 = findDangly(edge[0], new_dict, dangly1)
            size2 = findDangly(edge[1], new_dict, dangly2)
            if size1 < size2:
                edge_dictionary[edge] = size1
                hpq.heappush(dangly_size_heap, -size1, random.random(), edge)
            else:
                edge_dictionary[edge] = size2
                hpq.heappush(dangly_size_heap, -size2, random.random(), edge)
    while (budget > 0 and len(dangly_size_heap) > 0):
        cut = hpq.heappop(dangly_size_heap)[2]
        graph.remove(cut)
        budget -= 1
    if budget > 0:
        dict = tuples_to_dict(graph)
        new_graph2 = []
        for edge in graph:
            if budget > 0 and dict[edge[0]] == 3 or dict[edge[1]] == 3: 
                budget -= 1
            else:
                new_graph2.append(edge)
        return new_graph2
    return graph           





