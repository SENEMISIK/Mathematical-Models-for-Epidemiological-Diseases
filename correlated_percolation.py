# Correlated SIR

import numpy as np
import graphs
import random
 
def tuples_to_dict(graph, N):
  graph_dict = {}
  for i in range(N):
    graph_dict[i] = []
  for edge in graph:
    graph_dict[edge[0]].append(edge[1])
  return graph_dict

def percolation(neighbors_per_node, transmissionRate, recoveryRates):
    newGraph = []
    for node in neighbors_per_node:
        recoveryTime = np.random.exponential(1/recoveryRates[node])
        for neighbor in neighbors_per_node[node]:
            transmissionTime = np.random.exponential(1/transmissionRate)
            if (transmissionTime <= recoveryTime):
                newGraph.append([node, neighbor])
    return newGraph              

def find_connected_nodes(node, graph_dict, connected_component):
  if node not in connected_component:
    connected_component.append(node)
  for neighbor in graph_dict[node]:
    if neighbor not in connected_component:
      find_connected_nodes(neighbor, graph_dict, connected_component)

def find_entire_connection(infected_nodes, neighbors_per_node):
  connected_nodes = []
  for node in infected_nodes:
    find_connected_nodes(node, neighbors_per_node, connected_nodes)
  return connected_nodes

def calculateFinalInfection(numOfInfectedNodes, numOfNodes, numOfEdges, numOfTrials, transmissionRate, recoveryRates):
  num_infected = []
  for _ in range(numOfTrials):
    graph = graphs.erdos_renyi_graph(numOfNodes, numOfEdges)
    neighbors_per_node = tuples_to_dict(graph, numOfNodes)
    newGraph = percolation(neighbors_per_node, transmissionRate, recoveryRates)
    new_neighbors_per_node = tuples_to_dict(newGraph, numOfNodes)
    infected_nodes = find_entire_connection(random.sample([i for i in range(0, numOfNodes)], numOfInfectedNodes), new_neighbors_per_node)
    num_infected.append(len(infected_nodes))
  return np.mean(num_infected)
