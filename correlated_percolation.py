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

def percolation(graph, neighbors_per_node, transmissionRate, recoveryRate):
    graph = []
    for node in neighbors_per_node:
        recoveryTime = np.random.exponential(1/recoveryRate)
        for neighbor in neighbors_per_node[node]:
            transmissionTime = np.random.exponential(1/transmissionRate)
            if (transmissionTime <= recoveryTime):
                graph.append([node, neighbor])
    return graph              

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

def calculateFinalInfection(numOfInfectedNodes, numOfNodes, numOfEdges, numOfTrials, transmissionRate, recoveryRate):
  num_infected = []
  for _ in range(numOfTrials):
    graph = graphs.erdos_renyi_graph(numOfNodes, numOfEdges)
    neighbors_per_node = tuples_to_dict(graph, numOfNodes)
    graph = percolation(graph, neighbors_per_node, transmissionRate, recoveryRate)
    infected_nodes = find_entire_connection(random.sample([i for i in range(0, numOfNodes)], numOfInfectedNodes), graph, neighbors_per_node)
    num_infected.append(len(infected_nodes))
  return np.mean(num_infected)
