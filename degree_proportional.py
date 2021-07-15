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
    graph = []
    for node in neighbors_per_node:
        recoveryTime = np.random.exponential(1/recoveryRates[node])
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


def recovery_rates(graph, initial_recovery_rate, budget, numOfNodes):
    degDict = {}
    sum = len(graph)
    for edge in graph:
        if edge[0] not in degDict:
            degDict[edge[0]] = 0
        degDict[edge[0]] += 1
    recoveryRates = {}
    for node in range(numOfNodes):
        # recoveryRates[node] = initial_recovery_rate + (budget/numOfNodes)
        if node in degDict:
          recoveryRates[node] = initial_recovery_rate + (degDict[node]/sum)*budget
        else: 
          recoveryRates[node] = initial_recovery_rate
    return recoveryRates 

def calculateFinalInfection(numOfInfectedNodes, numOfNodes, numOfEdges, numOfTrials, transmissionRate, initial_recovery_rate, budget):
  num_infected = []
  for _ in range(numOfTrials):
    graph = graphs.erdos_renyi_graph(numOfNodes, numOfEdges)
    neighbors_per_node = tuples_to_dict(graph, numOfNodes)
    recoveryRates = recovery_rates(graph, initial_recovery_rate, budget, numOfNodes)
    graph = percolation(neighbors_per_node, transmissionRate, recoveryRates)
    neighbors_per_node = tuples_to_dict(graph, numOfNodes)
    infected_nodes = find_entire_connection(random.sample([i for i in range(0, numOfNodes)], numOfInfectedNodes), neighbors_per_node)
    num_infected.append(len(infected_nodes))
  return np.mean(num_infected)


    
        
    
        
    
    


