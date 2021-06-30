# PERCOLATION

import numpy as np
import graphs
import random

# Calculates the probability that the edge should not be removed -- the probability that transmission happens before recovery
def calculate_perc_probability(beta, rho):
  return (beta/(beta + rho))

def coinToss(beta, rho):
  perc_prob = calculate_perc_probability(beta, rho)
  #print (perc_prob)
  random_num = np.random.rand()
  #print (random_num)
  if random_num < perc_prob:
    return True #don't remove
  return False

# Tosses a coin for each edge and remove those that coinToss outputs False
def percolation(graph, transmissionRate, recoveryRate):
  newGraph = []
  for i in graph:
    if (coinToss(transmissionRate, recoveryRate)):
      newGraph.append(i)
  return newGraph

def percolation2(graph, perc_prob):
  random_num = np.random.rand()
  newGraph = []
  for i in graph:
    if (random_num < perc_prob):
      newGraph.append(i)
  return newGraph
      
def tuples_to_dict(graph, N):
  graph_dict = {}
  for i in range(N):
    graph_dict[i] = []
  for edge in graph:
    graph_dict[edge[0]].append(edge[1])
  return graph_dict

def find_connected_nodes(node, graph_dict, connected_component):
  if node not in connected_component:
    connected_component.append(node)
  for neighbor in graph_dict[node]:
    if neighbor not in connected_component:
      find_connected_nodes(neighbor, graph_dict, connected_component)

def find_entire_connection(infected_nodes, graph, N):
  graph_dict = tuples_to_dict(graph, N)
  #print (graph_dict)
  connected_nodes = []
  for node in infected_nodes:
    find_connected_nodes(node, graph_dict, connected_nodes)
  #print(connected_nodes)   
  return connected_nodes

def calculateFinalInfectionErdosReyni(numOfInfectedNodes, numOfNodes, numOfEdges, numOfTrials, transmissionRate, recoveryRate):
  num_infected = []
  for _ in range(numOfTrials):
    graph = graphs.erdos_renyi_graph(numOfNodes, numOfEdges)
    graph = percolation(graph, transmissionRate, recoveryRate)
    infected_nodes = find_entire_connection(random.sample([i for i in range(0, numOfNodes)], numOfInfectedNodes), graph, numOfNodes)
    num_infected.append(len(infected_nodes))
  return np.mean(num_infected)

def calculateFinalInfectionErdosReyni2(numOfInfectedNodes, numOfNodes, numOfEdges, numOfTrials, perc_prob):
  num_infected = []
  for _ in range(numOfTrials):
    graph = graphs.erdos_renyi_graph(numOfNodes, numOfEdges)
    graph = percolation2(graph, perc_prob)
    infected_nodes = find_entire_connection(random.sample([i for i in range(0, numOfNodes)], numOfInfectedNodes), graph, numOfNodes)
    num_infected.append(len(infected_nodes))
  return np.mean(num_infected)

def calculateFinalInfectionPrefAttach(numOfInfectedNodes, graph, curr_nodes, n, i, numOfTrials, transmissionRate, recoveryRate):
  num_infected = []
  for _ in range(numOfTrials):
    graph = graphs.pref_attachment(graph, curr_nodes, n, i)
    graph = percolation(graph, transmissionRate, recoveryRate)
    infected_nodes = find_entire_connection(random.sample([i for i in range(0, i+2)], numOfInfectedNodes), graph, i+2)
    num_infected.append(len(infected_nodes))
  return np.mean(num_infected)