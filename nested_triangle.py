import numpy as np
import graphs
import random

# TRIANGLE INSIDE THE CONFIGURATION MODEL

def config_model2(deg_dist, n):
  node_list = np.arange(n)
  degreeOfNodes = {}
  for key in deg_dist:
    num_nodes = int(round(deg_dist[key]*n))
    nodes = node_list[:num_nodes]
    for node in nodes:
      degreeOfNodes[node] = key
    node_list = node_list[num_nodes:]
  sum_degs = 0
  for key in deg_dist:
    sum_degs += key*n*deg_dist[key]
  half_edges = []
  for node in degreeOfNodes:
    deg = degreeOfNodes[node]
    for i in range(deg):
      half_edges.append(node)
  graph = []
  while half_edges != []:
    node1 = np.random.choice(half_edges)
    node2 = np.random.choice(half_edges)
    if (node1 != node2 and [node1, node2] not in graph):
      graph.append([node1, node2])
      graph.append([node2, node1])
      half_edges.remove(node1)
      half_edges.remove(node2)
  return graph

def tuples_to_dict(graph, N):
  graph_dict = {}
  for i in range(N):
    graph_dict[i] = []
  for edge in graph:
    graph_dict[edge[0]].append(edge[1])
  return graph_dict

import signal
from contextlib import contextmanager
import numpy as np

class TimeoutException(Exception): pass

@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

def config_model2_rec(deg_dist, n): 
  try:
    with time_limit(1):
      return config_model2(deg_dist, n)
  except TimeoutException as e:
    return config_model2(deg_dist, n)
  
def triangle(N):
  node_list = np.arange(N)
  triangles = {}
  for node in node_list:
    triangles[node] = [3 * node, 3 * node + 1, 3 * node + 2]
  triangleGraph = []
  for node in node_list:
    triangleGraph.append([node * 3, node * 3 + 1])
    triangleGraph.append([node * 3 + 1 , node * 3])
    triangleGraph.append([node * 3, node * 3 + 2])
    triangleGraph.append([node * 3 + 2 , node * 3])
    triangleGraph.append([node * 3 + 1, node * 3 + 2])
    triangleGraph.append([node * 3 + 2 , node * 3 + 1])
  graph = config_model2_rec({3:1}, N)
  dictionary = tuples_to_dict(graph, N)
  for node in dictionary:
    for neighbor in dictionary[node]:
      triangleGraph.append([triangles[node][0], triangles[neighbor][0]])
      triangleGraph.append([triangles[neighbor][0], triangles[node][0]])
      del triangles[node][0]
      del triangles[neighbor][0]
      dictionary[neighbor].remove(node)
  return triangleGraph  

# PERCOLATION

def tuples_to_dict(graph, N):
    graph_dict = {}
    for i in range(N):
        graph_dict[i] = []
    for edge in graph:
        graph_dict[edge[0]].append(edge[1])
    return graph_dict

def percolation(neighbors_per_node, transmissionRate, recovery_rates):
    newGraph = []
    node_rec_times = {}
    edge_transmit_times = {}
    for node in neighbors_per_node:
        recoveryTime = np.random.exponential(1/recovery_rates[node])
        for neighbor in neighbors_per_node[node]:
            transmissionTime = np.random.exponential(1/transmissionRate)
            if (transmissionTime <= recoveryTime):
                newGraph.append([node, neighbor])
                edge_transmit_times[[node, neighbor]] = transmissionTime
                if node not in node_rec_times:
                    node_rec_times[node] = recoveryTime
                
    return newGraph   

def percolation2(neighbors_per_node, transmissionRate, recoveryRate, node_rec_times, edge_transmit_times):
    newGraph = []
    for node in neighbors_per_node:
        recoveryTime = min(node_rec_times[node], np.random.exponential(1/recoveryRate))
        for neighbor in neighbors_per_node[node]:
            transmissionTime = edge_transmit_times([node,neighbor])
            if (transmissionTime <= recoveryTime):
                newGraph.append([node, neighbor])
    return newGraph, node_rec_times, edge_transmit_times 

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

def calculateFinalInfection(numOfInfectedNodes, numOfTriangles, numOfTrials, transmissionRate, budget1, budget2):
  num_infected1 = []
  num_infected2 = []
  recoveryRate1 = budget1 / (numOfTriangles*3)
  recoveryRate2 = budget2 / (numOfTriangles*3)
  for _ in range(numOfTrials):
    graph = triangle(numOfTriangles)
    neighbors_per_node = tuples_to_dict(graph, numOfTriangles*3)
    firstGraph, node_rec_times, edge_transmit_times = percolation(neighbors_per_node, transmissionRate, recoveryRate1)
    new_neighbors_per_node = tuples_to_dict(firstGraph, numOfTriangles*3)
    infected_nodes = find_entire_connection(random.sample([i for i in range(0, numOfTriangles*3)], numOfInfectedNodes), new_neighbors_per_node)
    num_infected1.append(len(infected_nodes))

    secondGraph = percolation2(neighbors_per_node, transmissionRate, recoveryRate2-recoveryRate1, node_rec_times, edge_transmit_times)
    new_neighbors_per_node2 = tuples_to_dict(secondGraph, numOfTriangles*3)
    infected_nodes2 = find_entire_connection(random.sample([i for i in range(0, numOfTriangles*3)], numOfInfectedNodes), new_neighbors_per_node2)
    num_infected2.append(len(infected_nodes2))

  return np.mean(num_infected1), np.mean(num_infected2)

# FRACTION

def strategyFraction(fraction, initial_recovery_rate, N, budget):
  recoveryRates = {}
  number = round(fraction * N)
  triangles = np.random.choice(np.arange(N), number)
  recoveryRate = round(budget/(3*number))
  for i in range(N):
    recoveryRates[3*i] = initial_recovery_rate
    recoveryRates[3*i + 1] = initial_recovery_rate
    recoveryRates[3*i + 2] = initial_recovery_rate
    if (i in triangles):
      recoveryRates[3*i] += recoveryRate
      recoveryRates[3*i + 1] += recoveryRate
      recoveryRates[3*i + 2] += recoveryRate
  return recoveryRates

def percolation1(neighbors_per_node, transmissionRate, recoveryRate):
  newGraph = []
  node_rec_times = {}
  edge_transmit_times = {}
  for node in neighbors_per_node:
      recoveryTime = np.random.exponential(1/recoveryRate)
      for neighbor in neighbors_per_node[node]:
          transmissionTime = np.random.exponential(1/transmissionRate)
          if (transmissionTime <= recoveryTime):
              newGraph.append([node, neighbor])
              edge_transmit_times[[node, neighbor]] = transmissionTime
              if node not in node_rec_times:
                  node_rec_times[node] = recoveryTime
  return newGraph, node_rec_times, edge_transmit_times

def calculateFinalInfection(fraction, numOfInfectedNodes, numOfTriangles, numOfTrials, transmissionRate, initialRecoveryRate, budget1, budget2):
  num_infected1 = []
  num_infected2 = []
  for _ in range(numOfTrials):
    graph = triangle(numOfTriangles)
    neighbors_per_node = tuples_to_dict(graph, numOfTriangles*3)
    recovery_rates = strategyFraction(fraction, initialRecoveryRate, numOfTriangles, budget1)
    firstGraph, node_rec_times, edge_transmit_times = percolation(neighbors_per_node, transmissionRate, initialRecoveryRate, recovery_rates)
    new_neighbors_per_node = tuples_to_dict(firstGraph, numOfTriangles*3)
    infected_nodes = find_entire_connection(random.sample([i for i in range(0, numOfTriangles*3)], numOfInfectedNodes), new_neighbors_per_node)
    num_infected1.append(len(infected_nodes))

    for node in recovery_rates:
      if recovery_rates[node] != initialRecoveryRate:
        newRecoveryRate = recovery_rates[node] + ((budget2 - budget1)/(numOfTriangles*3))
        newRecTime = min(node_rec_times[node], np.random.exponential(1/newRecoveryRate))
        node_rec_times[node] = newRecTime

    secondGraph = percolation2(neighbors_per_node, transmissionRate, node_rec_times, edge_transmit_times)
    new_neighbors_per_node2 = tuples_to_dict(secondGraph, numOfTriangles*3)
    infected_nodes2 = find_entire_connection(random.sample([i for i in range(0, numOfTriangles*3)], numOfInfectedNodes), new_neighbors_per_node2)
    num_infected2.append(len(infected_nodes2))

  return np.mean(num_infected1), np.mean(num_infected2)
