import random
import numpy as np

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
  numedges = sum_degs/2
  half_edges = []
  for node in degreeOfNodes:
    deg = degreeOfNodes[node]
    for i in range(deg):
      half_edges.append(node)
  graph = []
  while half_edges != []:
    node1 = np.random.choice(half_edges)
    node2 = np.random.choice(half_edges)
    # if ([node1, node2] not in graph):
    graph.append([node1, node2])
    if (node1 != node2):
      graph.append([node2, node1])
    half_edges.remove(node1)
    if (node1 != node2):
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
import configuration_model
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
      if (node != neighbor):
        del triangles[neighbor][0]
      dictionary[neighbor].remove(node)
  return triangleGraph  

def strategy1(initial_recovery_rate, N, budget):
    recoveryRates = {}
    recoveryRate = round(budget/(3*N))
    for i in range(N):
        recoveryRates[3*i] = initial_recovery_rate + recoveryRate
        recoveryRates[3*i + 1] = initial_recovery_rate + recoveryRate
        recoveryRates[3*i + 2] = initial_recovery_rate + recoveryRate
    return recoveryRates

def strategy2(initial_recovery_rate, N, budget):
    recoveryRates = {}
    recoveryRate = round(budget/(2*N))
    for i in range(N):
        list = np,random.choice([0, 1, 2], 2)
        for num in list:
            recoveryRates[3*i + num] = initial_recovery_rate + recoveryRate
    return recoveryRates

def strategy3(initial_recovery_rate, N, budget):
    recoveryRates = {}
    recoveryRate = round(budget/N)
    for i in range(N):
        list = np,random.choice([0, 1, 2], 1)
        for num in list:
            recoveryRates[3*i + num] = initial_recovery_rate + recoveryRate
    return recoveryRates

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

def calculateFinalInfection1(numOfInfectedNodes, numOfTriangles, numOfTrials, transmissionRate, initial_recovery_rate, budget):
  num_infected = []
  for _ in range(numOfTrials):
    graph = triangle(numOfTriangles)
    neighbors_per_node = tuples_to_dict(graph, 3*numOfTriangles)
    recoveryRates = strategy1(initial_recovery_rate, numOfTriangles, budget)
    graph = percolation(neighbors_per_node, transmissionRate, recoveryRates)
    neighbors_per_node = tuples_to_dict(graph, 3*numOfTriangles)
    infected_nodes = find_entire_connection(random.sample([i for i in range(0, 3*numOfTriangles)], numOfInfectedNodes), neighbors_per_node)
    num_infected.append(len(infected_nodes))
  return np.mean(num_infected)

def calculateFinalInfection2(numOfInfectedNodes, numOfTriangles, numOfTrials, transmissionRate, initial_recovery_rate, budget):
    num_infected = []
    for _ in range(numOfTrials):
        graph = triangle(numOfTriangles)
        neighbors_per_node = tuples_to_dict(graph, 3*numOfTriangles)
        recoveryRates = strategy2(initial_recovery_rate, numOfTriangles, budget)
        graph = percolation(neighbors_per_node, transmissionRate, recoveryRates)
        neighbors_per_node = tuples_to_dict(graph, 3*numOfTriangles)
        infected_nodes = find_entire_connection(random.sample([i for i in range(0, 3*numOfTriangles)], numOfInfectedNodes), neighbors_per_node)
        num_infected.append(len(infected_nodes))
    return np.mean(num_infected)

def calculateFinalInfection3(numOfInfectedNodes, numOfTriangles, numOfTrials, transmissionRate, initial_recovery_rate, budget):
    num_infected = []
    for _ in range(numOfTrials):
        graph = triangle(numOfTriangles)
        neighbors_per_node = tuples_to_dict(graph, 3*numOfTriangles)
        recoveryRates = strategy3(initial_recovery_rate, numOfTriangles, budget)
        graph = percolation(neighbors_per_node, transmissionRate, recoveryRates)
        neighbors_per_node = tuples_to_dict(graph, 3*numOfTriangles)
        infected_nodes = find_entire_connection(random.sample([i for i in range(0, 3*numOfTriangles)], numOfInfectedNodes), neighbors_per_node)
        num_infected.append(len(infected_nodes))
    return np.mean(num_infected)


