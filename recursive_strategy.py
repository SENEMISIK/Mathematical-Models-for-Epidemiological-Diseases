import scipy.linalg as la
import numpy as np
import random
import matplotlib.pyplot as plt

def erdos_renyi_graph(N, M):
  graph = []
  node_list = [i for i in range(N)]
  numEdges = 0
  while numEdges < M:
    edge = random.sample(node_list, 2)
    if edge not in graph:
      numEdges += 1
      graph.append(edge)
  return graph

def find_degrees(graph_dict, node_list):
  degs = []
  for i in node_list:
    degs.append(len(graph_dict[i]))
  return degs

def barbell(n1, n2):
  graph = []
  for node in range(n1):
    for neighbor in range(n1):
      if (node != neighbor):
        graph.append([node, neighbor])
  for node in range(n1, n1+n2):
    for neighbor in range(n1, n1+n2):
      if (node != neighbor):
        graph.append([node, neighbor])
  node1 = np.random.randint(n1)
  node2 = np.random.randint(n1, n1+n2)
  graph.append([node1, node2])
  graph.append([node2, node1])
  return graph

def stochastic_block(n, r, P):
  graph = []
  node_list = np.arange(n)
  communities = np.array_split(node_list, r)
  for i in range(r):
    for j in range(r):
      C_i = communities[i]
      C_j = communities[j]
      p_ij = P[i][j]
      for node1 in C_i:
        for node2 in C_j:
          if node1 != node2:
            if np.random.rand() < p_ij:
                graph.append([node1, node2])
  return graph

def multi_barbell(N, numBoundaryEdges):
  n1 = int(N/2)
  n2 = int(N/2)
  graph = []
  boundaryNodes1 = random.sample([i for i in range(n1)], numBoundaryEdges)
  boundaryNodes2 = random.sample([i for i in range(n1, N)], numBoundaryEdges)
  for node in range(n1):
    for neighbor in range(n1):
      if (node != neighbor):
        graph.append([node, neighbor])
  for node in range(n1, n1+n2):
    for neighbor in range(n1, n1+n2):
      if (node != neighbor):
        graph.append([node, neighbor])
  for i in range(numBoundaryEdges):
    node1 = random.sample(boundaryNodes1, 1)[0]
    # print (node1)
    node2 = random.sample(boundaryNodes2, 1)[0]
    boundaryNodes1.remove(node1)
    boundaryNodes2.remove(node2)
    graph.append([node1, node2])
    graph.append([node2, node1])
  # print (graph)
  return graph

def edge_density(part1, part2, graph_dict, N):
  E = 0
  for node in part1:
    for neighbor in graph_dict[node]:
      if neighbor in part2:
        E += 1
  print(part1)
  print(part2)
  return N*(E/(len(part1)*len(part2)))

def sparsest_cut(graph_dict, node_list):
  L = graph_to_laplacian(graph_dict, node_list)
  results = la.eig(L)
  eigvalues = results[0].real
  eigvectors = results[1]
  idx = eigvalues.argsort()[::-1]
  eigvectors = eigvectors[:,idx]  
  second_smallest_vec = eigvectors[:, -2]
  vec = second_smallest_vec.T
  ids = vec.argsort()[::-1]
  curr_density = len(node_list)
  curr_i = 1
  for i in range(int(len(node_list)/4), int((3*len(node_list))/4)):
    part1 = []
    part2 = []
    for id in ids[:i]:
      part1.append(node_list[id])
    for j in ids[i:]:
      part2.append(node_list[j])
    density = edge_density(part1, part2, graph_dict, len(node_list))
    if density < curr_density and density > 0:
      curr_density = density
      curr_i = i
  part1 = ids[:curr_i]
  part2 = ids[curr_i:]
  nodes1 = []
  for i in part1:
    nodes1.append(node_list[i])
  nodes2 = []
  for i in part2:
    nodes2.append(node_list[i])
  return nodes1, nodes2

def find_boundary_nodes(graph_dict, part1, part2):
  nodes = []
  for node in graph_dict:
    for neighbor in graph_dict[node]:
      if (node in part1 and neighbor in part2) or (neighbor in part1 and node in part2):
        if node not in nodes:
          nodes.append(node)
        if neighbor not in nodes:
          nodes.append(neighbor)
  return nodes

def min_cut_antidotes(graph_dict, node_list, budget, part1, part2, recoveryRates, boundaryThreshold):
  nodes = find_boundary_nodes(graph_dict, part1, part2)
  antidote_amt = budget/len(nodes)
  if antidote_amt > boundaryThreshold:
    antidote_amt = boundaryThreshold
  for node in node_list:
    if node in nodes:
      recoveryRates[node] += antidote_amt
      budget -= antidote_amt
  return recoveryRates

def tuples_to_dict(graph, N):
  node_list = np.arange(N)    
  graph_dict = {}
  for i in node_list:
    graph_dict[i] = []
  for edge in graph: 
    graph_dict[edge[0]].append(edge[1])
  return graph_dict

def node_index(node, node_list):
  for index in range(len(node_list)):
    if (node_list[index] == node):
      return index

def graph_to_laplacian(graph_dict, node_list):
  # graph_dict = tuples_to_dict(graph, node_list)
  D = find_degrees(graph_dict, node_list)
  L = np.diag(D)
  for node in graph_dict:
      for neighbor in graph_dict[node]:
        L[node_index(node, node_list)][node_index(neighbor, node_list)] = -1
        L[node_index(neighbor, node_list)][node_index(node, node_list)] = -1
  return L

def measureSpectralGap(graph_dict, node_list):
  L = graph_to_laplacian(graph_dict, node_list)
  results = la.eig(L)
  eigvalues = results[0].real
  idx = eigvalues.argsort()[::-1]
  eigvalues = eigvalues[idx]
  # print (eigvalues)
  eig2 = eigvalues[-2]
  return eig2

def degreeProportional(graph_dict, recovery_rates, budget):
    degDict = {}
    sum = 0
    for node in graph_dict:
        degDict[node] = len(graph_dict[node])
        sum += len(graph_dict[node])
    for node in degDict:
        recovery_rates[node] += (degDict[node]/sum)*budget
      
def findNewGraph(graph_dict, nodes1, nodes2):
  new_graph_dict1 = {}
  len1 = 0
  new_graph_dict2 = {}
  len2 = 0
  for node in nodes1:
    new_graph_dict1[node] = []
  for node in nodes2:
    new_graph_dict2[node] = []
  for node in graph_dict:
    for neighbor in graph_dict[node]:
      if node in nodes1 and neighbor in nodes1:
        new_graph_dict1[node].append(neighbor)
        len1 += 1
      if node in nodes2 and neighbor in nodes2:
        new_graph_dict2[node].append(neighbor)
        len2 += 1
  return new_graph_dict1, len1, new_graph_dict2, len2
    
def newStrategy(graph_dict, recovery_rates, budget, gap_threshold, boundary_threshold, node_list):
  gap = measureSpectralGap(graph_dict, node_list)
  if (gap < gap_threshold):
    nodes1, nodes2 = sparsest_cut(graph_dict, node_list)
    graph_dict1, len1, graph_dict2, len2 = findNewGraph(graph_dict, nodes1, nodes2)
    min_cut_antidotes(graph_dict, node_list, budget, nodes1, nodes2, recovery_rates, boundary_threshold)
    if (budget > 0):
      budget1 = budget * (len1 / (len1 + len2))
      budget2 = budget * (len2 / (len1 + len2))
      newStrategy(graph_dict1, recovery_rates, budget1, gap_threshold, boundary_threshold, nodes1)
      newStrategy(graph_dict2, recovery_rates, budget2, gap_threshold, boundary_threshold, nodes2)
  else:
    degreeProportional(graph_dict, recovery_rates, budget) 
  return recovery_rates

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

# ERDOS RENYI

def erdos_renyi_graph(N, M):
  graph = []
  node_list = [i for i in range(N)]
  numEdges = 0
  while numEdges < M:
    edge = random.sample(node_list, 2)
    if edge not in graph:
      numEdges += 1
      graph.append(edge)
  return graph

def calculateRecInfection(InfectedNodes, graph, N, numTrials, transmissionRate, initial_rec, budget, gap_threshold, boundary_threshold):
  num_infected = []
  for _ in range(numTrials):
    graph_dict = tuples_to_dict(graph, N)
    recovery_rates = {}
    for i in range(N):
        recovery_rates[i] = initial_rec
    node_list = np.arange(N)
    recoveryRates = newStrategy(graph_dict, recovery_rates, budget, gap_threshold, boundary_threshold, node_list)
    new_graph = percolation(graph_dict, transmissionRate, recoveryRates)
    neighbors_per_node = tuples_to_dict(new_graph, N)
    infected_nodes = find_entire_connection(random.sample([i for i in range(0, N)], InfectedNodes), neighbors_per_node)
    num_infected.append(len(infected_nodes))
  return np.mean(num_infected)

def degree_proportional(graph, initial_recovery_rate, budget, numOfNodes):
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

def calculateDegreeInfection(InfectedNodes, graph, N, numTrials, transmissionRate, initial_rec, budget):
  num_infected = []
  for _ in range(numTrials):
    neighbors_per_node = tuples_to_dict(graph, N)
    recoveryRates = degree_proportional(graph, initial_rec, budget, N)
    new_graph = percolation(neighbors_per_node, transmissionRate, recoveryRates)
    neighbors_per_node = tuples_to_dict(new_graph, N)
    infected_nodes = find_entire_connection(random.sample([i for i in range(0, N)], InfectedNodes), neighbors_per_node)
    num_infected.append(len(infected_nodes))
  return np.mean(num_infected)