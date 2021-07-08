import random
import numpy as np
import math

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

# STAR GRAPH

def directed_star_graph(N):
  graph = []
  for i in range (2, N+1):
    graph.append([1, i])
  return graph

def undirected_star_graph(N):
  graph = []
  for i in range(2, N+1):
    graph.append([1, i])
    graph.append([i, 1])
  return graph

# BARBELL GRAPH

def barbell(n1, n2):
  graph = []
  for node in range(1, n1+1):
    for neighbor in range(1, n1+1):
      if (node != neighbor):
        graph.append([node, neighbor])
  for node in range(n1+1, n1+n2+1):
    for neighbor in range(n1+1, n1+n2+1):
      if (node != neighbor):
        graph.append([node, neighbor])
  node1 = np.random.randint(1, n1+1)
  node2 = np.random.randint(n1+1, n1+n2+1)
  graph.append([node1, node2])
  graph.append([node2, node1])
  return graph

# STOCHASTIC BLOCK

# n = number of nodes
# r = number of communities
# P = edge probabilities

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

# LATTICE GRAPH

def lattice_graph(N):
  size = int(math.sqrt(N))
  if (size**2 != N):
    size = size + 1
  lattice = []
  i = 1
  while(i <= N):
    row = []
    for _ in range(size):
      if (i > N):
        row.append(0)
      else:
        row.append(i)
        i += 1
    lattice.append(row)
  print(lattice)
  graph = []
  for i in range(len(lattice)):
    for j in range(size-1):
      if (lattice[i][j] != 0) and (lattice[i][j+1] != 0):
        graph.append([lattice[i][j], lattice[i][j+1]])
        graph.append([lattice[i][j+1], lattice[i][j]])
  for i in range(len(lattice)-1):
    for j in range(size):
      if (lattice[i][j] != 0) and (lattice[i+1][j] != 0):
        graph.append([lattice[i][j], lattice[i+1][j]])
        graph.append([lattice[i+1][j], lattice[i][j]])
  return graph

# PREFERENTIAL ATTACHMENT

def find_degree(node, graph):
  degree = 0
  for edge in graph:
    if edge[0] == node:
      degree += 1
  return degree

def make_deg_dict(graph, curr_nodes):
  degDict = {}
  for i in curr_nodes:
    degDict[i] = find_degree(i, graph)
  return degDict

def sum_degs(graph, curr_nodes):
  sum = 0
  for node in curr_nodes:
    deg = find_degree(node, graph)
    sum += deg
  return sum

def pref_attachment(graph, curr_nodes,n, N):
  degDict = make_deg_dict(graph, curr_nodes)
  for i in range(N):
    new_node = n + i
    degDict[new_node] = 0
    for node in curr_nodes:
      deg = degDict[node]
      sum = sum_degs(graph, curr_nodes)
      p_node = deg/sum
      if np.random.rand() < p_node:
        graph.append([node, new_node])
        degDict[node] += 1
        graph.append([new_node, node])
        degDict[new_node] += 1
    curr_nodes.append(new_node)
  return graph

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
      del triangles[neighbor][0]
      dictionary[neighbor].remove(node)
  return triangleGraph  
