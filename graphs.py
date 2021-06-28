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

# PREFERENTIAL ATTACHMENT

def find_degree(node, graph):
  degree = 0
  for edge in graph:
    if edge[0] == node:
      degree += 1
  return degree

def sum_degs(graph, curr_nodes):
  sum = 0
  for node in curr_nodes:
    deg = find_degree(node, graph)
    sum += deg
  return sum

def pref_attachment(graph, curr_nodes,n, N):
  for i in range(N):
    new_node = n + i
    for node in curr_nodes:
      deg = find_degree(node, graph)
      sum = sum_degs(graph, curr_nodes)
      p_node = deg/sum
      if np.random.rand() < p_node:
        graph.append([node, new_node])
    curr_nodes.append(new_node)
  return graph

