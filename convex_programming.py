import cvxpy as cp
import numpy as np

import random

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

def cvx_find_rho(A, beta, budget, N):
    # v = cp.Variable(N)
    # rho = cp.Variable(N)
    # objective = cp.Minimize(cp.sum(v))
    # constraints = [beta*((v.T)@A) + rho <= cp.multiply(v,rho), cp.sum(rho) == budget, v>= 0]
    # prob = cp.Problem(objective, constraints)
    # print("Optimal value", prob.solve())
    # print("Optimal var")
    rho = cp.Variable(N)
    objective = cp.Minimize(cp.norm(beta*A - cp.diag(rho)))
    constraints = [cp.sum(rho) <= budget, rho >= 0, rho <= 1]
    prob = cp.Problem(objective, constraints)
    print("Optimal value", prob.solve())
    print("Optimal var")
    #print (cp.sum(rho))
    return (rho.value) # A numpy ndarray.

def tuples_to_adj(graph, N):
    #print (N)
    adj_matrix = np.zeros((N, N))
    for edge in graph:
        adj_matrix[edge[0]][edge[1]] = 1
        adj_matrix[edge[1]][edge[0]] = 1
    #print (adj_matrix)
    return adj_matrix


