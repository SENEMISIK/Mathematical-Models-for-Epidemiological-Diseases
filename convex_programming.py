import cvxpy as cp
import numpy as np
import graphs
import random

def cvx_find_rho(A, beta, budget, N):
    v = cp.Variable(N)
    rho = cp.Variable(N)
    objective = cp.Minimize(cp.sum(v))
    constraints = [beta*(v.T@A) + rho <= v*rho, cp.sum(rho) == budget, v>= 0]
    prob = cp.Problem(objective, constraints)
    print("Optimal value", prob.solve())
    print("Optimal var")
    return(rho.value) # A numpy ndarray.

def tuples_to_adj(graph, N):
    adj_matrix = np.zeros(N, N)
    for edge in graph:
        adj_matrix[edge[0]][edge[1]] = 1
        adj_matrix[edge[1]][edge[0]] = 1
    return adj_matrix


