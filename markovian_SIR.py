# MARKOVIAN SIR

import heapq
import numpy as np

class Node:
  def __init__(self, name, status, pred_inf_time, rec_time):
    self.name = name
    self.status = status
    self.pred_inf_time = pred_inf_time
    self.rec_time = rec_time

class Event:
  def __init__(self, node, time, action):
    self.node = node
    self.time = time
    self.action = action

def tuples_to_dict(graph, N):
  graph_dict = {}
  for i in range(N):
    graph_dict[i] = []
  for edge in graph:
    graph_dict[edge[0]].append(edge[1])
  return graph_dict

def process_trans_SIR(nodes, graph_dict, node, time, transmissionRate, recoveryRate, times, S, I, R, Q, tmax):
  times.append(time)
  S.append(S[-1]-1)
  I.append(I[-1]+1)
  R.append(R[-1])
  node.status = 'i'
  node.rec_time = time + np.random.exponential(recoveryRate) #check if 1/recovery
  if (node.rec_time < tmax):
    newEvent = Event(node, node.rec_time, 'recover')
    heapq.heappush(Q, (node.rec_time, newEvent))
  for neighbor in graph_dict[node.name]:
    v = nodes[neighbor]
    find_trans_SIR(Q, time, transmissionRate, node, v, tmax)

def find_trans_SIR(Q, time, transmissionRate, node, neighbor, tmax):
  if (neighbor.status == 's'):
    inf_time = time + np.random.exponential(transmissionRate)
    if (inf_time < min(node.rec_time, neighbor.pred_inf_time, tmax)):
      newEvent = Event(neighbor, inf_time, 'transmit')
      heapq.heappush(Q, (newEvent.time, newEvent))
      neighbor.pred_inf_time = inf_time

def process_rec_SIR(node, time, times, S, I, R):
  times.append(time)
  S.append(S[-1])
  I.append(I[-1]-1)
  R.append(R[-1]+1)
  node.status = 'r'

def process_SIR_model(nodes, graph_dict, transmissionRate, recoveryRate, infected_nodes, tmax):
  times = [0]
  S = [len(nodes)]
  I = [0]
  R = [0]
  Q = []
  for node in nodes:
    node.status = 's'
    node.pred_inf_time = tmax
    if node.name in infected_nodes:
      node.pred_inf_time = 0
      event = Event(node, 0, 'transmit')
      heapq.heappush(Q, (event.time, event))
  while (Q != []):
    event = heapq.heappop(Q)[1]
    if (event.action == 'transmit'):
      if (event.node.status == 's'):
        process_trans_SIR(nodes, graph_dict, event.node, event.time, transmissionRate, recoveryRate, times, S, I, R, Q, tmax)
    else:
      process_rec_SIR(event.node, event.time, times, S, I, R)
  return (times, S, I, R)
