import numpy as np
import percolation
import graphs

def config_model(deg_dist, n):
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

  graph = []
  i = 0
  print('a')

  while (i < numedges):
    listOfNodes = list(degreeOfNodes.items())
    node1 = np.random.choice(listOfNodes)[0]
    node2 = np.random.choice(listOfNodes)[0]
    if (node1 != node2 and [node1, node2] not in graph):
        graph.append([node1, node2])
        graph.append([node2, node1])
        degreeOfNodes[node1] -= 1
        degreeOfNodes[node2] -= 1
        i += 1
        if (degreeOfNodes[node1] == 0): del degreeOfNodes[node1]
        if (degreeOfNodes[node2] == 0): del degreeOfNodes[node2]
  return graph

def findComponentSizes(graph, n):
  graph_dict = graphs.tuples_to_dict(graph, n)
  node_list = [i for i in range(n)]
  component_sizes = []
  while node_list != []:
    node = np.random.choice(node_list)
    connected_nodes = []
    percolation.find_connected_nodes(node, graph_dict, connected_nodes)
    for i in connected_nodes:
      node_list.remove(i)
    component_sizes.append(len(connected_nodes))
  return component_sizes