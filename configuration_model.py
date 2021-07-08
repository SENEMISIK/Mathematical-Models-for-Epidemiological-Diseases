import numpy as np

def tuples_to_dict(graph, N):
  graph_dict = {}
  for i in range(N):
    graph_dict[i] = []
  for edge in graph:
    graph_dict[edge[0]].append(edge[1])
  return graph_dict

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
  

  while (len(degreeOfNodes) > 2):
    listOfNodes = list(degreeOfNodes.keys())
    node1 = np.random.choice(listOfNodes)
    node2 = np.random.choice(listOfNodes)
    # print (node1)
    # print (node2)
    if (node1 != node2 and [node1, node2] not in graph):
        # print(graph)
        graph.append([node1, node2])
        graph.append([node2, node1])
        degreeOfNodes[node1] -= 1
        degreeOfNodes[node2] -= 1
        i += 1
        if (degreeOfNodes[node1] == 0): del degreeOfNodes[node1]
        if (degreeOfNodes[node2] == 0): del degreeOfNodes[node2]
        # print(i)
  return graph

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

def find_connected_nodes(node, graph_dict, connected_component):
  if node not in connected_component:
    connected_component.append(node)
  for neighbor in graph_dict[node]:
    if neighbor not in connected_component:
      find_connected_nodes(neighbor, graph_dict, connected_component)

def findComponentSizes(graph, n):
  graph_dict = tuples_to_dict(graph, n)
  node_list = [i for i in range(n)]
  component_sizes = []
  while node_list != []:
    #print (node_list)
    node = np.random.choice(node_list)
    connected_nodes = []
    find_connected_nodes(node, graph_dict, connected_nodes)
    #print (connected_nodes)
    for i in connected_nodes:
      #print (i)
      node_list.remove(i)

    component_sizes.append(len(connected_nodes))
    # print(node_list)
  return component_sizes