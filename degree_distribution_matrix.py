from correlated_percolation import percolation
import numpy as np
import math

def generate_percolation_matrix(n, p, B):
    matrix = np.zeros( (n+1, n+1) )
    col = 0
    while (col <= n):
        row = 0
        while (row <= col):
            matrix[row][col] = (B/(B+p))**row * (p/(B+p))**(col-row) * math.comb(col, row)
            row += 1
        col += 1
    return matrix

def generate_correlated_percolation_matrix(n, p, B):
    matrix = np.zeros( (n+1, n+1) )
    col = 0
    while (col <= n):
        row = 0
        while (row <= col):
            if (row == 0):
                matrix[row][col] = p/(p + col * B)
            elif (row == col):
                list1 = []
                for i in range(1, col+1):
                    list1.append(i * B + p)
                result1 = np.prod(list1)
                matrix[row][col] = (math.factorial(row) * B**(row)) / result1
            else:
                list2 = []
                for i in range(col-row, col+1):
                    list2.append(i * B + p)
                result2 = np.prod(list2)
                matrix[row][col] = (math.comb(col, row) * math.factorial(row) * B**(row) * p) / result2
            row += 1
        col += 1
    print(matrix)
    return matrix

def degree_to_edge_distribution(probabilityDict):
    halfEdgeDict = {}
    sum = 0
    for key in probabilityDict:
        if (key != 0):
            halfEdgeDict[key - 1] = key * probabilityDict[key]
            sum += halfEdgeDict[key - 1]
    for key in halfEdgeDict:
        halfEdgeDict[key] = halfEdgeDict[key] / sum
    print("New Degree Distribution" +str(halfEdgeDict))
    return  halfEdgeDict

def calculate_extinction_probability(halfEdgeDict):
    halfEdgeDict[1] = halfEdgeDict[1] - 1
    n = len(halfEdgeDict)
    polynomial = np.zeros(n)
    for i in range(n):
        polynomial[n-i-1] = halfEdgeDict[i]
    roots = np.roots(polynomial)
    realroots = []
    for i in range(len(roots)):
        if np.isreal(roots[i]) and roots[i] >= 0:
            realroots.append(roots[i])
    prob = min(realroots)
    return np.real(prob)

def calculate_giant_component_size(extinction_prob, probabilityDict, n):
    result = 0
    for key in probabilityDict:
        result += n * probabilityDict[key] * (1 - (extinction_prob ** key))
    return np.real(result)

def degree_distribution_calculation1(numOfNodes, rho, beta, probabilityDict):
    maxInDegree = 0
    for key in probabilityDict:
        if (key > maxInDegree):
            maxInDegree = key
    percolation_matrix = generate_percolation_matrix(maxInDegree, rho, beta)
    correlated_percolation_matrix = generate_correlated_percolation_matrix(maxInDegree, rho, beta)
    
    probability_vector = []
    for key in probabilityDict:
        probability_vector.append(probabilityDict[key])
    normalPercolation = np.dot(percolation_matrix, probability_vector)

    normalPercolationDict = {}
    for i in range(len(normalPercolation)):
        normalPercolationDict[i] = normalPercolation[i]
    correlatedPercolation = np.dot(correlated_percolation_matrix, probability_vector)
    
    correlatedPercolationDict = {}
    for i in range(len(correlatedPercolation)):
        correlatedPercolationDict[i] = correlatedPercolation[i]

    normal_extinction = calculate_extinction_probability(degree_to_edge_distribution(normalPercolationDict))
    correlated_extinction = calculate_extinction_probability(degree_to_edge_distribution(correlatedPercolationDict))

    normal_size = calculate_giant_component_size(normal_extinction, normalPercolationDict, numOfNodes)
    percolated_size = calculate_giant_component_size(correlated_extinction, correlatedPercolationDict, numOfNodes)
    
    print("Normal Percolation: " + str(percolation_matrix))
    print("Normal Probability: " + str(normal_extinction))
    print("Size: " + str(normal_size))

    print(" ")
    
    print("Correlated Percolation: " + str(correlated_percolation_matrix))
    print("Correlated Probability: " + str(correlated_extinction))
    print("Size : " + str(percolated_size))

def apply_percolation(probabilityDict, matrix):
    probability_vector = []
    for key in probabilityDict:
        probability_vector.append(probabilityDict[key])
    afterPercolation = np.dot(matrix, probability_vector)
    newDict = {}
    for i in range(len(afterPercolation)):
        newDict[i] = afterPercolation[i]
    return newDict

def degree_distribution_calculation2(numOfNodes, rho, beta, probabilityDict):
    maxInDegree = 0
    for key in probabilityDict:
        if (key > maxInDegree):
            maxInDegree = key

    percolation_matrix = generate_percolation_matrix(maxInDegree-1, rho, beta)
    sir_matrix = generate_correlated_percolation_matrix(maxInDegree-1, rho, beta)
    
    second_degree_dict = degree_to_edge_distribution(probabilityDict)
    percolation_second_degree = apply_percolation(second_degree_dict, percolation_matrix)
    sir_second_degree = apply_percolation(second_degree_dict, sir_matrix)
    
    normal_extinction = calculate_extinction_probability(percolation_second_degree)
    correlated_extinction = calculate_extinction_probability(degree_to_edge_distribution(sir_second_degree))

    percolation_matrix = generate_percolation_matrix(maxInDegree, rho, beta)
    sir_matrix = generate_correlated_percolation_matrix(maxInDegree, rho, beta)

    PercolationDict = apply_percolation(probabilityDict, percolation_matrix)
    SIRDict = apply_percolation(probabilityDict, sir_matrix)

    print("Percolation: " + str(calculate_giant_component_size(normal_extinction, PercolationDict, numOfNodes)))
    print("Correlated Percolation: " + str(calculate_giant_component_size(correlated_extinction, SIRDict, numOfNodes)))
    