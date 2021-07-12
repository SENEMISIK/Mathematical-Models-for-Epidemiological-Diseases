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
    return matrix

def degree_to_edge_distribution(probabilityDict):
    halfEdgeDict = {}
    sum = 0
    for key in probabilityDict:
        halfEdgeDict[key - 1] = key * probabilityDict[key]
        sum += halfEdgeDict[key - 1]
        print(sum)
    print(halfEdgeDict)
    for key in halfEdgeDict:
        halfEdgeDict[key] = halfEdgeDict[key] / sum
    print(halfEdgeDict)
    return  halfEdgeDict

def calculate_extinction_probability(halfEdgeDict):
    halfEdgeDict[1] = halfEdgeDict[1] - 1
    n = len(halfEdgeDict)
    polynomial = np.zeros(n)

    for i in range(n):
        polynomial[n-i-1] = halfEdgeDict[i]
    print(polynomial)
    roots = np.roots(polynomial)
    for elem in roots:
        if np.iscomplex(elem) or elem < 0:
            roots.remove(elem)
    return roots


# def calculate_giant_component_size(extinction_prob)


