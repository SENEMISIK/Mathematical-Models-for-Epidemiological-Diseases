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
                list = []
                for i in range(1, col+1):
                    list.append(i * B + p)
                matrix[row][col] = (math.factorial(row) * B**(row)) / result
            else:
                list = []
                for i in range(col-row, col+1):
                    list.append(i * B + p)
                result = np.prod(list)
                matrix[row][col] = (math.comb(col, row) * math.factorial(row) * B**(row) * p) / result
            row += 1
        col += 1
    return matrix

def degree_to_edge_distribution(probabilityDict):
    halfEdgeDict = {}
    sum = 0
    for key in probabilityDict:
        halfEdgeDict[key - 1] = key * probabilityDict[key]
        sum += halfEdgeDict[key - 1]
    for key in halfEdgeDict:
        halfEdgeDict[key] = halfEdgeDict[key] / sum
    return  halfEdgeDict

# def calculate_extinction_probability(halfEdgeDict)
# def calculate_giant_component_size(extinction_prob)
    