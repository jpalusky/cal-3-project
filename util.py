import numpy as np
import math

def multiplyMatrices(matrix1, matrix2):
    matrix3 = np.zeros(shape=(matrix1.shape[0], matrix2.shape[1]))
    if matrix1.shape[1] != matrix2.shape[0]:
        print "Cannot multiply these matrices"
        return
    for rowIndex in range(0, matrix1.shape[0]):
            for colIndex in range(0, matrix2.shape[1]):
                matrix3[rowIndex, colIndex] = np.dot(matrix1[rowIndex, :], matrix2[:, colIndex])
    return matrix3

def triangular_inverse(matrix):
    answer = np.copy(matrix)
    for currentRow in range(1, answer.shape[0]):
        for currentCol in range(0, currentRow):
            answer[currentRow, currentCol] = -answer[currentRow, currentCol]
    return answer

def vector_length(matrix):
    return math.sqrt(np.dot(matrix[:, 0], matrix[:, 0]))

def matrix_max_norm(matrix):
    maxNorm = matrix[0, 0]
    for rowIndex in range(0, matrix.shape[0]):
        for colIndex in range(0, matrix.shape[1]):
            if matrix[rowIndex, colIndex] > maxNorm:
                maxNorm = matrix[rowIndex, colIndex]
    return maxNorm

def determinant2x2(matrix):
    if matrix.shape[0] != 2 and matrix.shape[1] != 2:
        print "this method only works with 2x2 matricies"
        return
    else:
        return matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0]

def inverse2x2(matrix):
    if matrix.shape[0] != 2 and matrix.shape[1] != 2:
        print "this method only works with 2x2 matricies"
        return
    a = matrix[0, 0]
    b = matrix[0, 1]
    c = matrix[1, 0]
    d = matrix[1, 1]
    matrixR = np.matrix([
        [d, -b],
        [-c, a]
    ])
    return 1/determinant2x2(matrix)*matrixR

def trace2x2(matrix):
    if matrix.shape[0] != 2 and matrix.shape[1] != 2:
        print "this method only works with 2x2 matricies"
        return
    return matrix[0,0]+matrix[1,1]
    
def vector_length(matrix):
    return math.sqrt(np.dot(matrix[:, 0], matrix[:, 0]))