import numpy as np
import math

def multiplyMatrices(matrix1, matrix2):
    matrix3 = np.zeros(shape=(matrix1.shape[0], matrix2.shape[1]))
    # iterate through rows of X
    for i in range(matrix1.shape[0]):
       # iterate through columns of Y
       for j in range(matrix2.shape[1]):
           # iterate through rows of Y
           for k in range(matrix1.shape[0]):
               matrix3[i,j] += matrix1[i,k] * matrix2[k,j]
    return matrix3

    # matrix3 = np.zeros(shape=(matrix1.shape[0], matrix2.shape[1]))
    # if matrix1.shape[1] != matrix2.shape[0]:
    #     print "Cannot multiply these matrices"
    #     return
    # elif matrix1.shape[1] == 1 and matrix2.shape[0] == 1:
    #     for rowIndex in range(0, matrix1.shape[0]):
    #         for colIndex in range(0, matrix1.shape[1] + 1):
    #             matrix3[rowIndex, colIndex] = round(np.dot(matrix1[rowIndex, 0], matrix2[0, colIndex]), 10)
    # elif matrix1.shape[0] == 1 and matrix2.shape[1] == 1:
    #             matrix3 = round(np.dot(matrix1, matrix2), 10)
    # else:
    #     for rowIndex in range(0, matrix1.shape[0]):
    #         for colIndex in range(0, matrix1.shape[1]):
    #             matrix3[rowIndex, colIndex] = round(np.dot(matrix1[rowIndex, :], matrix2[:, colIndex]), 10)
    # return matrix3

def triangular_inverse(matrix):
    answer = np.copy(matrix)
    for currentRow in range(1, answer.shape[0]):
        for currentCol in range(0, currentRow):
            answer[currentRow, currentCol] = -answer[currentRow, currentCol]
    return answer

def vector_length(matrix):
    return math.sqrt(np.dot(matrix[:, 0], matrix[:, 0]))

def matrix_max_norm(matrix):
    maxNorm = matrix[0,0]
    for rowIndex in range(0, matrix.shape[0]):
        for colIndex in range(0, matrix.shape[1]):
            if matrix[rowIndex, colIndex] > maxNorm:
                maxNorm = matrix[rowIndex, colIndex]
    return maxNorm
