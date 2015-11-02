#Final project for Calc 3 for Computer science.
#Welcome. We are the bomb.

import numpy as np

__author__ = 'JosiahMacbook'

matrixA = np.matrix([
    [3, 4, 5, 6, 7],
    [3, 4, 5, 6, 7],
    [3, 4, 5, 6, 7],
    [3, 4, 5, 6, 7],
    [3, 4, 5, 6, 7],

])
matrixB = np.matrix([
    [3, 4, 5, 6, 7],
    [3, 4, 5, 6, 7],
    [3, 4, 5, 6, 7],
    [3, 4, 5, 6, 7],
    [3, 4, 5, 6, 7],
])

def multiplyMatrices(matrix1, matrix2):
    matrix3 = np.zeros(shape=(matrix1.shape[0], matrix2.shape[1]))
    if matrix1.shape[1] != matrix2.shape[0]:
        print "Cannot multiply these matrices"
        return
    for rowIndex in range(0, matrix1.shape[0]):
        for colIndex in range(0, matrix1.shape[1]):
            matrix3[rowIndex, colIndex] = np.dot(matrix1[rowIndex, :], matrix2[:, colIndex])
    return matrix3


#Testing stuff
print multiplyMatrices(matrixA, matrixB)
