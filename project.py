#Final project for Calc 3 for Computer science.
#Welcome. We are the bomb.

import numpy as np
import Queue

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
matrixC = np.matrix([
    [2, 3, 4],
    [3, 4, 5],
    [5, 6, 7]
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

def lu_fact(matrix):
    lMatrixList = []
    rowIndex = 0
    colIndex = 0
    identityMatrix = np.identity(matrix.shape[0])
    while (rowIndex < matrix.shape[1]):
        if matrix[rowIndex, colIndex] != 0:
            currentLMatrix = identityMatrix
            for currentRow in range(rowIndex + 1, matrix.shape[0]):
                if matrix[currentRow, colIndex] > 0:
                    currentLMatrix[currentRow, :] = currentLMatrix[currentRow, :] - currentLMatrix[rowIndex, :]*(matrix[currentRow, colIndex]/matrix[rowIndex, colIndex])
                else:
                    currentLMatrix[currentRow, :] = currentLMatrix[currentRow, :] + currentLMatrix[rowIndex, :]*(matrix[currentRow, colIndex]/matrix[rowIndex, colIndex])
                currentLMatrix[currentRow, colIndex] = -currentLMatrix[currentRow, colIndex] #Changing the sign of the item so that we don't need to take the inverse later.
                lMatrixList.append(currentLMatrix)
        rowIndex = rowIndex + 2
        colIndex = colIndex + 2
    lMatrix = lMatrixList.pop()
    for currentMatrix in range(1, len(lMatrixList)):
        lMatrix = multiplyMatrices(lMatrix, lMatrixList.pop())
    uMatrix = multiplyMatrices(triangular_inverse(lMatrix),matrix)
    print "L Matrix"
    print lMatrix
    print "U Matrix:"
    print uMatrix
    print "Matrix: "
    print matrix
    print "Result Matrix: "
    print multiplyMatrices(lMatrix,uMatrix)

def triangular_inverse(matrix):
    answer = np.copy(matrix)
    for currentRow in range(1, answer.shape[0]):
        for currentCol in range(0, currentRow):
            answer[currentRow, currentCol] = -answer[currentRow, currentCol]
    return answer


#Testing stuff
print multiplyMatrices(matrixA, matrixB)

lu_fact(matrixC)
