#Final project for Calc 3 for Computer science.
#Welcome. We are the bomb.
#Test

import numpy as np
import math
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
    [1, 2, 0],
    [0, 4, 2],
    [0, 3, 1]
])
matrixD = np.matrix([
    [1, 2, 0],
    [1, 1, 1],
    [2, 1, 0]
])
matrixE = np.matrix([
    [1, 1, 1],
    [0, 1, 1],
    [0, 1, 2]
])

def multiplyMatrices(matrix1, matrix2):
    matrix3 = np.zeros(shape=(matrix1.shape[0], matrix2.shape[1]))
    if matrix1.shape[1] != matrix2.shape[0]:
        print "Cannot multiply these matrices"
        return
    elif matrix1.shape[1] == 1 and matrix2.shape[0] == 1:
        for rowIndex in range(0, matrix1.shape[0]):
            for colIndex in range(0, matrix1.shape[1] + 1):
                matrix3[rowIndex, colIndex] = round(np.dot(matrix1[rowIndex, 0], matrix2[0, colIndex]), 10)
    elif matrix1.shape[0] == 1 and matrix2.shape[1] == 1:
                matrix3 = round(np.dot(matrix1, matrix2), 10)
    else:
        for rowIndex in range(0, matrix1.shape[0]):
            for colIndex in range(0, matrix1.shape[1]):
                matrix3[rowIndex, colIndex] = round(np.dot(matrix1[rowIndex, :], matrix2[:, colIndex]), 10)
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

def qr_fact_househ(matrix):
    diagonalIndex = 0
    hMatrixQueue = Queue.Queue()
    while (diagonalIndex < matrix.shape[1] - 1):
        #Check to see if there are all zeros below the pivot
        allZeros = True
        for currentIndex in range(diagonalIndex + 1, matrix.shape[0]):
            if matrix[currentIndex, diagonalIndex] != 0:
                allZeros = False

        #If not all zeros
        if allZeros == False:
            #Creating the U matrix
            matrixU = np.zeros(shape=(matrix.shape[0] - diagonalIndex, 1))
            for currentIndex in range(0, matrix.shape[0] - diagonalIndex):
                matrixU[currentIndex, 0] = matrix[diagonalIndex + currentIndex, diagonalIndex]
            matrixU[0, 0] += vector_length(matrixU)

            #Creating the H matrix
            matrixH = np.identity(matrix.shape[0] - diagonalIndex)
            matrixH = matrixH - (2/(vector_length(matrixU)**2))*multiplyMatrices(matrixU, matrixU.transpose())
            matrixHFinal = matrixH

            #Putting H matrix in correct size
            if matrixH.shape[0] < matrix.shape[0]:
                matrixHFinal = np.identity(matrix.shape[0])
                for rowIndex in range(diagonalIndex, matrix.shape[0]):
                    for colIndex in range(diagonalIndex, matrix.shape[1]):
                        matrixHFinal[rowIndex, colIndex] = matrixH[rowIndex - diagonalIndex, colIndex - diagonalIndex]

            #Put H in a queue
            hMatrixQueue.put(matrixHFinal)
            #Use matrix to form R
            matrix = multiplyMatrices(matrixHFinal, matrix)
        diagonalIndex += 1
    #Form Q
    matrixQ = hMatrixQueue.get()
    while not hMatrixQueue.empty():
        matrixQ = multiplyMatrices(matrixQ, hMatrixQueue.get())

    #Return Q and R
    print matrixQ
    print matrix
    returnList = [matrixQ, matrix]
    return returnList


def qr_fact_givens(matrix):
    diagonalIndex = 0
    gMatrixQueue = Queue.Queue()
    #while we still have diagonal elements
    while (diagonalIndex < matrix.shape[1] - 1):
        # X is the pivot
        x = matrix[diagonalIndex, diagonalIndex]
        for currentIndex in range(diagonalIndex + 1, matrix.shape[0]):
            #y is at index below the pivot
            y = matrix[currentIndex, diagonalIndex]
            if y != 0:
                # set cos and sin
                cos = x/math.sqrt(x**2 + y**2)
                sin = -y/math.sqrt(x**2 + y**2)
                matrixG = np.identity(matrix.shape[0])
                matrixG[diagonalIndex, diagonalIndex] = cos
                matrixG[currentIndex, diagonalIndex] = sin
                matrixG[diagonalIndex, currentIndex] = -sin
                matrixG[currentIndex, currentIndex] = cos

                #add g matrix and update matrix and pivot
                gMatrixQueue.put(matrixG.transpose())
                matrix = multiplyMatrices(matrixG, matrix)
                x = matrix[diagonalIndex,diagonalIndex]
        diagonalIndex += 1

    #Form Q
    matrixQ = gMatrixQueue.get()
    while not gMatrixQueue.empty():
        matrixQ = multiplyMatrices(matrixQ, gMatrixQueue.get())

    print matrixQ
    print matrix
    returnList = [matrixQ, matrix]
    print "MULTIPLIED"
    print multiplyMatrices(matrixQ, matrix)
    return returnList


def vector_length(matrix):
    return math.sqrt(np.dot(matrix[:, 0], matrix[:, 0]))

def triangular_inverse(matrix):
    answer = np.copy(matrix)
    for currentRow in range(1, answer.shape[0]):
        for currentCol in range(0, currentRow):
            answer[currentRow, currentCol] = -answer[currentRow, currentCol]
    return answer


#Testing stuff
#print multiplyMatrices(matrixA, matrixB)
#lu_fact(matrixC)
#qr_fact_househ(matrixC)
qr_fact_givens(matrixD)