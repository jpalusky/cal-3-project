#Final project for Calc 3 for Computer science.
#Welcome. We are the bomb.
#Test

import numpy as np
import math
import Queue
import util

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
matrixF = np.matrix([
    [2, -1, 1],
    [3, 3, 9],
    [3, 3, 5]
])

matrixTest = np.matrix([
    [1, 1, 1, 1],
    [1, 2, 3, 4],
    [1, 3, 6, 10],
    [1, 4, 10, 20]
])

matrixBTest = np.matrix([
    [1],
    [1.0/2],
    [1.0/3],
    [1.0/4]
])

matrixBExample = np.matrix([
    [1],
    [2],
    [3],
])

def multiplyMatrices(matrix1, matrix2):
    matrix3 = np.zeros(shape=(matrix1.shape[0], matrix2.shape[1]))
    if matrix1.shape[1] != matrix2.shape[0]:
        print "Cannot multiply these matrices"
        return
    for rowIndex in range(0, matrix1.shape[0]):
            for colIndex in range(0, matrix2.shape[1]):
                matrix3[rowIndex, colIndex] = np.dot(matrix1[rowIndex, :], matrix2[:, colIndex])
    return matrix3

def lu_fact(matrix):

    originalMatrix = matrix
    lMatrixQueue = Queue.Queue()
    rowIndex = 0
    colIndex = 0
    while (rowIndex < matrix.shape[0]):
        if matrix[rowIndex, colIndex] != 0:
            for currentRow in range(rowIndex + 1, matrix.shape[0]):
                currentLMatrix = np.identity(matrix.shape[0])
                # THIS MAY BE BETTER --> if (matrixA[currentRow, colIndex] > 0 and matrixA[rowIndex, colIndex] > 0) or  (matrixA[currentRow, colIndex] < 0 and matrixA[rowIndex, colIndex] < 0):
                if matrix[currentRow, colIndex] > 0:
                    currentLMatrix[currentRow, :] = currentLMatrix[currentRow, :] - currentLMatrix[rowIndex, :]*(float(matrix[currentRow, colIndex])/float(matrix[rowIndex, colIndex]))
                else:
                    currentLMatrix[currentRow, :] = currentLMatrix[currentRow, :] + currentLMatrix[rowIndex, :]*(float(matrix[currentRow, colIndex])/float(matrix[rowIndex, colIndex]))
                matrix = multiplyMatrices(currentLMatrix, matrix)
                currentLMatrix[currentRow, colIndex] = -currentLMatrix[currentRow, colIndex] #Changing the sign of the item so that we don't need to take the inverse later.
                lMatrixQueue.put(currentLMatrix)
        rowIndex = rowIndex + 1
        colIndex = colIndex + 1

    #Form the L matrix
    lMatrix = lMatrixQueue.get()
    while not lMatrixQueue.empty():
        lMatrix = multiplyMatrices(lMatrix, lMatrixQueue.get())

    #Get error
    error = util.matrix_max_norm(multiplyMatrices(lMatrix, matrix) - originalMatrix)

    #Return
    returnList = [lMatrix, matrix, error]
    return returnList

def qr_fact_househ(matrix):
    originalMatrix = matrix
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
            matrixH = matrixH - (2.0/(vector_length(matrixU)**2))*multiplyMatrices(matrixU, matrixU.transpose())
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

    #Get error
    error = util.matrix_max_norm(multiplyMatrices(matrixQ, matrix) - originalMatrix)

    #Return Q and R
    returnList = [matrixQ, matrix, error]
    return returnList


def qr_fact_givens(matrix):
    originalMatrix = matrix
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

    #Get error
    error = util.matrix_max_norm(multiplyMatrices(matrixQ, matrix) - originalMatrix)

    returnList = [matrixQ, matrix, error]
    return returnList


def vector_length(matrix):
    return math.sqrt(np.dot(matrix[:, 0], matrix[:, 0]))

def triangular_inverse(matrix):
    answer = np.copy(matrix)
    for currentRow in range(1, answer.shape[0]):
        for currentCol in range(0, currentRow):
            answer[currentRow, currentCol] = -answer[currentRow, currentCol]
    return answer

def solve_lu_b(matrixA, matrixB):
    lu = lu_fact(matrixA)
    originalL = np.copy(lu[0])
    originalU = np.copy(lu[1])
    originalLU = multiplyMatrices(originalL, originalU)
    return [solve_b(lu[1], solve_b(lu[0], matrixB)), originalLU]

def solve_qr_b(matrixA, matrixB):
    qr = qr_fact_givens(matrixA)
    originalQ = np.copy(qr[0])
    originalR = np.copy(qr[1])
    return [originalQ, originalR, solve_b(qr[1], solve_b(qr[0], matrixB))]
    #return solve_b(qr[1], multiplyMatrices(qr[0].transpose(), matrixB))

def solve_b(matrixA, matrixB):
    rowIndex = 0
    colIndex = 0
    #Get into echelon form
    while rowIndex < matrixA.shape[0]:
        if matrixA[rowIndex, colIndex] != 0:
            for currentRow in range(rowIndex + 1, matrixA.shape[0]):
                if matrixA[currentRow, colIndex] != 0:
                    multiplier = (float(matrixA[currentRow, colIndex]) / matrixA[rowIndex, colIndex])
                    if (matrixA[currentRow, colIndex] > 0 and matrixA[rowIndex, colIndex] > 0) or (matrixA[currentRow, colIndex] < 0 and matrixA[rowIndex, colIndex] < 0):
                        matrixA[currentRow, :] -= matrixA[rowIndex, :]*multiplier
                        matrixB[currentRow, :] -= matrixB[rowIndex, :]*multiplier
                    else:
                        matrixA[currentRow, :] += matrixA[rowIndex, :]*multiplier
                        matrixB[currentRow, :] += matrixB[rowIndex, :]*multiplier
        rowIndex = rowIndex + 1
        colIndex = colIndex + 1

    rowIndex = matrixA.shape[0] - 1
    colIndex = matrixA.shape[0] - 1

    #Get into reduced echelon form
    while rowIndex >= 0:
        if matrixA[rowIndex, colIndex] != 0:
            #Make pivot 1
            matrixB[rowIndex, 0] = float(matrixB[rowIndex, 0]) / matrixA[rowIndex, colIndex]
            matrixA[rowIndex, colIndex] /= float(matrixA[rowIndex, colIndex])
            if rowIndex > 0:
                #Zero above
                for currentRow in range(rowIndex - 1, -1, -1):
                    if matrixA[currentRow, colIndex] != 0:
                        multiplier = float(matrixA[currentRow, colIndex])
                        matrixA[currentRow, :] -= matrixA[rowIndex, :]*multiplier
                        matrixB[currentRow, :] -= matrixB[rowIndex, :]*multiplier
        rowIndex = rowIndex - 1
        colIndex = colIndex - 1
    return matrixB

def form_paschal_matrix(n):
    matrix = np.zeros(shape=(n, n))
    for rowIndex in range(0, n):
        for colIndex in range(0, n):
            matrix[rowIndex, colIndex] = math.factorial((rowIndex + colIndex)) / (math.factorial(rowIndex)*math.factorial(colIndex))
    return matrix

def form_b_matrix(n):
    matrix = np.zeros(shape=(n, 1))
    for rowIndex in range(1, n + 1):
        matrix[rowIndex - 1, 0] = 1.0/rowIndex
    return matrix

def solve_paschal_lu():
    for n in range(2, 3):
        matrixA = form_paschal_matrix(n)
        matrixB = form_b_matrix(n)
        result = solve_lu_b(matrixA, matrixB)
        errorLU = util.matrix_max_norm(matrixA - result[1])
        print matrixA
        print result[0]
        print multiplyMatrices(matrixA, result[0])
        #errorP = util.matrix_max_norm(multiplyMatrices(matrixA, result[0]) - matrixB)
        print "n = " + str(n)
        print result[0]
        print errorLU
        #print errorP
        print "\n"

solve_paschal_lu()
#Testing stuff
#print multiplyMatrices(matrixA, matrixB)

#luList = lu_fact(matrixF)
#print luList[1]

#givensList = qr_fact_givens(matrixTest)
#print givensList[1]

#print solve_qr_b(matrixTest, matrixBTest)
#print solve_lu_b(matrixTest, matrixBTest)

#houseHolderList = qr_fact_househ(matrixTest)
#print houseHolderList[0]
#print houseHolderList[2]