# Part 1 - Final Project
import numpy as np
import math
import Queue
import util
__author__ = 'JosiahMacbook'

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

def lu_fact(matrix):
    if(isinstance(matrix, basestring)):
        matrix = np.loadtxt(matrix, unpack=False, delimiter=" ")
    originalMatrix = matrix
    lMatrixQueue = Queue.Queue()
    rowIndex = 0
    colIndex = 0
    while (rowIndex < matrix.shape[0]):
        if matrix[rowIndex, colIndex] != 0:
            for currentRow in range(rowIndex + 1, matrix.shape[0]):
                currentLMatrix = np.identity(matrix.shape[0])
                # THIS MAY BE BETTER --> if (matrixA[currentRow, colIndex] > 0 and matrixA[rowIndex, colIndex] > 0) or  (matrixA[currentRow, colIndex] < 0 and matrixA[rowIndex, colIndex] < 0):
                if (matrix[currentRow, colIndex] > 0 and matrix[rowIndex, colIndex] > 0) or (matrix[currentRow, colIndex] < 0 and matrix[rowIndex, colIndex] < 0):
                    currentLMatrix[currentRow, :] = currentLMatrix[currentRow, :] - currentLMatrix[rowIndex, :]*(float(matrix[currentRow, colIndex])/float(matrix[rowIndex, colIndex]))
                else:
                    currentLMatrix[currentRow, :] = currentLMatrix[currentRow, :] + currentLMatrix[rowIndex, :]*(float(matrix[currentRow, colIndex])/float(matrix[rowIndex, colIndex]))
                matrix = util.multiplyMatrices(currentLMatrix, matrix)
                currentLMatrix[currentRow, colIndex] = -currentLMatrix[currentRow, colIndex] #Changing the sign of the item so that we don't need to take the inverse later.
                lMatrixQueue.put(currentLMatrix)
        rowIndex = rowIndex + 1
        colIndex = colIndex + 1

    #Form the L matrix
    lMatrix = lMatrixQueue.get()
    while not lMatrixQueue.empty():
        lMatrix = util.multiplyMatrices(lMatrix, lMatrixQueue.get())

    #Get error
    error = util.matrix_max_norm(util.multiplyMatrices(lMatrix, matrix) - originalMatrix)

    #Return
    returnList = [lMatrix, matrix, error]
    return returnList

def qr_fact_househ(matrix):
    if(isinstance(matrix, basestring)):
        matrix = np.loadtxt(matrix, unpack=False, delimiter=" ")
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
            matrixU[0, 0] += util.vector_length(matrixU)

            #Creating the H matrix
            matrixH = np.identity(matrix.shape[0] - diagonalIndex)
            matrixH = matrixH - (2.0/(util.vector_length(matrixU)**2))*util.multiplyMatrices(matrixU, matrixU.transpose())
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
            matrix = util.multiplyMatrices(matrixHFinal, matrix)
        diagonalIndex += 1
    #Form Q
    matrixQ = hMatrixQueue.get()
    while not hMatrixQueue.empty():
        matrixQ = util.multiplyMatrices(matrixQ, hMatrixQueue.get())

    #Get error
    error = util.matrix_max_norm(util.multiplyMatrices(matrixQ, matrix) - originalMatrix)

    #Return Q and R
    returnList = [matrixQ, matrix, error]
    return returnList

def qr_fact_givens(matrix):
    if(isinstance(matrix, basestring)):
        matrix = np.loadtxt(matrix, unpack=False, delimiter=" ")
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
                matrix = util.multiplyMatrices(matrixG, matrix)
                x = matrix[diagonalIndex,diagonalIndex]
        diagonalIndex += 1

    #Form Q
    matrixQ = gMatrixQueue.get()
    while not gMatrixQueue.empty():
        matrixQ = util.multiplyMatrices(matrixQ, gMatrixQueue.get())

    #Get error
    error = util.matrix_max_norm(util.multiplyMatrices(matrixQ, matrix) - originalMatrix)

    returnList = [matrixQ, matrix, error]
    return returnList

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

def solve_lu_b(matrixInput):
    if(isinstance(matrixInput, basestring)):
        matrixInput = np.loadtxt(matrixInput, unpack=False, delimiter=" ")
    matrixA = matrixInput[:, 0:matrixInput.shape[1]-1]
    matrixB = np.matrix(matrixInput[:, [matrixInput.shape[1] - 1]])
    matrixBCopy = np.copy(matrixB)
    matrixACopy = np.copy(matrixA)
    lu = lu_fact(matrixACopy)
    matrixLCopy = np.copy(lu[0])
    matrixUCopy = np.copy(lu[1])
    xSolution = solve_b(matrixUCopy, solve_b(matrixLCopy, matrixBCopy))
    return [xSolution, util.matrix_max_norm(util.multiplyMatrices(matrixA, xSolution) - matrixB), util.matrix_max_norm(util.multiplyMatrices(lu[0], lu[1]) - matrixA)]

def solve_qr_b(matrixInput, function):
    if(isinstance(matrixInput, basestring)):
        matrixInput = np.loadtxt(matrixInput, unpack=False, delimiter=" ")
    matrixA = matrixInput[:, 0:matrixInput.shape[1]-1]
    matrixB = np.matrix(matrixInput[:, [matrixInput.shape[1] - 1]])
    matrixBCopy = np.copy(matrixB)
    matrixACopy = np.copy(matrixA)
    qr = function(matrixACopy)
    matrixQCopy = np.copy(qr[0])
    matrixRCopy = np.copy(qr[1])
    xSolution = solve_b(matrixRCopy, solve_b(matrixQCopy, matrixBCopy))
    return [xSolution, util.matrix_max_norm(util.multiplyMatrices(matrixA, xSolution) - matrixB), util.matrix_max_norm(util.multiplyMatrices(qr[0], qr[1]) - matrixA)]

def solve_givens_b(matrixInput):
    if(isinstance(matrixInput, basestring)):
        matrixInput = np.loadtxt(matrixInput, unpack=False, delimiter=" ")
    matrixA = matrixInput[:, 0:matrixInput.shape[1]-1]
    matrixB = np.matrix(matrixInput[:, [matrixInput.shape[1] - 1]])
    matrixBCopy = np.copy(matrixB)
    matrixACopy = np.copy(matrixA)
    qr = qr_fact_givens(matrixACopy)
    matrixQCopy = np.copy(qr[0])
    matrixRCopy = np.copy(qr[1])
    xSolution = solve_b(matrixRCopy, solve_b(matrixQCopy, matrixBCopy))
    return [xSolution, util.matrix_max_norm(util.multiplyMatrices(matrixA, xSolution) - matrixB), util.matrix_max_norm(util.multiplyMatrices(qr[0], qr[1]) - matrixA)]

def solve_househ_b(matrixInput):
    if(isinstance(matrixInput, basestring)):
        matrixInput = np.loadtxt(matrixInput, unpack=False, delimiter=" ")
    matrixA = matrixInput[:, 0:matrixInput.shape[1]-1]
    matrixB = np.matrix(matrixInput[:, [matrixInput.shape[1] - 1]])
    matrixBCopy = np.copy(matrixB)
    matrixACopy = np.copy(matrixA)
    qr = qr_fact_househ(matrixACopy)
    matrixQCopy = np.copy(qr[0])
    matrixRCopy = np.copy(qr[1])
    xSolution = solve_b(matrixRCopy, solve_b(matrixQCopy, matrixBCopy))
    return [xSolution, util.matrix_max_norm(util.multiplyMatrices(matrixA, xSolution) - matrixB), util.matrix_max_norm(util.multiplyMatrices(qr[0], qr[1]) - matrixA)]

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

def solve_paschal(solveFunction, message, decompPlot, xPlot):
    for n in range(2, 13):
        matrixA = form_paschal_matrix(n)
        matrixB = form_b_matrix(n)
        augmentedMatrix = np.concatenate((matrixA, matrixB), axis=1)
        result = solveFunction(augmentedMatrix)
        print "n = " + str(n)
        print "X solution:"
        print result[0]
        print message
        print result[2]
        print "Px - b error"
        print result[1]
        print "\n"
        decompPlot.append(result[2])
        xPlot.append(result[1])

lu_lu_plot = []
lu_px_plot = []
givens_qr_plot = []
givens_px_plot = []
househ_qr_plot = []
househ_px_plot = []
solve_paschal(solve_lu_b, "LU - P error", lu_lu_plot, lu_px_plot)
solve_paschal(solve_givens_b, "QR - P error", givens_qr_plot, givens_px_plot)
solve_paschal(solve_househ_b, "QR - P error", househ_qr_plot, househ_px_plot)