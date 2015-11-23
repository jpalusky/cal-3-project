#Part 3 - Final Project

import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import random
import util


def power_method(matrix, ev, error, n, inverse):
    if matrix.shape[0] != matrix.shape[0]:
        print "matrix must be square"
        return
    else:
        matrixU = ev
        matrixW = np.matrix([
            [1],
            [0]
        ])
        eValue = 0

        for i in range(1, n):
            eValueOld = eValue
            oldMatrixU = matrixU
            matrixU = util.multiplyMatrices(matrix, matrixU)
            eValue = np.dot(matrixW[:, 0].transpose(), matrixU[:, 0]) / np.dot(matrixW[:, 0].transpose(),
                                                                               oldMatrixU[:, 0])
            if (abs(eValueOld - eValue) < error):
                eVector = matrixU / util.vector_length(matrixU)
                if (inverse == True):
                    eValue = 1 / eValue
                return [eValue, eVector, i]
        return "failure"

x1 = []
y1 = []
c1 = []
x2 = []
y2 = []
c2 = []
for i in range(0, 999):
    out = ""
    matrixR = np.matrix([
        [random.uniform(-2, 2), random.uniform(-2, 2)],
        [random.uniform(-2, 2), random.uniform(-2, 2)]
    ])
    if matrixR[0, 0] * matrixR[1, 1] - matrixR[0, 1] * matrixR[1, 0] != 0:
        estimate = np.matrix([[1],
                              [0]])
        largest = power_method(matrixR, estimate, 0.00005, 100, False)
        if largest != "failure":
            trace = str(util.trace2x2(matrixR))
            determinant = str(util.determinant2x2(matrixR))
            x1.append(determinant)
            y1.append(trace)
            c1.append(largest[2])
        matrixRinverse = util.inverse2x2(matrixR)
        smallest = power_method(matrixRinverse, estimate, 0.00005, 100, True)
        out = ""
        if smallest != "failure":
            trace = str(util.trace2x2(matrixRinverse))
            determinant = str(util.determinant2x2(matrixRinverse))
            x2.append(determinant)
            y2.append(trace)
            c2.append(smallest[2])

    if (i == 0):
        print "Matrix A"
        print matrixR
        print ""
        print "Largest Eigenvalue " + str(largest[0])
        print "Largest Eigenvector " + str(largest[1])
        print "Number of iterations " + str(largest[2])
        print ""
        print "Matrix A inverse"
        print matrixRinverse
        print ""
        print "Smallest Eigenvalue " + str(smallest[0])
        print "Smallest Eigenvector " + str(smallest[1])
        print "Number of iterations " + str(smallest[2])
