import numpy as np
from numpy import linalg as LA
import random
import util

def power_method(matrix, initialEV, error, n):
    if matrix.shape[0] != matrix.shape[0]:
        print "matrix must be square"
        return
    i = 0
    lastEV = 0
    ev = normalize(initialEV)
    ev = np.matrix(ev)
    vectorY = util.multiplyMatrices(matrix, ev)
    ev = normalize(vectorY)
    ev = np.matrix(ev)
    evTransposed = ev.getT()
    numeratorMatrix = util.multiplyMatrices(evTransposed, matrix)
    numeratorMatrix = util.multiplyMatrices(numeratorMatrix, ev)
    numerator = numeratorMatrix[0, 0]
    denominatorMatrix = util.multiplyMatrices(evTransposed, ev)
    denominator = denominatorMatrix[0, 0]
    newEV = numerator / denominator
    while (i < n or abs(newEV - lastEV) > error):
        lastEV = newEV
        ev = normalize(initialEV)
        ev = np.matrix(ev)
        vectorY = util.multiplyMatrices(matrix, ev)
        ev = normalize(vectorY)
        ev = np.matrix(ev)
        evTransposed = ev.getT()
        numeratorMatrix = util.multiplyMatrices(evTransposed, matrix)
        numeratorMatrix = util.multiplyMatrices(numeratorMatrix, ev)
        numerator = numeratorMatrix[0, 0]
        denominatorMatrix = util.multiplyMatrices(evTransposed, ev)
        denominator = denominatorMatrix[0, 0]
        newEV = numerator / denominator
        i += 1
    return [newEV, ev, i]


def normalize(v):
    norm = LA.norm(v)
    return norm * v


for x in range(0, 999):
    matrixR = np.matrix([
        [random.uniform(-2, 2), random.uniform(-2, 2)],
        [random.uniform(-2, 2), random.uniform(-2, 2)]
    ])

    if matrixR[0, 0] * matrixR[1, 1] - matrixR[0, 1] * matrixR[1, 0] != 0:
        estimate = np.matrix([[1],
                             [0]])
        print power_method(matrixR, estimate, 0.00005, 100)