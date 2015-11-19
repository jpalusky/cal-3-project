import numpy as np
from numpy import linalg as LA
import random
import util



def power_method(matrix, ev, error, n):
    if matrix.shape[0] != matrix.shape[0]:
        print "matrix must be square"
        return
    else:
        eValue=0
        eVector=ev
        for i in range(1, n):
            eValueOld=eValue
            eVectorOld=eVector
            eVector=matrix*eVector
            eValue=eVector[0,0]
            eVector=1/eValue*eVector
            if(abs(eValueOld-eValue)<error):
                return [eValue, eVector, i]
    return "failure"


    # i = 1
    # lastEV = 1
    # ev = normalize(ev)
    # ev = np.matrix(ev)
    # vectorY = util.multiplyMatrices(matrix, ev)
    # ev = normalize(vectorY)
    # ev = np.matrix(ev)
    # evTransposed = ev.getT()
    # numeratorMatrix = util.multiplyMatrices(evTransposed, matrix)
    # numeratorMatrix = util.multiplyMatrices(numeratorMatrix, ev)
    # numerator = numeratorMatrix[0, 0]
    # denominatorMatrix = util.multiplyMatrices(evTransposed, ev)
    # denominator = denominatorMatrix[0, 0]
    # newEV = numerator / denominator
    # while (i < n and abs(newEV - lastEV) > error):
    #     lastEV = newEV
    #     ev = normalize(ev)
    #     ev = np.matrix(ev)
    #     vectorY = util.multiplyMatrices(matrix, ev)
    #     ev = normalize(vectorY)
    #     ev = np.matrix(ev)
    #     evTransposed = ev.getT()
    #     numeratorMatrix = util.multiplyMatrices(evTransposed, matrix)
    #     numeratorMatrix = util.multiplyMatrices(numeratorMatrix, ev)
    #     numerator = numeratorMatrix[0, 0]
    #     denominatorMatrix = util.multiplyMatrices(evTransposed, ev)
    #     denominator = denominatorMatrix[0, 0]
    #     newEV = numerator / denominator
    #     i += 1
    # if(i<n):
    #     return [newEV, ev, i]
    # else:
    #     return "failure"


def normalize(v):
    norm = LA.norm(v)
    return norm * v


plot1 = open("plot1.txt", "w")
plot2 = open("plot2.txt", "w")

for x in range(0, 999):
    out = ""
    matrixR = np.matrix([
        [random.uniform(-2, 2), random.uniform(-2, 2)],
        [random.uniform(-2, 2), random.uniform(-2, 2)]
    ])
    if matrixR[0, 0] * matrixR[1, 1] - matrixR[0, 1] * matrixR[1, 0] != 0:
        estimate = np.matrix([[1],
                              [0]])
        largest=power_method(matrixR, estimate, 0.00005, 100)
        if largest!="failure":
            #print "Largest  " + str(power_method(matrixR, estimate, 0.00005, 100))
            trace=str(util.trace2x2(matrixR))
            #print "trace " + trace
            determinant=str(util.determinant2x2(matrixR))
            #print "determinant " + determinant
            out=out+trace+","+determinant+","+str(largest[2])+"\n"
            plot1.write(out)
        matrixR = util.inverse2x2(matrixR)
        smallest=power_method(matrixR, estimate, 0.00005, 100)
        out=""
        if smallest!="failure":
            #print "Smallest eigenvalue " + smallest
            trace=str(util.trace2x2(matrixR))
            determinant=str(util.determinant2x2(matrixR))
            out=out+trace+","+determinant+","+str(smallest[2])+"\n"
            plot2.write(out)
plot1.close()
plot2.close()
