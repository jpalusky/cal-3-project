import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
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

def normalize(v):
    norm = LA.norm(v)
    return norm * v


x1=[]
y1=[]
c1=[]
x2=[]
y2=[]
c2=[]
for i in range(0, 999):
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
            determinant=str(util.determinant2x2(matrixR))
            x1.append(determinant)
            y1.append(trace)
            c1.append(largest[2])
        matrixR = util.inverse2x2(matrixR)
        smallest=power_method(matrixR, estimate, 0.00005, 100)
        out=""
        if smallest!="failure":
            #print "Smallest eigenvalue " + smallest
            trace=str(util.trace2x2(matrixR))
            determinant=str(util.determinant2x2(matrixR))

            x2.append(determinant)
            y2.append(trace)
            c2.append(smallest[2])



fig = plt.figure()

ax1 = fig.add_subplot(211)
ax1.scatter(x1, y1, c=c1, cmap='seismic')
ax2 = fig.add_subplot(212)
ax2.scatter(x2, y2, c=c2, cmap='seismic')

plt.show()