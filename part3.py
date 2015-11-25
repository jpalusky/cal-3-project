import numpy as np
import matplotlib.pyplot as plt
import random
import util

def power_method(matrix, ev, error, n, inverse):
    if matrix.shape[0] != matrix.shape[0]:
        print "matrix must be square"
        return
    else:
        matrixU = ev
        matrixW = np.zeros(shape=(ev.shape[0],1))
        matrixW[0,0]=1
        eValue = 0

        for i in range(1, n):
            eValueOld = eValue
            oldMatrixU = matrixU
            matrixU = util.multiplyMatrices(matrix, matrixU)
            eValue = np.dot(matrixW.transpose(), matrixU) / np.dot(matrixW.transpose(),
                                                                               oldMatrixU)
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
    largest=[]
    smallest=[]
    matrixR = np.matrix([
        [random.uniform(-2, 2), random.uniform(-2, 2)],
        [random.uniform(-2, 2), random.uniform(-2, 2)]
    ])
    if matrixR[0, 0] * matrixR[1, 1] - matrixR[0, 1] * matrixR[1, 0] != 0:
        estimate = np.matrix([[1],
                              [0]])
        largest = power_method(matrixR, estimate, 0.00005, 100, False)
        if largest != "failure":
            # print matrixR
            # print "Largest  " + str(power_method(matrixR, estimate, 0.00005, 100))
            trace = str(util.trace2x2(matrixR))
            determinant = str(util.determinant2x2(matrixR))
            x1.append(determinant)
            y1.append(trace)
            c1.append(largest[2])
        matrixRinverse = util.inverse2x2(matrixR)
        smallest = power_method(matrixRinverse, estimate, 0.00005, 100, True)
        out = ""
        if smallest != "failure":
            # print "Smallest eigenvalue " + smallest
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

matrixTest = np.matrix([
    [1, 1, 1, 1],
    [1, 2, 3, 4],
    [1, 3, 6, 10],
    [1, 4, 10, 20]
])
estimate = np.matrix([[1],
                      [0],
                      [0],
                      [0]])
result=power_method(matrixTest,estimate,0.00005, 100, False)
print 'Largest Eigenvalue ' + str(result[0])
print "Largest Eigenvector " + str(result[1])
print "Number of iterations " + str(result[2])
print ""


matrixF = np.matrix([
    [2, -1, 1],
    [3, 3, 9],
    [3, 3, 5]
])
estimate = np.matrix([[1],
                      [0],
                      [0]])
result=power_method(matrixF,estimate,0.00005, 100, False)
# plt.scatter(x1, y1, c=c1, cmap='seismic')
# plt.xlabel('determinant')
# plt.ylabel('trace')
# plt.title("Power Method on matrix A")
#
# plt.show()
# plt.scatter(x2, y2, c=c2, cmap='seismic')
# plt.xlabel('determinant')
# plt.ylabel('trace')
# plt.title(r'Power method on matrix $A^{-1}$')
# plt.show()
# fig = plt.figure()
# ax1 = fig.add_subplot(211)
# ax1.scatter(x1, y1, c=c1, cmap='seismic')
# ax2 = fig.add_subplot(212)
# ax2.scatter(x2, y2, c=c2, cmap='seismic')
# ax1.set_title('Matrix A')
# ax1.set_xlabel('determinant')
# ax1.set_ylabel('trace')
# ax2.set_title('Matrix A Inverse')
# ax2.set_xlabel('determinant')
# ax2.set_ylabel('trace')
# plt.tight_layout(pad=0.4, w_pad=0, h_pad=1.0)
# plt.show()
