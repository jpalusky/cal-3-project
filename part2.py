#Part 2 - Final Project
from util import multiplyMatrices
import math
import numpy as np
import random
import xlwt
A = np.matrix([
    [1., 1./2, 1./3],
    [1./2, 1., 1./4],
    [1./3, 1./4, 1.]
])

B = np.matrix([
    [0.1],
    [0.1],
    [0.1]
])

x_exact = np.matrix([
    [9.0/190.],
    [28./475],
    [33./475]
])

def dot_product(x1, x2):
    return np.dot(np.transpose(x1), x2)

def inverse_lower_3x3(matrix):
    ans = np.zeros((3, 3))
    a, b, c = matrix[0, 0], matrix[1, 0], matrix[1, 1]
    d, e, f = matrix[2, 0], matrix[2, 1], matrix[2, 2]
    ans[0, 0] = 1. / a
    ans[1, 0] = float(-b) / (a * c)
    ans[1, 1] = 1. / c
    ans[2, 0] = float(-c * d + b * e) / (a * c * f)
    ans[2, 1] = float(-e) / (c * f)
    ans[2, 2] = 1. / f
    return ans

def iter(x0, sigma, m, S, T):
    S_inv = inverse_lower_3x3(S)
    counter = 1
    x_cur = x0
    while (counter <= m):
        x_prev = x_cur
        x_cur = multiplyMatrices(S_inv, multiplyMatrices(T, x_cur) +  B)
        err = x_cur - x_prev
        if math.sqrt(dot_product(err, err)) <= sigma:
            return (x0, x_cur, counter)
        counter += 1
    return None

def jacobi_iter(x0, sigma, m):
    S = np.zeros((3, 3))
    for r in range(A.shape[0]):
            S[r, r] = A[r, r]
    T = -(A - S)
    return iter(x0, sigma, m, S, T)

def gs_iter(x0, sigma, m):
    S = np.zeros((3, 3))
    for r in range(A.shape[0]):
        for c in range(r + 1):
            S[r, c] = A[r, c]
    T = -(A - S)
    return iter(x0, sigma, m, S, T)

def vector_100():
    jacobi_list = []
    gs_list = []

    for i in range(100):
        x0 = np.matrix([
            [random.uniform(-1, 1)],
            [random.uniform(-1, 1)],
            [random.uniform(-1, 1)]
        ])
        jacobi_list.append(jacobi_iter(x0, 0.00005, 100))
        gs_list.append(gs_iter(x0, 0.00005, 100))

    return (jacobi_list, gs_list)

def getAverage(jacobi_list, gs_list):
    x_jacobi = np.zeros((3, 1))
    x_gs = np.zeros((3, 1))
    count_ratio = 0.

    for i in range(100):
        x_jacobi += jacobi_list[i][1]
        x_gs += gs_list[i][1]
        count_ratio += float(jacobi_list[i][2]) / gs_list[i][2]

    x_jacobi /= 100
    x_gs /= 100
    count_ratio /= 100
    err_jacobi = x_jacobi - x_exact
    err_gs = x_gs - x_exact
    err_jacobi = math.sqrt(dot_product(err_jacobi, err_jacobi))
    err_gs = math.sqrt(dot_product(err_gs, err_gs))
    return (x_jacobi, x_gs, err_jacobi, err_gs, count_ratio)

def plot(filename):

    book = xlwt.Workbook()
    sheet1 = book.add_sheet("Sheet1")
    sheet1.write(0, 0, "Jacobi")
    sheet1.write(0, 3, "Gauss-Seidel")
    sheet1.write(1, 0, "Initial error")
    sheet1.write(1, 1, "steps")
    sheet1.write(1, 3, "Initial error")
    sheet1.write(1, 4, "steps")
    sheet1.write(0, 6, "x-appr-jacobi")
    sheet1.write(1, 6, "x-appr-gs")
    sheet1.write(2, 6, "err-jacobi")
    sheet1.write(3, 6, "err-gs")
    sheet1.write(4, 6, "count-ratio-ave")

    jacobi_list, gs_list = vector_100()
    for i in range(100):
        ini_error = jacobi_list[i][1] - jacobi_list[i][0]
        ini_error = math.sqrt(dot_product(ini_error, ini_error))
        sheet1.write(i + 2, 0, ini_error)
        sheet1.write(i + 2, 1, jacobi_list[i][2])

        ini_error = gs_list[i][1] - gs_list[i][0]
        ini_error = math.sqrt(dot_product(ini_error, ini_error))
        sheet1.write(i + 2, 3, ini_error)
        sheet1.write(i + 2, 4, gs_list[i][2])


    x_jacobi, x_gs, err_jacobi, err_gs, count_ratio = getAverage(jacobi_list, gs_list) 
    sheet1.write(0, 7, "{}".format(x_jacobi))
    sheet1.write(1, 7, "{}".format(x_gs))
    sheet1.write(2, 7, "{}".format(err_jacobi))
    sheet1.write(3, 7, "{}".format(err_gs))
    sheet1.write(4, 7, "{}".format(count_ratio))
   
    book.save(filename)

# x0 = np.matrix([
#     [-3],
#     [-0.1],
#     [5]
# ])
# x0, soln, counter = jacobi_iter(x0, 0.00005, 100)
# print("x0: {}".format(x0))
# print("solution: {}".format(soln))
# print("number of it'ns: {}".format(counter))

plot("math2605_project_part2_plot_2.xls")

