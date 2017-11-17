import os
from ctypes import *

if 1:
    # compile cython
    command = r'..\..\Anaconda3\python setup.py build_ext --inplace'
    os.system(command)

    # compile c
    os.system("gcc covariance.c -shared -O3 -s -march=native -o covariance.dll")

dll = cdll.LoadLibrary("covariance.dll")
import covariance_cython
import numpy as np
from numba import jit
import time, sys
np.random.seed(0)

def random_positive_semidefinite_matrix(d):
    A = np.random.rand(d, d)
    return np.dot(A, A.T)

c = 7
n_by_c = 20*100
m = 2.0
d = 3

v = np.random.rand(c, d)
covariances1 = np.array([random_positive_semidefinite_matrix(d) for _ in range(c)])

x = np.concatenate([
    np.random.multivariate_normal(v[i], covariances1[i], n_by_c)
    for i in range(c)], axis=0)

u = np.zeros((c, n_by_c*c))
for i in range(c):
    u[i, i*n_by_c:i*n_by_c+n_by_c] = 1

n = n_by_c*c

um = u**m

def calculate_covariances2(x, v, um):
    covariances = np.zeros((c, d, d))
    
    for i in range(c):
        for k in range(n):
            xv = (x[k] - v[i]).reshape((d, 1))
            
            covariances[i] += um[i, k]*np.dot(xv, xv.T)

        covariances[i] /= np.sum(um[i])
    
    return covariances

@jit(cache=True)
def calculate_covariances3(x, v, um):
    covariances = np.zeros((c, d, d))
    
    for i in range(c):
        for k in range(n):
            xv = (x[k] - v[i]).reshape((d, 1))
            
            covariances[i] += um[i, k]*np.dot(xv, xv.T)

        covariances[i] /= np.sum(um[i])
    
    return covariances

def calculate_covariances4(x, v, um):
    covariances = np.zeros((c, d, d))
    
    for i in range(c):
        xv = (x - v[i]).reshape((n, 1, d))
        uxv = um[i].reshape((n, 1, 1))*xv
        uxv = uxv.reshape((n, d, 1))

        covariances[i] = np.sum(np.matmul(uxv, xv), axis=0)

        covariances[i] /= np.sum(um[i])
    
    return covariances

def calculate_covariances5(x, v, um):
    covariances = np.zeros((c, d, d))

    for i in range(c):
        xv = x - v[i]
        uxv = um[i, :, np.newaxis]*xv
        covariances[i] = np.einsum('ni,nj->ij', uxv, xv)/np.sum(um[i])
    
    return covariances

def calculate_covariances6(x, v, um):
    covariances = np.zeros((c, d, d))

    for i in range(c):
        xv = x - v[i]
        covariances[i] = np.einsum('n,n...i,nj...->ij', um[i], xv, xv)/np.sum(um[i])
    
    return covariances

def calculate_covariances7(x, v, um):
    xv = x[np.newaxis, :] - v[:, np.newaxis, :]
    return np.einsum('cn,cn...i,cnj...->cij', um, xv, xv)/np.sum(um, axis=1)[:, np.newaxis, np.newaxis]

@jit(cache=True)
def calculate_covariances10(x, v, ums):
    n = x.shape[0]
    c = v.shape[0]
    d = v.shape[1]
    
    covariances = np.zeros((c, d, d))

    xv = np.zeros(d)
    
    for i in range(c):
        sum_u = 0.0
        for k in range(n):
            um = ums[i, k]
            sum_u += um

            for p in range(d):
                xv[p] = x[k, p] - v[i, p]

            for p in range(d):
                for q in range(d):
                    covariances[i, p, q] += um * xv[p] * xv[q]

        covariances[i] /= sum_u
    
    return covariances

for _ in range(5):
    t1 = time.clock()

    covariances2 = calculate_covariances2(x, v, um)

    t2 = time.clock()

    covariances3 = calculate_covariances3(x, v, um)

    t3 = time.clock()

    covariances4 = calculate_covariances4(x, v, um)

    t4 = time.clock()

    covariances5 = calculate_covariances5(x, v, um)

    t5 = time.clock()

    covariances6 = calculate_covariances6(x, v, um)

    t6 = time.clock()

    covariances7 = calculate_covariances7(x, v, um)

    t7 = time.clock()

    covariances8 = np.zeros((c, d, d))
    covariance_cython.calculate_covariances8(x, v, um, covariances8)

    t8 = time.clock()

    covariances9 = np.zeros((c, d, d))

    dll.calculate_covariances9.argtypes = [
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
        
        c_int,
        c_int,
        c_int]

    dll.calculate_covariances9(
        x.ctypes.data,
        v.ctypes.data,
        u.ctypes.data,
        covariances9.ctypes.data,
        n, c, d)

    t9 = time.clock()

    covariances10 = calculate_covariances10(x, v, um)

    t10 = time.clock()

    print("python        : %.8f seconds"%(t2 - t1))
    print("numba jit     : %.8f seconds (%5.1f times faster)"%(t3 - t2, (t2 - t1)/(t3 - t2)))
    print("numpy         : %.8f seconds (%5.1f times faster)"%(t4 - t3, (t2 - t1)/(t4 - t3)))
    print("einsum x,v    : %.8f seconds (%5.1f times faster)"%(t5 - t4, (t2 - t1)/(t5 - t4)))
    print("einsum x,v,u  : %.8f seconds (%5.1f times faster)"%(t6 - t5, (t2 - t1)/(t6 - t5)))
    print("einsum x,v,u,c: %.8f seconds (%5.1f times faster)"%(t7 - t6, (t2 - t1)/(t7 - t6)))
    print("cython        : %.8f seconds (%5.1f times faster)"%(t8 - t7, (t2 - t1)/(t8 - t7)))
    print("c             : %.8f seconds (%5.1f times faster)"%(t9 - t8, (t2 - t1)/(t9 - t8)))
    print("numba jit more: %.8f seconds (%5.1f times faster)"%(t10 - t9, (t2 - t1)/(t10 - t9)))
    print("")

print("max difference:")
print(np.max(np.abs(covariances1 - covariances2)))
print(np.max(np.abs(covariances1 - covariances3)))
print(np.max(np.abs(covariances1 - covariances4)))
print(np.max(np.abs(covariances1 - covariances5)))
print(np.max(np.abs(covariances1 - covariances6)))
print(np.max(np.abs(covariances1 - covariances7)))
print(np.max(np.abs(covariances1 - covariances8)))
print(np.max(np.abs(covariances1 - covariances9)))
print(np.max(np.abs(covariances1 - covariances10)))

if 0:
    px,py = x.T
    plt.plot(px, py, 'bo')
    px,py = v.T
    plt.plot(px, py, 'ro')
    plt.show()


