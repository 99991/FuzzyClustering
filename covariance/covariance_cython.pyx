from numpy cimport ndarray
cimport numpy as np
cimport cython
from libc.stdlib cimport malloc, free

@cython.boundscheck(False)
def calculate_covariances8(
    ndarray[np.float64_t, ndim=2] x not None,
    ndarray[np.float64_t, ndim=2] v not None,
    ndarray[np.float64_t, ndim=2] ums not None,
    ndarray[np.float64_t, ndim=3] covariances not None
):
    cdef Py_ssize_t n = x.shape[0]
    cdef Py_ssize_t c = v.shape[0]
    cdef Py_ssize_t d = v.shape[1]
    
    cdef Py_ssize_t i, j, p, q
    cdef double sum_u, um
    cdef double *xv = <double *>malloc(d * sizeof(double))
    
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

    free(xv)

