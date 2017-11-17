Performance comparison of different methods to calculate covariance matrices.

Results:
```
python        : 1.06358322 seconds
numba jit     : 0.10889833 seconds (  9.8 times faster)
numpy         : 0.01389834 seconds ( 76.5 times faster)
einsum x,v    : 0.00571945 seconds (186.0 times faster)
einsum x,v,u  : 0.01894124 seconds ( 56.2 times faster)
einsum x,v,u,c: 0.02027552 seconds ( 52.5 times faster)
cython        : 0.00544233 seconds (195.4 times faster)
c             : 0.00196037 seconds (542.5 times faster)
numba jit more: 0.00224946 seconds (472.8 times faster)
```

Note that the numba jit code has to be unrolled all the way to be fast.
