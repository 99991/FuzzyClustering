import numpy as np
import scipy.spatial

def pairwise_squared_distances(A, B):
    return scipy.spatial.distance.cdist(A, B)**2

def calculate_covariances(x, u, v, m):
    c, n = u.shape
    d = v.shape[1]
    
    um = u**m

    covariances = np.zeros((c, d, d))

    for i in range(c):
        xv = x - v[i]
        uxv = um[i, :, np.newaxis]*xv
        covariances[i] = np.einsum('ni,nj->ij', uxv, xv)/np.sum(um[i])
    
    return covariances

def pc(x, u, v, m):
    c, n = u.shape
    return np.square(u).sum()/n

def npc(x, u, v, m):
    n, c = u.shape
    return 1 - c/(c - 1)*(1 - pc(x, u, v, m))

def fhv(x, u, v, m):
    covariances = calculate_covariances(x, u, v, m)
    return sum(np.sqrt(np.linalg.det(cov)) for cov in covariances)

def fs(x, u, v, m):
    n = x.shape[0]
    c = v.shape[0]

    um = u**m

    v_mean = v.mean(axis=0)

    d2 = pairwise_squared_distances(x, v)
    
    distance_v_mean_squared = np.linalg.norm(v - v_mean, axis=1, keepdims=True)**2

    return np.sum(um.T*d2) - np.sum(um*distance_v_mean_squared)

def xb(x, u, v, m):
    n = x.shape[0]
    c = v.shape[0]

    um = u**m
    
    d2 = pairwise_squared_distances(x, v)
    v2 = pairwise_squared_distances(v, v)
    
    v2[v2 == 0.0] = np.inf

    return np.sum(um.T*d2)/(n*np.min(v2))

def bh(x, u, v, m):
    n, d = x.shape
    c = v.shape[0]

    d2 = pairwise_squared_distances(x, v)
    v2 = pairwise_squared_distances(v, v)
    
    v2[v2 == 0.0] = np.inf

    V = np.sum(u*d2.T, axis=1)/np.sum(u, axis=1)

    return np.sum(u**m*d2.T)/n*0.5*np.sum(np.outer(V, V)/v2)

def bws(x, u, v, m):
    n, d = x.shape
    c = v.shape[0]

    x_mean = x.mean(axis=0)

    covariances = calculate_covariances(x, u, v, m)

    sep = np.einsum("ik,ij->", u**m, np.square(v - x_mean))
    comp = sum(np.trace(covariance) for covariance in covariances)

    return sep/comp

methods = [pc, npc, fhv, fs, xb, bh, bws]
targets = "max max min min min min max".split()
