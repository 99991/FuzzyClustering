import numpy as np

def random_positive_semidefinite_matrix(d):
    Q = np.random.randn(d, d)
    eigvals = np.random.rand(d)
    return Q.T @ np.diag(eigvals) @ Q
    
    while True:
        A = np.random.rand(d, d)
        A += A.T
        if np.all(np.linalg.eigvals(A) > 0):
            return A

def make_spiral_clusters(c, cluster_size, n_noise, d=2):
    angle = np.linspace(0, 2*np.pi, c, endpoint=False)
    radius = np.linspace(10, 30, c)
    vx = np.cos(angle)*radius
    vy = np.sin(angle)*radius
    v = np.stack([vx, vy], axis=1)

    covariances = np.array([random_positive_semidefinite_matrix(d) for _ in range(c)])

    x = np.concatenate([np.random.multivariate_normal(v[i], covariances[i], cluster_size)
        for i in range(c)], axis=0)

    u = np.random.rand(n_noise, d)
    noise = np.min(x, axis=0)*u + (1 - u)*np.max(x, axis=0)

    x = np.concatenate([x, noise], axis=0)

    return x, v
