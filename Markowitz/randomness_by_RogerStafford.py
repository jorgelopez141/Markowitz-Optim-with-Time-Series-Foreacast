#%%
import numpy as np

# def randfixedsum(n, m, s, a, b):
#     # Check the arguments
#     if not (m == int(m) and n == int(n) and m >= 0 and n >= 1):
#         raise ValueError("n must be a whole number and m a non-negative integer.")
#     if not (n * a <= s <= n * b and a < b):
#         raise ValueError("Inequalities n*a <= s <= n*b and a < b must hold.")
    
#     # Rescale to a unit cube: 0 <= x(i) <= 1
#     s = (s - n * a) / (b - a)
    
#     # Construct the transition probability table, t
#     k = max(min(int(np.floor(s)), n - 1), 0)
#     s = max(min(s, k + 1), k)
#     s1 = s - np.arange(k, k - n, -1)
#     s2 = np.arange(k + n, k, -1) - s
#     w = np.zeros((n, n + 1))
#     w[0, 1] = np.finfo(float).max
#     t = np.zeros((n - 1, n))
    
#     tiny = np.finfo(float).tiny
#     for i in range(1, n):
#         tmp1 = w[i-1, 1:i+2] * s1[:i+1] / i
#         tmp2 = w[i-1, :i+1] * s2[-i-1:] / i
#         w[i, 1:i+2] = tmp1 + tmp2
#         tmp3 = w[i, 1:i+2] + tiny
#         tmp4 = (s2[-i-1:] > s1[:i+1])
#         t[i-1, :i+1] = (tmp2 / tmp3) * tmp4 + (1 - tmp1 / tmp3) * (~tmp4)
    
#     # Derive the polytope volume v
#     v = n ** (3 / 2) * (w[n-1, k+1] / np.finfo(float).max) * (b - a) ** (n - 1)
    
#     # Compute the matrix x
#     x = np.zeros((n, m))
#     if m == 0:
#         return x, v

#     rt = np.random.rand(n - 1, m)
#     rs = np.random.rand(n - 1, m)
#     s = np.full(m, s)
#     j = np.full(m, k + 1, dtype=int)
#     sm = np.zeros(m)
#     pr = np.ones(m)
    
#     for i in range(n - 2, -1, -1):
#         e = (rt[n - 2 - i, :] <= t[i, j - 1])
#         sx = rs[n - 2 - i, :] ** (1 / (i + 1))
#         sm += (1 - sx) * pr * s / (i + 2)
#         pr *= sx
#         x[n - 2 - i, :] = sm + pr * e
#         s -= e
#         j -= e
    
#     x[n - 1, :] = sm + pr * s
    
#     # Randomly permute the order in the columns of x and rescale
#     for i in range(m):
#         x[:, i] = np.random.permutation(x[:, i])
    
#     x = (b - a) * x + a
#     return x, v

# # Example usage
# x, v = randfixedsum(5, 10, 100, 0, 100)
# print("printing x")
# print(x)
# print("printing v")
# print(v)

#%%

import numpy as np

def randfixedsum(n, m, s, a, b):
    if not (m == int(m) and n == int(n) and m >= 0 and n >= 1):
        raise ValueError("n must be a whole number and m a non-negative integer.")
    if not (n * a <= s <= n * b and a < b):
        raise ValueError("Inequalities n*a <= s <= n*b and a < b must hold.")
    
    s = (s - n * a) / (b - a)
    k = max(min(int(np.floor(s)), n - 1), 0)
    s = max(min(s, k + 1), k)
    s1 = s - np.arange(k, k - n, -1)
    s2 = np.arange(k + n, k, -1) - s
    w = np.zeros((n, n + 1))
    w[0, 1] = 1.0  # Scale down to avoid overflow
    t = np.zeros((n - 1, n))
    
    tiny = np.finfo(float).tiny
    for i in range(1, n):
        tmp1 = w[i-1, 1:i+2] * s1[:i+1] / i
        tmp2 = w[i-1, :i+1] * s2[-i-1:] / i
        w[i, 1:i+2] = tmp1 + tmp2
        tmp3 = w[i, 1:i+2] + tiny
        tmp4 = (s2[-i-1:] > s1[:i+1])
        t[i-1, :i+1] = (tmp2 / tmp3) * tmp4 + (1 - tmp1 / tmp3) * (~tmp4)
    
    v = n ** (3 / 2) * (w[n-1, k+1]) * (b - a) ** (n - 1)
    x = np.zeros((n, m))
    if m == 0:
        return x, v

    rt = np.random.rand(n - 1, m)
    rs = np.random.rand(n - 1, m)
    s = np.full(m, s)
    j = np.full(m, k + 1, dtype=int)
    sm = np.zeros(m)
    pr = np.ones(m)
    
    for i in range(n - 2, -1, -1):
        e = (rt[n - 2 - i, :] <= t[i, j - 1])
        sx = rs[n - 2 - i, :] ** (1 / (i + 1))
        sm += (1 - sx) * pr * s / (i + 2)
        pr *= sx
        x[n - 2 - i, :] = sm + pr * e
        s -= e
        j -= e
    
    x[n - 1, :] = sm + pr * s

    for i in range(m):
        x[:, i] = np.random.permutation(x[:, i])
    
    x = (b - a) * x + a
    return x, v

#Example usage
#x, v = randfixedsum(10000, 30, 1, 0, 1)






