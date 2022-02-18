import numpy as np
from itertools import product
from LPsolver import lp
from math import factorial

def oneroundLP(w, f):
    def ffunc(players):
        return f[len(players)]

    def wfunc(players):
        return w[len(players)]

    def nashfunc(jselect, NJ):
        if jselect == 1:
            return ffunc(NJ + [j])
        elif jselect == 3:
            return -ffunc(NJ + [j])
        else:
            return 0

    n = len(w)-1  # number of players
    n_c = 4**n - 1  # number of allocation types
    partition = list(product([1, 2, 3, 4], repeat=n))[:-1]  # resource types

    c = np.zeros((n_c,), dtype='float')
    cons_1 = np.zeros((n, n_c), dtype='float')
    A = np.zeros((1, n_c), dtype='float')

    for j in range(n):
        for i, p in enumerate(partition):
            # allocations for all agents
            Na = [k for k in range(n) if p[k]==1]
            Nx = [k for k in range(n) if p[k]==2]
            Nb = [k for k in range(n) if p[k]==3]

            # allocations for first j agents
            NJ = [k for k in range(j) if p[k]<=2]
            jselect= p[j]

            c[i] = -wfunc(Nb + Nx)  # maximize welfare of optimal allocation
            A[0, i] = wfunc(Na + Nx)  # set welfare of 1 round best response to 1

            cons_1[j][i] = nashfunc(jselect, NJ)  # best response constraint

    cons_2 = np.identity(n_c)
    G = -np.vstack((cons_1, cons_2))
    h = np.zeros((n_c+n, 1))
    b = np.array([[1]], dtype='float')
    
    return c, G, h, A, b

def get_sol(w, f):
    args = oneroundLP(w, f)
    sol = lp('cvxopt', *args)
    if sol is not None:
        print(f, -round(sol['min']**-1, 3))

    n = len(w)-1
    partition = list(product([1, 2, 3, 4], repeat=n))[:-1]
    for i, p in enumerate(partition):
        val = round(sol['argmin'][i], 4)
        if val != 0:
            print(p, val)
    


w = [0., 1., 1.5]
f = [0., 1., .5]

poa = 0
# for f3 in np.arange(0, 2, .1):
#     f[3] = f3
#     for f2 in np.arange(0, 2, .1):
#         f[2] = f2
#         args = oneroundLP(w, f)
#         sol = lp('cvxopt', *args)
#         if sol is not None:
#             poa_temp = -round(sol['min']**-1, 3)
#         if poa_temp >= poa:
#             poa = poa_temp
# print(f, poa)

#f[2] = w[2] - w[1]
#f[3] = w[3] - w[2]


def gairing(n):
    B = 1./(n-1)/factorial(n-1)
    y = lambda j: sum([1./factorial(i) for i in range(j, n)])
    return [0] + [round(factorial(j-1)*(B + y(j))/(B + y(1)), 3) for j in range(1, n+1)]

print(gairing(4))

get_sol(w, f)