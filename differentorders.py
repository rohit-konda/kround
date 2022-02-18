from LPsolver import lp
from itertools import product
import numpy as np
import warnings
warnings.filterwarnings('ignore')




def partition(acN, n): return list(product(product([1, 0], repeat=acN), repeat=n))[:-1]

def cover(rSig, act, n):
    # rSig: what players cover, act:list of actions
    cov = [0]*n
    for i in range(n):
        if act[i] != -1 and rSig[i][act[i]] == 1:
            cov[i] = 1
    return cov

def UtilCons(part, f, i, a1, a2, acNoti, n):
    # constraint according to partition that has U_i(act1) >= U_i(act2)
    brCons = np.zeros((1, len(part)), dtype='float')

    for j, p in enumerate(part):
        sel1 = p[i][a1]
        sel2 = p[i][a2]
        if sel1 == 1 and sel2 == 0:
            brCons[0][j] = f[sum(cover(p, acNoti, n))+1]
        elif sel1 == 0 and sel2 == 1:
            brCons[0][j] = -f[sum(cover(p, acNoti, n))+1]

    return brCons

def acWelf(part, w, n, act):
    # welfare of action act (list) based on resource partition
    welfArr = np.zeros((len(part),), dtype='float')
    for i, p in enumerate(part):
        welfArr[i] = w[sum(cover(p, act, n))]
    return welfArr

def BrUtil(part, f, n):
    constraints = []
    for i in range(n):
        for perm in list(product([1, 0], repeat=n)):
            acNoti = [e1 + e2 for e1, e2 in zip([-1]*n, list(perm))]
        constraints.append(UtilCons(part, f, i, 0, 1, acNoti, n))
    
    return np.vstack(tuple(constraints))


def arbitraryorder(w, f):
    n = len(w)-1
    n_ac = 2
    part = partition(n_ac, n)
    n_c = len(part)
    abr = 0
    aopt = 1

    C = acWelf(part, w, n, [abr]*n)
    G = -np.vstack((BrUtil(part, f, n), np.identity(n_c)))
    H = np.zeros((len(G), 1))
    A = np.expand_dims(acWelf(part, w, n, [aopt]*n), axis=0)
    B = np.ones((1, 1))

    return (C, G, H, A, B)

if __name__ == '__main__':
    w = [0, 1, 1, 1]
    f = [0, 1, 0, 0]
    args = arbitraryorder(w, f)
    sol = lp('cvxopt', *args)
    print(sol['min'])
