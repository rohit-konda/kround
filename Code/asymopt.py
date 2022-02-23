from LPsolver import lp
from itertools import product
import numpy as np
from games.price_of_anarchy import res_poa, res_opt_f, worst_game
import sys
import matplotlib.pyplot as plt

np.set_printoptions(threshold=sys.maxsize)

def nashcons(w, n, mu):
    numCons = int(n*(n+1)/2 - 1)
    G2 = np.zeros((numCons, n), dtype='float')
    h2 = np.zeros((numCons, 1), dtype='float')
    i = 0
    for a in range(1, n):
        for b in range(0, a+1):
            if a > 1:
                G2[i, a-1] =  float(a)
                h2[i, 0] = w[a]*mu - w[b]
            else:
                h2[i, 0] = w[a]*mu - w[b] - 1.
            G2[i, a] = -float(b)
            i += 1
    return G2[1:, :], h2[1:,:]

def brcons(w, n):
    numCons = int(n*(n+1)/2 - 1)
    G1 = np.zeros((numCons, n), dtype='float')
    h1 = np.zeros((numCons, 1), dtype='float')
    i = 0
    for a in range(1, n):
        for b in range(0, a+1):
            G1[i, 0] = -w[a]
            G1[i,1:a] = 1.
            G1[i, a] = -float(b)
            h1[i, 0] = -w[b] - 1.
            #print(a, b, 'CHECK:', G1[i, :], h1[i, :])
            i += 1
    return G1[[0, 1, 2, 3, 4, 5, 7, 8], :], h1[[0, 1, 2, 3, 4, 5, 7, 8], :]

def program(w, n, mu):
    c = np.zeros((n,))
    c[0] = 1

    G1,h1 = brcons(w, n)
    G2, h2 = nashcons(w, n, mu)

    G = np.vstack((G1, G2))
    h = np.vstack((h1, h2))

    return c, G, h

def getOpt(w, n, mu, returnall=False):
    args = program(w, n, mu)
    sol = lp('cvxopt', *args, returnall=returnall)
    if sol is not None:
        pob = round(sol['min']**-1, 4)
        f = [0, 1] + [round(e, 4) for e in sol['argmin'][1:]]
        return pob, f

def fopt(n, c):
    f = [0., 1.] + [0.]*(n-1)
    for i in range(1, n):
        f[i+1] = i * f[i] - (1 + (1-c)*(i-1))*(1-c/np.exp(1))**-1 + 1
    return f

def abcovering(n, a, b):
    from math import factorial
    p = (1 - a*b**b*np.exp(-b)/factorial(b))**-1
    f = [0, 1.]
    for i in range(1, n):
        val1 = 1 - a
        Vab = (1-a)*i + a*min(i, b)
        val2 = 1/b*(i*f[i] - Vab*p) + 1
        val = max(val1, val2)
        f.append(val)

    w  = [0] + [(1-a)*i + a*min(i, b) for i in range(1, n+1)]
    return w, f


def Cwelfare(n, c, k):
    w = [0] + [(1-c)*i + c*min(i, k) for i in range(1, n+1)]
    _, f = res_opt_f(w, 'cvxopt')
    return w, f



def getPoB(w, f, n):
    val = [1, 0, 0]
    for a in range(1, n):
        for b in range(0, n):
            mu = (sum(f[:a+1]) - b*f[a+1] + w[b])/w[a]
            if mu > val[0]:
                val[0] = mu
                val[1] = a
                val[2] = b
    return val

            
def getscale(c, k, n):
    w, f =  Cwelfare(n, c, k)
    pob, a, b = getPoB(w, f, n)
    print(c, a, b)
    return pob**-1



if __name__ == '__main__':
    # n = 20
    # c = 0
    # w = [0.] + [1. + c*i for i in range(n)]
    # mu = (1 - (1-c)/np.exp(1))**-1 
    # print(mu)
    # pob, f = getOpt(w, n, mu)
    # print('PoB: ', pob, f)

    # print(res_poa(f[:], w[:], 'cvxopt')**-1)

    c = .5
    n = 10
    w = [0., 1, 2, 3, 4, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6]
    f = [0., 1, 1, 1, 1, 2/11, 2/11, 2/11, 2/11, 2/11, 2/11]
    pob, a, b = getPoB(w, f, n)
    print(w)
    print(f)
    print(pob**-1, a, b)

    # n = 20
    # k = 4
    # dc = .01
    # C = np.arange(0, 1+dc, dc)
    # val = [getscale(c, k, n) for c in C]
    # plt.plot(C, val)
    # plt.plot(C, 1-C/2)
    # plt.plot(C, (1+C)**-1)
    # plt.show()