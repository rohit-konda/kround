import matplotlib.pyplot as plt
import numpy as np
from itertools import product
from games.analysis.resource_poa import ResourcePoA
from math import factorial
from games.misc.solver import lp
from itertools import product
from math import exp


class ResourcePoA2(ResourcePoA):
    @classmethod
    def dual_poa(cls, f, w, d):
        cls._check_args(f, w)
        N = len(w)-1
        I_r = cls.I_r(N)
        num = len(I_r)

        G = np.zeros((num+1, 2), dtype='float')
        h = np.zeros((num+1, 1), dtype='float')
        c = np.array([[0], [1]], dtype='float')  # variables = [lambda , mu]

        for i, (a, x, b) in enumerate(I_r):
            G[i, 0] = (1+d)*a*f[a+x] - (1-d)*b*f[a+x+1] if a+x < N else (1+d)*a*f[a+x]
            G[i, 1] = -w[a+x]
            h[i] = -w[b+x]
        G[num][0] = -1

        sol = lp('cvxopt', c , G, h)

        if sol is not None:
            return round(1./sol['min'], 3)
        else:
            return 0

    @classmethod
    def function_poa(cls, w, d):
        cls._check_w(w)
        N = len(w)-1
        I = cls.I(N)
        num = len(I)

        G = np.zeros((num+1, N+1), dtype='float')
        h = np.zeros((num+1, 1), dtype='float')
        h[num] = -1
        c = np.zeros((N+1, 1), dtype='float')
        c[0] = 1
        Bdel = (1+d)/(1-d)

        for i, (a, x, b) in enumerate(I):
            G[i, a+x] = (1+d)*a
            if a+x < N:
                G[i, a+x+1] = -(1-d)*b
            G[i, 0] = -w[a+x]
            h[i] = -w[b+x]
        G[num][0] = -1

        sol = lp('cvxopt', c , G, h)
        if sol is not None:
            f = sol['argmin']
            f = [0] + [e/f[1] for e in f[1:]]
            return f


def modgairing(n, d):
    from math import factorial
    C = (1+d)/(1-d)
    B = 1./(n-1)/factorial(n-1)/C**n
    y = lambda j: sum([(C**(-i))/factorial(i) for i in range(j, n)])
    return [0] + [round((C**(j-1))*factorial(j-1)*(B + y(j))/(B + y(1)), 3) for j in range(1, n+1)]


def gensubmodw(n, pow=1):
    w = [0., 1.] + [0]*(n-1)
    for i in range(2, len(w)):
        w[i] = round(w[i-1] + (w[i-1] - w[i-2])*np.power(np.random.random(), pow), 3)
    return w

def figPoaVDelt():
    step = .01
    delta = np.arange(0, 1+step, step)
    poa = 1 - 1/np.exp((1-delta)/(1+delta))
    plt.plot(delta, poa)
    plt.plot(delta, poa[0] - delta*poa[0], 'k--')
    plt.show()

def figwrongDeltSC():
    def modpoa(n, dtrue, df):
        B = (1+dtrue)/(1-dtrue)
        Bd = (1+df)/(1-df)
        fgard = modgairing(n, df)
        C =  (np.exp(Bd**-1) - 1)**-1
        curv1 = (B*(Bd**-1)*fgard[2] + B*Bd**-1*C - fgard[2] + 1) **-1
        curv2 = (1 + B*Bd**-1*C)**-1
        return curv1, curv2
    n = 10
    dfs = np.arange(0, 1, .01)
    w = [0] + [1]*n

    poa4 = [modpoa(n, .4, df) for df in dfs]
    poa3 = [modpoa(n, .3, df) for df in dfs]
    poa2 = [modpoa(n, .2, df) for df in dfs]

    poa4 = [min(c1, c2) for c1, c2 in poa4]
    mp4 = max(poa4)
    poa3 = [min(c1, c2) for c1, c2 in poa3]
    mp3 = max(poa3)
    poa2 = [min(c1, c2) for c1, c2 in poa2]
    mp2 = max(poa2)

    plt.plot(dfs, poa4, 'g')
    plt.plot(dfs, poa3, 'r')
    plt.plot(dfs, poa2, 'b')
    plt.plot([.4, .4], [0, mp4], 'g--')
    plt.plot([.4, .4], [0, mp4], 'g.')
    plt.plot([.3, .3], [0, mp3], 'r--')
    plt.plot([.3, .3], [0, mp3], 'r.')
    plt.plot([.2, .2], [0, mp2], 'b--')
    plt.plot([.2, .2], [0, mp2], 'b.')
    plt.xlim([0, 1])
    plt.ylim([.3, .5])
    plt.show()

def figwrongDeltSM():
    def modpoa(n, w, dtrue, df):
        fopt = ResourcePoA2.function_poa(w, df)
        if fopt is not None:
            return ResourcePoA2.dual_poa(fopt, w, dtrue)

    n = 10
    dtrue = .3
    dfs = np.arange(0, 1, .05)

    for _ in range(30):
        w = gensubmodw(n)
        poas = [modpoa(n, w, dtrue, df) for df in dfs if modpoa is not None]
        plt.plot(dfs, poas, 'r', alpha=.4)

    plt.plot([dtrue, dtrue], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.show()


#figwrongDeltSC()
figwrongDeltSM()