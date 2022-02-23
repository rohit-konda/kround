from LPfunctions import *
from LPsolver import lp
from games.price_of_anarchy import res_poa, res_opt_f, worst_game
from kroundempty import kroundLP
from prettytable import PrettyTable
import matplotlib.pyplot as plt



def pobUpperWithf(w, f):
    def uppart(N):
        part = [()]*round((N+1)*(N+2)/2)
        c = 0
        for a in range(N+1):
            for x in range(0, N-a+1):
                part[c] = (a, x)
                c+=1
        return part

    N = len(w) - 1
    partition = uppart(N)
    NC = len(partition)
    C = np.ones((1,), dtype='float')
    G = np.zeros((NC, 1), dtype='float')
    H = np.zeros((len(G), 1), dtype='float')

    for i, (a, x) in enumerate(partition):
        G[i, 0] = -w[a+x]
        H[i, 0] = -w[x] - sum([f[j+1] for j in range(a)])
    return C, G, H




def I_r(N):
    from itertools import combinations
    ind = []
    for i in range(0, N+1):
        not_a = [(0, j, i) for j in range(N+1-i)]
        not_x = [(j, 0, i) for j in range(N+1-i)]
        not_b = [(j, i, 0) for j in range(N+1-i)]
        ind = ind + not_a + not_b + not_x

    ind += [(j[0], j[1]-j[0]-1, N-j[1]+1) for j in combinations(range(N+2), 2)]
    return [j for j in list(set(ind)) if j != (0, 0, 0)]

def upper(N):
    part = [()]*round((N+1)*(N+2)/2)
    c = 0
    for a in range(N+1):
        for x in range(0, N-a+1):
            part[c] = (a, x)
            c+=1
    return part

def lower(N):
    part = [()]*(N*(N+1))
    c = 0
    for a in range(1, N+1):
        for b in range(N+1):
            x = max(a+b-N, 0)
            part[c] = (a-x, x, b-x)
            c+=1
    return part

def poaconst(N, w, X):
    partition = I_r(N)
    NC = len(partition)
    const = np.zeros((NC, N + 1), dtype='float')
    const_h = np.zeros((len(const), 1))

    for i, (a, x, b) in enumerate(partition):
        if a+x > 0:
            const[i, a+x] = -a
        if b > 0:
            const[i, a+x+1] = b
        const_h[i] = w[b+x] - X * w[a+x]
    return const, const_h

def upperconst(N, w):
    partition = upper(N)
    NC = len(partition)
    const = np.zeros((NC, N+1), dtype='float')
    const_h = np.zeros((len(const), 1), dtype='float')

    for i, (a, x) in enumerate(partition):
        const[i, 0] = w[a+x]
        for j in range(1, a+1):
            const[i, j] = -1
        const_h[i] = w[x]
    return const, const_h

def lowerconst(N, w):
    partition = lower(N)
    NC = len(partition)
    const = np.zeros((NC, N+1), dtype='float')
    const_h = np.zeros((len(const), 1), dtype='float')

    for i, (a, x, b) in enumerate(partition):
        const[i, 0] = w[a+x]
        for j in range(1, a+1):
            const[i, j] = -1
        if b > 0:
                const[i, a+x+1] = b
        const_h[i] = w[b+x]
    return const, const_h

def pobconst(w, X, upper=True):
    N = len(w)-1
    C = np.zeros((N+1,), dtype='float')
    nonnegcons = np.identity(N+1,  dtype='float')
    nonnegcons_h = np.zeros((N+1, 1), dtype='float')
    C[0] = 1

    if upper:
        pobcons, pobcons_h = upperconst(N, w)
    else:
        pobcons, pobcons_h = lowerconst(N, w)

    poacons, poacons_h = poaconst(N, w, X)
    G = -np.vstack((pobcons, poacons, nonnegcons))
    H = -np.vstack((pobcons_h, poacons_h, nonnegcons_h))
    return C, G, H

def plotpobpoagenw(w, step, upper):
    poarange = []
    pobrange = []
    for poa in np.arange(step, 1+step, step):
        X = poa**-1
        args = pobconst(w, X, upper)
        sol = lp('cvxopt', *args)
        if sol is not None:
            poarange.append(poa)
            pobrange.append(round(sol['min']**-1, 4))
    plt.plot(poarange, pobrange, 'k')

def plottrad():
    w = [0, 1, 2, 3] + [4]*15
    step = .005
    plotpobpoagenw(w, step, True)
    plotpobpoagenw(w, step, False)
    plt.show()


def kcovw(N, k):
    return [i for i in range(k)] + [k for _ in range(N-k+1)]

def foptkcov(N, k):
    from math import factorial
    f = [0, 1] + [0]*(N-1)
    p = (1 - (k**k * np.exp(-k))/factorial(k))**-1

    for i in range(1, N):
        f[i+1] = 1/k*(i * f[i] - p*min(i, k)) + 1
    return f


if __name__ == '__main__':
    k = 7

    # for N in range(2, 30):
    #     args = pobUpperWithf(kcovw(N, 1), foptkcov(N, k))
    #     sol = lp('cvxopt', *args)
    #     if sol is not None:
    #         print(N, round(1/sol['min'], 3))

    for i in range(1, 10):
        print([round(e, 2) for e in np.diff(foptkcov(20, i))])
