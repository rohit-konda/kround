from LPsolver import lp
from itertools import product
import numpy as np
import warnings
warnings.filterwarnings('ignore')


def getGame(acN, n, val):
    part = partition(acN, n)
    for i in range(len(part)):
        v = round(val[i], 4)
        if v > 0:
            print(v, part[i])


def partition(acN, n): return list(product(product([1, 0], repeat=acN), repeat=n))[:-1]


def cover(rSig, act, n):
    # rSig: what players cover, act:list of actions
    cov = [0]*n
    for i in range(n):
        if act[i] != -1 and rSig[i][act[i]] == 1:
            cov[i] = 1
    return cov


def acWelf(part, w, n, act):
    # welfare of action act (list) based on resource partition
    welfArr = np.zeros((len(part),), dtype='float')
    for i, p in enumerate(part):
        welfArr[i] = w[sum(cover(p, act, n))]
    return welfArr


def kBrUtil(part, f, k, n, Emp=True):
    if Emp:
        brCons = np.zeros((n*k*k, len(part)), dtype='float')
        iterRange = product(range(n), range(k), range(k+1))
    else:
        brCons = np.zeros((n*k*(k+1), len(part)), dtype='float')
        iterRange = product(range(n), range(1, k+1), range(k+2))

    c = 0
    for j, b, acOth in iterRange:
        # Uj(a_b, a_b>i, a_b-1<i) >= Uj(acOth, a_b>i, a_b-1<i)
        if acOth != b:
            for i, p in enumerate(part):
                acNotI = [b]*j + [-1] + [b-1]*(n-j-1)            
                if p[j][b] == 1 and p[j][acOth] == 0: # if resource is covered by j in b but not acOth
                    brCons[c][i] = f[sum(cover(p, acNotI, n))+1]
                elif p[j][b] == 0 and p[j][acOth] == 1:
                    brCons[c][i] = -f[sum(cover(p, acNotI, n))+1]
            c += 1
    return brCons

def kNashUtil(part, f, k, n, Emp=True):
    if Emp:
        nashCons = np.zeros((n*k, len(part)), dtype='float')
        iterRange = product(range(n), range(k+1))
    else:
        nashCons = np.zeros((n*(k+1), len(part)), dtype='float')
        iterRange = product(range(n), range(k+2))

    c = 0
    b = 0
    for j, acOth in iterRange:
        # Uj(a_b, a_b-i) >= Uj(acOth, a_b-i)
        if acOth != b:
            for i, p in enumerate(part):
                acNotI = [b]*j + [-1] + [b]*(n-j-1)            
                if p[j][b] == 1 and p[j][acOth] == 0: # if resource is covered by j in b but not acOth
                    nashCons[c][i] = f[sum(cover(p, acNotI, n))+1]
                elif p[j][b] == 0 and p[j][acOth] == 1:
                    nashCons[c][i] = -f[sum(cover(p, acNotI, n))+1]
            c += 1
    return nashCons


def kroundEmp(w, f, k, n, part=None, printR=False, nash=False):
    acN = k+1
    if part is None:
        part = partition(acN, n)

    aKbr = k-1
    aOpt = k

    c = acWelf(part, w, n, [aKbr]*n)
    A = np.expand_dims(acWelf(part, w, n, [aOpt]*n), axis=0)
    if nash:
        G = -np.vstack((kBrUtil(part, f, k, n), kNashUtil(part, f, k, n), np.identity(len(part))))
    else:
        G = -np.vstack((kBrUtil(part, f, k, n), np.identity(len(part))))
    h = np.zeros((len(G), 1))
    b = np.ones((1, 1))

    return c, G, h, A, b


def dualkLP(w, f, k, n, reduced=False):
    varN = n*k*k + 1
    acN = k+1
    part = partition(acN, n)
    aKbr = k-1
    aOpt = k

    c = np.array([1] + [0]*(varN-1), dtype='float') + int(reduced)*.001*np.ones((varN,))
    ineqCons = kBrUtil(part, f, k, n).T
    eqCons = -np.expand_dims(acWelf(part, w, n, [aKbr]*n).T, axis=0).T
    dualCons = -np.identity(varN)[1:]
    G = np.vstack((np.hstack((eqCons, ineqCons)), dualCons))
    h = np.vstack((-np.expand_dims(acWelf(part, w, n, [aOpt]*n), axis=0).T, np.zeros((varN-1, 1))))
    return c, G, h


def kroundNotEmp(w, f, k, n, reduced=False):
    acN = k+2
    part = partition(acN, n)
    pN = len(part)
    aKbr = k
    aOpt = k+1

    c = acWelf(part, w, n, [aKbr]*n) + int(reduced)*.001*np.ones((pN,))
    A = np.expand_dims(acWelf(part, w, n, [aOpt]*n), axis=0)
    G = -np.vstack((kBrUtil(part, f, k, n, False), np.identity(pN)))
    h = np.zeros((len(G), 1))
    b = np.ones((1, 1))

    return c, G, h, A, b


def kwrapperLP(w, f, k, dual=False, returnall=False, reduced=False, empty=True, part=None, nash=False):
    if not dual:
        if empty:
            args = kroundEmp(w, f, k, len(w)-1, part=part, nash=nash)
        else:
            args = kroundNotEmp(w, f, k, len(w)-1, reduced=reduced)
    else:
        args = dualkLP(w, f, k, len(w)-1, reduced=reduced)

    sol = lp('cvxopt', *args, returnall=returnall)
    if not returnall:
        if sol is not None:
            pob = round(sol['min'], 4)
            argmin = [round(e, 4) for e in sol['argmin']]
            return pob, argmin
        else:
            return -1, []
    else:
        if sol is not None:
            return sol


def prunegame(values, pob, w, f, k, ERROR=.001, **kwargs):
    def normres(par): 
        return sum([item for plysel in par for item in plysel])
    resources = [p for v, p in zip(values, partition(k+1, len(w)-1)) if v > 0]  # resources where value > 0
    resources.sort(reverse=True, key=normres)  # sort resources that players select less
    tightres = []
    for i in range(len(resources)):
        resi = list(resources[i+1:] + tightres)
        pobi, _ = kwrapperLP(w, f, k, part=resi, **kwargs)

        if pobi < 0 or abs(pobi - pob) > ERROR:
            tightres.append(resources[i])
    pobend, valend = kwrapperLP(w, f, k, part=tightres, **kwargs)
    return pobend, valend, tightres


def getPruned(w, f, k, **kwargs):
    pob, values = kwrapperLP(w, f, k, **kwargs)
    pobend, valend, tightres = prunegame(values, pob, w, f, k, **kwargs)
    print('PoB: ', pobend)
    print('Game: ')
    for i in range(len(tightres)):
        print(valend[i], tightres[i])

def runarray():
    dc = .1
    df1 = .1
    df2 = .1
    for c in np.arange(0, 1+dc, dc):
        for f1 in np.arange(0, 1 + df1, df1):
            for f2 in np.arange(0, 1 + df2, df2):
                f = [0, 1, round(f1, 2), round(f2, 2)]
                w = [0, 1, round(1+c, 2), round(1+2*c, 2)]
                pob, _ = kwrapperLP(w, f, 2)
                print(w, f, pob)


def abcoveringf(n, a, b):
    from math import factorial
    p = (1 - a*b**b*np.exp(-b)/factorial(b))**-1
    f = [0, 1.]
    for i in range(1, n):
        val1 = 1 - a
        Vab = (1-a)*i + a*min(i, b)
        val2 = 1/b*(i*f[i] - Vab*p) + 1
        val = max(val1, val2)
        f.append(val)
    return f

if __name__ =='__main__':
    n = 4
    w = [0] + [1]*n
    f = abcoveringf(n, 1, 1)
    k = 2
    pob, _ = kwrapperLP(w, f, 2, nash=True)
    print(pob)
    # getPruned(w, f, k, nash=True)
