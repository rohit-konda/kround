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


def kroundEmp(w, f, k, n, part=None, printR=False):
    acN = k+1
    if part is None:
        part = partition(acN, n)

    aKbr = k-1
    aOpt = k

    c = acWelf(part, w, n, [aKbr]*n)
    A = np.expand_dims(acWelf(part, w, n, [aOpt]*n), axis=0)
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


def kwrapperLP(w, f, k, dual=False, returnall=False, reduced=False, empty=True, part=None):
    if not dual:
        if empty:
            args = kroundEmp(w, f, k, len(w)-1, part=part)
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

def runarray():
    dc = .1
    df1 = .05
    df2 = .05
    pobmax = 0
    for c in np.arange(0, 1+dc, dc):
        for f1 in np.arange(0, 1 + df1, df1):
            for f2 in np.arange(0, 1 + df2, df2):
                f = [0, 1, round(f1, 2), round(f2, 2)]
                w = [0, 1, round(1+c, 2), round(1+2*c, 2)]
                pob, _ = kwrapperLP(w, f, 1, empty=False)
                if pob >= pobmax:
                    pobmax = pob
                    optf = f
                    optw = w
        print(round(c, 1), optw, optf, pob)


if __name__ =='__main__':
    # w = [0, 1, 4, 10]
    # f = [0, 1, 1, 1]
    # n = 3
    # k = 1
    # pob, vals = kwrapperLP(w, f, k, empty=False)
    # # pob2, val, endres = prunegame(val, pob, w, f, k)
    # part = partition(2, n)
    # for i, p in enumerate(part):
    #     v = vals[i]
    #     if v > 0:
    #         print(p, v*10)
    # print(w, f)
    # print(pob**-1)
    #runarray()

    pob, _ = kwrapperLP([0, 1, 1, 1], [0, 1, 1, 1], 2, empty=False)
    print(pob)

    # res = [(v, p) for v, p in zip(val, endres) if v > 0]
    # minval = min([e[0] for e in res])
    # for v, p in res:
    #     print(round(v/minval, 4), p)

    # c = .5
    # w = [0, 1, 1+c, 1+2*c, 1+3*c, 1+4*c]
    # f = [0, 1, c, c, c, c]
    # k = 1

    # pob, dual = kwrapperLP(w, f, k, dual=True, reduced=True)
    # print(pob, dual)
    pass