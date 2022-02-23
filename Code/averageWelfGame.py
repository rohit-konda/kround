import matplotlib.pyplot as plt
import numpy as np
from games.price_of_anarchy import res_opt_f, brute_opt
from games.types.wresource import WResourceFactory


def getw(C, n):
    return [0] + [1 + (1-C) * j for j in range(n)] 


def fpoa(w):
    return [round(e, 2) for e in res_opt_f(w, 'cvxopt')[1]]

def fmc(w):
    return [0] + [w[j] - w[j-1] for j in range(1, len(w))]

def f1b(C, b, n):
    B = (b+1)/b
    Bb = (B**b)/(B**b - C)
    fb = [0] + [(1-Bb)*B**(j-1) + Bb if j <= b+1 else (1-C)*Bb for j in range(1, n+1)]
    return [round(e, 2) for e in fb]

def decomp(w):
    C = 1 - w[-1] + w[-2]
    n = len(w) - 1
    nb = [(2*w[j] - w[j-1] - w[j+1])/C for j in range(1, n)]
    nb = nb + [1 - sum(nb)]
    return C, nb, n

def f1(w):
    C, nb, n = decomp(w)
    f = np.zeros((n+1,))
    for b in range(1, n+1):
        f += nb[b-1]*np.array(f1b(C, b, n))
    return f


def ranResAc(n):
    # generate random values and actions
    N = 2*n
    values = [0] + list(np.round(np.random.uniform(size=N), 3))
    actions = []
    for i in range(n):
        numAc  = np.random.choice(range(2, 4))
        Aci = [[np.random.choice(range(1, len(values)))] for _ in range(numAc)]
        for j in range(numAc):
            ac = Aci[j][0]
            Aci[j] = list(range(ac, min(ac+np.random.choice(range(2, 4)), N+1)))
        Aci = [[0]] + Aci # add null resource 
        actions.append(Aci)
    return values, actions


def mkGm(w, f, v, a): return WResourceFactory.make_game(a, v, w, f)


def getGames(numSamps, n, C):
    w = getw(C, n)
    paramG = [ranResAc(n) for _ in range(numSamps)]
    poaG = [mkGm(w, fpoa(w), v, a) for v, a in paramG]
    mcG = [mkGm(w, fmc(w), v, a) for v, a in paramG]
    oneG = [mkGm(w, f1(w), v, a) for v, a in paramG]
    return poaG, mcG, oneG


def bestResp(Game, i, act):
    def newac(j, a):
        a[i] = j
        return a
    act = np.copy(act)
    act[i] = np.argmax([Game.U_i(i, newac(j, act)) for j in range(len(Game.actions[i]))])
    return act


def brPath(Game, initial, plySeq):
    steps = len(plySeq)
    actions = [initial] + [None]*steps
    for c in range(1, steps+1):
        actions[c] = bestResp(Game, plySeq[c-1], actions[c-1])
    return actions


def wRound(WG, K, n): return [WG.welfare(act) for act in brPath(WG, [0]*n, list(range(n))*K)]


def getBRData(numSamp, k, n, C):
    wDataPoA = np.zeros((numSamp, k*n+1))
    wDataMC = np.zeros((numSamp, k*n+1))
    wDataOne = np.zeros((numSamp, k*n+1))
    poaG, mcG, oneG = getGames(numSamp, n, C)

    for c in range(numSamp):
        wDataPoA[c, :] = wRound(poaG[c], k, n)
        wDataMC[c, :] = wRound(mcG[c], k, n)
        wDataOne[c, :] = wRound(oneG[c], k, n)
        print(c)
    return wDataPoA, wDataMC, wDataOne


def onlyK(vec, n): return np.array(vec)[:, n::n]


def plotBRData():
    n = 20
    k = 5
    C = .8
    numSamp = 50
    dPoa, dMc, dOne = getBRData(numSamp, k, n, C)

    dMcK = onlyK(dMc, n)
    plt.boxplot(dMcK)
    
    dPoaK = onlyK(dPoa, n)
    plt.boxplot(dPoaK, positions=np.array(range(k))+1.2, manage_ticks=False)
    
    dOneK = onlyK(dOne, n)
    plt.boxplot(dOneK, positions=np.array(range(k))+.8, manage_ticks=False)
    plt.show()

    print(getw(C, n))
    print(dMcK, dPoaK, dOneK)

plotBRData()
# v, a = ranResAc(10)
# print(v)
# print(a)
