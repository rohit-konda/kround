from games.types.wresource import WResourceFactory
from games.analysis.search_nash import BrutePoA
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce


def fgairing(n):
    from math import factorial
    f = [0]*(n+1)
    for j in range(1, n+1):
        t1 = 1./((n-1)*factorial(n-1))
        t2 = sum([1./factorial(e) for e in range(j, n)])
        t3 = sum([1./factorial(e) for e in range(1, n)])
        f[j] = factorial(j-1) * (t1 + t2) / (t1 + t3)
    return f

def makeResActRand(n):
    numres = round(2*n)
    values = list(np.round(np.random.uniform(size=numres), 1)) # 
    all_actions = []
    for i in range(n):
        numaction = 3
        actions = [[]]*numaction
        for a in range(numaction):
            select = 2
            actions[a] = list(np.random.choice(range(numres), select, replace=False) + 1)
        actions = [[0]] + actions
        all_actions += [actions]

    # #
    # sel = reduce(lambda x, y: x + y, reduce(lambda x, y: x + y, all_actions))
    # val = [0 for e in range(numres+1)]
    # for r in sel:
    #     val[r] += 1
    # print(val)
    # #
    # e = .1
    # #print(np.arange(0, 1+e, e))
    # np.random.choice(np.arange(0, 1+e, e), numres, p=[.25, .25, 0, 0, 0, 0, 0, 0, 0, .25, .25])
    #

    values = [0.] + values
    return values, all_actions

def wsoftmax(WG):
    wsum = 0
    for i, pact in enumerate(WG.actions):
        numac = len(pact)
        wmaxi = np.zeros((numac,))
        blankac = [0 for _ in WG.players]
        for j in range(numac):
            blankac[i] = j
            wmaxi[j] = WG.welfare(blankac)
        wsum += np.amax(wmaxi)
    return wsum

def bestresponse(Game, i, action):
    best = 0
    best_util = 0
    tempaction = np.copy(action)

    temtempaction = np.copy(action)
    for j in range(len(Game.actions[i])):
        tempaction[i] = j
        util = Game.U_i(i, tempaction)
        if util > best_util:
            best = j
            best_util = util
    return best

def bestresponseround(Game, start=None):
    if start is not None:
        actions = start
    else:
        actions = [0 for _ in Game.players]  # start out at null action
    for i in range(len(Game.players)):
        actions[i] = bestresponse(Game, i, actions)
    return actions

def wpath(WG, K):
    newac = None
    welfarepath = [0]*K
    for i in range(K):
        newac = bestresponseround(WG, newac)
        welfarepath[i] =  WG.welfare(newac)
    return welfarepath

def wpath2(WG, K):
    welfarepath = [0]*K*n
    actions = [0 for _ in WG.players]
    c = 0
    for k in range(K):
        for i in range(len(WG.players)):
            actions[i] = bestresponse(WG, i, actions)
            welfarepath[c] =  WG.welfare(actions)
            c += 1
    return welfarepath


def fdec(n, pow=1):
    f = [0., 1.] + [0]*(n-1)
    for i in range(2, n+1):
        f[i] = round(f[i-1]*np.power(np.random.random(), pow), 3)
    return f





if __name__ == '__main__':
    n= 10

    w = [0] + [1]*n
    fMC = [0, 1] + [0]*(n-1)
    fshap = [0] + [1./j for j in range(1, n+1)]
    fgar = fgairing(n)
    K = 5
    SAMP = 200

    mcresults = np.zeros((SAMP, K*n))
    shapresults = np.zeros((SAMP, K*n))
    diffresults = np.zeros((SAMP, K*n))

    for i in range(SAMP):
        values, all_actions = makeResActRand(n)
        Wgmc = WResourceFactory.make_game(all_actions, values, w, fMC)
        Wgshap = WResourceFactory.make_game(all_actions, values, w, fshap)

        #welfmax = wsoftmax(Wgmc)
        #mceff = np.array([e/welfmax for e in wpath2(Wgmc, K)])
        #shapeff = [e/welfmax for e in wpath2(Wgshap, K)]
        #mcresults[i, :] = mceff  # efficiency along rounds
        #shapresults[i, :] = shapeff # efficiency along rounds

        diffeff = [e/e2 for e, e2 in zip(wpath2(Wgmc, K), wpath2(Wgshap, K))]
        diffresults[i, :] = diffeff
        print(i)

    roundax = range(1, K*n+1)

    diffavg = np.mean(diffresults, 0)
    mindiffround = np.amin(diffresults, 0)
    maxdiffround = np.amax(diffresults, 0)
    plt.plot(roundax, diffresults.T, 'r.', alpha=.1)
    plt.plot(roundax, diffavg, 'k--')
    plt.plot(roundax, mindiffround, 'r', alpha=.5)
    plt.plot(roundax, maxdiffround, 'r', alpha=.5)

    #plt.title('Comparison between optimal utility mechanisms through Simulation')
    #plt.xlabel('Best Response Steps')
    #plt.ylabel('Ratio of Efficiency')

    # minmcround = np.amin(mcresults, 0)
    # minshapround = np.amin(shapresults, 0)

    # mcavg = np.mean(mcresults, 0)
    # shapavg = np.mean(shapresults, 0)

    # roundax = range(1, K+1)

    # plt.plot(mcresults.T, 'r', alpha=.2)
    # plt.plot(minmcround, 'r')
    # plt.plot(mcavg, 'r-.')
    # plt.plot(shapresults.T, 'b', alpha=.2)
    # plt.plot(minshapround, 'b')
    # plt.plot(shapavg, 'b-.')

    ax = plt.gca()
    #ax.set_xticks(range(K*n+1))
    ax.set_xlim([0, K*n])
    ax.set_ylim([.9, 1.25])
    plt.grid()
    plt.show()