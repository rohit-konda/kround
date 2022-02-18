import matplotlib.pyplot as plt
import numpy as np

def fgairing(n):
    from math import factorial
    f = [0]*(n+1)
    for j in range(1, n+1):
        t1 = 1./((n-1)*factorial(n-1))
        t2 = sum([1./factorial(e) for e in range(j, n)])
        t3 = sum([1./factorial(e) for e in range(1, n)])
        f[j] = factorial(j-1) * (t1 + t2) / (t1 + t3)
    return f

def fMC(n): return [0, 1] + [0]*(n-1)

def setw(n): return [0] + [1]*n

def tradeoffFigf():
    def pobset(C):
        from math import factorial
        suml = 30
        sumval = 0
        for i in range(suml):
            tfact = factorial(i)
            tsum = 0
            for j in range(1, i+1):
                tsum += factorial(i)/factorial(j)
            sumval += max(tfact - (1-C)*tsum/C, 0)
        return (sumval + 1.)**(-1)
    step = .001
    poa = np.arange(step, .634, step)
    pob = [pobset(v) for v in poa]

    plt.plot(poa, pob, 'k')
    plt.title('PoA v.s. PoB tradeoff')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('PoA(f)')
    plt.ylabel('PoB(f, 1)')
    plt.show()

class SimFig():
    @classmethod
    def randomResActions(cls, n):
        values = list(np.round(np.random.uniform(size=2*n), 1))
        all_actions = []
        for i in range(n):
            actions = [[0], [0]]
            actions[0][0] = np.random.choice(range(n)) + 1
            actions[1][0] = np.random.choice(range(n, 2*n)) + 1
            actions = [[0]] + actions
            all_actions += [actions]

        values = [0.] + values
        return values, all_actions

    @classmethod
    def bestResp(cls, Game, i, act):
        best = 0
        bestUtil = 0
        act = np.copy(act)
        for j in range(len(Game.actions[i])):
            act[i] = j
            util = Game.U_i(i, act)
            if util > bestUtil:
                best = j
                bestUtil = util
        act[i] = best
        return act
        
    @classmethod
    def brPath(cls, Game, initial, plySeq):
        steps = len(plySeq)
        actions = [initial] + [None]*steps
        for c in range(1, steps+1):
            actions[c] = cls.bestResp(Game, plySeq[c-1], actions[c-1])
        return actions

    @classmethod
    def wRound(cls, WG, K, n): return [WG.welfare(act) for act in cls.brPath(WG, [0]*n, list(range(n))*K)]

    @classmethod
    def getSamples(cls, samples, K, n):
        from games.types.wresource import WResourceFactory
        wdataMC = np.zeros((samples, K*n+1))
        wdataGar = np.zeros((samples, K*n+1))
        diff = np.zeros((samples, K*n+1))
        for c in range(samples):
            values, all_actions = cls.randomResActions(n)
            WgMc = WResourceFactory.make_game(all_actions, values, setw(n), fMC(n))
            WgGar = WResourceFactory.make_game(all_actions, values, setw(n), fgairing(n))
            wdataMC[c, :] = cls.wRound(WgMc, K, n)
            wdataGar[c, :] = cls.wRound(WgGar, K, n)
            diff[c, :] = [e1/e2 if e2 > 0 else 1 for e1, e2 in zip(cls.wRound(WgMc, K, n), cls.wRound(WgGar, K, n))]
            print(c)
        return wdataMC, wdataGar, diff

    @classmethod
    def plotBRavg(cls):
        n = 10
        K = 4
        SAMP = 100
        steps = range(K*n+1)

        _, _, diffresults = cls.getSamples(SAMP, K, n)

        diffavg = np.mean(diffresults, 0)
        mindiffround = np.amin(diffresults, 0)
        maxdiffround = np.amax(diffresults, 0)
        plt.plot(steps, diffresults.T, 'r.', alpha=.1)
        plt.plot(steps, diffavg, 'k--')
        plt.plot(steps, mindiffround, 'r', alpha=.5)
        plt.plot(steps, maxdiffround, 'r', alpha=.5)
        plt.plot([0, K*n+1], [1, 1], 'k', alpha=.2)

        for c in range(K+1):
            plt.plot([c*n, c*n], [0, 2], 'k', alpha=.2)

        ax = plt.gca()
        ax.set_xlim([0, K*n])
        ax.set_ylim([.7, 1.4])
        #plt.grid()
        plt.show()


def plotcurve():
    dx = .01
    rangex = np.arange(0, 1+dx, dx)
    rangePOBopt = 1 - rangex/2.
    rangePOAopt = 1 - rangex/np.exp(1)
    rangeMC = (1 + rangex)**-1
    plt.plot(rangex, rangePOBopt)
    plt.plot(rangex, rangePOAopt)
    plt.plot(rangex, rangeMC)
    plt.xlim(0, 1)
    plt.ylim(0.5, 1)
    plt.show()


def abcovering(n, a, b):
    from math import factorial
    p = (1 - a*b**b*np.exp(-b)/factorial(b))**-1
    f = [0, 1.]
    for i in range(1, n):
        val1 = 1 - a
        Vab = (1-a)*i + a*min(i, b)
        val2 = 1/b*(i*f[i] - Vab*p) + 1
        val = max(val1, val2 - .001)
        f.append(val)

    w  = [0] + [(1-a)*i + a*min(i, b) for i in range(1, n+1)]
    return w, f


def tradeoffcurv():
    from games.price_of_anarchy import res_opt_f
    from asymopt import getPoB
    dc = .01
    n = 40
    ccs = np.arange(0, 1+dc, dc)
    pobs = [0]*len(ccs)
    poaccs = 1 - ccs/2
    for i, c in enumerate(ccs):
        w, f = abcovering(n, c, 1)
        print('HHHHEEEEEEELLLLLOOOOOOOO', c, w, f)
        pobs[i] = getPoB(w, f, n)[0]**(-1)
    # print(pobs)
    plt.plot(ccs, poaccs)
    plt.plot(ccs, pobs)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.show()


def submod():
    x = range(0, 11)
    y = [0]*11
    for i in range(10):
        y[i+1] = y[i] + 3/(4*x[i+1]) + 1/4

    plt.plot(x, y, '.-')
    plt.xlim(0, 10)
    plt.ylim(0, 5)
    plt.show()

def plotcurve2():
    dx = .01
    rangex = np.arange(0, 1+dx, dx)
    rangePOBopt = 1 - rangex/2.
    rangePOAopt = 1 - rangex/np.exp(1)
    rangeMC = (1 + rangex)**-1
    plt.plot(rangex, (rangePOAopt - rangeMC)/rangeMC)
    plt.plot(rangex, (rangePOBopt - rangeMC)/rangeMC)
    plt.xlim(0, 1)
    plt.ylim(0, .3)
    plt.show()

SimFig.plotBRavg()