from games.types.wresource import WResourceFactory

WResourceFactory.make_game()



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