from LPfunctions import *
from LPsolver import lp
from games.price_of_anarchy import res_poa, res_opt_f, worst_game
from kroundempty import kroundLP
from prettytable import PrettyTable
import matplotlib.pyplot as plt

def Uforac(partition, val, w, f, k):
    n = len(w) - 1
    pt = PrettyTable(['actions'] + [str(i) for i in range(n+1)])
    actions = list(product(range(0, k+1), repeat=n))
    for ac in actions:
        row = [0.]*(n+1)
        for i, p in enumerate(partition):
            if round(val[i], 4) != 0:
                selected = [p[pl][ac[pl]] for pl in range(n)]
                row[0] += val[i] * w[sum(selected)]
                for pl in range(n):
                    if selected[pl] == 1:
                        row[1+pl] += val[i] * f[sum(selected)]
        row = [ac] + [round(v, 3) for v in row]
        pt.add_row(row)
    print(pt)

def gensubmodw(n, pow=1):
    w = [0., 1.] + [0]*(n-1)
    for i in range(2, len(w)):
        w[i] = round(w[i-1] + (w[i-1] - w[i-2])*np.power(np.random.random(), pow), 3)
    return w

def MC(w): return [0] + [round(e, 4) for e in list(np.diff(w))]


def gendecf(n, pow=1):
    f = [0., 1.] + [0]*(n-1)
    for i in range(2, n+1):
        f[i] = round(f[i-1]*np.power(np.random.random(), pow), 3)
    return f

def genrandomwf(n):
    w = [0., 1.] + [0]*(n-1)
    f = [0., 1.] + [0]*(n-1)
    for i in range(2, len(w)):
        w[i] = round(np.random.random()*i, 3)
        f[i] = round(np.random.random(), 3)
    return w, f

def checkKempty():
    # w = [0, 1., 1.69, 1.8]
    # f = [0] + [round(e, 3) for e in list(np.diff(w))]
    # print(w, f)
    # args, partition = kroundLPempty(w, f, 2)

    # args2, partition2 = kroundLPempty(w, f, 3)
    
    # sol2 = lp('cvxopt', *args2)
    # print('3round: ', round(-sol2['min'], 3))
    
    # sol = lp('cvxopt', *args)
    # print('2round: ', round(-sol['min'], 3))
    # for i, p in enumerate(partition):
    #     val = round(sol['argmin'][i], 3)
    #     if val > 0:
    #         print(p, val)

    # Uforac(partition, sol['argmin'], w, f, 2)
    # Uforac(partition2, sol2['argmin'], w, f, 3)
    for kk in range(1000):
        #w = gensubmodw(2)
        #f = [0] + [round(e, 3) for e in list(np.diff(w))]
        w, f = genrandomwf(2)
        args, _ = kroundLPempty(w, f, 2)
        args2, _ = kroundLPempty(w, f, 3)
        sol1 = lp('cvxopt', *args)
        sol2 = lp('cvxopt', *args2)
        if sol1 is not None and sol2 is not None:
            val1 = round(-sol1['min'], 4)
            val2 = round(-sol2['min'], 4)
            if val1 < val2:
                print('w: ', w)
                print('f: ', f)
                print('2round: ', val1)
                print('3round: ', val2)
        print(kk)

def check_poapob():
    #for _ in range(30):
    w = [0, 1., 2., 2., 2., 2., 2., 2., 2., 2.] # gensubmodw(3)
    f = [0] + [round(e, 3) for e in list(np.diff(w))]
    args = oneemptydual(w, f)
    print('w: ', w)
    print('f: ', f)
    print('PoB: ', round(lp('cvxopt', *args)['min']**-1, 3))
    print('PoA: ', round(res_poa(f, w, 'cvxopt'), 3))

def check2v3():
    for kk in range(500):
        w3 = gensubmodw(3)
        f3 = [0] + [round(e, 3) for e in list(np.diff(w3))]
        w2 = w3[:-1]
        f2 = [0] + [round(e, 3) for e in list(np.diff(w2))]
        args, _ = kroundLPempty(w2, f2, 2)
        args2, _ = kroundLPempty(w3, f3, 2)
        sol1 = lp('cvxopt', *args)
        sol2 = lp('cvxopt', *args2)
        if sol1 is not None and sol2 is not None:
            val1 = round(-sol1['min'], 4)
            val2 = round(-sol2['min'], 4)
            if val1 > val2:
                print('w: ', w3, w2)
                print('f: ', f3, f2)
                print('2player: ', val1)
                print('3player: ', val2)

def checkincreasingn():
    N = 5
    for _ in range(50):
        wN = gensubmodw(N)
        print(wN)
        for n in range(2, N+1):
            wn = wN[:n+1]
            fn = [0] + [round(e, 3) for e in list(np.diff(wn))]
            args, _ = kroundLPempty(wn, fn, 1)
            sol = lp('cvxopt', *args)
            if sol is not None:
                print(n, round(-sol['min'], 4))

def poavpob2():
    for alp in np.arange(1., 3., .03):
        for bet in np.arange(1., 3, .03):
            ralp = round(alp, 4)
            rbet = round(bet, 4)
            w = [0, 1, ralp, rbet]
            f = [0] + [round(e, 4) for e in list(np.diff(w))]
            args = oneemptydual(w, f)
            sol = lp('cvxopt', *args)
            if sol is None:
                continue
            pob = sol['min']**-1
            poa, _ = res_opt_f(w, 'cvxopt')

            col = 1 - (poa - pob)*2
            plt.plot(ralp, rbet, '.', color=(col, col, col))
            plt.plot(ralp, rbet, col)
    plt.show()


def submodlin():
    n = 5
    k = 3
    w3 = gensubmodw(3)
    c = .1
    w = w3 + [e*c + w3[-1] for e in w]

def testonekround():
    # w = [0, 1., 1.5, 1.7]
    # f = [0, 1., .9, .4]
    # args = oneroundbest(w, f)
    # sol = lp('cvxopt', *args)
    # if sol is not None:
    #     print(round(sol['min'], 3))
    # args2 = kroundbest(w, f, 2)
    # sol2 = lp('cvxopt', *args2)
    # if sol2 is not None:
    #     print(round(sol2['min'], 3))

    for kk in range(100):
        # w = gensubmodw(3)
        # f = [0] + [round(e, 3) for e in list(np.diff(w))]
        w, f = genrandomwf(3)
        args = oneroundbest(w, f)
        args2 = kroundbest(w, f, 2)
        sol1 = lp('cvxopt', *args)
        sol2 = lp('cvxopt', *args2)
        if sol1 is not None and sol2 is not None:
            val1 = round(sol1['min'], 4)
            val2 = round(sol2['min'], 4)
            print('w: ', w)
            print('f: ', f)
            print('1round: ', val1)
            print('2round: ', val2)

def getcurvedw(n, c, inter):
    w = [0., 1.] + [0]*(n-1)
    for i in range(2, inter):
        w[i] = round(w[i-1] + (w[i-1] - w[i-2] - c)*np.random.random() + c, 3)
    for i in range(inter, n+1):
        w[i] = round(w[i-1] + c, 3)
    f = [0] + [round(e, 3) for e in list(np.diff(w))]
    return w, f

def curvaturehyp(n, c, inter):
    minval = 2
    maxval = 0
    minw = None
    maxw = None
    for _ in range(80):
        w, f = getcurvedw(n, c, inter)
        args = oneemptydual(w, f)
        sol = lp('cvxopt', *args)
        if sol is not None:
            val = round(sol['min'], 4)
            if val < minval:
                minval = val
                minw = w
            elif val > maxval:
                maxval = val
                maxw = w
    print(minval, maxval)
    print(minw, maxw)

    # c = .3
    # inter = 3
    # for n in range(inter+2, 9):
    #     print(n, inter)
    #     curvaturehyp(n, c, inter)

def verif2player():
    w = [0, 1, 3.]
    f = [0, 1., .5]
    print(f)
    args, partition = kroundLPempty(w, f, 2)
    #args = oneemptydual(w, f)
    sol = lp('cvxopt', *args)
    pob = round(sol['min'], 4)
    print(pob)

    for i, p in enumerate(partition):
        val = round(sol['argmin'][i]*100, 1)
        if val != 0:
            print(p, val)


def checklowerbound(w, f):
    n = len(w)-1
    minval = 1
    for x in range(0, n):
        for a in range(1, n-x+1):
            b = min(a, n-a-x)
            num = w[a+x]
            if a + x == n:
                den = w[a] + w[b+x]
            else:
                den = w[a] - b*f[a+x+1] + w[b+x]
            val = num/den
            if val < minval:
                minval = val
    print('lowerbound', round(minval, 4))

    args, partition = kroundLPempty(w, f, 1)
    sol = lp('cvxopt', *args)

    if sol is not None:
        pob = round(sol['min'], 4)
        print('pob', pob)

def worstcase():
    w = gensubmodw(5)
    w = [round(e, 1) for e in w]
    f = [0] + [round(e, 1) for e in list(np.diff(w))]
    print(w, f)

    args, partition = kroundLPempty(w, f, 1)
    sol = lp('cvxopt', *args)

    if sol is not None:
        pob = round(sol['min'], 4)
        print(pob)

        for i, p in enumerate(partition):
            val = round(sol['argmin'][i]*100, 1)
            if val != 0:
                print(p, val)

def getpob(w, f, k):
    args, partition = kroundLPempty(w, f, k)
    sol = lp('cvxopt', *args)
    f = [round(e, 3) for e in f]

    if sol is not None:
        pob = round(sol['min'], 4)
        return pob

def testmcmax():
    k = 1
    n = 3
    for _ in range(30):
        w = gensubmodw(n)
        fmc = [0] + [round(e, 3) for e in list(np.diff(w))]
        pobmc = getpob(w, fmc, k)
        print(pobmc, w, fmc)
        if pobmc is not None:
            for _ in range(50):
                _, f = genrandomwf(n)
                pob = getpob(w, f, k)
                if pob is not None and pob > pobmc:
                    print('FALSE: ', pob, f)


def getworstcmax():
    w = [0, 1, 1.4, 1.7]
    f1 = [0] + list(np.diff(w))
    f2 = [0, 1., .6, .5]
    k = 2

    args1, partition1 = kroundLPempty(w, f1, k)
    sol1 = lp('cvxopt', *args1)

    if sol1 is not None:
        pob1 = round(sol1['min'], 4)
        print(pob1)

        for i, p in enumerate(partition1):
            val = round(sol1['argmin'][i]*100, 1)
            if val != 0:
                print(p, val)

    args2, partition2 = kroundLPempty(w, f2, k)
    sol2 = lp('cvxopt', *args2)

    if sol2 is not None:
        pob2 = round(sol2['min'], 4)
        print(pob2)

        for i, p in enumerate(partition2):
            val = round(sol2['argmin'][i]*100, 1)
            if val != 0:
                print(p, val)


def MCMax():
    c = 1
    for a in np.arange(0., 1., .02):
        for b in np.arange(a, 2*a, .02):
            alp = 1+a
            bet = 1+b
            ralp = round(alp, 4)
            rbet = round(bet, 4)
            w = [0, 1, ralp, rbet]
            fmc = [0] + [round(e, 4) for e in list(np.diff(w))]
            args = oneemptydual(w, fmc)
            sol = lp('cvxopt', *args)
            if sol is not None:
                pob = round(sol['min']**-1, 5)
                ismax = True
                for _ in range(200):
                    _, f = genrandomwf(3)
                    args = oneemptydual(w, f)
                    solother = lp('cvxopt', *args)
                    if solother is not None:
                        pobother = round(solother['min']**-1, 5)
                        if pobother > pob + .001:
                            ismax = False
                            print('False')
                            break
                if ismax:
                    plt.plot(ralp, rbet, '.k')
                else:
                    plt.plot(ralp, rbet, '.r')
            print(c)
            c += 1
    plt.show()

def checkmcnotmaxgame():
    w = [0, 1, 1.8, 2.333]
    f = [0] + [round(e, 4) for e in list(np.diff(w))]
    args, partition = kroundLPempty(w, f, 1)
    sol = lp('cvxopt', *args)
    pobmc = round(sol['min']**-1, 5)
    poa = round(res_poa(f, w, 'cvxopt'), 5)
    print(poa, pobmc**-1)

    for i, p in enumerate(partition):
        val = round(sol['argmin'][i]*100, 1)
        if val != 0:
            print(p, val)


def checkdual3player():
    # w = [0, 1, 2, 3]
    # f = [0] + [round(e, 4) for e in list(np.diff(w))]
    # args = oneemptydual(w, f, True)
    # sol = lp('cvxopt', *args)
    # val = [round(e, 3) for e in sol['argmin']]
    # print(val)

    c = 1
    for a in np.arange(0., 1.+.02, .01):
        for b in np.arange(a, 2*a+.02, .01):
            alp = 1+a
            bet = 1+b
            ralp = round(alp, 4)
            rbet = round(bet, 4)
            w = [0, 1, ralp, rbet]
            fmc = [0] + [round(e, 4) for e in list(np.diff(w))]
            args, _ = kroundLPempty(w, fmc, 1)
            sol = lp('cvxopt', *args)
            argstrue, _ = kroundLPempty(w, fmc, 2)
            soltrue = lp('cvxopt', *argstrue)
            if sol is not None and soltrue is not None:
                pob = round(sol['min'], 3)
                pobtrue = round(soltrue['min'], 3)
                if pob == pobtrue:
                    plt.plot(ralp, rbet, '.k')
                else:
                    plt.plot(ralp, rbet, '.r')
            print(c)
            c += 1
    plt.show()


def pobcharacter():
    c = 1
    for a in np.arange(0., 1.+.02, .01):
        for b in np.arange(a, 2*a+.02, .01):
            alp = 1+a
            bet = 1+b
            ralp = round(alp, 4)
            rbet = round(bet, 4)
            w = [0, 1, ralp, rbet]
            fmc = [0] + [round(e, 4) for e in list(np.diff(w))]
            args = oneemptydual(w, fmc)
            sol = lp('cvxopt', *args)
            if sol is not None:
                pob = round(sol['min']**-1, 3)
                col = (1-pob)*1.8
                plt.plot(ralp, rbet, 's', color=(col, col, col))
            print(c)
            c += 1
    plt.show()


def pobchargame():
    c = 1
    for a in np.arange(0., 1.+.02, .03):
        for b in np.arange(a, 2*a+.02, .03):
            alp = 1+a
            bet = 1+b
            ralp = round(alp, 4)
            rbet = round(bet, 4)
            w = [0, 1, ralp, rbet]
            fmc = [0] + [round(e, 4) for e in list(np.diff(w))]
            args = oneemptydual(w, fmc)
            sol = lp('cvxopt', *args)
            if sol is not None:
                pob = round(sol['min']**-1, 3)

                poa = round(res_poa(fmc, w, 'cvxopt'), 3)

                poa1 = round((2. - fmc[2])**-1, 3)
                poa2 = round(w[2]/(w[2]+fmc[2]-fmc[3]), 3)

                pob1 = round((1.+fmc[2]+fmc[2]**2)/(2.+fmc[2]), 3)
                pob2 = round((1.+fmc[2]+fmc[3])/(2.+fmc[2]), 3)
                pob3 = round((1.+fmc[2]/2.+fmc[3]/2.)/(2.), 3)

                if poa == poa1:
                    plt.plot(ralp, rbet, '.k')
                elif poa == poa2:
                    plt.plot(ralp, rbet, '.b')
                else:
                    plt.plot(ralp, rbet, '.r')
            print(c)
            c += 1
    plt.show()


def checknoboth():
    for _ in range(10):
        w = gensubmodw(5)
        fmc = [0] + [round(e, 4) for e in list(np.diff(w))]
        args = oneemptydual(w, fmc)
        sol = lp('cvxopt', *args)

        args2 = onenobothprimal(w, fmc)
        sol2 = lp('cvxopt', *args2)

        if sol is not None and sol2 is not None:
            pob = round(sol['min'], 3)
            pob2 = -round(sol2['min'], 3)

            if pob != pob2:
                print('FALSE')
                print(pob, pob2)
                print(w, fmc)


def check1reduced(n):
    for _ in range(100):
        w = gensubmodw(n)
        fmc = [0] + [round(e, 4) for e in list(np.diff(w))]
        args = oneemptydual(w, fmc)
        sol = lp('cvxopt', *args)

        args2, part = reducedoneLPempty(w, fmc, True)
        sol2 = lp('cvxopt', *args2)

        if sol is not None and sol2 is not None:
            pob = round(sol['min'], 3)
            pob2 = round(sol2['min']**-1, 3)

            if pob != pob2:
                print('FALSE')
                print(pob, pob2)
                print(w, fmc)

def primaltricks():
    w = [round(e, 1) for e in gensubmodw(3)]
    f = [0] + [round(e, 4) for e in list(np.diff(w))]
    print(w, f)

    args, partition = reducedoneLPempty(w, f, True, True)
    sol = lp('cvxopt', *args)

    if sol is not None:
        pob = round(sol['min'], 4)
        print(pob)

        for i, p in enumerate(partition):
            val = round(sol['argmin'][i]*100, 1)
            if val != 0:
                print(p, val)

    args = oneemptydual(w, f)
    sol = lp('cvxopt', *args)
    if sol is not None:
        print([round(e, 3) for e in sol['argmin']])

def checkoutnonempty():
    w = [round(e, 1) for e in gensubmodw(2)]
    f = [0] + [round(e, 4) for e in list(np.diff(w))]
    print(w, f)

    args, partition = kroundLPnonempty(w, f, 1)
    sol = lp('cvxopt', *args)

    if sol is not None:
        pob = round(sol['min'], 4)
        print(pob)

        for i, p in enumerate(partition):
            val = round(sol['argmin'][i]*100, 1)
            if val != 0:
                print(p, val)


def pobnonempty(w, f, k=1):
    args, partition = kroundLPnonempty(w, f, k)
    sol = lp('cvxopt', *args)

    if sol is not None:
        pob = round(sol['min'], 4)
        values = [round(e*100, 1) for e in sol['argmin']]
        return pob, partition, values
    else:
        return 0, [], []


def getlinearpoa(w):
    maxval = 0
    n = len(w) - 1
    lamb = max([w[k]/k for k in range(1, n+1)])
    for l in range(int(n/2+1)):
        #print(l)
        for j in range(l+1, n+1):
            val = (w[l] + lamb * (j - l))/w[j]
            if val > maxval:
                maxval = val
    return 1./maxval

def getlinearpoa2(w):
    minval = 1
    n = len(w) - 1
    for l in range(int(n/2-1)):
        for j in range(l+1, n+1):
            for k in range(1, n+1):
                val = (k*w[j])/(k*w[l] + w[k]*(j - l))
                if val < minval:
                    minval = val
    return minval


def checklinear():
    # w = [0, 1, 1, 2]
    # f = [0] + [1]*3
    # poa = round(getlinearpoa(w), 4)
    # poa2 = round(res_poa(f, w, 'cvxopt'), 4)
    # print(poa, poa2)

    for _ in range(500):
        n = 8
        w = gensubmodw(n)
        f = [0] + [1]*n
        if 0 in w[1:]:
            print('iszero')
        else:
            poa = round(getlinearpoa(w), 5)
            poa2 = round(w[-1]/n, 5)
            if poa != poa2:
                print(poa, poa2, w)

def checklater():
    n = 2
    for i in range(100):
        w, f = genrandomwf(n)
        pob, _, _ = pobnonempty(w, f, 1)
        pob2, _, _ = pobnonempty(w, f, 2)
        poa = round(res_poa(f, w, 'cvxopt'), 4)
        print(w, f, pob, pob2, poa)


def test_order():
    orders = [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]]
    for _ in range(200):
        w = gensubmodw(3)
        f = [0] + [round(e, 4) for e in list(np.diff(w))]
        pobs = [0.]*6
        for i, order_2 in enumerate(orders):
            order = [[0, 1, 2], order_2]

            args, _ = arbitrarykorderempty(w, f, order)
            sol = lp('cvxopt', *args)

            if sol is not None:
                pob = round(sol['min'], 4)
                pobs[i] = pob

        pob_order = [e for e, p in sorted(list(zip(orders, pobs)), key=lambda x: x[1])]
        print(pob_order)


def cache():
    w = [0, 1, 1.5, 2.]
    pobmax = 0
    f = [0.0, 1.0, 0.655, 0.64]
    # args, partition = kroundLPempty(w, f, 1)
    # sol = lp('cvxopt', *args)

    # for i, p in enumerate(partition):
    #     val = round(sol['argmin'][i]*100, 1)
    #     if val != 0:
    #         print(p, val)

    # for _ in range(1000):
    #     _, f = genrandomwf(3)
    #     #f = [0.0, 1.0, 0.655, 0.64]  # .8422
    #     args, _ = kroundLPempty(w, f, 1)
    #     sol = lp('cvxopt', *args)

    #     if sol is not None:
    #         pob = round(sol['min'], 4)
    #         if pobmax < pob:
    #             print(pob, f)
    #             pobmax = pob

    # w = [0, 1, 1.5, 1.5]
    # f = [0] + [round(e, 4) for e in list(np.diff(w))]

    # args, partition = arbitrarykorderempty(w, f, [[0, 1, 2], [2, 1, 0]], reduced=True)
    # sol = lp('cvxopt', *args)

    # print(sol['min'])

    # for i, p in enumerate(partition):
    #     val = round(sol['argmin'][i]*100, 1)
    #     if val != 0:
    #         print(p, val)

    # print()
    # args, partition = kroundLPempty(w, f, 2, reduced=True)
    # sol = lp('cvxopt', *args)
    # print(sol['min'])
    # for i, p in enumerate(partition):
    #     val = round(sol['argmin'][i]*100, 1)
    #     if val != 0:
    #         print(p, val)

def test_samepob():
    for l in [4]:
        for a in [.7]:
            b = round(.7*2 - a, 2)
            f = [0, 1, a, b] + [.5]*l
            w = [sum(f[:i+1]) for i in range(len(f))]
            
            args, partition = reducedoneLPempty(w, f, True)
            sol = lp('cvxopt', *args)

            if sol is not None:
                 pob = round(sol['min']**-1, 4)
                 print(a, pob)

                 for i, p in enumerate(partition):
                     val = round(sol['argmin'][i]*100)
                     if val != 0:
                         print(p, val)

def test_spread():
    for _ in range(50):
        w, f = gensubmodw(7, .2)
        print(f)
        for k in range(1, 7):
            fp = f[:k+1]
            minpob = 2
            maxpob = 1
            for las in np.arange(0, fp[-1]*1.05, fp[-1]/5):
                fplas = fp + [las]
                wplas = [sum(f[:i+1]) for i in range(len(fplas))]
                
                args, _ = reducedoneLPempty(wplas, fplas, True)
                sol = lp('cvxopt', *args)

                if sol is not None:
                    pob = round(sol['min']**-1, 4)                    
                    if pob < minpob:
                        minpob = pob
                    if pob > maxpob:
                        maxpob = pob
            diff = round(maxpob - minpob, 4)
            norm_diff = round(diff/las, 4)
            print(diff, norm_diff)


def checksetcovering():
    w = [0, 1, 1, 1]
    f = [0, 1, -.2, -1]

    args, partition = kroundLPempty(w, f, 1)
    sol = lp('cvxopt', *args)

    if sol is not None:
        pob = round(sol['min'], 4)
        values = [round(e*100, 1) for e in sol['argmin']]
        
        print(pob**-1)

        for v, p in zip(values, partition):
            if v > 0:
                print(p, v)

def checkpoa():
    w = [0, 1, 1, 1, 1, 1]
    f = [0, 1, .5, .3, -.04, .1]
    p = res_poa(f, w, 'cvxopt')
    print(p)
    game = worst_game(f, w, 'cvxopt', TOL=4)
    print(game.actions)
    print([round(v, 3) for v in game.values])

def pobempty(w, f, k=1, reduced=False):
    args, partition = kroundLPempty(w, f, k, reduced)
    sol = lp('cvxopt', *args)

    if sol is not None:
        pob = round(sol['min'], 4)
        values = [round(e*100, 1) for e in sol['argmin']]
        return pob, partition, values
    else:
        return 0, [], []



def pobpoatradupper(w, X, upper):
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

    def PBI(N, upper):
        if upper:
            part = [()]*((N+1)*(N+2)/2)
            c = 0
            for a in range(N+1):
                for x in range(a, N+1):
                    part[c] = (a, x, 0)
                    c+=1
        else:
            part = [()]*(N*(N+1))
            c = 0
            for a in range(1, N+1):
                for b in range(N+1):
                    x = max(a+b-N, 0)
                    part[c] = (a-x, x, b-x)
                    c+=1
        return part

    def poaconst():
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

            # print(a, x, b)
            # print(const_h[i])
            # print(const[i,:])

        return const, const_h


    def pobconst(upper=False):
        partition = PBI(N, upper)
        NC = len(partition)
        const = np.zeros((NC, N+1), dtype='float')
        const_h = np.zeros((len(const), 1), dtype='float')

        for i, (a, x, b) in enumerate(partition):
            const[i, 0] = w[a+x]
            for j in range(1, a+1):
                const[i, j] = -1
            if b != 0:
                const[i, a+x+1] = b
            const_h[i] = w[b+x]
        return const, const_h


    N = len(w)-1
    C = np.zeros((N+1,), dtype='float')
    nonnegcons = np.identity(N+1,  dtype='float')
    nonnegcons_h = np.zeros((N+1, 1), dtype='float')
    C[0] = 1

    pobcons, pobcons_h = pobconst(upper)
    poacons, poacons_h = poaconst()
    G = -np.vstack((pobcons, poacons, nonnegcons))
    H = -np.vstack((pobcons_h, poacons_h, nonnegcons_h))
    
    return C, G, H

def plotpobpoagenw(w, step):
    poarange = []
    pobrange = []
    for poa in np.arange(step, 1+step, step):
        X = poa**-1
        args = pobpoatradupper(w, X)
        sol = lp('cvxopt', *args)
        if sol is not None:
            poarange.append(poa)
            pobrange.append(round(sol['min']**-1, 4))
    plt.plot(poarange, pobrange, 'k')
    plt.show()


def approxdualpob(w, f, reduced=False):
    args = step1redLPmod(w, f, reduced)
    sol = lp('cvxopt', *args)

    if sol is not None:
        return sol

def makefig():
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
    plt.xlabel('PoA(w, f, n)')
    plt.ylabel('PoB(w, f, n, 1)')
    plt.show()



def genwzero(n, pow=.3):
    w = [0., 1.] + [0]*(n-1)
    for i in range(2, len(w)-1):
        w[i] = round(w[i-1] + (w[i-1] - w[i-2])*np.power(np.random.random(), pow), 3)
    w[-1] = w[-2]
    return w

def modgairing(n, d):
    from math import factorial
    B = 1./(n-1)/factorial(n-1)
    y = lambda j: sum([1/factorial(i) for i in range(j, n)])
    return [0] + [round(factorial(j-1)*(B + y(j))/(B + y(1)), 5) for j in range(1, n+1)]

def getclosetohalf():
    n = 6
    for _ in range(1000):
        w = gensubmodw(n, .5)
        f = MC(w)

        args = oneemptydual(w, f, True)
        sol = lp('cvxopt', *args)
        if sol is not None:
            lambdas =  all([round(e, 2) == 1 for e in sol['argmin'][1:]])
            if sol['min'] < 1.9 and lambdas:
                print(w, f)
                print(sol)


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

def getcoeff(w, n):
    # get coefficients for decomposition
    n1 = 2*w[1] - w[2]
    nk = [2*w[k] - w[k-1] - w[k+1] for k in range(2, n)]
    nn = w[1] - sum(nk) - n1 
    return [n1] + nk + [nn]

def ftractable(w, n):
    coeff = getcoeff(w, n)
    f = np.zeros((n+1,))
    for k, nk in enumerate(coeff):
        foptk = abcoveringf(n, 1, k+1)
        f += nk*np.array(foptk)
    return f


if __name__ == '__main__':
    n = 10
    w = [0, 1, 2] + [3]*(n-2)
    ftract = ftractable(w, n)
    # print(ftract)
    # print(abcoveringf(n, 1, 3))

    for _ in range(100):
        w = gensubmodw(n, .3)
        _, fopt = res_opt_f(w, 'cvxopt')
        ftract = ftractable(w, n)
        fopt = [round(e, 4) for e in fopt]
        ftract = [round(e, 4) for e in ftract]
        if not all([fopt[i] >= ftract[i] for i in range(n+1)]):
            print('Wrong', w)
            print(fopt, ftract)



    #print(w, f)

    # args, partition = oneemptyprimal(w, f)
    # sol = lp('cvxopt', *args)
    # print(round(-sol['min']**-1, 4))
    # args, partition = oneemptyprimalsum(w, f)
    # sol = lp('cvxopt', *args)
    # print(round(-sol['min']**-1, 4))
    # print(res_poa(f, w, 'cvxopt'))
    # for i, p in enumerate(partition):
    #         val = round(sol['argmin'][i], 3)
    #         if val > 0:
    #             print(p, val)

    # pob, partition, values = pobempty(w, f, 1, True)
    # print(pob)

    # # pobmax = .7
    # # for _ in range(10000):
    # #     f = gendecf(3)
    # #     pob, _, _ = pobempty(w, f)
    # #     if pob > pobmax:
    # #         print(f, pob)
    # #         pobmax = pob


    # for i in range(len(partition)):
    #     if values[i] > 0:
    #         print(partition[i], round(values[i]/.376))
    
    #makefig()

    #print(res_poa([0,1,.4,.3,-.1], [0, 1, 1, 1, 1], 'cvxopt'))

    # n = 15
    # fgar = modgairing(n, 0)
    # print(fgar)
    # for i in range(1000):
    #     w = genwzero(n, .2)
    #     _, f = res_opt_f(w, 'cvxopt')
    #     if any([e2 < e1 for e1, e2 in zip(fgar[1:], f[1:])]):
    #         print(f, fgar)
    #     else:
    #         print(i)


    #getclosetohalf()
    # w = [0, 1, 2, 2.7, 3, 3.2]
    # f = [0, 1, 1, .7, .3, .2]

    # args = oneemptydual(w, f, True)
    # sol = lp('cvxopt', *args)
    # print(sol)

    # w = [0, 1, 1.5, 2, 2, 2]
    # f = [0, 1, .3, .2, 0, 0]

    # poa, par, val = pobempty(w, f, 1, True)

    # print(poa)
    # for p, v in zip(par, val):
    #     if v > 0:
    #         print(p, v)
