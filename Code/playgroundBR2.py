from kroundLP import *
from games.price_of_anarchy import res_poa, res_opt_f, worst_game
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")





def optPOB(n, B, mu):
    f = [0.] + [0.]*n
    for i in range(1, n+1):
        t1 = 2.**(max(i-2., 0))
        t2 = sum([2.**(max(i-2.-j, 0))*(1. - mu*((1.-B) + B*j)) for j in range(1, i)])
        f[i]  = t1 + t2
    return [round(e, 4) for e in f]


def closedPOB(n, B, mu):
    f = [0., 1.] + [0.]*(n-1)
    for i in range(2, n+1):
        t1 = 2.**(i-2)
        t2 = (1. - 2.**(2.-i))
        t3 = (1. - mu + mu*B)
        t4 = (2. - (i-1.)*2.**(3.-i) + (i-2.)*2.**(2.-i))
        t5 = mu*B
        t6 = 1. - mu*(1.-B+B*(i-1.))
        f[i] = t1*(1. + t2*t3 - t4*t5) + t6
    return [round(e, 4) for e in f]






def closedPOB2(n, B, mu):
    f = [0., 1.] + [0.]*(n-1)
    for i in range(2, n+1):
        t1 = 2.**(i-1)
        t2 = 2.**(i-2)*(B-1) - 2.**(i-1)*B + B
        f[i] = t1 + t2*mu
    return [round(e, 4) for e in f]


def getf(w, mu, n):
    f = [0., 1.]
    for i in range(1, n):
        val = (sum(f) + (1-mu)*w[i])/i
        f.append(val)
    return f


def koptPOBrecurv(n, k, B, mu, w):
    f = [0., 1.] + [0.]*(n-1)
    for i in range(1, n):
        Mi = (min(i, k))**-1
        t1 = 1 - Mi*mu*w[i]
        t2 = Mi*sum(f)
        f[i+1]  = t2 + t1
    return [round(e, 4) for e in f]

def koptPOBnotsimp(n, k, B, mu, w):
    def Mi(i): return (min(i, k))**-1
    def trans(a, b): return Mi(a) * prod([(1 + Mi(i)) for i in range(a+1, b-1)]) 
    f = [0., 1.] + [0.]*(n-1)
    for i in range(2, n+1):
        f[i] = trans(1, i) + sum([trans(m, i)*(1-Mi(m)*mu*w[m]) for m in range(1, i-2)]) + (1-Mi(i)*mu*w[i])
    return [round(e, 4) for e in f]

def kcovw(B, k, n):
    w = [0.] + [0.]*n
    for i in range(1, n+1):
        w[i] = B*i + (1-B)*min(i, k)
    return [round(e, 4) for e in w]

def ranf(n):
    f = [0., 1.] + [0]*(n-1)
    for i in range(2, n+1):
        f[i] = round(np.random.random()*4+1, 3)
    return f


if __name__ == '__main__':
    # n = 10
    # B = .5
    # k = 2
    # mu = (1/2 + B/2)**-1 - .075
    # w = kcovw(B, k, n)
    # f = koptPOBrecurv(n, k, B, mu, w)
    # f2 =  koptPOBnotsimp(n, k, B, mu, w)
    # print(w)
    # print(f)
    # print(f2)

    # w = [0, 1, 3, 5, 7, 9]
    # for _ in range(1000):
    #     f = ranf(5)
    #     pob, _ = kwrapperLP(w, f, 1)
    #     print(f, pob)
    #     if pob >= 5/9:
    #         print('BETTER: ', f, pob)





    n = 4
    b = 7
    w = [0, 1, 2, 4, 8, 16] #[0] + [1 + b*i for i in range(n+1)]
    f = [0, 1, 1, 1, 1, 1] # + [w[i]/i for i in range(1, n+2)]
    print(w, f)

    pob, values = kwrapperLP(w, f, 1, dual=True)
    partition = partition(2, len(w)-1)
    print(pob, values)

    lamb = 16/5
    for p in partition:
        br = [e[0] for e in p]
        opt = [e[1] for e in p]
        cons = w[sum(opt)]
        for i in range(5):
            brsubi = sum(br[:i])
            v1 = br[i]*f[brsubi+1] - opt[i]*f[brsubi+1]
            cons += v1*lamb
        if sum(br) > 0:
            val = round(cons/w[sum(br)], 4)
            if val > 3.1:
                print(val, p)

    # for i in range(len(values)):
    #     if values[i] > 0:
    #         print(values[i], partition[i])



    # c = .5
    # mu = 2 - c -.5
    # n = 30
    # w = [0.] + [1.+c*i for i in range(n)]
    # f = getf(w, mu, n)
    # print(w)
    # print([round(e, 3) for e in f])



    # n = 3
    # k = 2
    # part = partition(k+1, n)
    # w = [0, 1, 2, 2, 2]

    # pobmax = 0
    # d = .1
    # for f2 in np.arange(0, 1 + d, d):
    #     for f3 in np.arange(0, 1 + d, d):
    #         for f4 in np.arange(0, 1 + d, d):
    #             f = [0, 1, f2, f3, f4]
    #             w = [round(e, 2) for e in w]
    #             f = [round(e, 2) for e in f]
    #             pob, _ = kwrapperLP(w, f, k, empty=False)
    #             print(pob, ' : ', w, f)
    #             if pobmax < pob:
    #                 pobmax = pob
    # print('MAX: ', pobmax)

    # c = .5
    # w = [0, 1, 1+c, 1+2*c]
    # f = [0, 1, c+.1, c-.1]
    # n = 3
    # k = 2
    # pob, val = kwrapperLP(w, f, k)

    # print('MC: ', pob)
    # # print(res_poa([0, 1, .45, .25], [0, 1, 1, 1], 'cvxopt'))

    # for v, p in zip(val, partition(k+1, n)):
    #     if v  > 0:
    #         print(v, p)

    # c = .5
    # n = 5
    # w = [0, 1, 2, 3] + [4 + c*i for i in range(n)]
    # f = [0, 1, 1, 1, 1] + [c]*(n-1)

    # print(w, f)

    # poa = res_poa(f, w, 'cvxopt')
    # wg = worst_game(f, w, 'cvxopt')
    # print(wg.values)
    # print(wg.actions)
    # print(poa, (2-c)**-1)
    
    # n = 4
    # B = 0.5
    # mu = (1/2 + B/2)**-1
    # print(mu, B*mu)

    # w = [0., 1, 2, 3, 4, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6]
    # f = [0., 1, 1, 1, 1, 2/11, 2/11, 2/11, 2/11, 2/11, 2/11]
    # # print(optPOB(n, B, mu))
    # # print(closedPOB(n, B, mu))
    # # print(closedPOB2(n, B, mu))
    # print(w)
    # print(f)
    # print('PoA: ', res_poa(f, w, 'cvxopt'))
    # Game = worst_game(f, w, 'cvxopt')
    # print(Game.actions)
    # print(Game.values)