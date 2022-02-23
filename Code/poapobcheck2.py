import numpy as np
from LPsolver import lp

def I(N):
    from itertools import combinations
    ind = []
    for i in range(1, N+1):
        all_i = [(j[0], j[1]-j[0]-1, i-j[1]+1) for j in combinations(range(i+2), 2)]
        ind += all_i
    return ind

def dual_poa(f, w):
    N = len(w)-1
    Irange = I(N)
    num = len(Irange)

    G = np.zeros((num+1, 2), dtype='float')
    h = np.zeros((num+1, 1), dtype='float')
    c = np.array([[0], [1]], dtype='float')  # variables = [lambda , mu]

    for i, (a, x, b) in enumerate(Irange):
        G[i, 0] = a*f[a+x] - b*f[a+x+1] if a+x < N else a*f[a+x]
        G[i, 1] = -w[b+x]
        h[i] = w[a+x]
    G[num][0] = -1

    return c, G, h

def primal_poa(f, w):
    N = len(w)-1
    Irange = I(N)
    num = len(Irange)

    c = -np.array([w[b+x] for a, x, b in Irange], dtype='float')
    cons_1 = [a*f[a+x] - b*f[a+x+1] if a+x < N else a*f[a+x] for a, x, b in Irange]
    cons_2 = np.identity(num)
    G = -np.vstack((cons_1, cons_2))
    A = np.array([[w[a+x] for a, x, b in Irange]], dtype='float')
    b = np.array([[1]], dtype='float')
    h = np.zeros((num+1, 1))

    return c, G, h, A, b



def wck(C, k, j): return (1-C)*j + C*min(j, k)



def kopt(n, C, k, mu):
    f = [0, 1.]
    for j in range(1, n):
        fj1 = 1/k*sum(f) + 1 - 1/k*wck(C, k, j)*mu
        f.append(fj1)
    return [round(e, 3) for e in f]

def kopt2(n, C, k, mu):
    f = [0, 1.] + [0]*(n-1)
    D = float((k+1)/k)
    for j in range(2, n+1):
        auto = 1/k*D**(j-2)
        iden = (1 - 1/k*mu*wck(C, k, j-1))
        inp1 = sum([1/k*D**(j-tau-2)*(1-mu*(tau*(1-C)/k + C)) for tau in range(1, j-1)])
        inp2 = sum([1/k*D**(j-tau-2)*(mu*C-mu*C*tau/k) for tau in range(1, min(j-1, k))])
        inpt = sum([1/k*D**(j-tau-2)*(1-1/k*mu*wck(C, k, tau)) for tau in range(1, j-1)])

        in1 = (1-mu*C)*(1-D**(2-j))
        in2 = (mu*(C-1))*(D - (j-1)*D**(3-j) + (j-2)*D**(2-j))
        in3 = (1 - mu*D) + mu*C/k - (1./(k+1))*(mu*(C-1))*(j-1)*D**(3-j) + (mu-1)*D**(2-j)
        #(1-mu)+(mu*C/k) - 1/k*(j-1)*(mu*C-mu)*D**(2-j)+(mu-1)*D**(2-j)


        in21 = (mu*C)*(1-D**(1-k))
        in22 = -(mu*C)*(D - k*D**(2-k) + (k-1)*D**(1-k))
        in23 = (mu*C)*(D**(1-k) - 1./k)


        in4 = (mu*C)*(D**(1-k)) + (1 - mu)*D + (1./k)*(mu*(1-C))*(j-1)*D**(2-j) + (mu-1)*D**(2-j)

        in5 = (mu*C)*(D**(1-k)) + (1 - mu)*D

        auto = D**(j-1)
        iden = -1/k*mu*(j-1)
        inp6 = sum([1/k*D**(j-tau-2)*(1-mu*tau/k) for tau in range(1, j-1)])
        inp61 = mu*((j-1)*D**(2-j) - (j-2)*D**(1-j) - 1)
        
        if j-1 < k:
            f[j] = (1-mu)*D**(j-1) + mu #auto + iden + D**(j-1)*inp61 #+ inp6
        else:
            f[j] = (1-C)*mu

        #f[j] = D**(j-2)*(in5) + (1-C)*mu
    return [round(e, 3) for e in f]



if __name__ == '__main__':
    # w = [0, 1, 2, 2.1, 2.7, 3.4]
    # f = [0, 1, .8, .33, .2, .1]
    

    # args = dual_poa(f, w)
    # sol = lp('cvxopt', *args)
    # print(-round(sol['min'], 3))

    # args = primal_poa(f, w)
    # sol = lp('cvxopt', *args)
    # print(-round(sol['min']**(-1), 3))

    n = 20
    C = .9
    k = 10
    D = float((k+1)/k)
    mu = D**k/(D**k - C)
    print(mu*(1-C))
    f = kopt(n, C, k, mu)
    f2 = kopt2(n, C, k, mu)
    print(f)
    print(f2)

    import matplotlib.pyplot as plt
    x = range(0, n+1)
    plt.plot(x, f2, '.')
    plt.xlim(0, n+1)
    plt.ylim(0, 1)
    plt.show()
