from kroundLP import *
from games.price_of_anarchy import res_poa, res_opt_f, worst_game
import matplotlib.pyplot as plt
import warnings
from functools import reduce
warnings.filterwarnings("ignore")


def Mi(j, k): return (min(j, k))**-1

def prod(li):
    if not li:
        return 1
    else:
       return reduce(lambda a, b: a*b, li)

def trans(a, b, k):
    if a == b:
        return 1
    elif b > a:
        return Mi(b-1, k) * prod([(1 + Mi(i, k)) for i in range(a, b-1, 1)])
    else:
        raise ValueError('b < a')

def trans2(a, b, k):
    if a == b:
        return 1
    elif a<= k+1 and b <= k+1:
        return 1./a
    elif a<= k+1 and b > k+1:
        return 1./a*((k+1.)/k)**(b-k-1)
    else:
        return 1./k*((k+1.)/k)**(b-a-1)


def A(t, k): return np.array([[1, 1], [Mi(t, k), Mi(t, k)]])

def H(n): return sum([1./i for i in range(1, n+1)])

a = 5
b = 20
k = 10
# matrices = [A(i, k) for i in range(a, b)]
# matrices.reverse()
# print(matrices)
# ppp = reduce(lambda x, y: np.dot(x, y), matrices)
# print('MAT', ppp)
# print('TRANS', trans(a, b, k))
# print('TRANS2', trans2(a, b, k))
# print(A(3, 2))


def koptPOBrecurv(n, k, B, mu, w):
    f = [0., 1.] + [0.]*(n-1)
    for i in range(1, n):
        t1 = 1 - Mi(i, k)*mu*w[i]
        t2 = Mi(i, k)*sum(f)
        f[i+1]  = t2 + t1
    return [round(e, 4) for e in f]

def koptPOBnotsimp(n, k, mu, w, B):
    f = [0., 1.] + [0.]*(n-1)
    for i in range(2, n+1):
        f[i] = trans2(1, i, k) + sum([trans2(m+1, i, k)*(1-Mi(m, k)*mu*w[m]) for m in range(1, i)])


        # if i > k+1:
        #     print(i)
        #     print([round(trans2(m+1, i, k)*(1-Mi(m, k)*mu*w[m]), 4) for m in range(k+1, i)])
        #     print()

        #     # print([round((1-Mi(m, k)*mu*w[m]), 4) for m in range(k+1, i+1)])
        #     # print([round((1 - (B*m/k + (1-B))*mu), 4) for m in range(k+1, i+1)])
    return [round(e, 4) for e in f]


def koptPOBsimp2(n, k, mu):
    f = [0., 1.] + [0.]*(n-1)
    for i in range(2, n+1):
        if i <= k+1:
            f[i] = 1 + (H(i-1) - 1)*(1-mu) + 1-mu
        else:
            t1 = ((k+1.)/k)**(i-k-1)
            t2 = (H(k+1)-1)*((k+1.)/k)**(i-k-1)*(1-mu)
            t3 = sum([(1 - (B*m/k + (1-B))*mu)*(1./k*((k+1.)/k)**(i-m-2)) for m in range(k+1, i-1)])
            t4 = 1 - (B*(i-1)/k + (1-B))*mu
            f[i] = t1 + t2 + t3 + t4
    return [round(e, 4) for e in f]

def koptPOBsimp(n, k, mu):
    f = [0., 1.] + [0.]*(n-1)
    for i in range(2, n+1):
        t1 = 0
        t2 = 0
        t3 = 0
        t4 = 0
        f[i] = t1 + t2 + t3 + t4
    return [round(e, 4) for e in f]

def kcovw(B, k, n):
    w = [0.] + [0.]*n
    for i in range(1, n+1):
        w[i] = B*i + (1-B)*min(i, k)
    return [round(e, 4) for e in w]



if __name__ == '__main__':
    n = 19
    B = .5
    k = 13
    mu = (H(k+1) + k/(k+1))/(B + H(k+1) - 1./(k+1))
    w = kcovw(B, k, n)
    f = koptPOBrecurv(n, k, B, mu, w)
    f2 =  koptPOBnotsimp(n, k, mu, w, B)
    f3 =  koptPOBsimp2(n, k, mu)
    print(w)
    print(f)
    print(f2)
    print(f3)
    print([0., 1.] + [round(2 - mu + (H(i-1) - 1)*(1-mu), 4) for i in range(2, k+2)])
    print(round(B*mu, 4))
    # pass