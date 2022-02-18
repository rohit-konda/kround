from kroundLP import *
from games.price_of_anarchy import res_poa, res_opt_f, worst_game
import matplotlib.pyplot as plt
import warnings


def abcovering(n, a, b):
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


def Cwelfare(n, c, k): return [0] + [(1-c)*i + c*min(i, k) for i in range(1, n+1)]


def cons(w, f, j, l): 
    return sum(f[:j+1]) - l*f[j+1] + w[l]

def mu(w, f, j, l): return cons(w, f, j, l)/w[j]



if __name__ == '__main__':
    n = 20
    c = .5
    k = 1
    dc = .1
    conss =np.zeros((n-1,))
    f = abcovering(n, c, k)
    w = Cwelfare(n, c, k)
    for j in range(1, n):
        conss[j-1] = cons(w, f, j, 1)

    plt.plot(range(1, n), conss, w[2:], 'k.')
    plt.show()
