import numpy as np
from itertools import product
from LPsolver import lp
from LPfunctions import kroundLPempty
from matplotlib import pyplot as plt
from playground import gensubmodw, gendecf
from games.price_of_anarchy import res_poa

def nashBR(w, f, reduced=False):
    n = len(w)-1
    nac = 3
    partition = list(product(product([1, 0], repeat=nac), repeat=n))[:-1]
    n_c = len(partition)

    C = np.zeros((n_c,), dtype='float')
    br_cons = np.zeros((n*4, n_c), dtype='float')
    nash_cons = np.zeros(((n-1)*2, n_c), dtype='float')
    A = np.zeros((1, n_c), dtype='float')

    a_nash = 0
    a_k = 1
    a_opt = 2
    
    for i, p in enumerate(partition):
        opt_i = [pl for pl in range(n) if p[pl][a_opt]==1]
        kbest_i = [pl for pl in range(n) if p[pl][a_k]==1]
        if reduced:
            C[i] = w[len(kbest_i)] + .0001*sum([sum(pi) for pi in p])
        else:
            C[i] = w[len(kbest_i)]
        A[0, i] = w[len(opt_i)]
    
        c = 0
        for j in range(n):
            for b in range(2):
                prevac = [pl for pl in range(j) if p[pl][b]==1]
                nextac = [pl for pl in range(j+1, n) if p[pl][b-1]==1] if b > 0 else []
                other_ac = nextac + prevac

                for a_other in range(nac):
                    if a_other == b:  
                        continue
                    if p[j][b] == 1 and p[j][a_other] == 0:
                        br_cons[c][i] = f[len(other_ac + [j])]
                    elif p[j][b] == 0 and p[j][a_other] == 1:
                        br_cons[c][i] = -f[len(other_ac + [j])]
                    c += 1
        c2 = 0
        for j in range(n-1):
            other_ac = [pl for pl in range(n) if p[pl][a_nash]==1 and pl != j]
            for a_other in [1, 2]:
                if p[j][a_nash] == 1 and p[j][a_other] == 0:
                    nash_cons[c2][i] = f[len(other_ac + [j])]
                elif p[j][a_nash] == 0 and p[j][a_other] == 1:
                    nash_cons[c2][i] = -f[len(other_ac + [j])]
                c2 += 1

    cons_2 = np.identity(n_c)
    G = -np.vstack((br_cons, nash_cons, cons_2))
    H = np.zeros((len(G), 1))
    B = np.ones((1, 1))
    return (C, G, H, A, B), partition


def justnashBR(w, f, reduced=False):
    n = len(w)-1
    nac = 2
    partition = list(product(product([1, 0], repeat=nac), repeat=n))[:-1]
    n_c = len(partition)

    C = np.zeros((n_c,), dtype='float')
    br_cons = np.zeros((n*4, n_c), dtype='float')
    nash_cons = np.zeros(((n-1)*2, n_c), dtype='float')
    A = np.zeros((1, n_c), dtype='float')

    a_nash = 0
    a_k = 0
    a_opt = 1
    
    for i, p in enumerate(partition):
        opt_i = [pl for pl in range(n) if p[pl][a_opt]==1]
        kbest_i = [pl for pl in range(n) if p[pl][a_k]==1]
        if reduced:
            C[i] = w[len(kbest_i)] + .0001*sum([sum(pi) for pi in p])
        else:
            C[i] = w[len(kbest_i)]
        A[0, i] = w[len(opt_i)]
    
        c = 0
        for j in range(n):
            for b in range(2):
                prevac = [pl for pl in range(j) if p[pl][b]==1]
                nextac = [pl for pl in range(j+1, n) if p[pl][b-1]==1] if b > 0 else []
                other_ac = nextac + prevac

                for a_other in range(nac):
                    if a_other == b:  
                        continue
                    if p[j][b] == 1 and p[j][a_other] == 0:
                        br_cons[c][i] = f[len(other_ac + [j])]
                    elif p[j][b] == 0 and p[j][a_other] == 1:
                        br_cons[c][i] = -f[len(other_ac + [j])]
                    c += 1
        c2 = 0
        for j in range(n-1):
            other_ac = [pl for pl in range(n) if p[pl][a_nash]==1 and pl != j]
            a_other = 1
            if p[j][a_nash] == 1 and p[j][a_other] == 0:
                nash_cons[c2][i] = f[len(other_ac + [j])]
            elif p[j][a_nash] == 0 and p[j][a_other] == 1:
                nash_cons[c2][i] = -f[len(other_ac + [j])]
            c2 += 1

    cons_2 = np.identity(n_c)
    G = -np.vstack((br_cons, nash_cons, cons_2))
    H = np.zeros((len(G), 1))
    B = np.ones((1, 1))
    return (C, G, H, A, B), partition

def plotdiff(w):
    for a in np.arange(0, 1, .1):
        for b in np.arange(0, 1, .1):

            args, _ = kroundLPempty(w, [0, 1, a, b], 2)
            sol = lp('cvxopt', *args)
            args2, _ = nashBR(w, [0, 1, a, b], False)
            sol2 = lp('cvxopt', *args2)

            if sol is not None and sol2 is not None:
                pob2 = round(sol['min'], 3)
                pobnsh = round(sol2['min'], 3)
                if pob2 - pobnsh < .005:
                    plt.plot(a, b, 'k.')
                else:
                    plt.plot(a, b, 'r.')
            else:
                plt.plot(a, b, 'b.')
    plt.show()


def run4check():
    w = [round(e, 2) for e in gensubmodw(3, .7)]
    f = [round(e, 2) for e in gendecf(3, .7)]
    print(w, f)

    args, partition = kroundLPempty(w, f, 1)
    sol = lp('cvxopt', *args)
    if sol is not None:
        print('1round', round(sol['min'], 2))

    args, partition = kroundLPempty(w, f, 2)
    sol = lp('cvxopt', *args) #, returnall=True)
    # print(sol)
    if sol is not None:
        print('2round', round(sol['min'], 2))

    args2, partition2 = nashBR(w, f)
    sol2 = lp('cvxopt', *args2)#, returnall=True)
    # print(sol2)
    if sol2 is not None:
        print('nashround', round(sol2['min'], 2))

    print()


def runpoacheck():
    w = [0, 1, 1.5] #[round(e, 2) for e in gensubmodw(2, 1)]
    f = [0, 1, .5] # [round(e, 2) for e in gendecf(2, 1)]
    args, partition = justnashBR(w, f)
    sol = lp('cvxopt', *args)
    args2, _ = kroundLPempty(w, f, 1)
    sol2 = lp('cvxopt', *args2)
    if sol is not None:
        print(w, f)
        print('justnashround', round(sol['min'], 2))
        print('nashround', round(sol2['min'], 2))
        print('poa', round(res_poa(f, w, 'cvxopt'), 3))

    for i in range(len(partition)):
        val = round(sol['argmin'][i]*10, 2)
        if val != 0:
            print(partition[i], '   ', val)


if __name__ == '__main__':
    # [runpoacheck() for i in range(1)]
    #[0.0, 1.0, 1.99, 2.83] [0.0, 1.0, 0.79, 0.16] - different than 

    w = [0, 1, 1, 1]
    f = [0, 1, 3/7, 2/7]

    args, partition = nashBR(w, f, False)

    sol = lp('cvxopt', *args)
    if sol is not None:
        print('nashround', round(sol['min'], 2))

        for i in range(len(partition)):
            val = round(sol['argmin'][i]*40/3.92, 2)
            if val != 0:
                print(partition[i], '   ', val)
