import numpy as np
from itertools import product

def kroundLPnonempty(w, f, k):
    n = len(w)-1
    n_ac = k+2
    partition = list(product(product([1, 0], repeat=n_ac), repeat=n))[:-1]
    n_c = len(partition)

    C = np.zeros((n_c,), dtype='float')
    br_cons = np.zeros((n*(n_ac-1)*(n_ac-2), n_c), dtype='float')
    A = np.zeros((1, n_c), dtype='float')

    a_k = -2
    a_opt = -1
    
    for i, p in enumerate(partition):
        opt_i = [pl for pl in range(n) if p[pl][a_opt]==1]
        kbest_i = [pl for pl in range(n) if p[pl][a_k]==1]
        C[i] = w[len(kbest_i)]
        A[0, i] = w[len(opt_i)]
    
        c = 0
        for j in range(n):
            for b in range(k):
                bnext = b + 1
                nextac = [pl for pl in range(j+1, n) if p[pl][b]==1]
                prevac = [pl for pl in range(j) if p[pl][bnext]==1]
                other_ac = nextac + prevac

                for a_other in range(n_ac):
                    if a_other == bnext:  
                        continue

                    if p[j][bnext] == 1 and p[j][a_other] == 0:
                        br_cons[c][i] = f[len(other_ac + [j])]
                    elif p[j][bnext] == 0 and p[j][a_other] == 1:
                        br_cons[c][i] = -f[len(other_ac + [j])]
                    c += 1

    cons_2 = np.identity(n_c)
    G = -np.vstack((br_cons, cons_2))
    H = np.zeros((len(G), 1))
    B = np.ones((1, 1))
    return (C, G, H, A, B), partition

def kroundLPempty(w, f, k, reduced=False):
    n = len(w)-1
    n_ac = k+1
    partition = list(product(product([1, 0], repeat=n_ac), repeat=n))[:-1]
    n_c = len(partition)

    C = np.zeros((n_c,), dtype='float')
    br_cons = np.zeros((n*n_ac*k, n_c), dtype='float')
    A = np.zeros((1, n_c), dtype='float')

    a_k = -2
    a_opt = -1
    
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
            for b in range(k):
                prevac = [pl for pl in range(j) if p[pl][b]==1]
                nextac = [pl for pl in range(j+1, n) if p[pl][b-1]==1] if b > 0 else []
                other_ac = nextac + prevac

                for a_other in range(n_ac):
                    if a_other == b:  
                        continue
                    if p[j][b] == 1 and p[j][a_other] == 0:
                        br_cons[c][i] = f[len(other_ac + [j])]
                    elif p[j][b] == 0 and p[j][a_other] == 1:
                        br_cons[c][i] = -f[len(other_ac + [j])]
                    c += 1

    cons_2 = np.identity(n_c)
    print(np.shape(br_cons), np.shape(cons_2))

    G = -np.vstack((br_cons, cons_2))
    H = np.zeros((len(G), 1))
    B = np.ones((1, 1))
    return (C, G, H, A, B), partition

def oneemptydual(w, f, mindual=False):
    n = len(w) - 1
    partition = list(product(product([1, 0], repeat=2), repeat=n))

    C = np.zeros((n+1,), dtype='float')
    C[0] = 1
    if mindual:
        C += .0001
    G = np.zeros((len(partition), n+1))
    cons_2 = -np.hstack((np.zeros((n, 1)), np.identity(n)))
    G = np.vstack((G, cons_2))
    H = np.zeros((len(G), 1))
    
    for i, p in enumerate(partition):
        brall = [e[0] for e in p]
        optall = [e[1] for e in p]
        G[i, 0] = -w[sum(brall)]
        H[i, 0] = -w[sum(optall)]
        for j in range(n):
            brsubj = int(sum(brall[:j]))
            allj = p[j]
            if allj == (1, 0):
                G[i, j+1] = f[brsubj+1]
            elif allj == (0, 1):
                G[i, j+1] = -f[brsubj+1]

    return C, G, H

def oneemptyprimal(w, f):
    def nashfunc(jselect, NJ):
        if jselect == 1:
            return f[len(NJ + [j])]
        elif jselect == 3:
            return -f[len(NJ + [j])]
        else:
            return 0

    n = len(w)-1  # number of players
    n_c = 4**n - 1  # number of allocation types
    partition = list(product([1, 2, 3, 4], repeat=n))[:-1]  # resource types

    c = np.zeros((n_c,), dtype='float')
    cons_1 = np.zeros((n, n_c), dtype='float')
    A = np.zeros((1, n_c), dtype='float')

    for j in range(n):
        for i, p in enumerate(partition):
            # allocations for all agents
            Na = [k for k in range(n) if p[k]==1]
            Nx = [k for k in range(n) if p[k]==2]
            Nb = [k for k in range(n) if p[k]==3]

            # allocations for first j agents
            NJ = [k for k in range(j) if p[k]<=2]
            jselect= p[j]

            c[i] = -w[len(Nb + Nx)]  # maximize welfare of optimal allocation
            A[0, i] = w[len(Na + Nx)]  # set welfare of 1 round best response to 1
            cons_1[j][i] = nashfunc(jselect, NJ)  # best response constraint

    cons_2 = np.identity(n_c)
    G = -np.vstack((cons_1, cons_2))
    h = np.zeros((n_c+n, 1))
    b = np.array([[1]], dtype='float')
    
    return (c, G, h, A, b), partition


def oneemptyprimalsum(w, f):
    def nashfunc(jselect, NJ):
        if jselect == 1:
            return f[len(NJ + [j])]
        elif jselect == 3:
            return -f[len(NJ + [j])]
        else:
            return 0

    n = len(w)-1  # number of players
    n_c = 4**n - 1  # number of allocation types
    partition = list(product([1, 2, 3, 4], repeat=n))[:-1]  # resource types

    c = np.zeros((n_c,), dtype='float')
    cons_1 = np.zeros((n, n_c), dtype='float')
    A = np.zeros((1, n_c), dtype='float')

    for j in range(n):
        for i, p in enumerate(partition):
            # allocations for all agents
            Na = [k for k in range(n) if p[k]==1]
            Nx = [k for k in range(n) if p[k]==2]
            Nb = [k for k in range(n) if p[k]==3]

            # allocations for first j agents
            NJ = [k for k in range(j) if p[k]<=2]
            jselect= p[j]

            c[i] = -w[len(Nb + Nx)]  # maximize welfare of optimal allocation
            A[0, i] = w[len(Na + Nx)]  # set welfare of 1 round best response to 1
            cons_1[j][i] = nashfunc(jselect, NJ)  # best response constraint

    cons_1 = np.sum(cons_1, 0)
    cons_2 = np.identity(n_c)
    G = -np.vstack((cons_1, cons_2))
    h = np.zeros((n_c+1, 1))
    b = np.array([[1]], dtype='float')
    
    return (c, G, h, A, b), partition


def oneroundbest(w, f):
    n = len(w)-1
    n_ac = 2
    partition = list(product(product([1, 0], repeat=n_ac), repeat=n))[:-1]
    n_c = len(partition)

    C = np.zeros((n_c,), dtype='float')
    br_cons = np.zeros((n, n_c), dtype='float')
    A = np.zeros((1, n_c), dtype='float')
    
    for i, p in enumerate(partition):
        startac = [pl for pl in range(n) if p[pl][0]==1]
        endac = [pl for pl in range(n) if p[pl][1]==1]
        C[i] = w[len(endac)]
        A[0, i] = w[len(startac)]
        
        for j in range(n):
            startalloc = [pl for pl in range(j+1, n) if p[pl][0]==1]
            endalloc = [pl for pl in range(j) if p[pl][1]==1]
            otheralloc = startalloc + endalloc


            if p[j][1] == 1 and p[j][0] == 0:
                br_cons[j][i] = f[len(otheralloc + [j])]
            elif p[j][1] == 0 and p[j][0] == 1:
                br_cons[j][i] = -f[len(otheralloc + [j])]


    cons_2 = np.identity(n_c)
    G = -np.vstack((br_cons, cons_2))
    H = np.zeros((len(G), 1))
    B = np.ones((1, 1))
    return C, G, H, A, B

def kroundbest(w, f, k):
    n = len(w)-1
    n_ac = k+1
    partition = list(product(product([1, 0], repeat=n_ac), repeat=n))[:-1]
    n_c = len(partition)

    C = np.zeros((n_c,), dtype='float')
    br_cons = np.zeros((n*k*(n_ac+1), n_c), dtype='float')
    A = np.zeros((1, n_c), dtype='float')
    
    for i, p in enumerate(partition):
        startac = [pl for pl in range(n) if p[pl][0]==1]
        endac = [pl for pl in range(n) if p[pl][1]==1]
        C[i] = w[len(endac)]
        A[0, i] = w[len(startac)]

        c = 0
        for j in range(n):
            for b in range(k):
                bnext = b + 1
                nextac = [pl for pl in range(j+1, n) if p[pl][b]==1]
                prevac = [pl for pl in range(j) if p[pl][bnext]==1]
                other_ac = nextac + prevac

                for a_other in range(n_ac):
                    if a_other == bnext:  
                        continue

                    if p[j][bnext] == 1 and p[j][a_other] == 0:
                        br_cons[c][i] = f[len(other_ac + [j])]
                    elif p[j][bnext] == 0 and p[j][a_other] == 1:
                        br_cons[c][i] = -f[len(other_ac + [j])]
                    c += 1
    cons_2 = np.identity(n_c)
    G = -np.vstack((br_cons, cons_2))
    H = np.zeros((len(G), 1))
    B = np.ones((1, 1))
    return C, G, H, A, B

def reducedoneLPempty(w, f, reduced=False, expand=False):
    def optless(p):
        p = list(p)
        optall = sum([1 for e in p if e[1] == 1])
        nashall = sum([1 for e in p if e[0] == 1])
        
        if (nashall < 1 and optall < 2) or optall < 1:
            return False

        for i in range(len(p)):
            if p[i][1] == 1:
                nashbef = sum([1 for e in p[:i] if e[0] == 1])
                if optall > nashbef:
                    return True
        return False

    def nashlast(p):
        p = list(p)[::-1]
        nashall = sum([1 for e in p if e[0] == 1])
        if nashall < 2:
            return False
        for e in p:
            if e == (0, 1):
                return False
            elif e[0] == 1:
                return True

    def par_reduce(p):
        if optless(p) or nashlast(p):
            return None
        else:
            return p
    
    n = len(w)-1
    partition = list(product(product([1, 0], repeat=2), repeat=n))[:-1]
    if reduced:
        for i, p in enumerate(partition):
            partition[i] = par_reduce(p)
        partition = [e for e in partition if e is not None]

    n_c = len(partition)
    C = np.zeros((n_c,), dtype='float')
    br_cons = np.zeros((n, n_c), dtype='float')
    A = np.zeros((1, n_c), dtype='float')
    
    for i, p in enumerate(partition):
        opt_i = [pl for pl in range(n) if p[pl][1]==1]
        best_i = [pl for pl in range(n) if p[pl][0]==1]

        if expand:
            C[i] = -w[len(opt_i)] - .0001*sum([sum(pi) for pi in p])
            A[0, i] = w[len(best_i)]
        else:
            C[i] = w[len(best_i)]
            A[0, i] = w[len(opt_i)]

        for j in range(n):
            other_ac = [pl for pl in range(j) if p[pl][0]==1]
            if p[j][0] == 1 and p[j][1] == 0:
                br_cons[j][i] = f[len(other_ac)+1]
            elif p[j][0] == 0 and p[j][1] == 1:
                br_cons[j][i] = -f[len(other_ac)+1]

    cons_2 = np.identity(n_c)
    G = -np.vstack((br_cons, cons_2))
    H = np.zeros((len(G), 1))
    B = np.ones((1, 1))
    return (C, G, H, A, B), partition

def arbitrarykorderempty(w, f, order, reduced=False):
    n = len(w)-1
    k = len(order)
    n_ac = k+1
    partition = list(product(product([1, 0], repeat=n_ac), repeat=n))[:-1]
    n_c = len(partition)

    C = np.zeros((n_c,), dtype='float')
    br_cons = np.zeros((n*n_ac*k, n_c), dtype='float')
    A = np.zeros((1, n_c), dtype='float')

    a_k = -2
    a_opt = -1
    
    for i, p in enumerate(partition):
        opt_i = [pl for pl in range(n) if p[pl][a_opt]==1]
        kbest_i = [pl for pl in range(n) if p[pl][a_k]==1]
        if reduced:
            C[i] = w[len(kbest_i)] + .0001*sum([sum(pi) for pi in p])
        else:
            C[i] = w[len(kbest_i)]
        A[0, i] = w[len(opt_i)]
    
        c = 0
        for b in range(k):
            order_k = order[b]
            for j in range(n):
                ind = order_k.index(j)
                prevac = [pl for pl in order_k[:ind] if p[pl][b]==1]
                nextac = [pl for pl in order_k[ind+1:] if p[pl][b-1]==1] if b > 0 else []
                other_ac = nextac + prevac

                for a_other in range(n_ac):
                    if a_other == b:  
                        continue
                    if p[j][b] == 1 and p[j][a_other] == 0:
                        br_cons[c][i] = f[len(other_ac + [j])]
                    elif p[j][b] == 0 and p[j][a_other] == 1:
                        br_cons[c][i] = -f[len(other_ac + [j])]
                    c += 1

    cons_2 = np.identity(n_c)
    G = -np.vstack((br_cons, cons_2))
    H = np.zeros((len(G), 1))
    B = np.ones((1, 1))
    return (C, G, H, A, B), partition


def step1redLPmod(w, f, reduced=False):
    def makepart():
        partition = [[[0, 0] for _ in range(n)] for _ in range(n*(n+1)+2*n)]

        c = 0
        for a in range(1, n+1):
            for b in range(n+1):
                for i in range(a):
                    partition[c][i][0] = 1
                for j in range(n-b, n):
                    partition[c][j][1] = 1
                c += 1
        
        for i in range(n):
            partition[c][i][0] = 1
            c += 1

        for i in range(n):
            partition[c][i][1] = 1
            c += 1

        return partition

    n = len(w) - 1
    partition = makepart()
    #print(partition)

    C = np.zeros((n+1,), dtype='float')
    C[0] = 1
    if reduced:
        C[1:] = .0001

    G = np.zeros((len(partition), n+1))
    cons_2 = -np.hstack((np.zeros((n, 1)), np.identity(n)))
    G = np.vstack((G, cons_2))
    H = np.zeros((len(G), 1))
    
    for i, p in enumerate(partition):
        #print('p: ', p)
        brall = [e[0] for e in p]
        optall = [e[1] for e in p]
        #print(brall, optall)
        G[i, 0] = -w[sum(brall)]
        H[i, 0] = -w[sum(optall)]
        for j in range(n):
            brsubj = int(sum(brall[:j]))
            allj = p[j]
            if allj == [1, 0]:
                G[i, j+1] = f[brsubj+1]
            elif allj == [0, 1]:
                G[i, j+1] = -f[brsubj+1]

    cons_3 = np.zeros((2, n+1))
    cons_3[0, n-1] = -1
    cons_3[1, n] = -1

    H3 = np.zeros((2, 1))
    H3[0, 0] = -1.578
    H3 [1, 0] = -1.578

    H = np.vstack((H, H3))
    G = np.vstack((G, cons_3))

    print(G, H)

    return C, G, H


if __name__ == '__main__':
    from LPsolver import lp
    w = [0, 1, 1, 1]
    f = [0, 1, 0, 0]
    args = arbitrarykorderempty(w, f, [[0, 1, 2], [2, 1, 0]])
    sol = lp('cvxopt', *args[0])
    if sol is not None:
        pob = round(sol['min'], 2)
        values = [round(e, 2) for e in sol['argmin']]
    print(pob)
    partition = list(product(product([1, 0], repeat=3), repeat=3))[:-1]
    for i, p in enumerate(partition):
        if values[i] > 0:
            print(p, values[i])




    # df = .05
    # for order in [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]]:
    #     for f2 in np.arange(0, 1+df, df):
    #         for f3 in np.arange(0, 1+df, df):
    #             f = [0, 1, f2, f3]
    #             args = arbitrarykorderempty(w, f, [[0, 1, 2], order])
    #             sol = lp('cvxopt', *args[0])
    #             if sol is not None:
    #                 pob = round(sol['min'], 2)
    #                 if pob > .74:
    #                     print([round(e, 2) for e in f], pob, order)