import numpy as np
from itertools import product
from LPsolver import lp
from games.price_of_anarchy import res_poa


def gensubmodw(n):
    w = [0., 1.] + [0]*(n-1)
    for i in range(2, len(w)):
        w[i] = round(w[i-1] + (w[i-1] - w[i-2])*np.random.random(), 3)
    return w


def checknull(p):
    return all([e == (0, 0) for e in p])

def checknobrnull(p):
    for i in range(len(p)-1):
        if p[i] == (1, 0) and p[i+1] == (0, 0):
            return True
    return False

def checknonullopt(p):
    for i in range(len(p)-1):
        if p[i] == (0, 0) and p[i+1] == (0, 1):
            return True
    return False

def checknooptbeg(p):
    beg = True
    for e in p:
        if e == (0, 1):
            return True
        elif e == (1, 0) or e == (1, 1):
            return False

def checknobothnull(p):
    for i in range(len(p)-1):
        if p[i] == (0, 0) and p[i+1] == (1, 1):
            return True
        elif p[i] == (1, 1) and p[i+1] == (0, 0):
            return True
    return False

def reduce_res(partition):
    for i, p in enumerate(partition):
        if checknull(p) or \
           checknobrnull(p) or \
           checknobothnull(p) or \
           checknonullopt(p):
            partition[i] = None


    return [p for p in partition if p is not None], partition

def oneMCdual(w, f, reduced=False):
    n = len(w) - 1
    partition = list(product(product([1, 0], repeat=2), repeat=n))
    if reduced:
        partition, _ = reduce_res(partition)

    C = np.zeros((n+1,), dtype='float')
    C[0] = 1
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


def printpart(n):
    def letterpart(e):
        if e == (0, 0):
            return 'n'
        elif e == (0, 1):
            return 'o'
        elif e == (1, 0):
            return 'b'
        elif e == (1, 1):
            return 'x'

    partition = list(product(product([1, 0], repeat=2), repeat=n))
    _, reducedpart = reduce_res(list(product(product([1, 0], repeat=2), repeat=n)))
    for i in range(len(partition)):
        if reducedpart[i] is not None:
            pass #print(list(map(letterpart, partition[i])), list(map(letterpart, reducedpart[i])))
        else:
            print(list(map(letterpart, partition[i])))

def getpob(w, f, reduced=False):
    args = oneMCdual(w, f, reduced)
    sol = lp('cvxopt', *args)
    if sol is not None:
        pob = round(sol['min'], 3)
        return pob
    else:
        return 0

if __name__ == '__main__':
    n = 6
    
    for i in range(100):
        w = gensubmodw(n)
        f = [0] + [round(e, 3) for e in list(np.diff(w))]
        pob = getpob(w, f)
        pob_reduced = getpob(w, f, True)
        if pob != pob_reduced:
            print('FALSE: ', w, f, pob, pob_reduced)
        print(i)

    # printpart(n)
