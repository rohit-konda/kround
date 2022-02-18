import numpy as np
from LPsolver import lp
import matplotlib.pyplot as plt

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

def PB_lower(N):
    part = [()]*(N*(N+1))
    c = 0
    for a in range(1, N+1):
        for b in range(N+1):
            x = max(a+b-N, 0)
            part[c] = (a-x, x, b-x)
            c+=1
    return part


def PB_upper(N):
    part = [()]*int((N+1)*(N+2)/2)
    c = 0
    for a in range(N+1):
        for x in range(a, N+1):
            part[c] = (a, x-a, 0)
            c+=1

    #print(part[1:])
    return part[1:]

# generate constratints for POA
def poaconst(N, X):
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
    return const, const_h

# generate constraints for POB
def pobconst(N, upper):
    if upper:
        partition = PB_upper(N)
    else:
        partition = PB_lower(N)
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

# For a fixed POA = X and w submodular welfare function, find the upper bound POB if upper=True, and lower bound POB if upper=False
def pobpoatradupper(w, X, upper):
    N = len(w)-1
    C = np.zeros((N+1,), dtype='float')
    nonnegcons = np.identity(N+1,  dtype='float')
    nonnegcons_h = np.zeros((N+1, 1), dtype='float')
    C[0] = 1

    finccons = np.zeros((N-1, N+1), dtype='float')
    for k in range(2, N+1):
        finccons[k-2, k] = -1
        finccons[k-2, k-1] = 1
    finccons_h = np.zeros((N-1, 1), dtype='float')

    pobcons, pobcons_h = pobconst(N, upper)
    poacons, poacons_h = poaconst(N, X)
    G = -np.vstack((pobcons, poacons, nonnegcons, finccons))
    H = -np.vstack((pobcons_h, poacons_h, nonnegcons_h, finccons_h))
    
    return C, G, H


# for a welfare w, solve the LP for PoA = [0, 1] with interval step. Get the upper curve if upper = True, lower bound curve if upper=False
def plotpobpoagenw(w, step, upper):
    poarange = []
    pobrange = []
    for poa in np.arange(step, 1+step, step):
        X = poa**-1
        args = pobpoatradupper(w, X, upper)
        sol = lp('cvxopt', *args)
        if sol is not None:
            poarange.append(poa)
            pobrange.append(round(sol['min']**-1, 4))
    return poarange, pobrange

if __name__ == '__main__':
    w = [0, 1, 1]  # welfare

    poarange, pobrange = plotpobpoagenw(w, step, False)  # plot POB lower bound
    plt.plot(poarange, pobrange, 'k')
    poarange, pobrange = plotpobpoagenw(w, step, True)  # plot POB upper bound
    plt.plot(poarange, pobrange, 'r')

    plt.show()