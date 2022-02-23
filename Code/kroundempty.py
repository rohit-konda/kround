import numpy as np
from itertools import product
from LPsolver import lp
from prettytable import PrettyTable


def kroundLP(w, f, k, empty=True):
    n = len(w)-1  # number of players
    n_ac = k+2  # number of actions in the worst case game

    # list[tuple(0, 1, length=n_ac)] - shows which players select resource in which action
    # Example [(0, 1), (1, 0)] means this resource is selected in 2nd action 
    # by 1st player and 1st action by 2nd player
    partition = list(product(product([1, 0], repeat=n_ac), repeat=n))[:-1]
    n_c = len(partition)  # number of resource types

    C = np.zeros((n_c,), dtype='float')  # objective function is cT x
    br_cons = np.zeros((n*(n_ac-1)*(n_ac-2), n_c), dtype='float')  # encodes correct better response constraints
    A = np.zeros((1, n_c), dtype='float')  # equality constraints


    # action structure : (a_0, a_1, ..., a_k, a_opt) for each player
    # players start at a_0, after 1 round, all players should select a_1 ...
    # after k rounds, players select the a_k. optimal allocation is a_opt
    a_k = -2
    a_opt = -1
    
    for i, p in enumerate(partition):
        # what players are selecting resource in optimal allocation
        opt_i = [pl for pl in range(n) if p[pl][a_opt]==1]
        # what players are selecting resource in a_k allocation
        kbest_i = [pl for pl in range(n) if p[pl][a_k]==1]
        C[i] = -w[len(opt_i)]  # maximize welfare of optimal allocation
        A[0, i] = w[len(kbest_i)]  # set welfare of k round best response to 1
        

        # Better response constraint is 
        # U_i(a^(b+1)_i, a^(b+1)_(<i), a^b_(>i)) - U_i(a^(other)_i, a^(b+1)_(<i), a^b_(>i)) >= 0 for all a_other
        # In other words, if players <i select action a_(k+1) and players >i select action a_k,
        # then best response for player i is to select a_(k+1) for all 0 - (k-1) actions
        # Here we take best response sequence as player 1 playing , 2 playing , ... n playing (k times)
        c = 0  # counter
        for j in range(n):
            for b in range(k):
                bnext = b + 1  # best response (b+1)
                # what players below i select the resource in a_b
                nextac = [pl for pl in range(j+1, n) if p[pl][b]==1]
                # what players above i select the resource in a_(b+1)
                prevac = [pl for pl in range(j) if p[pl][bnext]==1]  # 
                other_ac = nextac + prevac

                for a_other in range(n_ac):  # encode that b+1 is better response to all other actions a_other
                    if a_other == bnext:  # remove redundant constraints
                        continue

                    if p[j][bnext] == 1 and p[j][a_other] == 0:  # +f(|a_r|) if resource is selected in best response
                        br_cons[c][i] = f[len(other_ac + [j])]
                    elif p[j][bnext] == 0 and p[j][a_other] == 1:   # -f(|a_r|) if resource is selected in other response
                        br_cons[c][i] = -f[len(other_ac + [j])]
                    c += 1

    # to encode a_0 as the empty set,
    # make every resource selected in a_0 by any player to be 0.
    if empty:
        for i, p in enumerate(partition):
            # checks if resource is selected in a_0 by any player
            if any([bool(alloc[0]) for alloc in p]):
                empty_cons = np.zeros((1, n_c))
                empty_cons[0][i] = 1
                A = np.vstack((A, empty_cons))

        # equality constant
        B = np.vstack((np.ones((1, 1)), np.zeros((len(A)-1, 1))))
    else:
        B = np.ones((1, 1))

    cons_2 = np.identity(n_c)  # make sure all resource values are nonnegaive
    G = -np.vstack((br_cons, cons_2))
    H = np.zeros((len(G), 1))
    return C, G, H, A, B


# run LP
def get_sol(w, f, k, empty=True):
    args = kroundLP(w, f, k, empty)  # get constraints from LP
    sol = lp('cvxopt', *args)  # execute LP

    n = len(w)-1  # num players
    n_ac = k+2  # number of actions
    partition = list(product(product([1, 0], repeat=n_ac), repeat=n))[:-1]  # resource types
    return partition, sol

def print_sol(partition, sol):
    if sol is None:
        return

    print('POA: ', -round(sol['min']**-1, 3))  # print POA

    # for i, p in enumerate(partition):
    #     val = round(sol['argmin'][i], 4)  # resource values (round to 4 decimal places)
    #     if val != 0:
    #         print('TYPE:', p, ', VAL:', val)

# print table of utilities for 2 players
def Utable(partition, val, w, f, k):
    p1 = PrettyTable(['P1'] + [str(i) for i in range(k+2)])
    p2 = PrettyTable(['P2'] + [str(i) for i in range(k+2)])
    wt = PrettyTable(['W'] + [str(i) for i in range(k+2)])

    for i in range(k+2):
        row1 = [str(i)] + [0]*(k+2) 
        row2 = [str(i)] + [0]*(k+2)
        roww = [str(i)] + [0]*(k+2)
        for j in range(k+2):
            u1 = 0
            u2 = 0
            wv = 0
            for c, p in enumerate(partition):
                sel1 = p[0][i]
                sel2 = p[1][j]
                if sel1 == 1:
                    u1 += val[c] * f[sel1+sel2]
                if sel2 == 1:
                    u2 += val[c] * f[sel1+sel2]

                wv += val[c] * w[sel1+sel2]

            row1[j+1] = round(u1, 5)
            row2[j+1] = round(u2, 5)
            roww[j+1] = round(wv, 5)
        p1.add_row(row1)
        p2.add_row(row2)
        wt.add_row(roww)

    print(p1)
    print()
    print(p2)
    print()
    print(wt)


def Uforac(partition, val, w, f, k):
    n = len(w) - 1
    pt = PrettyTable(['actions'] + [str(i) for i in range(n+1)])
    actions = list(product(range(1, k+2), repeat=n))
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


# # PARAMETERS
# w = [0., 1., 1.5]  # welfare function
# #f = [0., 1., 0.]  # distribution function
# f = [0] + list(np.diff(w))
# k = 2 # number of best response rounds
# empty = True  # if starting at empty resource allocation

# partition, sol = get_sol(w, f, k, empty)  # GET SOLUTION

# # for w2 in np.arange(0, 3, .1):
# #     w[2] = w2
# #     print('w: ', w2)
# #     for f2 in np.arange(0, 3, .1):
# #         f[2] = f2
# #         partition, sol = get_sol(w, f, k, empty)
# #         if sol is not None:
# #             print('f: ', round(f2, 2), -round(sol['min']**-1, 3))
# #         #print_sol(partition, sol)


# print_sol(partition, sol)  # PRINT SOLUTION
# # Uforac(partition, sol['argmin'], w, f, k)
# Utable(partition, sol['argmin'], w, f, k)  # PRINT UTILITY AND WELFARE TABLES