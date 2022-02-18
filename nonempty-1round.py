import numpy as np
from scipy.special import factorial
from scipy.optimize import linprog



def Fgar(n, j):
    return factorial(j-1)*( 1/(n-1)/factorial(n-1) + np.sum( 1./factorial(range(j,n)) ) ) / ( 1/(n-1)/factorial(n-1) + np.sum( 1./factorial(range(1,n)) ) )



def primalLP(n, C, F, costMinGame=False):
    # Indices [S0, Sn, X]
    I = np.arange( 2**(3*n) )
    I = (((I[:,None] & (1 << np.arange(3*n)))) > 0).astype(int)

    # Objective: Minimize total congestion at optimum
    c = C[ np.sum(I[:,2*n:], axis=1) ]

    if not costMinGame:
        c = -1.*c

    # Subject to:
    A_ub = np.zeros( (2*n+np.shape(I)[0],np.shape(I)[0]), dtype=float )
    b_ub = np.zeros( (2*n+np.shape(I)[0],), dtype=float )

    for i in np.arange(n):
        A_ub[i,:] = F[ np.sum( np.hstack((I[:,n:(n+i+1)],I[:,(i+1):n])), axis=1 )*I[:,n+i] ] \
                    - F[ ( np.sum(np.hstack((I[:,n:(n+i)],I[:,(i+1):n])), axis=1)+I[:,2*n+i] )*I[:,2*n+i] ]
        A_ub[n+i,:] = F[ np.sum( np.hstack((I[:,n:(n+i+1)],I[:,(i+1):n])), axis=1 )*I[:,n+i] ] \
                    - F[ ( np.sum(np.hstack((I[:,n:(n+i)],I[:,i:n])), axis=1) )*I[:,i] ]

    if not costMinGame:
        A_ub[:2*n,:] = -1.*A_ub[:2*n,:]

    A_ub[2*n:,:] = -1.*np.eye(np.shape(I)[0])

    # Final allocation has congestion normalized
    A_eq = np.tile( C[ np.sum( I[:,n:2*n], axis=1 ) ], (1,1) )
    b_eq = np.ones( (1,), dtype=float )

    return linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, method='revised simplex')



if __name__ == "__main__":
    # costMinGame = True
    costMinGame = False
    n = 4

    # Cost function and player cost function
    x_ = np.arange( n+1, dtype=float )
    # C = x_**2.0
    # F = np.hstack((0., np.diff(C)))
    # F = x_

    C = np.zeros((n+1,))
    C[1:] = 1.
    F = np.zeros((n+1,))
    # F[1:] = np.diff(C)
    # F[1:] = 1./np.arange(1,n+1)
    F[1:] = [Fgar(n,j) for j in range(1,n+1) ]

    res = primalLP(n, C, F, costMinGame=costMinGame)


