import numpy as np
from math import factorial
from LPsolver import lp

def Fgar(n, j):
    return factorial(j-1)*( 1/(n-1)/factorial(n-1) + np.sum( 1./factorial(range(j,n)) ) ) / ( 1/(n-1)/factorial(n-1) + np.sum( 1./factorial(range(1,n)) ) )



def primalLP(n, C, F, costMinGame=False):
    # Indices [Sn, X]
    I = np.arange( 2**(2*n) )
    I = (((I[:,None] & (1 << np.arange(2*n)))) > 0).astype(int)

    # Objective: Minimize total congestion at optimum
    c = C[ np.sum(I[:,n:], axis=1) ]

    if not costMinGame:
        c = -1.*c

    # Subject to:
    A_ub = np.zeros( (n+np.shape(I)[0],np.shape(I)[0]), dtype=float )
    b_ub = np.zeros( (n+np.shape(I)[0],), dtype=float )

    for i in np.arange(n):
        A_ub[i,:] = F[ np.sum( I[:,0:(i+1)], axis=1 )*I[:,i] ] - F[ (np.sum( I[:,0:i], axis=1 ) + I[:,n+i])*I[:,n+i] ]

    if not costMinGame:
        A_ub[:n,:] = -1.*A_ub[:n,:]

    A_ub[n:,:] = -1.*np.eye(np.shape(I)[0])

    # Final allocation has congestion normalized
    A_eq = np.tile( C[ np.sum( I[:,:n], axis=1 ) ], (1,1) )
    b_eq = np.ones( (1,), dtype=float )

    return lp('cvxopt', c, A_ub, b_ub, A_eq, b_eq)



def dualLP(n, C, F, costMinGame=False):
    # Indices [Sn, X]
    I = np.arange( 2**(2*n) )
    I = (((I[:,None] & (1 << np.arange(2*n)))) > 0).astype(int)

    # `n+1` decision variables: [lam_1, ..., lam_n, mu]

    # Objective is with respect to `mu`
    if costMinGame:
        c = -1.*np.eye(n+1,dtype=float)[-1]
    else:
        c = np.eye(n+1,dtype=float)[-1]

    A_ub = np.zeros((n+np.shape(I)[0], n+1), dtype=float)
    b_ub = np.zeros((n+np.shape(I)[0],), dtype=float)

    # lam_i \geq 0
    A_ub[:n,:] = -1.*np.eye(n+1, dtype=float)[:n]

    if costMinGame:
        for i in np.arange(n):
            A_ub[n:,i] = F[ (np.sum( I[:,0:i], axis=1 ) + I[:,n+i])*I[:,n+i] ] - F[ np.sum( I[:,0:(i+1)], axis=1 )*I[:,i] ]

        A_ub[n:,-1] = C[ np.sum( I[:,:n], axis=1 ) ]
        b_ub[n:]    = C[ np.sum(I[:,n:], axis=1) ]
    else:
        for i in np.arange(n):
            A_ub[n:,i] = F[ np.sum( I[:,0:(i+1)], axis=1 )*I[:,i] ] - F[ (np.sum( I[:,0:i], axis=1 ) + I[:,n+i])*I[:,n+i] ]

        A_ub[n:,-1] = -1.*C[ np.sum( I[:,:n], axis=1 ) ]
        b_ub[n:]    = -1.*C[ np.sum(I[:,n:], axis=1) ]


    return lp('cvxopt', c, A_ub, b_ub)



def optimalLP(n, C, costMinGame=False):
    # Indices [Sn, X]
    I = np.arange( 2**(2*n) )
    I = (((I[:,None] & (1 << np.arange(2*n)))) > 0).astype(int)

    # `n+1` decision variables: [ -f1-, ..., -fn-, mu ]

    # Objective is with respect to `mu`
    if costMinGame:
        c = -1.*np.eye(n*n+1,dtype=float)[-1]
    else:
        c = np.eye(n*n+1,dtype=float)[-1]

    A_ub = np.zeros((np.shape(I)[0], n*n+1), dtype=float)
    b_ub = np.zeros((np.shape(I)[0],), dtype=float)

    if costMinGame:
        for i in np.arange(n):
            A_ub[np.arange(np.shape(I)[0]),n*i+(np.sum( I[:,0:i], axis=1 ) + I[:,n+i])*I[:,n+i]] += 1. 
            A_ub[np.arange(np.shape(I)[0]),n*i+np.sum( I[:,0:(i+1)], axis=1 )*I[:,i]] -= 1. 
        
        A_ub[:,-1] = C[ np.sum(I[:,:n],axis=1) ]
        b_ub = C[ np.sum(I[:,n:],axis=1) ]
    else:
        for i in np.arange(n):
            A_ub[np.arange(np.shape(I)[0]),n*i+np.sum( I[:,0:(i+1)], axis=1 )*I[:,i]] += 1. 
            A_ub[np.arange(np.shape(I)[0]),n*i+(np.sum( I[:,0:i], axis=1 ) + I[:,n+i])*I[:,n+i]] -= 1. 

        A_ub[:,-1] = -1.*C[ np.sum( I[:,:n], axis=1 ) ]
        b_ub[:]    = -1.*C[ np.sum(I[:,n:], axis=1) ]

    return lp('cvxopt', c, A_ub, b_ub)


if __name__ == "__main__":
    # costMinGame = True
    costMinGame = False
    n = 3

    # Cost function and player cost function
    # x_ = np.arange( n+1, dtype=float )
    # C = x_**2.0
    # F = np.hstack((0., np.diff(C)))
    # F = x_

    C = np.array([0., 1., 1.5, 1.5])
    F = np.zeros((n+1,))
    F[1:] = np.diff(C)
    # F[1:] = 1./np.arange(1,n+1)
    # F[1:] = [Fgar(n,j) for j in range(1,n+1) ]

    #reso = optimalLP(n, C, costMinGame=costMinGame)
    resp = primalLP(n, C, F, costMinGame=costMinGame)
    resd = dualLP(n, C, F, costMinGame=costMinGame)

    print(resd)
