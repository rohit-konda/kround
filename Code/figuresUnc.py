import matplotlib.pyplot as plt
import numpy as np


def modgairing(n, d):
    from math import factorial
    C = (1+d)/(1-d)
    B = 1./(n-1)/factorial(n-1)/C**n
    y = lambda j: sum([(C**(-i))/factorial(i) for i in range(j, n)])
    return [0] + [round((C**(j-1))*factorial(j-1)*(B + y(j))/(B + y(1)), 3) for j in range(1, n+1)]

def figF():

    plt.plot(f)
    plt.show()

def figPoaVDelt():
    step = .01
    delta = np.arange(0, 1+step, step)
    poa = 1 - 1/np.exp((1-delta)/(1+delta))
    plt.plot(delta, poa)
    plt.plot(delta, poa[0] - delta*poa[0], 'k--')
    plt.show()

def figwrongDelt():
    def modpoa(n, dtrue, df):
        B = (1+dtrue)/(1-dtrue)
        fgard = modgairing(n, df)
        if dtrue >= df:
            return (B - fgard[2] + 1)**-1
        else:
            return (1 + B*(n-1)*fgard[n])**-1
    n = 10
    dtrue = .3
    dfs = np.arange(0, .99, .01)
    poa = [modpoa(n, dtrue, df) for df in dfs]
    plt.plot(dfs, poa)
    plt.show()

#figPoaVDelt()
figwrongDelt()
