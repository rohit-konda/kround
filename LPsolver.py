#!/usr/bin/env python
# Author : Rohit Konda
# Copyright (c) 2020 Rohit Konda. All rights reserved.
# Licensed under the MIT License. See LICENSE file in the project root for full license information.

"""
Module for wrapping optimization solvers for use in the game package.
"""
from warnings import warn
from typing import Dict, Any, Optional
import numpy as np

def lp(solver: str, c: np.ndarray, G: np.ndarray, h: np.ndarray, A: np.ndarray=None, b: np.ndarray=None, progress: bool=False, returnall: bool=False) -> Dict[str, Any]:
    """ lp solver of the standard form: min c^T x subject to G x <= h , A x = b.
    
    Args:
        solver (str): which solver to use for solving the linear program.
        c (np.ndarray): Objective function.
        G (np.ndarray): Linear inequality constraint.
        h (np.ndarray): Linear inequality constraint.
        A (np.ndarray, optional): Linear equality constraint.
        b (np.ndarray, optional): Linear equality constraint.
        progress (bool, optional): Toggle to show progress of optimization solver.
    
    Returns:
        Dict[str, Any]: {'min': minimum of objective, 'argmin': value in which the minimum is obtained.}
    """
    wrapper = SolverWrapper(solver, progress)
    wrapper.returnall = returnall
    return wrapper.lp(c, G, h, A, b)


class SolverWrapper:

    """ Wrapper for solving Convex Programs.

    Args:
        solver (str): Which solver to use.
        progress (bool): Whether to show progress of solver.
    
    Attributes:
        SUPPORTED (List(str)): Which solvers are supported overall.
        LP_SUPPORTED (List(str)): Which solvers are supported for solving an LP.
        progress (bool): Whether to show progress of solver.
        returnall (bool): Whether to wrap solver output in a dict or just return everything.
        solver (str): Which solver to use.
    """
    
    SUPPORTED = ['cvxopt']
    LP_SUPPORTED = ['cvxopt']

    def __init__(self, solver: str, progress: bool):
        self.solver = solver
        self.check_solver(solver)
        self.progress = progress
        self.returnall = False
    
    def lp(self, c: np.ndarray, G: np.ndarray, h: np.ndarray, A: np.ndarray, b: np.ndarray) -> Optional[Dict[str, Any]]:
        """solver for LP. Form is min c^T x subject to G x <= h , A x = b.
        
        Args:
            c (np.ndarray): Objective function.
            G (np.ndarray): Linear inequality constraint.
            h (np.ndarray): Linear inequality constraint.
            A (np.ndarray): Linear equality constraint.
            b (np.ndarray): Linear equality constraint.
        
        Returns:
            Optional[Dict[str, Any]]: If not returnall, return {'min': minimum of objective, 'argmin': value in which the minimum is obtained.}.
        
        Raises:
            ImportError: If specified solver is not an implemented LP solver.
        """
        if self.solver == 'cvxopt':
            from cvxopt import matrix
            from cvxopt.solvers import lp
            
            c = matrix(c)
            G = matrix(G)
            h = matrix(h)
            A = matrix(A) if A is not None else None
            b = matrix(b) if b is not None else None
            sol = lp(c, G, h, A, b, options={'show_progress': self.progress})
            if sol['status'] != 'optimal':
                warn('no feasible solution found')
            else:
                if self.returnall:
                    return sol
                else:
                    return {'min': sol['primal objective'], 'argmin': list(sol['x'])}
        else:
            raise ImportError('Not a valid or implemented lp solver. Supported lp solvers include ' + ', '.join(self.LP_SUPPORTED) + '.')

    def check_solver(self, solver: str) -> None:
        """check if solver is supported and perform some necessary setup.
        
        Args:
            solver (str): Which solver to use.
        
        Raises:
            ImportError: If solver given is not an implemented solver.
        """
        if self.solver == 'cvxopt':
            pass
        else:
            raise ImportError('Not a valid or implemented solver. Supported solvers include ' + ', '.join(self.SUPPORTED) + '.')