"""
Copyright (c) 2023, Yakov Vaisbourd
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.


Miscellaneous functions supplementing the memmpy package.

"""

import numpy as np

MAX_ITER = 100
TOLERANCE = 1e-8


def find_root(func_val_der, lower_bound: np.ndarray, upper_bound: np.ndarray, initial_point: np.ndarray,
              max_iter: int = MAX_ITER, tol: float = TOLERANCE) -> np.ndarray:
    """
    Find root of a function using safeguarded Newton-Raphson method under monotonicity assumption.

    Parameters
    ----------
    func_val_der : callable
        Function that computes both the value and the derivative of the input function. The function should have
        the signature `f_val, f_der = func_val_der(x, index_not_converged)`, where `x` is a numpy array containing the
        values at which the function and derivative should be evaluated and `index_not_converged` is a boolean array
        that indicates which values of `x` have not yet converged to the root.
    lower_bound : np.ndarray
        The lower bound(s) of the search interval for the root.
    upper_bound : np.ndarray
        The upper bound(s) of the search interval for the root.
    initial_point : np.ndarray
        The initial point(s) for the root search.
    max_iter : int, optional
        Maximum number of iterations allowed for the root search. Default is set to the constant MAX_ITER.
    tol : float, optional
        Tolerance for the root search. Default is set to the constant TOLERANCE.

    Returns
    -------
    np.ndarray
        An array of roots, one for each initial point provided.

    Notes
    -----
    The function computes the root of a function using the safeguarded Newton-Raphson method under the assumption that
    the function is monotonically increasing. If the derivative of the function cannot be computed or the Newton step
    does not lead to any progress, then the function falls back to the bisection method. The search for the root is
    limited by the upper and lower bounds provided. This is an internal function that does not include input validation.

"""

    # Initialize variables
    x = np.copy(initial_point)
    lb = np.copy(lower_bound)
    ub = np.copy(upper_bound)
    x = np.clip(x, lb + TOLERANCE / 10, ub - TOLERANCE / 10)
    index_not_converged = ub - lb > tol

    step = np.empty_like(x)
    f_val = np.empty_like(x)
    f_der = np.empty_like(x)
    r = np.empty_like(ub, dtype=float)
    x_mid = np.empty_like(ub, dtype=float)
    convergence_lb = np.empty_like(ub, dtype=float)
    convergence_ub = np.empty_like(ub, dtype=float)
    use_r = np.zeros_like(ub, dtype=bool)
    diff = np.empty_like(ub, dtype=float)
    f_val_prev = np.empty_like(ub, dtype=float)

    # Compute function values and derivatives
    f_val[index_not_converged], f_der[index_not_converged] = func_val_der(x, index_not_converged)

    # Compute indicators
    index_derivative_exists = f_der > TOLERANCE # False also for nan values
    f_val_geq_tol = f_val > tol
    f_val_leq_tol = f_val < -tol

    # Update upper and lower bounds
    ub[f_val_geq_tol] = x[f_val_geq_tol]
    lb[f_val_leq_tol] = x[f_val_leq_tol]

    index_not_converged = np.logical_and(np.logical_or(f_val_geq_tol, f_val_leq_tol), ub - lb > tol)

    iter_counter = 0
    stopping_criteria_flag = index_not_converged.any()

    if not stopping_criteria_flag:
        x = (ub + lb) / 2

    while stopping_criteria_flag:
        iter_counter += 1

        np.copyto(diff, ub - lb, where=index_not_converged)
        np.copyto(f_val_prev, f_val, where=index_not_converged)
        np.copyto(r, (ub - lb) / 2, where=np.logical_not(use_r))
        np.copyto(r, 1 / (1 + iter_counter//10) * (ub - lb) / 2, where=use_r)
        np.copyto(x_mid, (ub + lb) / 2, where=index_not_converged)

        np.copyto(step, np.divide(f_val, f_der, where=np.logical_and(index_derivative_exists, index_not_converged)),
                  where=np.logical_and(index_derivative_exists, index_not_converged))

        index_bisection_step = np.zeros_like(ub, dtype=bool)
        np.copyto(index_bisection_step, np.logical_or(np.logical_not(index_derivative_exists), np.abs(step) <= tol),
                  where=index_not_converged)
        np.copyto(x, x - step, where=np.logical_and(index_not_converged, np.logical_not(index_bisection_step)))
        np.copyto(x, x_mid, where=np.logical_and(index_not_converged, index_bisection_step))

        np.copyto(convergence_lb, np.maximum(lb, x_mid - r), where=index_not_converged)
        np.copyto(convergence_ub, np.minimum(ub, x_mid + r), where=index_not_converged)

        np.copyto(x, np.clip(x, convergence_lb + TOLERANCE / 10, convergence_ub - TOLERANCE / 10),
                  where=index_not_converged) # For numerical stability

        f_val[index_not_converged], f_der[index_not_converged] = func_val_der(x, index_not_converged)

        index_derivative_exists = f_der > TOLERANCE # False also for nan values
        f_val_geq_tol = f_val > tol
        f_val_leq_tol = f_val < -tol

        np.copyto(ub, x, where=np.logical_and(f_val_geq_tol, index_not_converged))
        np.copyto(lb, x, where=np.logical_and(f_val_leq_tol, index_not_converged))

        use_r = np.zeros_like(ub, dtype=bool)
        np.copyto(use_r, np.logical_and(diff / 2 < ub - lb, np.abs(f_val_prev) / 2 < np.abs(f_val)),
                  where=index_not_converged)

        index_not_converged = np.logical_and(np.logical_or(f_val_geq_tol, f_val_leq_tol), ub - lb > tol)

        if iter_counter >= max_iter or not index_not_converged.any():
            stopping_criteria_flag = False

    return x


def power_method(lin_map, lin_map_adj, dom_shape, max_iter=100):
    """
    Computes the spectral norm of a linear operator using the power method.

    Parameters
    ----------
    lin_map : callable
        A function representing the linear operator. Should accept a
        numpy array of shape `dom_shape` and return a numpy array of shape
        `(codom_shape, )`, where `codom_shape` is the shape of the operator's
        codomain.
    lin_map_adj : callable
        A function representing the adjoint of the linear operator. Should
        accept a numpy array of shape `(codom_shape, )` and return a numpy
        array of shape `dom_shape`.
    dom_shape : tuple of int
        The shape of the operator's domain.
    max_iter : int, optional
        The maximum number of iterations to perform. Defaults to 100.

    Returns
    -------
    float
        The spectral norm of the linear operator.

    Raises
    ------
    ValueError
        If `max_iter` is not a positive integer.

    Notes
    -----
    The power method is an iterative algorithm that finds the largest
    eigenvalue (in absolute value) of a matrix or the spectral norm of a
    linear operator. This implementation works with general linear operators
    represented by their action on vectors.

    """
    if not isinstance(max_iter, int) or max_iter <= 0:
        raise ValueError("max_iter must be a positive integer")

    x = np.random.rand(*dom_shape)
    for k in range(max_iter):
        x = x / np.linalg.norm(x, ord='fro')
        x = lin_map_adj(lin_map(x))

    x = x / np.linalg.norm(x, ord='fro')
    x = lin_map(x)

    return np.linalg.norm(x, ord='fro')


def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', print_end='\r'):
    """
    Prints a progress bar in the console to track the progress of a loop.
    From: https://stackoverflow.com/questions/3173320/text-progress-bar-in-terminal-with-block-characters

    Parameters
    ----------
    iteration : int
        Current iteration number.
    total : int
        Total number of iterations.
    prefix : str, optional
        Prefix string to print before the progress bar (default '').
    suffix : str, optional
        Suffix string to print after the progress bar (default '').
    decimals : int, optional
        Number of decimal places to use when displaying the progress percentage (default 1).
    length : int, optional
        Length of the progress bar in characters (default 100).
    fill : str, optional
        Character to use for the filled portion of the progress bar (default '█').
    print_end : str, optional
        String to use to end the progress bar line (default '\r').

    Returns
    -------
    None
        This function does not return anything; it only prints to the console.

    Examples
    --------
    >>> for i in range(10):
    ...     print_progress_bar(i+1, 10, prefix='Progress:', suffix='Complete', length=50)
    Progress: |██████████████████████████████████████| 100.0% Complete

    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=print_end)

    if iteration == total:
        print()
