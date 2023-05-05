"""
Copyright (c) 2023, Yakov Vaisbourd
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.


Optimization algorithms supplementing the memmpy package. The module includes the implementation of the Bregman proximal
gradient algorithm (including accelerated/fast variants) and the Chambolle-Pock algorithm.

"""

import numpy as np
from memmpy.misc import print_progress_bar
from memmpy.misc import power_method
from typing import NamedTuple


class MathFunction:
    """
    A class representing a mathematical function.

    Attributes:
    - val: A function that takes in an input and returns a value.
    - grad: A function that takes in an input and returns the gradient of the function at that input.
    """
    def __init__(self, val, grad=None):
        self.val = val
        self.grad = grad


class KernelFunction(MathFunction):
    """
    A class representing a kernel function that induces the Bregman proximal gradient method.

    Attributes:
    - val: A function that takes in an input and returns a value.
    - grad: A function that takes in an input and returns the gradient of the function at that input.
    - grad_dual: A function that takes in two inputs and returns the gradient of the dual (convex conjugate) of the
                 function.
    - is_normal: A boolean indicating whether the kernel is normal or not.
    """
    def __init__(self, val, grad, grad_dual, is_normal=False):
        super().__init__(val, grad)
        self.grad_dual = grad_dual
        self.is_normal = is_normal

    def val_bregman(self, y, x):
        """
        Calculates the Bregman distance between two points y and x.

        Args:
        - y: The first argument of the Bregman distance.
        - x: The second argument of the Bregman distance (used to construct the linear approximation).

        Returns:
        - The Bregman distance between y and x.
        """
        return self.val(y) - self.val(x) - np.dot(self.grad(x), y - x)


class ObjectiveFunction(MathFunction):
    """
    A class representing an objective function.

    Attributes:
    - val: A function that takes in an input and returns a value.
    - grad: A function that takes in an input and returns the gradient of the function at that input.
    - residual: A function that takes in an input and returns the residual of the function at that input.
    - kernel: A kernel function that is used in the objective function.
    - smoothness_constant: A float representing the smoothness constant of the function.
    - proximal_operator: A function that takes in an input and returns the proximal operator of the function at
                         that input.
    """
    def __init__(self, val, grad=None, residual=None, kernel=None, smoothness_constant=None, proximal_operator=None):
        super().__init__(val, grad)
        self.residual = residual
        self.kernel = kernel
        self.smoothness_constant = smoothness_constant
        self.proximal_operator = proximal_operator


class LinearMap:
    """
    A class representing a linear map.

    Attributes:
    - lin_map: A function that takes in an input and returns the linear mapping of that input.
    - lin_map_adj: A function that takes in an input and returns the adjoint of the linear mapping at that input.
    - lin_map_norm: A float representing the norm of the linear mapping.
    """
    def __init__(self, lin_map, lin_map_adj, lin_map_norm=None):
        self.lin_map = lin_map
        self.lin_map_adj = lin_map_adj
        self.lin_map_norm = lin_map_norm


class ProblemData(NamedTuple):
    """
    A named tuple representing the problem data of the optimization problem of the form:
                                minimize f(x) + g(Lx)
    If L is None then it is assumed to be equal to the identity matrix.

    Attributes:
    - f: An objective function.
    - g: An objective function.
    - L: A linear map.
    - x_true: A numpy array representing the true value of x.
    - observed_signal: A numpy array representing the observed signal.
    """
    f: ObjectiveFunction
    g: ObjectiveFunction = None
    L: LinearMap = None
    x_true: np.ndarray = None
    observed_signal: np.ndarray = None


class Parameters(NamedTuple):
    """
    Parameters used for optimization algorithms.

    Attributes:
        initial_point (np.ndarray): The initial point for optimization.
        initial_point_dual (np.ndarray, optional): The initial dual point for optimization.
        step_size (float, optional): The step size to be used in the algorithm.
        step_size_dual (float, optional): The step size for dual variables.
        is_fast (bool, optional): A flag indicating whether to use a fast implementation of the algorithm.
        cp_par (float, optional): A parameter used in the algorithm (only for Chambolle-Pock algorithm).
        max_iter (int, optional): The maximum number of iterations for the algorithm.
        obj_decrease_tol (float, optional): The tolerance for objective decrease in the algorithm.
        grad_map_norm_tol (float, optional): The tolerance for the gradient map norm in the algorithm.
        verbose (bool, optional): A flag indicating whether to print progress information during optimization.
    """
    initial_point: np.ndarray
    initial_point_dual: np.ndarray = None
    step_size: float = None
    step_size_dual: float = None
    is_fast: bool = False
    cp_par: float = None
    max_iter: int = None
    obj_decrease_tol: float = None
    grad_map_norm_tol: float = None
    verbose: bool = False


class Results(NamedTuple):
    """
    Results of optimization algorithm.

    Attributes:
        opt_sol (np.ndarray): The optimized solution.
        obj_val (np.ndarray): The objective function value at the optimized solution. If max_iter was included as a
            parameter to the algorithm then the whole sequence of objective values will be provided.
    """
    opt_sol: np.ndarray
    obj_val: np.ndarray


def bpg(data: ProblemData, pars: Parameters) -> Results:
    """
    Solves the optimization problem:

        min f(x) + g(x)

    using the Bregman proximal gradient algorithm. The function f must have a smooth adaptable kernel function and
    constant L, and the function g must have a proximal operator. If the proximal operator of g is not provided,
    the algorithm will reduce to the Bregman algorithm. In case that the kernel function is the squared Euclidean norm
    (i.e., f is gradient Lipschitz continuous) then the algorithm reduces to the proximal gradient algorithm. For
    further details please refer to the manuscript:

        A Descent Lemma Beyond Lipschitz Gradient Continuity: First-Order Methods Revisited and Applications
        by H. H. Bauschke, J. Bolte, M. Teboulle.

    Fast implementations included in this method are the fast iterative shrinkage thresholding algorithm (FISTA) when
    the kernel is the squared Euclidean norm (i.e., f is gradient Lipschitz continuous) presented in the manuscript:

        A Fast Iterative Shrinkage Thresholding Algorithm for Linear Inverse Problems by A. Beck, M. Teboulle,

    and the improved interior gradient algorithm (IGA) presented in the manuscript:
        Interior gradient and proximal methods for convex and conic optimization by A.Auslender,M. Teboulle.


    Parameters
    ----------
    data : ProblemData
        An object containing the objective function f and, optionally, the objective function g.
    pars : Parameters
        An object containing the algorithm parameters, including the initial point, the step size (if the step size is
        not provided, the algorithm will compute it as 1/L, where L is the smooth adaptability constant of f), the
        maximum number of iterations, a flag to use the fast variant of the algorithm, a tolerance for the norm of the
        gradient mapping, and a flag to enable verbose output.

    Returns
    -------
    Results
        An object containing the solution of the optimization problem and the objective function values at each
        iteration.

    Raises
    ------
    ValueError
        If the step size or the smoothness constant of f is not specified.
    RuntimeError
        If the proximal operator of g is not specified.

    """

    # Initialization and input validation
    iter_index = 0
    x = np.copy(pars.initial_point)

    if pars.step_size is None and data.f.smoothness_constant is not None:
        step_size = 1 / data.f.smoothness_constant
    elif pars.step_size is None and data.f.smoothness_constant is None:
        ValueError(
            'The step size or the smoothness constant of the function f must be specified.')
    else:
        step_size = pars.step_size

    if data.g is not None:
        if data.g.proximal_operator is None:
            raise RuntimeError('The proximal operator of the function g must be specified.')
        x = data.g.proximal_operator(x, step_size)

    if pars.max_iter is not None:
        obj_val = np.zeros(pars.max_iter)
    else:
        obj_val = np.zeros(1)

    if data.f.residual is None:
        obj_val[0] = data.f.val(x)
    else:
        r = data.f.residual(x)
        obj_val[0] = data.f.val(r, True)

    if data.g is not None:
        obj_val[0] += data.g.val(x)

    fast_str = ""
    t_step = 1
    y = x
    z = x
    if pars.is_fast:
        t = 1
        fast_str = "Fast"
        y = np.copy(x)
        if not data.f.kernel.is_normal:
            z = np.copy(x)

    p = z

    if pars.verbose:
        print(fast_str, "Bregman Proximal Gradient output:")
        print('+', '-' * 10, '+', '-' * 20, '+')
        print('| {:<10} | {:<20} |'.format('iter_index', 'obj_val'))
        print('+', '-' * 10, '+', '-' * 20, '+')
        print('| {:<10} | {:<20} |'.format(iter_index, obj_val[iter_index]))

    # Main Loop
    stopping_criteria_flag = True
    while stopping_criteria_flag:
        iter_index += 1

        # The behavior of the algorithm varies depending on the selected method. There are three methods available:
        # BPG, FISTA, and IGA. When using the BPG method, the variables x, y, z, and p are set to the same value,
        # resulting in a one-sequence method. With the FISTA method, x and p take one value, while y and z take another,
        # making it a two-sequence method. Finally, with the IGA method, the variable z takes the value of p, creating
        # a three-sequence method.

        if pars.is_fast or pars.grad_map_norm_tol:
            t_prev = t
            x_prev = np.copy(x)

        if data.f.residual is None:
            np.copyto(p, data.f.kernel.grad_dual(
                data.f.kernel.grad(z) - t_step * step_size * data.f.grad(y)))
        else:
            np.copyto(p, data.f.kernel.grad_dual(
                data.f.kernel.grad(z) - t_step * step_size * data.f.grad(r, True)))

        if data.g is not None:
            np.copyto(p, data.g.proximal_operator(p, t_step * step_size))

        if pars.is_fast:
            t = (1 + np.sqrt(1 + 4 * t ** 2)) / 2
            if data.f.kernel.is_normal:
                y = x + (x - x_prev) * (t_prev - 1) / t
                z = y
            else:
                t_step = t
                x = (1 - (1 / t_prev)) * x_prev + (1 / t_prev) * z
                y = (1 - (1 / t)) * x + (1 / t) * z

        if data.f.residual is not None:
            r = data.f.residual(y)

        # Validate stopping criteria
        if pars.max_iter is not None:
            iter_output = iter_index
            if iter_index >= pars.max_iter - 1:
                stopping_criteria_flag = False
        else:
            iter_output = 0

        # Compute objective value
        obj_val_prev = obj_val
        if data.f.residual is None or pars.is_fast:
            obj_val[iter_output] = data.f.val(x)
        else:
            obj_val[iter_output] = data.f.val(r, True)

        if data.g is not None:
            obj_val[iter_output] += data.g.val(x)

        if pars.grad_map_norm_tol is not None:
            if np.linalg.norm((x - x_prev) / step_size) < pars.grad_map_norm_tol:
                stopping_criteria_flag = False

        # Fast methods are not necessarily monotonic, thus the objective value stopping criteria is not used for the
        # fast variants
        if pars.obj_decrease_tol is not None and not pars.is_fast:
            if obj_val_prev - obj_val[iter_output] < pars.obj_decrease_tol:
                stopping_criteria_flag = False

        # Produce output
        if pars.verbose:
            print('| {:<10} | {:<20} |'.format(iter_index, obj_val[iter_output]))
        else:
            print_progress_bar(iter_index + 1, pars.max_iter)

    if pars.verbose:
        print('+', '-' * 10, '+', '-' * 20, '+')

    return Results(x, obj_val)


def cp(data: ProblemData, pars: Parameters) -> Results:
    """
    Solves the optimization problem:

        min f(x) + g(Lx)

    using the Chambolle-Pock algorithm. The functions f and g are convex functions, and L is a linear operator. For
    further details please refer to the manuscript:
        A first-order primal-dual algorithm for convex problems with applications to imaging by A.Chambolle,T.Pock

    Parameters
    ----------
    data : ProblemData
        An object containing the functions f and g, and the linear operator L.
    pars : Parameters
        An object containing the algorithm parameters, including the primal and dual initial points, the step size and
        dual step size (if either of the step size is not provided, the algorithm will compute it as to satisfy
                                step_size * step_size_dual * L ** 2 < 1
        where L is the norm of the linear mapping), the maximum number of iterations, the (Chambolle-Pock) variable
        relaxation parameter (see parameter theta in the manuscript mentioned above), and a flag to enable verbose
        output.

    Returns
    -------
    Results
        An object containing the solution of the optimization problem and the objective function values at each
        iteration.

    Raises
    ------
    ValueError
        If the (Chambolle-Pock) variable relaxation parameter is negative, if the primal and dual step sizes and the
        linear mapping norm does not satisfy:
                            step_size*step_size_dual*(L.lin_map_norm ** 2) < 1.
    RuntimeError
        If either the dual initial point, the proximal operator of f, the function g or its proximal operator, or the
        linear mapping, is not specified.

    """

    # Initialization and input validation
    iter_index = 0
    x = np.copy(pars.initial_point)
    x_bar = np.copy(pars.initial_point)
    x_prev = np.copy(pars.initial_point)

    if pars.cp_par is None:
        cp_par = 1
    elif pars.cp_par < 0:
        raise ValueError('The Chambolle-Pock parameter (cp_par) must be non-negative.')
    else:
        cp_par = pars.cp_par

    if pars.initial_point_dual is None:
        raise RuntimeError('The dual initial point must be specified.')

    y = np.copy(pars.initial_point_dual)

    lin_map_norm = data.L.lin_map_norm
    if lin_map_norm is None and (pars.step_size is None or pars.step_size_dual is None):
        lin_map_norm = power_method(data.L.lin_map, data.L.lin_map_adj, x.shape)

    if pars.step_size is None and pars.step_size_dual is None:
        step_size = 0.99 / lin_map_norm
        step_size_dual = step_size
    elif pars.step_size is None:
        if pars.step_size_dual * (lin_map_norm ** 2) < 1:
            step_size_dual = pars.step_size_dual
            step_size = 0.99 / (step_size_dual * (lin_map_norm ** 2))
        else:
            raise ValueError('It must hold that (step_size*step_size_dual*(L.lin_map_norm ** 2) < 1).')
    elif pars.step_size_dual is None:
        if pars.step_size * (lin_map_norm ** 2) < 1:
            step_size = pars.step_size
            step_size_dual = 0.99 / (step_size * (lin_map_norm ** 2))
        else:
            raise ValueError('It must hold that (step_size*step_size_dual*(L.lin_map_norm ** 2) < 1).')
    else:
        step_size = pars.step_size
        step_size_dual = pars.step_size_dual

    if data.f.proximal_operator is None:
        raise RuntimeError('The proximal operator of the function f must be specified.')

    if data.g is None or data.L is None:
        raise RuntimeError('The function g and the linear mapping L must be specified.')

    if data.g.proximal_operator is None:
        raise RuntimeError('The proximal operator of the function g must be specified.')


    # Compute the proximal operator of the conjugate of g using the extended Moreau decomposition
    def g_dual_proximal_operator(_y, _step_size_dual):
        return _y - _step_size_dual * data.g.proximal_operator(_y / _step_size_dual, 1 / _step_size_dual)

    if pars.max_iter is not None:
        obj_val = np.zeros(pars.max_iter)
    else:
        obj_val = np.zeros(1)

    obj_val[iter_index] = data.f.val(x) + data.g.val(data.L.lin_map(x))
    # The objective value is expected to be inf unless the functions f and g are finite.

    if pars.verbose:
        print("Chambolle-Pock method output:")
        print('+', '-' * 10, '+', '-' * 20, '+')
        print('| {:<10} | {:<20} |'.format('iter_index', 'obj_val'))
        print('+', '-' * 10, '+', '-' * 20, '+')
        print('| {:<10} | {:<20} |'.format(iter_index, obj_val[iter_index]))

    # Main Loop
    stopping_criteria_flag = True
    while stopping_criteria_flag:
        iter_index += 1
        x_prev = np.copy(x)

        y = g_dual_proximal_operator(y + step_size_dual * data.L.lin_map(x_bar), step_size_dual)
        x = data.f.proximal_operator(x - step_size * data.L.lin_map_adj(y), step_size)
        x_bar = x + cp_par * (x - x_prev)

        if pars.max_iter is not None:
            iter_output = iter_index
            if iter_index >= pars.max_iter - 1:
                stopping_criteria_flag = False
        else:
            iter_output = 0

        obj_val[iter_output] = data.f.val(x) + data.g.val(data.L.lin_map(x))

        if pars.verbose:
            print('| {:<10} | {:<20} |'.format(iter_index, obj_val[iter_output]))
        else:
            print_progress_bar(iter_index + 1, pars.max_iter)
            # stdout.flush()

    if pars.max_iter is None:
        obj_val[0] = data.f.val(x) + data.g.val(data.L.lin_map(x))

    return Results(x, obj_val)