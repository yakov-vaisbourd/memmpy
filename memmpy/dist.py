"""
Copyright (c) 2023, Yakov Vaisbourd
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.


The maximum entropy on the mean (MEM) is an information driven statistical estimation method.
This package contains an implementation of the Cramér rate function and Bregman proximal operators for many well known
reference distributions.

"""

import memmpy.misc as misc
import numpy as np

find_root = misc.find_root
TOLERANCE = misc.TOLERANCE
MAX_ITER = misc.MAX_ITER

WARM_START_NUM = 500
WARM_START_INTERVAL = 3
WARM_START_MAX = 10


# super class
class generic_dist(object):
    """
    A generic distribution class.

    Attributes
    ----------
    root_tol : float
        The tolerance level for the root search method.
    root_max_iter : int
        The maximum number of iterations for the root search method.
    warm_start_num : int
        The number of warm starts for the root search method.
    warm_start_interval : int
        The warm start interval for the root search method.
    warm_start_max : int
        The maximum number of warm starts for the root search method.
    warm_start_on : bool
        Whether to use warm start for the root search method.
    """

    def __init__(self):
        self.root_tol = TOLERANCE
        self.root_max_iter = MAX_ITER

        self.warm_start_num = WARM_START_NUM
        self.warm_start_interval = WARM_START_INTERVAL
        self.warm_start_max = WARM_START_MAX

        self.warm_start_on = (self.warm_start_interval > 0) and (self.warm_start_max > 0)

    def _validateArguments(self, arg, arg_name, validation_type=[], arg2=None, arg2_name=None):
        """
        Validate arguments satisfy the desired conditions.

        Parameters
        ----------
        arg : float or np.ndarray
            The argument to be validated.
        arg_name : str
            The name of the argument.
        validation_type : list of str
            The types of validation to be performed on the argument.
        arg2 : float or np.ndarray, optional
            A second argument to be validated, by default None.
        arg2_name : str, optional
            The name of the second argument, by default None.

        Raises
        ------
        ValueError
            If the argument is not valid.
        NotImplementedError
            If the validation type is not recognized.
        """

        if not np.isscalar(arg) and not isinstance(arg, np.ndarray):
            raise ValueError('Argument ' + arg_name + ' must be a scalar or an ndarray (numpy).')

        if not np.all(np.isfinite(arg)):
            raise ValueError('Argument ' + arg_name + ' must contain only finite numbers.')

        validation_type = [validation_type] if isinstance(validation_type, str) else validation_type
        for val_type in validation_type:
            match val_type:
                case 'scalar':
                    if not np.isscalar(arg) and not np.size(arg) == 1:
                        raise ValueError('Argument ' + arg_name + ' must be a scalar.')
                case 'positive':
                    if np.any(arg <= 0):
                        raise ValueError('Argument ' + arg_name + ' must be positive in all entries.')
                case 'non-negative':
                    if np.any(arg < 0):
                        raise ValueError('Argument ' + arg_name + ' must be non-negative in all entries.')
                case 'positive scalar':
                    if (not np.isscalar(arg) and not np.size(arg) == 1) or np.any(arg <= 0):
                        raise ValueError('Argument ' + arg_name + ' must be a positive scalar.')
                case 'simplex':
                    if arg2 is None:
                        arg2 = 1
                    if np.any(arg < 0) or (np.sum(arg) != arg2):
                        raise ValueError(
                            'Argument ' + arg_name + ' must be in the ' + arg2 +
                            ' simplex (non-negative values which sum to ' + arg2 + ').')
                case 'simplex_ri':
                    if arg2 is None:
                        arg2 = 1
                    if np.any(arg <= 0) or (np.sum(arg) != arg2):
                        raise ValueError(
                            'Argument ' + arg_name + ' must be in the ' + arg2 +
                            ' simplex relative interior (positive values which sum to ' + arg2 + ').')
                case 'integer':
                    if not (isinstance(np.atleast_1d(arg)[0], np.integer)):
                        raise ValueError('Argument ' + arg_name + ' must be an integer (possibly ndarray).')
                case 'dimensions':
                    if not (np.isscalar(arg) and np.isscalar(arg2)) and (
                            (not np.isscalar(arg) and not np.isscalar(arg2)) and arg.shape != arg2.shape):
                        raise ValueError('Arguments ' + arg_name + ' and ' + arg2_name +
                                         ' must be scalars or one dimensional ndarray of the same size.')
                case 'ordered':
                    if np.any(arg2 < arg):
                        raise ValueError('All components of argument ' + arg_name + ' must be strictly smaller than  '
                                                                                    '' + arg2_name + '.')
                case _:
                    return NotImplementedError

    def _verifyPars(self, params, is_map=False):
        """
        Verify the validity of the input parameters.

        Parameters
        ----------
        params : tuple or list of ndarrays, optional
            The input parameters to be verified.
        is_map : bool, optional
            Flag indicating if the function is being called for a map, by default False.

        Returns
        -------
        list of ndarrays or tuple or list of None
            The verified input parameters.

        Raises
        ------
        ValueError
            If some but not all of the input parameters are None.

        Notes
        -----
        This function checks if the input parameters are valid, i.e., if they are either all None or all ndarrays. If
        some but not all of the parameters are None, a ValueError is raised. If the function is being called for a map,
        it returns the input parameters without any modifications. If the function is not being called for a map, it
        converts each parameter to an ndarray using np.atleast_1d, unless the parameter is None.
        """

        isNone = [par is None for par in params]

        if any(isNone) and not all(isNone):
            raise ValueError('All or none of the arguments must be not None.')

        # todo: if we won't have map in the final version then this should be updated - including the documentation
        if not is_map:
            if not any(isNone):
                return [np.atleast_1d(np.asarray(par)) for par in params]
            else:
                return params

    def init_parameters_root_search_method(self):
        """
        Initializes the parameters used in the root search method.

        Parameters
        ----------
        self : object
            Instance of the class.

        Returns
        -------
        None

        Notes
        -----
        TOLERANCE : float
            The tolerance level for the root search method.
        MAX_ITER : int
            The maximum number of iterations for the root search method.
        WARM_START_NUM : int
            The number of initial starting points that will be recorded to be used in the
            warm start procedure.
        WARM_START_INTERVAL : int
            The interval between calls to the operator warm start iterations for the root search method.
        WARM_START_MAX : int
            The maximum number of function calls for which the warm start procedure will
            record initialization values for the root search method.

        The warm start feature is used to speed up the root search method by using the previous solution as a starting
        (initial) point for the next iteration. This feature can subsequently improve the performance of optimization
        algorithm which repeatedly employ the proximal operator (at each iteration). The warm start procedure records
        the result of the root search method as a function of the point of evaluation and use it for the initialization
        of the root search method in subsequent calls. The variables control the number of recorded starting points
        (WARM_START_NUM) and the and the recording of starting points protocol (WARM_START_INTERVAL, WARM_START_MAX).
        Since the collecting initial points incur additional computational burden it is limited only to WARM_START_MAX
        calls of the proximal operator in which the recording take place only at each WARM_START_INTERVAL iteration. The
        warm start procedure is only enabled if the warm start interval and maximum number of iterations are greater
        than zero.

        """

        self.root_tol = TOLERANCE
        self.root_max_iter = MAX_ITER

        self.warm_start_num = WARM_START_NUM
        self.warm_start_interval = WARM_START_INTERVAL
        self.warm_start_max = WARM_START_MAX

        self.warm_start_on = (self.warm_start_interval > 0) and (self.warm_start_max > 0)

    def update_parameters_root_search_method(self, root_tol=None, root_max_iter=None,
                                             warm_start_num=None, warm_start_interval=None, warm_start_max=None):

        """
          Update the parameters of the root search method.

          Parameters
          ----------
          root_tol : float, optional
              The tolerance value used in the root search method. Must be a positive scalar.
          root_max_iter : int or float, optional
              The maximum number of iterations used in the root search method. Must be a positive scalar or integer.
          warm_start_num : int or float, optional
              The number of recorded starting points in the warm start procedure. Must be a positive scalar or integer.
          warm_start_interval : int or float, optional
              The interval at which starting points are recorded in the warm start procedure. Must be a non-negative
              scalar or integer.
          warm_start_max : int or float, optional
              The maximum number of calls to the proximal operator in which recording occurs. Must be a non-negative
              scalar or integer.

          Raises
          ------
          ValueError
              If any of the input arguments are invalid.

          Notes
          -----
          The warm start feature is used to speed up the root search method by using the previous solution as a starting
          (initial) point for the next iteration. This feature can subsequently improve the performance of optimization
          algorithm which repeatedly employ the proximal operator (at each iteration). The warm start procedure records
          the result of the root search method as a function of the point of evaluation and use it for the
          initialization of the root search method in subsequent calls. The variables control the number of recorded
          starting points ('warm_start_num') and the and the recording of starting points protocol
          ('warm_start_interval', 'warm_start_max'). Since the collecting initial points incur additional computational
          burden it is limited only to 'warm_start_max' calls of the proximal operator in which the recording take place
          only at each 'warm_start_interval' iteration. The warm start procedure is only enabled if the warm start
          interval and maximum number of iterations are greater than zero.

          """

        if root_tol is not None:
            generic_dist.__validateArguments(self, root_tol, 'root_tol', 'positive scalar')
            self.root_tol = root_tol
        if root_max_iter is not None:
            generic_dist.__validateArguments(self, root_max_iter, 'root_max_iter', ['positive scalar', 'integer'])
            self.root_max_iter = root_max_iter

        if warm_start_num is not None:
            generic_dist.__validateArguments(self, warm_start_num, 'warm_start_num', ['positive scalar', 'integer'])
            self.warm_start_num = warm_start_num

        if warm_start_interval is not None:
            generic_dist.__validateArguments(self, warm_start_interval, 'warm_start_interval', ['non-negative',
                                                                                                'scalar', 'integer'])
            self.warm_start_interval = warm_start_interval

        if warm_start_max is not None:
            generic_dist._validateArguments(self, warm_start_max, 'warm_start_max', ['non-negative', 'scalar',
                                                                                     'integer'])

            self.warm_start_max = warm_start_max

        self.warm_start_on = (self.warm_start_interval > 0) and (self.warm_start_max > 0)

    # Abstract methods, see subclasses for details.
    def cramer(self, x):
        raise NotImplementedError("The function should have been implemented in a subclasses.")

    def bregman_prox_gen(self, x, t):
        raise NotImplementedError("The function should have been implemented in a subclasses.")

    def kernel_val(self, x):
        raise NotImplementedError("The function should have been implemented in a subclasses.")

    def kernel_grad(self, x):
        raise NotImplementedError("The function should have been implemented in a subclasses.")

    def kernel_grad_dual(self, z):
        raise NotImplementedError("The function should have been implemented in a subclasses.")

# Distributions
class normal_gen(generic_dist):
    """
    Class representing a Normal distribution.

    The Normal distribution is parameterized by its mean and standard deviation.
    """
    def __init__(self, mu=None, sigma=None):
        """
        Constructs a Normal distribution with specified mean and standard deviation.

        Parameters
        ----------
        mu : array_like, optional
            Mean of the distribution. If None, distribution parameters are not specified.
        sigma : array_like, optional
            Standard deviation of the distribution. If None, distribution parameters are not specified.
        """

        mu, sigma = super()._verifyPars((mu, sigma))

        if mu is not None:
            self.__validateArguments(mu, sigma)

        self.mu = mu
        self.sigma = sigma
        super().__init__()

    def __validateArguments(self, mu, sigma):
        """
        Validates that the distribution parameters are valid.

        Parameters
        ----------
        mu : array_like
            Mean of the distribution.
        sigma : array_like
            Standard deviation of the distribution.

        Raises
        ------
        ValueError
            If any of the distribution parameters is not valid.
        """

        super()._validateArguments(mu, 'mu')
        super()._validateArguments(sigma, 'sigma', 'positive')
        super()._validateArguments(mu, 'mu', 'dimensions', sigma, 'sigma')

    def __verifyPars(self, mu, sigma):
        """
        Verifies that the distribution parameters are valid.

        Parameters
        ----------
        mu : array_like, optional
            Mean of the distribution. If None, the current mean is used.
        sigma : array_like, optional
            Standard deviation of the distribution. If None, the current standard deviation is used.

        Returns
        -------
        tuple
            Tuple of the verified mean and standard deviation.

        Raises
        ------
        ValueError
            If both mu and sigma are None.

        Notes
        -----
        The function validates that either all None or all are not None. Furthermore, if the parameters all None, then
        the function retrieves the current parameters.
        """

        mu, sigma = super()._verifyPars((mu, sigma))

        if mu is None:
            if self.mu is not None:
                mu = self.mu
                sigma = self.sigma
            else:
                raise ValueError('Distribution parameters were not specified.')

        self.__validateArguments(mu, sigma)

        return mu, sigma

    def freeze(self, mu=None, sigma=None):
        """
        Freezes the distribution by fixing its mean and standard deviation.

        Parameters
        ----------
        mu : array_like, optional
            Mean of the distribution. If None, the current mean is used.
        sigma : array_like, optional
            Standard deviation of the distribution. If None, the current standard deviation is used.

        Returns
        -------
        self
            Returns the instance itself with frozen parameters.

        Raises
        ------
        ValueError
            If both mu and sigma are None.
        """

        mu, sigma = self.__verifyPars(mu, sigma)
        self.mu = mu
        self.sigma = sigma
        self.mu
        return self

    def cramer(self, x, mu=None, sigma=None, entry_wise=False):
        """
        Evaluates the Cramer rate function at the given point(s).

        Parameters
        ----------
        x : array_like
            Point(s) at which to evaluate the Cramer rate function.
        mu : array_like, optional
            Mean of the distribution. If None, the current mean is used.
        sigma : array_like, optional
            Standard deviation of the distribution. If None, the current standard deviation is used.
        entry_wise : bool, optional
            If True, the function returns an array where each element corresponds to the
            Cramer rate evaluated at the corresponding element of `x`. If False (default),
            the sum of the Cramer rate over all elements in `x` is returned.

        Returns
        -------
        float or ndarray
            The value of the Cramer rate function at the given point.
        """

        mu, sigma = self.__verifyPars(mu, sigma)

        val = ((x - mu) ** 2) / (2 * sigma)

        if entry_wise:
            return val

        return np.sum(val)

    def bregman_prox_gen(self, kernel, mu=None, sigma=None):
        """
        Generate the Bregman proximal operator of the Cramer rate function corresponding to the specified Bregman
        divergence (kernel).

        Parameters
        ----------
        kernel : {'Normal', 'Poisson', 'Gamma'}
           The name of the Bregman divergence kernel.
        mu : array_like, optional
           The mean parameter of the distribution. Default is None.
        sigma : array_like, optional
           The scale parameter of the distribution. Default is None.

        Returns
        -------
        breg_prox : callable
           A callable function that computes the Bregman proximal operator.

        Raises
        ------
        ValueError
           If the distribution parameters are not specified.

        NotImplementedError
           If the specified Bregman divergence kernel is not implemented.

        Notes
        -----
        This function generates the proximal operator corresponding to the specified Bregman divergence kernel.
        """

        mu, sigma = self.__verifyPars(mu, sigma)
        mean_val = mu

        match kernel:
            case 'Normal':
                def breg_prox(x, t):
                    return (sigma * x + t * mu) / (t + sigma)
            case 'Poisson':
                def breg_prox(x, t):
                    x = np.atleast_1d(x)

                    def func_val_der(u, index_not_converged):
                        s_u = u[index_not_converged]
                        s_x = x[index_not_converged]

                        if mu.size == 1:
                            s_mu = mu
                            s_sigma = sigma
                        else:
                            s_mu = mu[index_not_converged]
                            s_sigma = sigma[index_not_converged]

                        val = t * (s_u - s_mu) / s_sigma + np.log(s_u) - np.log(s_x)
                        der = t / s_sigma + 1 / s_u

                        return val, der

                    if self.warm_start_on:
                        try:
                            breg_prox.ref_counter += 1
                        except AttributeError:
                            breg_prox.ref_counter = 0

                    lb_val = np.maximum(np.minimum(mean_val, x), 0)
                    ub_val = np.maximum(np.maximum(mean_val, x), 1)

                    if self.warm_start_on and (breg_prox.ref_counter > 0):
                        initial_point = breg_prox.ref_root[
                            np.minimum(np.searchsorted(breg_prox.ref_var, x.reshape(-1)), self.warm_start_num - 1)]
                    else:
                        initial_point = (lb_val + ub_val) / 2

                    res = find_root(func_val_der, lower_bound=lb_val, upper_bound=ub_val,
                                    initial_point=initial_point, max_iter=self.root_max_iter, tol=self.root_tol)

                    effective_warm_start_num = min(self.warm_start_num, x.size)

                    if (self.warm_start_on and (breg_prox.ref_counter % self.warm_start_interval == 0)
                            and (breg_prox.ref_counter <= self.warm_start_max)):
                        breg_prox.ref_var = np.linspace(x.min(), x.max(), effective_warm_start_num)
                        idx_srt = np.argsort(x.view().reshape(-1))
                        breg_prox.ref_root = res.reshape(-1)[
                            idx_srt[np.searchsorted(x.view().reshape(-1)[idx_srt], breg_prox.ref_var)]]

                    return res

            case 'Gamma':  # x is assumed to be positive due to domain assumption.
                def breg_prox(x, t):
                    x = np.atleast_1d(x)
                    return (mu * t / sigma - 1 / x + np.sqrt((mu * t / sigma - 1 / x) ** 2 + 4 * t / sigma)) / (
                            2 * t / sigma)
            case _:
                return NotImplementedError

        return breg_prox

    def kernel_val(self, x):
        """
        Evaluates the kernel function (for Normal distribution) at the given point(s).

        Parameters
        ----------
        x : array_like
            Point(s) at which to evaluate the kernel function.

        Returns
        -------
        float
            The value of the kernel function at the given point.

        Notes
        -----
        The kernel function induces the Bregman distance. This function is used to specify
        the Bregman proximal gradient method and should be smooth adaptable to the objective.
        In case of the Normal distribution, smooth the kernel function is smooth adaptable to
        any function with Lipschitz continuous gradients.
        """

        return x.dot(x) / 2

    def kernel_grad(self, x):
        """
        Evaluates the kernel function gradient (for Normal distribution) at the given point(s).

        Parameters
        ----------
        x : array_like
            Point(s) at which to evaluate the kernel function gradient.

        Returns
        -------
        ndarray
            The gradient of the kernel function at the given point.
        """

        return x

    def kernel_grad_dual(self, z):
        """
        Evaluates the gradient of the kernel function convex conjugate (for Normal distribution) at the given point(s).

        Parameters
        ----------
        x : array_like
            Point(s) at which to evaluate the gradient of the kernel function convex conjugate .

        Returns
        -------
        ndarray
            The gradient of the kernel function convex conjugate at the given point.
        """

        return z

normal = normal_gen()

class multnormal_gen(generic_dist):
    """
    Class representing a multivariate normal distribution.

    The multivariate normal distribution is parameterized by its mean vector and covariance matrix.

    Notes
    -----
    The following operators that involve the covariance matrix are used within the `multnormal_gen`
    class (Sigma stands for the covariance matrix, as default, sigma is set to the identity matrix):
    - `cov_mat(x)`: yielding the product Sigma x
    - `cov_mat_inv(x)`: yielding the product Sigma^{-1} x
    - `res_cov_mat(x, rho)`: resolvent of the covariance matrix, yielding (rho I + Sigma)^{-1} x
    """

    def __init__(self, mu=None, cov_mat=None, cov_mat_inv=None, res_cov_mat=None):
        """
        Constructs a multivariate normal distribution with specified mean and standard deviation.

        Parameters
        ----------
        mu : array-like, optional
            Mean of the distribution. If None, distribution parameters are not specified.
        cov_mat : callable, optional
            A function that takes a vector `x` and returns the product of the
            covariance matrix and `x`. If None, distribution parameters are not specified.
        cov_mat_inv : callable, optional
            A function that takes a vector `x` and returns the product of the
            inverse of the covariance matrix and `x`. If None, distribution parameters are
            not specified.
        res_cov_mat : callable, optional
            A function that takes a vector `x` and a scalar `rho` and returns
            the resolvent of the covariance matrix, (rho I + Sigma)^{-1} x. If None,
            distribution parameters are not specified.
        """

        mu, = super()._verifyPars((mu,))
        super()._verifyPars((cov_mat, cov_mat_inv, res_cov_mat), is_map=True)
        if mu is None and cov_mat is not None:
            raise ValueError('Covariance mappings can be defined only if the standard parameters were specified.')

        is_cov_mat_identity = False
        sigma = None
        if mu is not None:
            if cov_mat is None:
                cov_mat = lambda x: x
                cov_mat_inv = lambda x: x
                res_cov_mat = lambda x, rho: x / (1 + rho)
                is_cov_mat_identity = True
                sigma = 1
            else:
                v = np.random.rand(*mu.shape)
                if np.all(cov_mat(v) / np.linalg.norm(cov_mat(v)) == v / np.linalg.norm(v)):
                    is_cov_mat_identity = True
                    sigma = cov_mat(v)[0] / v[0]
                    if sigma <= 0:
                        raise ValueError('The covariance matrix must be positive definite')

            self.__validateArguments(mu, cov_mat, cov_mat_inv, res_cov_mat)

        self.mu = mu
        self.cov_mat_inv = cov_mat_inv
        self.res_cov_mat = res_cov_mat
        self.is_cov_mat_identity = is_cov_mat_identity
        self.sigma = sigma

        super().__init__()

    def __validateArguments(self, mu, cov_mat=None, cov_mat_inv=None, res_cov_mat=None):
        """
        Validates that the distribution parameters are valid.

        Parameters
        ----------
        mu : array-like
            Mean of the distribution.
        cov_mat : callable
            A function that takes a vector `x` and returns the product of the
            covariance matrix and `x`.
        cov_mat_inv : callable
            A function that takes a vector `x` and returns the product of the
            inverse of the covariance matrix and `x`.
        res_cov_mat : callable
            A function that takes a vector `x` and a scalar `rho` and returns
            the resolvent of the covariance matrix, (rho I + Sigma)^{-1} x.

        Raises
        ------
        ValueError
            If any of the distribution parameters is not valid.
        """

        super()._validateArguments(mu, 'mu')
        super()._verifyPars((cov_mat, cov_mat_inv, res_cov_mat), is_map=True)

    def __verifyPars(self, mu, cov_mat=None, cov_mat_inv=None, res_cov_mat=None):
        """
        Verifies that the distribution parameters are valid.

        Parameters
        ----------
        mu : array-like, optional
            Mean of the distribution. If None, the current mean is used.
        cov_mat : callable, optional
            A function that takes a vector `x` and returns the product of the
            covariance matrix and `x`. If None, the current covariance matrix is used.
        cov_mat_inv : callable, optional
            A function that takes a vector `x` and returns the product of the
            inverse of the covariance matrix and `x`. If None, the current inverse of the covariance
            matrix is used.
        res_cov_mat : callable, optional
            A function that takes a vector `x` and a scalar `rho` and returns
            the resolvent of the covariance matrix, (rho I + Sigma)^{-1} x. If None, the current
            resolvent of the covariance matrix is used.

        Returns
        -------
        tuple
            Tuple of the verified mean and covariance matrix operators.

        Raises
        ------
        ValueError
            If the distribution parameters are None.

        Notes
        -----
        The function validates that either all None or all are not None. Furthermore, if the parameters all None, then
        the function retrieves the current parameters.
        """

        mu, = super()._verifyPars((mu,))
        super()._verifyPars((cov_mat, cov_mat_inv, res_cov_mat), is_map=True)

        if mu is None and cov_mat is not None:
            raise ValueError('Covariance mappings can be defined only if the standard parameters were specified.')

        is_cov_mat_identity = False
        sigma = None
        if mu is not None:
            if np.isscalar(mu) or np.size(mu) == 1:
                raise ValueError('For one dimensional distributions use the Normal prior.')
            if cov_mat is None:
                cov_mat = lambda x: x
                cov_mat_inv = lambda x: x
                res_cov_mat = lambda x, rho: x / (1 + rho)
                is_cov_mat_identity = True
                sigma = 1
            else:
                v = np.random.rand(*mu.shape)
                if np.all(cov_mat(v) / np.linalg.norm(cov_mat(v)) == v / np.linalg.norm(v)):
                    is_cov_mat_identity = True
                    sigma = cov_mat(mu)[0] / mu[0]
                    if sigma <= 0:
                        raise ValueError('The covariance matrix must be positive definite')

        else:
            if self.mu is not None:
                mu = self.mu
                cov_mat = self.cov_mat
                cov_mat_inv = self.cov_mat_inv
                res_cov_mat = self.res_cov_mat
                is_cov_mat_identity = self.is_cov_mat_identity
                sigma = self.sigma
            else:
                raise ValueError('Distribution parameters were not specified.')

        self.__validateArguments(mu, cov_mat, cov_mat_inv, res_cov_mat)

        return mu, cov_mat, cov_mat_inv, res_cov_mat, is_cov_mat_identity, sigma

    def freeze(self, mu=None, cov_mat=None, cov_mat_inv=None, res_cov_mat=None):
        """
        Freezes the distribution by fixing its mean and covariance matrix operators.

        Parameters
        ----------
        mu : array-like, optional
            Mean of the distribution. If None, the current mean is used.
        cov_mat : callable, optional
            A function that takes a vector `x` and returns the product of the
            covariance matrix and `x`. If None, the current covariance matrix is used.
        cov_mat_inv : callable, optional
            A function that takes a vector `x` and returns the product of the
            inverse of the covariance matrix and `x`. If None, the current inverse of the covariance
            matrix is used.
        res_cov_mat : callable, optional
            A function that takes a vector `x` and a scalar `rho` and returns
            the resolvent of the covariance matrix, (rho I + Sigma)^{-1} x. If None, the current
            resolvent of the covariance matrix is used.

        Returns
        -------
        self
            Returns the instance itself with frozen parameters.

        Raises
        ------
        ValueError
            If the distribution parameters are None.
        """

        mu, cov_mat, cov_mat_inv, res_cov_mat, is_cov_mat_identity, sigma = \
            self.__verifyPars(mu, cov_mat, cov_mat_inv, res_cov_mat)

        self.mu = mu
        self.cov_mat = cov_mat
        self.cov_mat_inv = cov_mat_inv
        self.res_cov_mat = res_cov_mat
        self.is_cov_mat_identity = is_cov_mat_identity
        self.sigma = sigma
        return self

    def cramer(self, x, mu=None, cov_mat=None, cov_mat_inv=None, res_cov_mat=None):
        """
        Evaluates the Cramer rate function at the given point(s).

        Parameters
        ----------
        x : array_like
            Point(s) at which to evaluate the Cramer rate function.
        mu : array-like, optional
            Mean of the distribution. If None, the current mean is used.
        cov_mat : callable, optional
            A function that takes a vector `x` and returns the product of the
            covariance matrix and `x`. If None, the current covariance matrix is used.
        cov_mat_inv : callable, optional
            A function that takes a vector `x` and returns the product of the
            inverse of the covariance matrix and `x`. If None, the current inverse of the covariance
            matrix is used.
        res_cov_mat : callable, optional
            A function that takes a vector `x` and a scalar `rho` and returns
            the resolvent of the covariance matrix, (rho I + Sigma)^{-1} x. If None, the current
            resolvent of the covariance matrix is used.

        Returns
        -------
        float
            The value of the Cramer rate function at the given point.
        """

        mu, cov_mat, cov_mat_inv, res_cov_mat, is_cov_mat_identity, sigma = \
            self.__verifyPars(mu, cov_mat, cov_mat_inv, res_cov_mat)

        if is_cov_mat_identity:
            return np.sum(((x - mu) ** 2) / (2 * sigma))
        else:
            return np.tensordot(x - mu, cov_mat_inv(x - mu), x.ndim) / 2

    def bregman_prox_gen(self, kernel, mu=None, cov_mat=None, cov_mat_inv=None, res_cov_mat=None):
        """
        Generate the Bregman proximal operator of the Cramer rate function corresponding to the specified Bregman
        divergence (kernel).

        Parameters
        ----------
        kernel : {'Normal', 'Poisson', 'Gamma'}
           The name of the Bregman divergence kernel.
        mu : array-like, optional
            Mean of the distribution. Default is None.
        cov_mat : callable, optional
            A function that takes a vector `x` and returns the product of the
            covariance matrix and `x`. Default is None.
        cov_mat_inv : callable, optional
            A function that takes a vector `x` and returns the product of the
            inverse of the covariance matrix and `x`. Default is None.
        res_cov_mat : callable, optional
            A function that takes a vector `x` and a scalar `rho` and returns
            the resolvent of the covariance matrix, (rho I + Sigma)^{-1} x. Default is None.

        Returns
        -------
        breg_prox : callable
           A callable function that computes the Bregman proximal operator.

        Raises
        ------
        ValueError
           If the distribution parameters are not specified.

        NotImplementedError
           If the specified Bregman divergence kernel is not implemented. Only the normal
           linear model is supported.

        Notes
        -----
        This function generates the proximal operator corresponding to the specified Bregman divergence kernel.
        """

        mu, cov_mat, cov_mat_inv, res_cov_mat, is_cov_mat_identity, sigma = \
            self.__verifyPars(mu, cov_mat, cov_mat_inv, res_cov_mat)

        match kernel:
            case 'Normal':
                def breg_prox(x, t):
                    if is_cov_mat_identity:
                        return (sigma * x + t * mu) / (t + sigma)
                    else:
                        return res_cov_mat(cov_mat(x) + t * mu, t)
            case 'Poisson':
                def breg_prox(x, t):
                    return NotImplementedError
            case 'Gamma':
                def breg_prox(x, t):
                    return NotImplementedError
            case _:
                return NotImplementedError

        return breg_prox

multnormal = multnormal_gen()

class bernoulli_gen(generic_dist):
    """
    Class representing a Bernoulli distribution.

    The Bernoulli distribution is parameterized by the probability of success.
    """
    def __init__(self, p=None):
        """
        Constructs a Bernoulli distribution with specified probability of success.

        Parameters
        ----------
        p : array_like, optional
            Probability of success. If None, distribution parameters are not specified.
        """
        p, = super()._verifyPars((p,))

        if p is not None:
            self.__validateArguments(p)

        self.p = p
        super().__init__()

    def __validateArguments(self, p):
        """
        Validates that the distribution parameter is valid.

        Parameters
        ----------
        p : array_like
            Probability of success.

        Raises
        ------
        ValueError
            If the distribution parameter is not valid.
        """
        super()._validateArguments(p, 'p', 'non-negative')
        super()._validateArguments(1 - p, '1-p', 'non-negative')

    def __verifyPars(self, p):
        """
        Verifies that the distribution parameter is valid.

        Parameters
        ----------
        p : array_like
            Probability of success. If None, the current probability of success is used.

        Returns
        -------
        tuple
            Tuple of the verified probability of success.

        Raises
        ------
        ValueError
            If p is None.

        Notes
        -----
        The function validates that either all None or all are not None. Furthermore, if the parameters all None, then
        the function retrieves the current parameters.
        """

        p, = super()._verifyPars((p,))

        if p is None:
            if self.p is not None:
                p = self.p
            else:
                raise ValueError('Distribution parameters were not specified.')

        self.__validateArguments(p)

        return p

    def freeze(self, p=None):
        """
        Freezes the distribution by fixing its probability of success.

        Parameters
        ----------
        p : array_like, optional
            Probability of success. If None, the current probability of success is used.

        Returns
        -------
        self
            Returns the instance itself with frozen parameters.

        Raises
        ------
        ValueError
            If p is None.
        """
        p = self.__verifyPars(p)
        self.p = p
        return self

    def cramer(self, x, p=None, entry_wise=False):
        """
        Evaluates the Cramer rate function at the given point(s).

        Parameters
        ----------
        x : array_like
            Point(s) at which to evaluate the Cramer rate function.
        p : array_like, optional
            Probability of success. If None, the current probability of success is used.
        entry_wise : bool, optional
                    If True, the function returns an array where each element corresponds to the
                    Cramer rate evaluated at the corresponding element of `x`. If False (default),
                    the sum of the Cramer rate over all elements in `x` is returned.

        Returns
        -------
        float or ndarray
            The value of the Cramer rate function at the given point.
        """
        p = self.__verifyPars(p)
        x = np.atleast_1d(x)
        p_bnd = np.logical_or(p==0, p==1)
        x_int = np.logical_and(0 < x, x < 1)

        val = np.zeros_like(x)
        out_of_domain = np.logical_or(np.logical_and(p_bnd, x != p), np.abs(x) > 1)
        val[out_of_domain] = np.inf

        if not np.all(out_of_domain):
            idx_to_update = np.logical_and(np.logical_not(out_of_domain), x_int)
            p_update = p if p.size == 1 else p[idx_to_update]

            val[idx_to_update] = x[idx_to_update]*np.log(x[idx_to_update]/p_update) \
                                 + (1-x[idx_to_update])*np.log((1-x[idx_to_update])/(1-p_update))

        if entry_wise:
            return val

        return np.sum(val)

    def bregman_prox_gen(self, kernel, p=None):
        """
        Generate the Bregman proximal operator of the Cramer rate function corresponding to the specified Bregman
        divergence (kernel).

        Parameters
        ----------
        kernel : {'Normal', 'Poisson', 'Gamma'}
           The name of the Bregman divergence kernel.
        p : array_like, optional
           Probability of success. Default is None.

        Returns
        -------
        breg_prox : callable
           A callable function that computes the Bregman proximal operator.

        Raises
        ------
        ValueError
           If the distribution parameters are not specified.

        NotImplementedError
           If the specified Bregman divergence kernel is not implemented.

        Notes
        -----
        This function generates the proximal operator corresponding to the specified Bregman divergence kernel.
        """

        p = self.__verifyPars(p)
        match kernel:
            case 'Normal':
                def breg_prox(x, t):
                    x = np.atleast_1d(x)

                    # The degenerate case # todo: make sure you have this treatment for all discrete dist.
                    if p.size == 1:
                        if p == 0 or p == 1:
                            return p * np.ones_like(x)
                        else:
                            mask_p_intr = np.ones_like(x, dtype=bool)
                            x_p_intr = x
                            p_intr = p
                            res = np.empty_like(x)
                    else:
                        res = np.copy(p)
                        mask_p_intr = np.logical_and(0 < p, p < 1)
                        p_intr = p[mask_p_intr]
                        x_p_intr = x[mask_p_intr]

                    def func_val_der(u, index_not_converged):
                        s_x = x_p_intr[index_not_converged]

                        if p.size == 1:
                            s_p = p
                        else:
                            s_p = p_intr[index_not_converged]

                        s_u = u[index_not_converged]

                        val = (s_u - s_x) - t * np.log(s_p * (1 - s_u) / (s_u * (1 - s_p)))
                        der = (s_u * (1 - s_u) + t) / (s_u * (1 - s_u))

                        return val, der

                    if self.warm_start_on:
                        try:
                            breg_prox.ref_counter += 1
                        except AttributeError:
                            breg_prox.ref_counter = 0

                    if self.warm_start_on and (breg_prox.ref_counter > 0):
                        initial_point = breg_prox.ref_root[
                            np.minimum(np.searchsorted(breg_prox.ref_var, x_p_intr.reshape(-1)), self.warm_start_num-1)]
                    else:
                        initial_point = np.full_like(x_p_intr, 0.5)

                    res[mask_p_intr] = find_root(func_val_der, lower_bound=np.zeros_like(x_p_intr) + self.root_tol,
                                                 upper_bound=np.ones_like(x_p_intr) - self.root_tol,
                                                 initial_point=initial_point,
                                                 max_iter=self.root_max_iter, tol=self.root_tol)

                    effective_warm_start_num = min(self.warm_start_num, x_p_intr.size)

                    if (self.warm_start_on and (breg_prox.ref_counter % self.warm_start_interval == 0)
                                and (breg_prox.ref_counter <= self.warm_start_max)):
                        breg_prox.ref_var = np.linspace(x_p_intr.min(), x_p_intr.max(), effective_warm_start_num)
                        idx_srt = np.argsort(x_p_intr.view().reshape(-1))
                        breg_prox.ref_root = res[mask_p_intr].reshape(-1)[
                            idx_srt[np.searchsorted(x_p_intr.view().reshape(-1)[idx_srt], breg_prox.ref_var)]]

                    return res

            case 'Poisson':
                def breg_prox(x, t):
                    x = np.atleast_1d(x)

                    if p.size == 1:
                        if p == 0 or p == 1:
                            return p * np.ones_like(x)
                        else:
                            mask_p_intr = np.ones_like(x, dtype=bool)
                            x_p_intr = x
                            p_intr = p
                            res = np.empty_like(x)
                    else:
                        res = np.copy(p)
                        mask_p_intr = np.logical_and(0 < p, p < 1)
                        p_intr = p[mask_p_intr]
                        x_p_intr = x[mask_p_intr]

                    def func_val_der(u, index_not_converged):
                        index_active = index_not_converged
                        s_x = x_p_intr[index_active]

                        if p.size == 1:
                            s_p = p
                        else:
                            s_p = p_intr[index_active]

                        s_u = u[index_active]

                        val = np.log(s_u) - np.log(s_x) - t * np.log(s_p * (1 - s_u) / (s_u * (1 - s_p)))
                        der = (t * (1 - s_u) + 1) / (s_u * (1 - s_u))

                        return val, der

                    if self.warm_start_on:
                        try:
                            breg_prox.ref_counter += 1
                        except AttributeError:
                            breg_prox.ref_counter = 0

                    if self.warm_start_on and (breg_prox.ref_counter > 0):
                        initial_point = breg_prox.ref_root[
                            np.minimum(np.searchsorted(breg_prox.ref_var, x_p_intr.reshape(-1)),
                                       self.warm_start_num - 1)]
                    else:
                        initial_point = np.full_like(x_p_intr, 0.5)

                    res[mask_p_intr] = find_root(func_val_der, lower_bound=np.zeros_like(x_p_intr) + self.root_tol,
                                                 upper_bound=np.ones_like(x_p_intr) - self.root_tol,
                                                 initial_point=initial_point,
                                                 max_iter=self.root_max_iter, tol=self.root_tol)

                    effective_warm_start_num = min(self.warm_start_num, x_p_intr.size)

                    if (self.warm_start_on and (breg_prox.ref_counter % self.warm_start_interval == 0)
                            and (breg_prox.ref_counter <= self.warm_start_max)):
                        # print('breg_prox.ref_counter:', breg_prox.ref_counter)
                        breg_prox.ref_var = np.linspace(x_p_intr.min(), x_p_intr.max(), effective_warm_start_num)
                        idx_srt = np.argsort(x_p_intr.view().reshape(-1))
                        breg_prox.ref_root = res[mask_p_intr].reshape(-1)[
                            idx_srt[np.searchsorted(x_p_intr.view().reshape(-1)[idx_srt], breg_prox.ref_var)]]

                    return res

            case 'Gamma':  # x is assumed to be positive due to domain assumption.
                def breg_prox(x, t):
                    x = np.atleast_1d(x)

                    if p.size == 1:
                        if p == 0 or p == 1:
                            return p * np.ones_like(x)
                        else:
                            mask_p_intr = np.ones_like(x, dtype=bool)
                            x_p_intr = x
                            p_intr = p
                            res = np.empty_like(x)
                    else:
                        res = np.copy(p)
                        mask_p_intr = np.logical_and(0 < p, p < 1)
                        p_intr = p[mask_p_intr]
                        x_p_intr = x[mask_p_intr]

                    def func_val_der(u, index_not_converged):
                        index_active = index_not_converged
                        s_x = x_p_intr[index_active]

                        if p.size == 1:
                            s_p = p
                        else:
                            s_p = p_intr[index_active]

                        s_u = u[index_active]

                        val = 1 / s_x - 1 / s_u - t * np.log(s_p * (1 - s_u) / (s_u * (1 - s_p)))
                        der = (t * (s_u - 1) + 1) / ((s_u ** 2) * (1 - s_u))

                        return val, der

                    if self.warm_start_on:
                        try:
                            breg_prox.ref_counter += 1
                        except AttributeError:
                            breg_prox.ref_counter = 0

                    if self.warm_start_on and (breg_prox.ref_counter > 0):
                        initial_point = breg_prox.ref_root[
                            np.minimum(np.searchsorted(breg_prox.ref_var, x_p_intr.reshape(-1)),
                                       self.warm_start_num - 1)]
                    else:
                        initial_point = np.full_like(x_p_intr, 0.5)

                    res[mask_p_intr] = find_root(func_val_der, lower_bound=np.zeros_like(x_p_intr) + self.root_tol,
                                                 upper_bound=np.ones_like(x_p_intr) - self.root_tol,
                                                 initial_point=initial_point,
                                                 max_iter=self.root_max_iter, tol=self.root_tol)

                    effective_warm_start_num = min(self.warm_start_num, x_p_intr.size)

                    if (self.warm_start_on and (breg_prox.ref_counter % self.warm_start_interval == 0)
                            and (breg_prox.ref_counter <= self.warm_start_max)):
                        # print('breg_prox.ref_counter:', breg_prox.ref_counter)
                        breg_prox.ref_var = np.linspace(x_p_intr.min(), x_p_intr.max(), effective_warm_start_num)
                        idx_srt = np.argsort(x_p_intr.view().reshape(-1))
                        breg_prox.ref_root = res[mask_p_intr].reshape(-1)[
                            idx_srt[np.searchsorted(x_p_intr.view().reshape(-1)[idx_srt], breg_prox.ref_var)]]

                    return res

            case _:
                return NotImplementedError

        return breg_prox
bernoulli = bernoulli_gen()

class norminvgauss_gen(generic_dist):
    """
    Class representing a normal inverse Gaussian distribution.

    The normal inverse Gaussian distribution is parameterized by its location, tail, asymmetry and scale parameters.
    """

    def __init__(self, mu=None, alpha=None, beta=None, delta=None):
        """
        Constructs a normal inverse Gaussian distribution with specified location, tail, asymmetry and scale parameters.

        Parameters
        ----------
        mu : array_like, optional
            Location parameter of the distribution. If None, distribution parameters are not specified.
        alpha : array_like, optional
            Tail parameter of the distribution. If None, distribution parameters are not specified.
        beta : array_like, optional
            Asymmetry parameter of the distribution. If None, distribution parameters are not specified.
        delta : array_like, optional
            Scale parameter of the distribution. If None, distribution parameters are not specified.
        """

        mu, alpha, beta, delta = super()._verifyPars((mu, alpha, beta, delta))

        if mu is not None:
            self.__validateArguments(mu, alpha, beta, delta)

        self.mu = mu
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        super().__init__()

    def __validateArguments(self, mu, alpha, beta, delta):
        """
        Validates that the distribution parameters are valid.

        Parameters
        ----------
        mu : array_like
            Location parameter of the distribution.
        alpha : array_like
            Tail parameter of the distribution.
        beta : array_like
            Asymmetry parameter of the distribution.
        delta : array_like
            Scale parameter of the distribution.

        Raises
        ------
        ValueError
            If any of the distribution parameters is not valid.
        """

        super()._validateArguments(mu, 'mu')
        super()._validateArguments(alpha, 'alpha')
        super()._validateArguments(beta, 'beta')
        super()._validateArguments(delta, 'delta', 'positive')
        super()._validateArguments(mu, 'mu', 'dimensions', alpha, 'alpha')
        super()._validateArguments(mu, 'mu', 'dimensions', beta, 'beta')
        super()._validateArguments(mu, 'mu', 'dimensions', delta, 'delta')
        if np.any(alpha < np.abs(beta)):
            raise ValueError('The arguments must satisfy alpha>=|beta|.')

    def __verifyPars(self, mu, alpha, beta, delta):
        """
        Verifies that the distribution parameters are valid.

        Parameters
        ----------
        mu : array_like, optional
            Location parameter of the distribution. If None, the current location parameter is used.
        alpha : array_like, optional
            Tail parameter of the distribution. If None, the current tail parameter is used.
        beta : array_like, optional
            Asymmetry parameter of the distribution. If None, the current asymmetry parameter is used.
        delta : array_like, optional
            Scale parameter of the distribution. If None, the current scale parameter is used.

        Returns
        -------
        tuple
            Tuple of the verified location, tail, asymmetry and scale parameters.

        Raises
        ------
        ValueError
            If all parameters are None.

        Notes
        -----
        The function validates that either all None or all are not None. Furthermore, if the parameters all None, then
        the function retrieves the current parameters.
        """

        mu, alpha, beta, delta = super()._verifyPars((mu, alpha, beta, delta))

        if mu is None:
            if self.mu is not None:
                mu = self.mu
                alpha = self.alpha
                beta = self.beta
                delta = self.delta
            else:
                raise ValueError('Distribution parameters were not specified.')

        self.__validateArguments(mu, alpha, beta, delta)

        return mu, alpha, beta, delta

    def freeze(self, mu=None, alpha=None, beta=None, delta=None):
        """
        Freezes the distribution by fixing its location, tail, asymmetry and scale parameters.

        Parameters
        ----------
        mu : array_like, optional
            Location parameter of the distribution. If None, the current location parameter is used.
        alpha : array_like, optional
            Tail parameter of the distribution. If None, the current tail parameter is used.
        beta : array_like, optional
            Asymmetry parameter of the distribution. If None, the current asymmetry parameter is used.
        delta : array_like, optional
            Scale parameter of the distribution. If None, the current scale parameter is used.

        Returns
        -------
        self
            Returns the instance itself with frozen parameters.

        Raises
        ------
        ValueError
            If all parameters are None.
        """

        mu, alpha, beta, delta = self.__verifyPars(mu, alpha, beta, delta)
        self.mu = mu
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        return self

    def cramer(self, x, mu=None, alpha=None, beta=None, delta=None, entry_wise=False):
        """
        Evaluates the Cramer rate function at the given point(s).

        Parameters
        ----------
        x : array_like
            Point(s) at which to evaluate the Cramer rate function.
        mu : array_like, optional
            Location parameter of the distribution. If None, the current location parameter is used.
        alpha : array_like, optional
            Tail parameter of the distribution. If None, the current tail parameter is used.
        beta : array_like, optional
            Asymmetry parameter of the distribution. If None, the current asymmetry parameter is used.
        delta : array_like, optional
            Scale parameter of the distribution. If None, the current scale parameter is used.
        entry_wise : bool, optional
                    If True, the function returns an array where each element corresponds to the
                    Cramer rate evaluated at the corresponding element of `x`. If False (default),
                    the sum of the Cramer rate over all elements in `x` is returned.

        Returns
        -------
        float or ndarray
            The value of the Cramer rate function at the given point.
        """

        mu, alpha, beta, delta = self.__verifyPars(mu, alpha, beta, delta)

        val = alpha * np.sqrt(delta ** 2 + (x - mu) ** 2) - beta * (x - mu) - delta * np.sqrt(alpha ** 2 - beta ** 2)

        if entry_wise:
            return val
        else:
            return np.sum(val)

    def bregman_prox_gen(self, kernel:str, mu=None, alpha=None, beta=None, delta=None):
        """
        Generate the Bregman proximal operator of the Cramer rate function corresponding to the specified Bregman
        divergence (kernel).

        Parameters
        ----------
        kernel : {'Normal', 'Poisson', 'Gamma'}
           The name of the Bregman divergence kernel.
        mu : array_like
            Location parameter of the distribution. Default is None.
        alpha : array_like
            Tail parameter of the distribution. Default is None.
        beta : array_like
            Asymmetry parameter of the distribution. Default is None.
        delta : array_like
            Scale parameter of the distribution. Default is None.


        Returns
        -------
        breg_prox : callable
           A callable function that computes the Bregman proximal operator.

        Raises
        ------
        ValueError
           If the distribution parameters are not specified.

        NotImplementedError
           If the specified Bregman divergence kernel is not implemented.

        Notes
        -----
        This function generates the proximal operator corresponding to the specified Bregman divergence kernel.
        """

        mu, alpha, beta, delta = self.__verifyPars(mu, alpha, beta, delta)
        mean_val = mu + delta * beta / np.sqrt(alpha ** 2 - beta ** 2)
        match kernel:
            case 'Normal':
                def breg_prox(x: np.ndarray, t: float):
                    x = np.atleast_1d(x)
                    aux1 = (t * beta + x - mu) ** 2

                    def func_val_der(u, index_not_converged):
                        if alpha.size == 1:
                            s_alpha = alpha
                            s_beta = beta
                            s_delta = delta
                            s_mu = mu
                        else:
                            s_alpha = alpha[index_not_converged]
                            s_beta = beta[index_not_converged]
                            s_delta = delta[index_not_converged]
                            s_mu = mu[index_not_converged]

                        s_x = x[index_not_converged]
                        s_u = u[index_not_converged]
                        s_aux = np.sqrt(s_delta ** 2 + (s_u - s_mu) ** 2)

                        val = t * s_alpha * (s_u - s_mu) / s_aux - t * s_beta + s_u - s_x
                        der = t * s_alpha * (s_delta ** 2) / (s_aux ** 3) + 1

                        return val, der

                    lb_val = np.minimum(mean_val, x)
                    ub_val = np.maximum(mean_val, x)

                    if self.warm_start_on:
                        try:
                            breg_prox.ref_counter += 1
                        except AttributeError:
                            breg_prox.ref_counter = 0

                    if self.warm_start_on and (breg_prox.ref_counter > 0):
                        initial_point = breg_prox.ref_root[
                            np.minimum(np.searchsorted(breg_prox.ref_var, x), self.warm_start_num - 1)]
                    else:
                        initial_point = (lb_val + ub_val) / 2

                    res = find_root(func_val_der, lower_bound=lb_val, upper_bound=ub_val,
                                    initial_point=initial_point, max_iter=self.root_max_iter, tol=self.root_tol)

                    effective_warm_start_num = min(self.warm_start_num, 1 if np.isscalar(x) else x.size)

                    if (self.warm_start_on and (breg_prox.ref_counter % self.warm_start_interval == 0)
                                and (breg_prox.ref_counter <= self.warm_start_max)):
                        breg_prox.ref_var = np.linspace(x.min(), x.max(), effective_warm_start_num)
                        idx_srt = np.argsort(x.view().reshape(-1))
                        breg_prox.ref_root = res.view().reshape(-1)[
                            idx_srt[np.searchsorted(x.view().reshape(-1)[idx_srt], breg_prox.ref_var)]]

                    return res

            case 'Poisson':
                def breg_prox(x, t):
                    x = np.atleast_1d(x)

                    def func_val_der(u, index_not_converged):

                        if alpha.size == 1:
                            s_alpha = alpha
                            s_beta = beta
                            s_delta = delta
                            s_mu = mu
                        else:
                            s_alpha = alpha[index_not_converged]
                            s_beta = beta[index_not_converged]
                            s_delta = delta[index_not_converged]
                            s_mu = mu[index_not_converged]

                        s_x = x[index_not_converged]
                        s_u = u[index_not_converged]
                        s_aux = np.sqrt(s_delta ** 2 + (s_u - s_mu) ** 2)

                        val = t * s_alpha * (s_u - s_mu) / s_aux - t * s_beta + np.log(s_u) - np.log(s_x)
                        der = t * s_alpha * (s_delta ** 2) / (s_aux ** 3) + 1 / s_u

                        return val, der

                    lb_val = np.maximum(np.minimum(mean_val, x), self.root_tol)
                    ub_val = np.maximum(np.maximum(mean_val, x), self.root_tol)

                    if self.warm_start_on:
                        try:
                            breg_prox.ref_counter += 1
                        except AttributeError:
                            breg_prox.ref_counter = 0

                    if self.warm_start_on and (breg_prox.ref_counter > 0):
                        initial_point = breg_prox.ref_root[
                            np.minimum(np.searchsorted(breg_prox.ref_var, x), self.warm_start_num - 1)]
                    else:
                        initial_point = (lb_val + ub_val) / 2

                    res = find_root(func_val_der, lower_bound=lb_val, upper_bound=ub_val,
                                    initial_point=initial_point, max_iter=self.root_max_iter, tol=self.root_tol)

                    effective_warm_start_num = min(self.warm_start_num, 1 if np.isscalar(x) else x.size)

                    if (self.warm_start_on and (breg_prox.ref_counter % self.warm_start_interval == 0)
                            and (breg_prox.ref_counter <= self.warm_start_max)):
                        breg_prox.ref_var = np.linspace(x.min(), x.max(), effective_warm_start_num)
                        idx_srt = np.argsort(x.view().reshape(-1))
                        breg_prox.ref_root = res.view().reshape(-1)[
                            idx_srt[np.searchsorted(x.view().reshape(-1)[idx_srt], breg_prox.ref_var)]]

                    return res

            case 'Gamma':  # x is assumed to be positive due to domain assumption.
                def breg_prox(x, t):
                    x = np.atleast_1d(x)

                    def func_val_der(u, index_not_converged):

                        if alpha.size == 1:
                            s_alpha = alpha
                            s_beta = beta
                            s_delta = delta
                            s_mu = mu
                        else:
                            s_alpha = alpha[index_not_converged]
                            s_beta = beta[index_not_converged]
                            s_delta = delta[index_not_converged]
                            s_mu = mu[index_not_converged]

                        s_x = x[index_not_converged]
                        s_u = u[index_not_converged]
                        s_aux = np.sqrt(s_delta ** 2 + (s_u - s_mu) ** 2)

                        val = t * s_alpha * (s_u - s_mu) / s_aux - t * s_beta - 1 / s_u + 1 / s_x
                        der = t * s_alpha * (s_delta ** 2) / (s_aux ** 3) + 1 / (s_u ** 2)

                        return val, der

                    lb_val = np.maximum(np.minimum(mean_val, x), self.root_tol)
                    ub_val = np.maximum(np.maximum(mean_val, x), self.root_tol)

                    if self.warm_start_on:
                        try:
                            breg_prox.ref_counter += 1
                        except AttributeError:
                            breg_prox.ref_counter = 0

                    if self.warm_start_on and (breg_prox.ref_counter > 0):
                        initial_point = breg_prox.ref_root[
                            np.minimum(np.searchsorted(breg_prox.ref_var, x), self.warm_start_num - 1)]
                    else:
                        initial_point = (lb_val + ub_val) / 2

                    res = find_root(func_val_der, lower_bound=lb_val, upper_bound=ub_val,
                                    initial_point=initial_point, max_iter=self.root_max_iter, tol=self.root_tol)

                    effective_warm_start_num = min(self.warm_start_num, 1 if np.isscalar(x) else x.size)

                    if (self.warm_start_on and (breg_prox.ref_counter % self.warm_start_interval == 0)
                            and (breg_prox.ref_counter <= self.warm_start_max)):
                        breg_prox.ref_var = np.linspace(x.min(), x.max(), effective_warm_start_num)
                        idx_srt = np.argsort(x.view().reshape(-1))
                        breg_prox.ref_root = res.view().reshape(-1)[
                            idx_srt[np.searchsorted(x.view().reshape(-1)[idx_srt], breg_prox.ref_var)]]

                    return res

            case _:
                return NotImplementedError

        return breg_prox

norminvgauss = norminvgauss_gen()


class multnorminvgauss_gen(generic_dist):
    """
    Class representing a multivariate normal inverse Gaussian distribution.

    The multivariate normal inverse Gaussian distribution is parameterized by location, tail,
    asymmetry, scale and inter-correlations matrix. It is assumed that its determinant is equal
    to one (not validated in the class).

    icor

    Notes
    -----
    The following operators that involve the inter-correlations matrix are used within the `multnorminvgauss_gen`
    class (Sigma stands for the inter-correlations matrix, as default, sigma is set to the identity matrix):
    - `icor_mat(x)`: yielding the product Sigma x
    - `icor_mat_inv(x)`: yielding the product Sigma^{-1} x
    - `res_icor_mat_inv(x, rho)`: resolvent of the inter-correlations matrix inverse, yielding
        (rho^{-1} I + Sigma^{-1})^{-1} x
    """

    def __init__(self, mu=None, alpha=None, beta=None, delta=None, icor_mat=None, icor_mat_inv=None,
                 res_icor_mat_inv=None):
        """
        Constructs a multivariate normal inverse Gaussian distribution with specified location,
        tail, asymmetry, scale and inter-correlations matrix.

        Parameters
        ----------
        mu : array_like, optional
            Location parameter of the distribution. If None, distribution parameters are not specified.
        alpha : array_like, optional
            Tail parameter of the distribution. If None, distribution parameters are not specified.
        beta : array_like, optional
            Asymmetry parameter of the distribution. If None, distribution parameters are not specified.
        delta : array_like, optional
            Scale parameter of the distribution. If None, distribution parameters are not specified.
        icor_mat : callable, optional
            A function that takes a vector `x` and returns the product of the
            inter-correlations matrix and `x`. If None, distribution parameters are not specified.
        icor_mat_inv : callable, optional
            A function that takes a vector `x` and returns the product of the
            inverse of the inter-correlations matrix and `x`. If None, distribution parameters are
            not specified.
        res_icor_mat_inv : callable, optional
            A function that takes a vector `x` and a scalar `rho` and returns
            the resolvent of the inter-correlations matrix inverse, (rho^{-1} I + Sigma^{-1})^{-1} x.
            If None, distribution parameters are not specified.
        """

        mu, alpha, beta, delta = super()._verifyPars((mu, alpha, beta, delta))
        super()._verifyPars((icor_mat, icor_mat_inv, res_icor_mat_inv), is_map=True)
        if mu is None and icor_mat is not None:
            raise ValueError('Inter-correlations mappings can be defined only if the standard parameters '
                             'were specified.')

        is_icor_mat_identity = False
        sigma = None
        if mu is not None:
            if icor_mat is None:
                icor_mat = lambda x: x
                icor_mat_inv = lambda x: x
                res_icor_mat_inv = lambda x, rho: x * rho / (1 + rho)
                is_icor_mat_identity = True
                sigma = 1
            else:
                if np.all(icor_mat(mu) / np.linalg.norm(icor_mat(mu)) == mu / np.linalg.norm(mu)):
                    is_icor_mat_identity = True
                    sigma = icor_mat(mu)[0] / mu[0]
                    if sigma <= 0:
                        raise ValueError('The inter-correlations matrix must be positive definite')

            self.__validateArguments(mu, alpha, beta, delta, icor_mat, icor_mat_inv, res_icor_mat_inv)

        self.mu = mu
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        self.icor_mat = icor_mat
        self.icor_mat_inv = icor_mat_inv
        self.res_icor_mat_inv = res_icor_mat_inv
        self.is_icor_mat_identity = is_icor_mat_identity
        self.sigma = sigma

        if mu is not None:
            self._gamma = np.sqrt(alpha ** 2 - np.tensordot(beta, icor_mat(beta), beta.ndim))
        else:
            self._gamma = None

        super().__init__()

    def __validateArguments(self, mu, alpha, beta, delta, icor_mat, icor_mat_inv, res_icor_mat_inv):
        """
        Validates that the distribution parameters are valid.

        Parameters
        ----------
        mu : array_like
            Location parameter of the distribution.
        alpha : array_like
            Tail parameter of the distribution.
        beta : array_like
            Asymmetry parameter of the distribution.
        delta : array_like
            Scale parameter of the distribution.
        icor_mat : callable
            A function that takes a vector `x` and returns the product of the
            inter-correlations matrix and `x`.
        icor_mat_inv : callable
            A function that takes a vector `x` and returns the product of the
            inverse of the inter-correlations matrix and `x`.
        res_icor_mat_inv : callable
            A function that takes a vector `x` and a scalar `rho` and returns
            the resolvent of the inter-correlations matrix inverse, (rho^{-1} I + Sigma^{-1})^{-1} x.

        Raises
        ------
        ValueError
            If any of the distribution parameters is not valid.
        """

        super()._validateArguments(mu, 'mu')
        super()._validateArguments(alpha, 'alpha', 'scalar')
        super()._validateArguments(beta, 'beta')
        super()._validateArguments(delta, 'delta', 'positive scalar')
        super()._validateArguments(mu, 'mu', 'dimensions', beta, 'beta')
        super()._verifyPars((icor_mat, icor_mat_inv, res_icor_mat_inv), is_map=True)

        if alpha ** 2 < np.tensordot(beta, icor_mat(beta), beta.ndim):
            raise ValueError('The arguments must satisfy alpha>=np.inner(beta, icor_mat(beta)).')

    def __verifyPars(self, mu, alpha, beta, delta, icor_mat, icor_mat_inv, res_icor_mat_inv):
        """
        Verifies that the distribution parameters are valid.

        Parameters
        ----------
        mu : array_like, optional
            Location parameter of the distribution. If None, the current location parameter is used.
        alpha : array_like, optional
            Tail parameter of the distribution. If None, the current tail parameter is used.
        beta : array_like, optional
            Asymmetry parameter of the distribution. If None, the current asymmetry parameter is used.
        delta : array_like, optional
            Scale parameter of the distribution. If None, the current scale parameter is used.
        icor_mat : callable, optional
            A function that takes a vector `x` and returns the product of the
            inter-correlations matrix and `x`. If None, the current inter-correlations matrix is used.
        icor_mat_inv : callable, optional
            A function that takes a vector `x` and returns the product of the
            inverse of the inter-correlations matrix and `x`. If None, the current inverse
            inter-correlations matrix is used.
        res_icor_mat_inv : callable, optional
            A function that takes a vector `x` and a scalar `rho` and returns
            the resolvent of the inter-correlations matrix inverse, (rho^{-1} I + Sigma^{-1})^{-1} x.
            If None, the current resolvent inter-correlations matrix inverse is used.

        Returns
        -------
        tuple
            Tuple of the verified mean and inter-correlations matrix operators.

        Raises
        ------
        ValueError
            If the distribution parameters are None.

        Notes
        -----
        The function validates that either all None or all are not None. Furthermore, if the parameters all None, then
        the function retrieves the current parameters.
        """

        mu, alpha, beta, delta = super()._verifyPars((mu, alpha, beta, delta))
        super()._verifyPars((icor_mat, icor_mat_inv, res_icor_mat_inv), is_map=True)

        if mu is None and icor_mat is not None:
            raise ValueError('Inter-correlations mappings can be defined only if the standard parameters '
                             'were specified.')

        is_icor_mat_identity = False
        sigma = None
        if mu is not None:
            if icor_mat is None:
                icor_mat = lambda x: x
                icor_mat_inv = lambda x: x
                res_icor_mat_inv = lambda x, rho: x * rho / (1 + rho)
                is_icor_mat_identity = True
                sigma = 1
            else:
                if np.linalg.norm(icor_mat(mu) / np.linalg.norm(icor_mat(mu)) - mu / np.linalg.norm(mu)) < misc.TOLERANCE:
                    is_icor_mat_identity = True
                    sigma = icor_mat(mu)[0] / mu[0]
                    if sigma <= 0:
                        raise ValueError('The inter-correlations matrix must be positive definite')

            _gamma = np.sqrt(alpha ** 2 - np.tensordot(beta, icor_mat(beta), beta.ndim))
        else:
            if self.mu is not None:
                mu = self.mu
                alpha = self.alpha
                beta = self.beta
                delta = self.delta
                icor_mat = self.icor_mat
                icor_mat_inv = self.icor_mat_inv
                res_icor_mat_inv = self.res_icor_mat_inv
                _gamma = self._gamma
                is_icor_mat_identity = self.is_icor_mat_identity
                sigma = self.sigma
            else:
                raise ValueError('Distribution parameters were not specified.')

        self.__validateArguments(mu, alpha, beta, delta, icor_mat, icor_mat_inv, res_icor_mat_inv)

        return mu, alpha, beta, delta, icor_mat, icor_mat_inv, res_icor_mat_inv, _gamma, is_icor_mat_identity, sigma

    def freeze(self, mu=None, alpha=None, beta=None, delta=None, icor_mat=None, icor_mat_inv=None,
               res_icor_mat_inv=None):
        """
        Freezes the distribution by fixing its mean and covariance matrix operators.

        Parameters
        ----------
        mu : array_like, optional
            Location parameter of the distribution. If None, the current location parameter is used.
        alpha : array_like, optional
            Tail parameter of the distribution. If None, the current tail parameter is used.
        beta : array_like, optional
            Asymmetry parameter of the distribution. If None, the current asymmetry parameter is used.
        delta : array_like, optional
            Scale parameter of the distribution. If None, the current scale parameter is used.
        icor_mat : callable, optional
            A function that takes a vector `x` and returns the product of the
            inter-correlations matrix and `x`. If None, the current inter-correlations matrix is used.
        icor_mat_inv : callable, optional
            A function that takes a vector `x` and returns the product of the
            inverse of the inter-correlations matrix and `x`. If None, the current inverse
            inter-correlations matrix is used.
        res_icor_mat_inv : callable, optional
            A function that takes a vector `x` and a scalar `rho` and returns
            the resolvent of the inter-correlations matrix inverse, (rho^{-1} I + Sigma^{-1})^{-1} x.
            If None, the current resolvent inter-correlations matrix inverse is used.

        Returns
        -------
        self
            Returns the instance itself with frozen parameters.

        Raises
        ------
        ValueError
            If the distribution parameters are None.
        """
        mu, alpha, beta, delta, icor_mat, icor_mat_inv, res_icor_mat_inv, _gamma, is_icor_mat_identity, sigma = \
            self.__verifyPars(mu, alpha, beta, delta, icor_mat, icor_mat_inv, res_icor_mat_inv)


        self.mu = mu
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        self.icor_mat = icor_mat
        self.icor_mat_inv = icor_mat_inv
        self.res_icor_mat_inv = res_icor_mat_inv
        self._gamma = np.sqrt(alpha ** 2 - np.tensordot(beta, icor_mat(beta), beta.ndim))
        self.is_icor_mat_identity = is_icor_mat_identity
        self.sigma = sigma
        return self

    def cramer(self, x, mu=None, alpha=None, beta=None, delta=None,
               icor_mat=None, icor_mat_inv=None, res_icor_mat_inv=None):
        """
        Evaluates the Cramer rate function at the given point(s).

        Parameters
        ----------
        x : array_like
            Point(s) at which to evaluate the Cramer rate function.
        mu : array_like, optional
            Location parameter of the distribution. If None, the current location parameter is used.
        alpha : array_like, optional
            Tail parameter of the distribution. If None, the current tail parameter is used.
        beta : array_like, optional
            Asymmetry parameter of the distribution. If None, the current asymmetry parameter is used.
        delta : array_like, optional
            Scale parameter of the distribution. If None, the current scale parameter is used.
        icor_mat : callable, optional
            A function that takes a vector `x` and returns the product of the
            inter-correlations matrix and `x`. If None, the current inter-correlations matrix is used.
        icor_mat_inv : callable, optional
            A function that takes a vector `x` and returns the product of the
            inverse of the inter-correlations matrix and `x`. If None, the current inverse
            inter-correlations matrix is used.
        res_icor_mat_inv : callable, optional
            A function that takes a vector `x` and a scalar `rho` and returns
            the resolvent of the inter-correlations matrix inverse, (rho^{-1} I + Sigma^{-1})^{-1} x.
            If None, the current resolvent inter-correlations matrix inverse is used.

        Returns
        -------
        float
            The value of the Cramer rate function at the given point.
        """

        mu, alpha, beta, delta, icor_mat, icor_mat_inv, res_icor_mat_inv, _gamma, is_icor_mat_identity, sigma = \
            self.__verifyPars(mu, alpha, beta, delta, icor_mat, icor_mat_inv, res_icor_mat_inv)
        return (alpha * np.sqrt(delta ** 2 + np.tensordot(x - mu, icor_mat_inv(x - mu), x.ndim))
                - np.tensordot(beta, x - mu, x.ndim) - delta * _gamma)

    def bregman_prox_gen(self, kernel, mu=None, alpha=None, beta=None, delta=None,
                         icor_mat=None, icor_mat_inv=None, res_icor_mat_inv=None):
        """
        Generate the Bregman proximal operator of the Cramer rate function corresponding to the specified Bregman
        divergence (kernel).

        Parameters
        ----------
        kernel : {'Normal', 'Poisson', 'Gamma'}
           The name of the Bregman divergence kernel.
        mu : array_like, optional
            Location parameter of the distribution. Default is None.
        alpha : array_like, optional
            Tail parameter of the distribution. Default is None.
        beta : array_like, optional
            Asymmetry parameter of the distribution. Default is None.
        delta : array_like, optional
            Scale parameter of the distribution. Default is None.
        icor_mat : callable, optional
            A function that takes a vector `x` and returns the product of the
            inter-correlations matrix and `x`. Default is None.
        icor_mat_inv : callable, optional
            A function that takes a vector `x` and returns the product of the
            inverse of the inter-correlations matrix and `x`. Default is None.
        res_icor_mat_inv : callable, optional
            A function that takes a vector `x` and a scalar `rho` and returns
            the resolvent of the inter-correlations matrix inverse, (rho^{-1} I + Sigma^{-1})^{-1} x.
            Default is None.

        Returns
        -------
        breg_prox : callable
           A callable function that computes the Bregman proximal operator.

        Raises
        ------
        ValueError
           If the distribution parameters are not specified.

        NotImplementedError
           If the specified Bregman divergence kernel is not implemented. Only the normal
           linear model is supported.

        Notes
        -----
        This function generates the proximal operator corresponding to the specified Bregman divergence kernel.
        """

        mu, alpha, beta, delta, icor_mat, icor_mat_inv, res_icor_mat_inv, _gamma, is_icor_mat_identity, sigma = \
            self.__verifyPars(mu, alpha, beta, delta, icor_mat, icor_mat_inv, res_icor_mat_inv)

        match kernel:
            case 'Normal':
                def breg_prox(x, t):
                    x = np.atleast_1d(x)
                    aux1 = t * beta + x - mu

                    if is_icor_mat_identity:
                        aux1 = np.linalg.norm(t * beta + x - mu) ** 2
                    else:
                        aux1 = t * beta + x - mu

                    def func_val_der(u, index_not_converged):
                        # For this case index_not_converged is not relevant since the problem involved in the prox
                        # computation is one dimensional. For this reason the warm start is not implemented either.
                        if is_icor_mat_identity:
                            val = (delta * u) ** 2 + aux1 * sigma * (u / (sigma + u)) ** 2 - (alpha * t) ** 2
                            der = 2 * u * (delta ** 2) + 2 * aux1 * u * (sigma ** 2) / ((sigma + u) ** 3)
                        else:
                            aux2 = res_icor_mat_inv(aux1, u)
                            val = (delta * u) ** 2 + np.tensordot(aux2, icor_mat_inv(aux2), aux2.ndim) - \
                                  (alpha * t) ** 2
                            der = None  # Use bisection in this case

                        return val, der

                    rho = find_root(func_val_der, lower_bound=np.zeros(1), upper_bound=alpha * t / delta,
                                    initial_point=alpha * t / (2 * delta), max_iter=self.root_max_iter,
                                    tol=self.root_tol)

                    return res_icor_mat_inv(t * beta + x + rho * icor_mat_inv(mu), rho) / rho

            case 'Poisson':
                def breg_prox(x, t):
                    return NotImplementedError

            case 'Gamma':  # x is assumed to be positive due to domain assumption.
                def breg_prox(x, t):
                    return NotImplementedError

            case _:
                return NotImplementedError

        return breg_prox


multnorminvgauss = multnorminvgauss_gen()


class gamma_gen(generic_dist):
    """
    Class representing a Gamma distribution.

    The Gamma distribution is parameterized by its shape and (inverse) scale parameters.
    """
    def __init__(self, alpha=None, beta=None):
        """
        Constructs a Gamma distribution with specified mean and standard deviation.

        Parameters
        ----------
        alpha : array_like, optional
            Shape parameter. If None, distribution parameters are not specified.
        beta : array_like, optional
            Inverse scale parameter. If None, distribution parameters are not specified.
        """

        alpha, beta = super()._verifyPars((alpha, beta))

        if alpha is not None:
            self.__validateArguments(alpha, beta)

        self.alpha = alpha
        self.beta = beta
        super().__init__()

    def __validateArguments(self, alpha, beta):
        """
        Validates that the distribution parameters are valid.

        Parameters
        ----------
        alpha : array_like
            Shape parameter.
        beta : array_like
            Inverse scale parameter.
        Raises
        ------
        ValueError
            If any of the distribution parameters is not valid.
        """

        super()._validateArguments(alpha, 'alpha', 'positive')
        super()._validateArguments(beta, 'beta', 'positive')
        super()._validateArguments(alpha, 'alpha', 'dimensions', beta, 'beta')

    def __verifyPars(self, alpha, beta):
        """
        Verifies that the distribution parameters are valid.

        Parameters
        ----------
        alpha : array_like, optional
            Shape parameter. If None, the current shape is used.
        beta : array_like, optional
            Inverse scale parameter. If None, the current inverse scale is used.

        Returns
        -------
        tuple
            Tuple of the verified mean and standard deviation.

        Raises
        ------
        ValueError
            If both alpha and beta are None.

        Notes
        -----
        The function validates that either all None or all are not None. Furthermore, if the parameters all None, then
        the function retrieves the current parameters.
        """

        alpha, beta = super()._verifyPars((alpha, beta))

        if alpha is None:
            if self.alpha is not None:
                alpha = self.alpha
                beta = self.beta
            else:
                raise ValueError('Distribution parameters were not specified.')

        self.__validateArguments(alpha, beta)

        return alpha, beta

    def freeze(self, alpha=None, beta=None):
        """
        Freezes the distribution by fixing its shape and inverse scale parameters.

        Parameters
        ----------
        alpha : array_like, optional
            Shape parameter. If None, the current shape is used.
        beta : array_like, optional
            Inverse scale parameter. If None, the current inverse scale is used.

        Returns
        -------
        self
            Returns the instance itself with frozen parameters.

        Raises
        ------
        ValueError
            If both alpha and beta are None.
        """

        alpha, beta = self.__verifyPars(alpha, beta)
        self.alpha = alpha
        self.beta = beta
        return self

    def cramer(self, x, alpha=None, beta=None, entry_wise=False):
        """
        Evaluates the Cramer rate function at the given point(s).

        Parameters
        ----------
        x : array_like
            Point(s) at which to evaluate the Cramer rate function.
        alpha : array_like
            Shape parameter. If None, the current shape is used.
        beta : array_like
            Inverse scale parameter. If None, the current inverse scale is used.
        entry_wise : bool, optional
                    If True, the function returns an array where each element corresponds to the
                    Cramer rate evaluated at the corresponding element of `x`. If False (default),
                    the sum of the Cramer rate over all elements in `x` is returned.

        Returns
        -------
        float or ndarray
            The value of the Cramer rate function at the given point.
        """

        alpha, beta = self.__verifyPars(alpha, beta)

        val = beta * x - alpha + alpha * np.log(alpha / (beta * x))
        if entry_wise:
            return val

        return np.sum(val)

    def bregman_prox_gen(self, kernel, alpha=None, beta=None):
        """
        Generate the Bregman proximal operator of the Cramer rate function corresponding to the specified Bregman
        divergence (kernel).

        Parameters
        ----------
        kernel : {'Normal', 'Poisson', 'Gamma'}
           The name of the Bregman divergence kernel.
        alpha : array_like, optional
            Shape parameter. Default is None.
        beta : array_like, optional
            Inverse scale parameter. Default is None.

        Returns
        -------
        breg_prox : callable
           A callable function that computes the Bregman proximal operator.

        Raises
        ------
        ValueError
           If the distribution parameters are not specified.

        NotImplementedError
           If the specified Bregman divergence kernel is not implemented.

        Notes
        -----
        This function generates the proximal operator corresponding to the specified Bregman divergence kernel.
        """

        alpha, beta = self.__verifyPars(alpha, beta)
        mean_val = alpha / beta

        match kernel:
            case 'Normal':
                def breg_prox(x, t):
                    x = np.atleast_1d(x)
                    return (x - t * beta + np.sqrt((x - t * beta) ** 2 + 4 * t * alpha)) / 2

            case 'Poisson':
                def breg_prox(x, t):
                    x = np.atleast_1d(x)

                    def func_val_der(u, index_not_converged):
                        s_u = u[index_not_converged]
                        s_x = x[index_not_converged]

                        if alpha.size == 1:
                            s_alpha = alpha
                            s_beta = beta
                        else:
                            s_alpha = alpha[index_not_converged]
                            s_beta = beta[index_not_converged]

                        val = t * s_beta - t * s_alpha / s_u + np.log(s_u / s_x)
                        der = t * s_alpha / (s_u ** 2) + 1 / s_u

                        return val, der

                    if self.warm_start_on:
                        try:
                            breg_prox.ref_counter += 1
                        except AttributeError:
                            breg_prox.ref_counter = 0

                    lb_val = np.maximum(np.minimum(mean_val, x), 0)
                    ub_val = np.maximum(np.maximum(mean_val, x), 1)

                    if self.warm_start_on and (breg_prox.ref_counter > 0):
                        initial_point = breg_prox.ref_root[
                            np.minimum(np.searchsorted(breg_prox.ref_var, x.reshape(-1)), self.warm_start_num - 1)]
                    else:
                        initial_point = (lb_val + ub_val) / 2

                    res = find_root(func_val_der, lower_bound=lb_val, upper_bound=ub_val,
                                    initial_point=initial_point, max_iter=self.root_max_iter, tol=self.root_tol)

                    effective_warm_start_num = min(self.warm_start_num, x.size)

                    if (self.warm_start_on and (breg_prox.ref_counter % self.warm_start_interval == 0)
                            and (breg_prox.ref_counter <= self.warm_start_max)):
                        breg_prox.ref_var = np.linspace(x.min(), x.max(), effective_warm_start_num)
                        idx_srt = np.argsort(x.view().reshape(-1))
                        breg_prox.ref_root = res.reshape(-1)[
                            idx_srt[np.searchsorted(x.view().reshape(-1)[idx_srt], breg_prox.ref_var)]]

                    return res

            case 'Gamma':  # x is assumed to be positive due to domain assumption.
                def breg_prox(x, t):
                    x = np.atleast_1d(x)
                    return x * (t * alpha + 1) / (x * t * beta + 1)

            case _:
                return NotImplementedError

        return breg_prox

    def kernel_val(self, x):
        """
        Evaluates the kernel function (for Gamma distribution) at the given point(s).

        Parameters
        ----------
        x : array_like
            Point(s) at which to evaluate the kernel function.

        Returns
        -------
        float
            The value of the kernel function at the given point.

        Notes
        -----
        The kernel function induces the Bregman distance. This function is used to specify
        the Bregman proximal gradient method and should be smooth adaptable to the objective.
        """

        return -np.sum(np.log(x))

    def kernel_grad(self, x):
        """
        Evaluates the kernel function gradient (for Gamma distribution) at the given point(s).

        Parameters
        ----------
        x : array_like
            Point(s) at which to evaluate the kernel function gradient.

        Returns
        -------
        ndarray
            The gradient of the kernel function at the given point.
        """

        return -1 / x

    def kernel_grad_dual(self, z):
        """
        Evaluates the gradient of the kernel function convex conjugate (for Gamma distribution) at the given point(s).

        Parameters
        ----------
        x : array_like
            Point(s) at which to evaluate the gradient of the kernel function convex conjugate .

        Returns
        -------
        ndarray
            The gradient of the kernel function convex conjugate at the given point.
        """

        return -1 / z

gamma = gamma_gen()

class poisson_gen(generic_dist):
    """
    Class representing a Poisson distribution.

    The Poisson distribution is parameterized by its rate.
    """
    def __init__(self, _lambda=None):
        """
        Constructs a Poisson distribution with specified rate.

        Parameters
        ----------
        _lambda : array_like, optional
            Rate of the distribution. If None, distribution parameters are not specified.
        """
        _lambda, = super()._verifyPars((_lambda,))

        if _lambda is not None:
            self.__validateArguments(_lambda)

        self._lambda = _lambda
        super().__init__()

    def __validateArguments(self, _lambda):
        """
        Validates that the distribution parameters are valid.

        Parameters
        ----------
        _lambda : array_like
            Rate of the distribution.

        Raises
        ------
        ValueError
            If the distribution parameter is not valid.
        """
        super()._validateArguments(_lambda, '_lambda', 'positive')

    def __verifyPars(self, _lambda):
        """
        Verifies that the distribution parameters are valid.

        Parameters
        ----------
        _lambda : array_like, optional
            Rate of the distribution. If None, the current rate is used.

        Returns
        -------
        tuple
            The verified rate.

        Raises
        ------
        ValueError
            If rate is None.

        Notes
        -----
        If the parameter is None, then the function retrieves the current parameter.
        """
        _lambda, = super()._verifyPars((_lambda,))

        if _lambda is None:
            if self._lambda is not None:
                _lambda = self._lambda
            else:
                raise ValueError('Distribution parameters were not specified.')

        self.__validateArguments(_lambda)

        return _lambda

    def freeze(self, _lambda=None):
        """
        Freezes the distribution by fixing its rate.

        Parameters
        ----------
        _lambda : array_like, optional
            Rate of the distribution. If None, the current rate is used.

        Returns
        -------
        self
            Returns the instance itself with frozen parameter.
        """
        _lambda = self.__verifyPars(_lambda)
        self._lambda = _lambda
        return self

    def cramer(self, x, _lambda=None, entry_wise=False):
        """
        Evaluates the Cramer rate function at the given point(s).

        Parameters
        ----------
        x : array_like
            Point(s) at which to evaluate the Cramer rate function.
        _lambda : array_like, optional
            Rate of the distribution. If None, the current rate is used.
        entry_wise : bool, optional
                    If True, the function returns an array where each element corresponds to the
                    Cramer rate evaluated at the corresponding element of `x`. If False (default),
                    the sum of the Cramer rate over all elements in `x` is returned.

        Returns
        -------
        float or ndarray
            The value of the Cramer rate function at the given point.
        """
        _lambda = self.__verifyPars(_lambda)
        x = np.atleast_1d(x)

        x_int = x > 0
        x_bnd = x == 0

        if _lambda.size == 1:
            _lambda_x_bnd = _lambda
            _lambda_x_int = _lambda
        else:
            _lambda_x_bnd = _lambda[x_bnd]
            _lambda_x_int = _lambda[x_int]

        val = np.empty_like(x)
        val[x_bnd] = _lambda_x_bnd
        val[x_int] = x[x_int] * np.log(x[x_int] / _lambda_x_int) - x[x_int] + _lambda_x_int
        val[np.logical_not(np.logical_or(x_bnd, x_int))] = np.inf

        if entry_wise:
            return val

        return np.sum(val)

    def bregman_prox_gen(self, kernel, _lambda=None):
        """
        Generate the Bregman proximal operator of the Cramer rate function corresponding to the specified Bregman
        divergence (kernel).

        Parameters
        ----------
        kernel : {'Normal', 'Poisson', 'Gamma'}
           The name of the Bregman divergence kernel.
        _lambda : array_like, optional
           The rate parameter of the distribution. Default is None.

        Returns
        -------
        breg_prox : callable
           A callable function that computes the Bregman proximal operator.

        Raises
        ------
        ValueError
           If the distribution parameters are not specified.

        NotImplementedError
           If the specified Bregman divergence kernel is not implemented.

        Notes
        -----
        This function generates the proximal operator corresponding to the specified Bregman divergence kernel.
        """

        _lambda = self.__verifyPars(_lambda)
        mean_val = _lambda

        match kernel:
            case 'Normal':
                def breg_prox(x, t):
                    x = np.atleast_1d(x)

                    def func_val_der(u, index_not_converged):
                        s_u = u[index_not_converged]
                        s_x = x[index_not_converged]

                        if _lambda.size == 1:
                            s_lambda = _lambda
                        else:
                            s_lambda = _lambda[index_not_converged]

                        val = np.log(s_u / s_lambda) + (s_u - s_x) / t
                        der = 1 / s_u + 1 / t

                        return val, der

                    if self.warm_start_on:
                        try:
                            breg_prox.ref_counter += 1
                        except AttributeError:
                            breg_prox.ref_counter = 0

                    lb_val = np.maximum(np.minimum(mean_val, x), 0)
                    ub_val = np.maximum(np.maximum(mean_val, x), 1)

                    if self.warm_start_on and (breg_prox.ref_counter > 0):
                        initial_point = breg_prox.ref_root[
                            np.minimum(np.searchsorted(breg_prox.ref_var, x.reshape(-1)), self.warm_start_num-1)]
                    else:
                        initial_point = (lb_val + ub_val) / 2

                    res = find_root(func_val_der, lower_bound=lb_val, upper_bound=ub_val,
                                    initial_point=initial_point, max_iter=self.root_max_iter, tol=self.root_tol)

                    effective_warm_start_num = min(self.warm_start_num, x.size)

                    if (self.warm_start_on and (breg_prox.ref_counter % self.warm_start_interval == 0)
                            and (breg_prox.ref_counter <= self.warm_start_max)):
                        breg_prox.ref_var = np.linspace(x.min(), x.max(), effective_warm_start_num)
                        idx_srt = np.argsort(x.view().reshape(-1))
                        breg_prox.ref_root = res.reshape(-1)[
                            idx_srt[np.searchsorted(x.view().reshape(-1)[idx_srt], breg_prox.ref_var)]]

                    return res

            case 'Poisson':
                def breg_prox(x, t):
                    x = np.atleast_1d(x)
                    return (x ** (t / (1+t))) * (_lambda ** (1 / (1+t)))

            case 'Gamma':  # x is assumed to be positive due to domain assumption.
                def breg_prox(x, t):
                    x = np.atleast_1d(x)

                    def func_val_der(u, index_not_converged):
                        s_u = u[index_not_converged]
                        s_x = x[index_not_converged]

                        if _lambda.size == 1:
                            s_lambda = _lambda
                        else:
                            s_lambda = _lambda[index_not_converged]

                        val = np.log(s_u / s_lambda) - (1 / s_u - 1 / s_x) / t
                        der = 1 / s_u + 1 / (t * s_u ** 2)

                        return val, der

                    if self.warm_start_on:
                        try:
                            breg_prox.ref_counter += 1
                        except AttributeError:
                            breg_prox.ref_counter = 0

                    lb_val = np.maximum(np.minimum(mean_val, x), 0)
                    ub_val = np.maximum(np.maximum(mean_val, x), 1)

                    if self.warm_start_on and (breg_prox.ref_counter > 0):
                        initial_point = breg_prox.ref_root[
                            np.minimum(np.searchsorted(breg_prox.ref_var, x.reshape(-1)), self.warm_start_num-1)]
                    else:
                        initial_point = (lb_val + ub_val) / 2

                    res = find_root(func_val_der, lower_bound=lb_val, upper_bound=ub_val,
                                    initial_point=initial_point, max_iter=self.root_max_iter, tol=self.root_tol)

                    effective_warm_start_num = min(self.warm_start_num, x.size)

                    if (self.warm_start_on and (breg_prox.ref_counter % self.warm_start_interval == 0)
                            and (breg_prox.ref_counter <= self.warm_start_max)):
                        breg_prox.ref_var = np.linspace(x.min(), x.max(), effective_warm_start_num)
                        idx_srt = np.argsort(x.view().reshape(-1))
                        breg_prox.ref_root = res.reshape(-1)[
                            idx_srt[np.searchsorted(x.view().reshape(-1)[idx_srt], breg_prox.ref_var)]]

                    return res

            case _:
                return NotImplementedError

        return breg_prox

    def kernel_val(self, x):
        """
        Evaluates the kernel function (for Poisson distribution) at the given point(s).

        Parameters
        ----------
        x : array_like
            Point(s) at which to evaluate the kernel function.

        Returns
        -------
        float
            The value of the kernel function at the given point.

        Notes
        -----
        The kernel function induces the Bregman distance. This function is used to specify
        the Bregman proximal gradient method and should be smooth adaptable to the objective.
        """

        return np.sum(x * np.log(x))

    def kernel_grad(self, x):
        """
        Evaluates the kernel function gradient (for Poisson distribution) at the given point(s).

        Parameters
        ----------
        x : array_like
            Point(s) at which to evaluate the kernel function gradient.

        Returns
        -------
        ndarray
            The gradient of the kernel function at the given point.
        """

        return np.log(x) + 1

    def kernel_grad_dual(self, z):
        """
        Evaluates the gradient of the kernel function convex conjugate (for Poisson distribution) at the given point(s).

        Parameters
        ----------
        x : array_like
            Point(s) at which to evaluate the gradient of the kernel function convex conjugate .

        Returns
        -------
        ndarray
            The gradient of the kernel function convex conjugate at the given point.
        """

        return np.exp(z - 1)

poisson = poisson_gen()

class unifdsc_gen(generic_dist):
    """
    Class representing a discrete uniform distribution.

    The discrete uniform distribution is parameterized by its lower and upper bounds.
    """
    def __init__(self, a=None, b=None):
        """
        Constructs a discrete uniform distribution with specified lower and upper bounds.

        Parameters
        ----------
        a : array_like, optional
            Lower bound of the distribution. If None, distribution parameters are not specified.
        b : array_like, optional
            Upper bound of the distribution. If None, distribution parameters are not specified.
        """

        a, b = super()._verifyPars((a, b))

        if a is not None:
            self.__validateArguments(a, b)

        self.a = a
        self.b = b
        super().__init__()

    def __validateArguments(self, a, b):
        """
        Validates that the distribution parameters are valid.

        Parameters
        ----------
        a : array_like
            Lower bound of the distribution.
        b : array_like
            Upper bound of the distribution.

        Raises
        ------
        ValueError
            If any of the distribution parameters is not valid.
        """

        super()._validateArguments(a, 'a')
        super()._validateArguments(b, 'b')
        super()._validateArguments(a, 'a', 'dimensions', b, 'b')
        super()._validateArguments(a, 'a', 'ordered', b, 'b')

    def __verifyPars(self, a, b):
        """
        Verifies that the distribution parameters are valid.

        Parameters
        ----------
        a : array_like, optional
            Lower bound of the distribution. If None, the current mean is used.
        b : array_like, optional
            Upper bound of the distribution. If None, the current standard deviation is used.

        Returns
        -------
        tuple
            Tuple of the verified lower and upper bounds.

        Raises
        ------
        ValueError
            If both a and b are None.

        Notes
        -----
        The function validates that either all None or all are not None. Furthermore, if the parameters all None, then
        the function retrieves the current parameters.
        """

        a, b = super()._verifyPars((a, b))

        if a is None:
            if self.a is not None:
                a = self.a
                b = self.b
            else:
                raise ValueError('Distribution parameters were not specified.')

        self.__validateArguments(a, b)

        return a, b

    def freeze(self, a=None, b=None):
        """
        Freezes the distribution by fixing its lower and upper bounds.

        Parameters
        ----------
        a : array_like, optional
            Lower bound of the distribution. If None, the current mean is used.
        b : array_like, optional
            Upper bound of the distribution. If None, the current standard deviation is used.

        Returns
        -------
        self
            Returns the instance itself with frozen parameters.

        Raises
        ------
        ValueError
            If both a and b are None.
        """

        a, b = self.__verifyPars(a, b)
        self.a = a
        self.b = b
        return self

    def cramer(self, x, a=None, b=None, entry_wise=False):
        """
        Evaluates the Cramer rate function at the given point(s).

        Parameters
        ----------
        x : array_like
            Point(s) at which to evaluate the Cramer rate function.
        a : array_like, optional
            Lower bound of the distribution. If None, the current mean is used.
        b : array_like, optional
            Upper bound of the distribution. If None, the current standard deviation is used.
        entry_wise : bool, optional
                    If True, the function returns an array where each element corresponds to the
                    Cramer rate evaluated at the corresponding element of `x`. If False (default),
                    the sum of the Cramer rate over all elements in `x` is returned.

        Returns
        -------
        float or ndarray
            The value of the Cramer rate function at the given point.
        """

        a, b = self.__verifyPars(a, b)
        x = np.atleast_1d(x)

        if not entry_wise:
            if np.any(x < a) or np.any(x > b):
                return np.inf

        val = np.empty_like(x, dtype=float)
        out_of_domain = np.logical_or(x < a, x > b)
        val[out_of_domain] = np.inf

        mean_val = (a + b) / 2
        n = b - a + 1

        def func_val_der(u, index_not_converged):
            s_u = u[index_not_converged]
            s_x = x[index_not_converged]
            s_n = n[index_not_converged]
            s_a = a[index_not_converged]
            s_b = b[index_not_converged]
            s_mean_val = mean_val[index_not_converged]

            pos_not_converged = s_u > 0
            neg_not_converged = s_u < 0

            aux_expm1 = np.ones_like(s_n, dtype=float)
            aux_expm1_inv = np.ones_like(s_n, dtype=float)
            aux_expnm1 = np.ones_like(s_n, dtype=float)
            aux_expnm1_inv = np.ones_like(s_n, dtype=float)

            aux_expm1[neg_not_converged] = np.expm1(s_u[neg_not_converged])
            aux_expm1[np.isnan(aux_expm1)] = -1
            aux_expnm1[neg_not_converged] = np.expm1(s_n[neg_not_converged] * s_u[neg_not_converged])
            aux_expnm1[np.isnan(aux_expnm1)] = -1
            aux_expm1_inv[pos_not_converged] = np.expm1(- s_u[pos_not_converged])
            aux_expm1_inv[np.isnan(aux_expm1_inv)] = -1
            aux_expnm1_inv[pos_not_converged] = np.expm1(- s_n[pos_not_converged] * s_u[pos_not_converged])
            aux_expnm1_inv[np.isnan(aux_expnm1_inv)] = -1

            zer_not_converged = np.logical_or.reduce((s_u == 0, aux_expm1 == 0, aux_expm1_inv == 0, aux_expnm1 == 0,
                                                      aux_expnm1_inv == 0))

            pos_not_converged = np.logical_and(s_u > 0, np.logical_not(zer_not_converged))
            neg_not_converged = np.logical_and(s_u < 0, np.logical_not(zer_not_converged))

            val = np.empty_like(s_n, dtype=float)
            der = np.empty_like(s_n, dtype=float)

            val[zer_not_converged] = s_mean_val[zer_not_converged] - s_x[zer_not_converged]
            der[zer_not_converged] = np.nan

            val[neg_not_converged] = np.divide(s_n[neg_not_converged] * (aux_expnm1[neg_not_converged] + 1),
                                               aux_expnm1[neg_not_converged]) - \
                                     np.divide(aux_expm1[neg_not_converged] + 1, aux_expm1[neg_not_converged]) + \
                                     s_a[neg_not_converged] - s_x[neg_not_converged]

            val[pos_not_converged] = np.divide(aux_expnm1_inv[pos_not_converged] - s_n[pos_not_converged] * \
                                               aux_expm1_inv[pos_not_converged],
                                     aux_expnm1_inv[pos_not_converged] * aux_expm1_inv[pos_not_converged]) + \
                                     s_a[pos_not_converged] - s_x[pos_not_converged]

            der[zer_not_converged] = np.nan

            der[neg_not_converged] = np.divide((aux_expm1[neg_not_converged] + 1) *
                                               (aux_expnm1[neg_not_converged] ** 2) - (aux_expnm1[neg_not_converged]
                                                                                       + 1) *
                                               ((s_n[neg_not_converged] * aux_expm1[neg_not_converged]) ** 2),
                                               ((aux_expm1[neg_not_converged] * aux_expnm1[neg_not_converged]) ** 2))

            der[pos_not_converged] = np.divide(
                (aux_expm1_inv[pos_not_converged] + 1) * (aux_expnm1_inv[pos_not_converged] ** 2) - (
                            aux_expnm1_inv[pos_not_converged] + 1) * ((s_n[pos_not_converged] *
                                                                       aux_expm1_inv[pos_not_converged]) ** 2),
                ((aux_expm1_inv[pos_not_converged] * aux_expnm1_inv[pos_not_converged]) ** 2))

            return val, der

        # trivial cases (to avoid numerical instability)
        res = np.empty_like(n, dtype=float)
        res[x == mean_val] = 0
        res[np.logical_or(x == a, x == b)] = np.log(n[np.logical_or(x == a, x == b)])

        pos_theta = np.logical_and(b > x, x > mean_val)
        neg_theta = np.logical_and(mean_val > x, x > a)
        lb = np.zeros_like(x, dtype=float)
        ub = np.zeros_like(x, dtype=float)
        lb[neg_theta] = np.log((x[neg_theta] - a[neg_theta] + TOLERANCE) / (x[neg_theta] - a[neg_theta] + 1))
        ub[pos_theta] = np.log((b[pos_theta] - x[pos_theta] + 1) / (b[pos_theta] - x[pos_theta] + TOLERANCE))

        theta = find_root(func_val_der, lower_bound=lb, upper_bound=ub, initial_point=(lb + ub) / 2,
                          max_iter=self.root_max_iter, tol=self.root_tol)

        nnz_theta = theta != 0
        pos_theta = np.logical_and(pos_theta, nnz_theta)
        neg_theta = np.logical_and(neg_theta, nnz_theta)

        res[np.logical_not(nnz_theta)] = 0

        res[pos_theta] = (x[pos_theta] - b[pos_theta]) * theta[pos_theta] - np.log(np.divide(np.expm1(-n[pos_theta] *
                                                                                                      theta[pos_theta]),
                                                          n[pos_theta] * np.expm1(-theta[pos_theta])))

        pos_theta = np.logical_and(pos_theta, np.isnan(res))
        res[pos_theta] = (x[pos_theta] - b[pos_theta]) * theta[pos_theta] + np.log(n[pos_theta])

        res[neg_theta] = (x[neg_theta] - a[neg_theta]) * theta[neg_theta] - np.log(np.divide(np.expm1(n[neg_theta] *
                                                                                                      theta[neg_theta]),
                                                          n[neg_theta] * np.expm1(theta[neg_theta])))

        neg_theta = np.logical_and(neg_theta, np.isnan(res))
        res[neg_theta] = (x[neg_theta] - a[neg_theta]) * theta[neg_theta] + np.log(n[neg_theta])

        if entry_wise:
            return res

        return np.sum(res)

    def bregman_prox_gen(self, kernel, a=None, b=None):
        """
        Generate the Bregman proximal operator of the Cramer rate function corresponding to the specified Bregman
        divergence (kernel).

        Parameters
        ----------
        kernel : {'Normal', 'Poisson', 'Gamma'}
           The name of the Bregman divergence kernel.
        a : array_like, optional
           The lower bound of the distribution. Default is None.
        b : array_like, optional
           The upper bound of the distribution. Default is None.

        Returns
        -------
        breg_prox : callable
           A callable function that computes the Bregman proximal operator.

        Raises
        ------
        ValueError
           If the distribution parameters are not specified.

        NotImplementedError
           If the specified Bregman divergence kernel is not implemented.

        Notes
        -----
        This function generates the proximal operator corresponding to the specified Bregman divergence kernel.
        """

        a, b = self.__verifyPars(a, b)

        mean_val = (a + b) / 2
        n = b - a + 1

        match kernel:
            case 'Normal':
                def breg_prox(x, t):
                    x = np.atleast_1d(x)

                    def func_val_der(u, index_not_converged):

                        s_u = u[index_not_converged]
                        s_x = x[index_not_converged]
                        s_n = n[index_not_converged]
                        s_a = a[index_not_converged]
                        s_mean_val = mean_val[index_not_converged]

                        pos_not_converged = s_u > 0
                        neg_not_converged = s_u < 0

                        aux_expm1 = np.empty_like(s_n, dtype=float)
                        aux_expm1_inv = np.empty_like(s_n, dtype=float)
                        aux_expnm1 = np.empty_like(s_n, dtype=float)
                        aux_expnm1_inv = np.empty_like(s_n, dtype=float)

                        aux_expm1[neg_not_converged] = np.expm1(s_u[neg_not_converged])
                        aux_expm1[np.isnan(aux_expm1)] = -1
                        aux_expnm1[neg_not_converged] = np.expm1(s_n[neg_not_converged] * s_u[neg_not_converged])
                        aux_expnm1[np.isnan(aux_expnm1)] = -1
                        aux_expm1_inv[pos_not_converged] = np.expm1(- s_u[pos_not_converged])
                        aux_expm1_inv[np.isnan(aux_expm1_inv)] = -1
                        aux_expnm1_inv[pos_not_converged] = np.expm1(- s_n[pos_not_converged] * s_u[pos_not_converged])
                        aux_expnm1_inv[np.isnan(aux_expnm1_inv)] = -1

                        zer_not_converged = np.logical_or.reduce((s_u == 0, aux_expm1 == 0, aux_expm1_inv == 0,
                                                                  aux_expnm1 == 0, aux_expnm1_inv == 0))
                        pos_not_converged = np.logical_and(s_u > 0, np.logical_not(zer_not_converged))
                        neg_not_converged = np.logical_and(s_u < 0, np.logical_not(zer_not_converged))

                        val = np.empty_like(s_n, dtype=float)
                        der = np.empty_like(s_n, dtype=float)

                        val[zer_not_converged] = s_mean_val[zer_not_converged] - s_x[zer_not_converged]
                        der[zer_not_converged] = np.nan

                        val[neg_not_converged] = np.divide(s_n[neg_not_converged] * (aux_expnm1[neg_not_converged] + 1)
                                                           , aux_expnm1[neg_not_converged]) - \
                                  np.divide(aux_expm1[neg_not_converged] + 1, aux_expnm1[neg_not_converged]) + \
                                                 s_a[neg_not_converged] + t * s_u[neg_not_converged] - \
                                                 s_x[neg_not_converged]

                        val[pos_not_converged] = np.divide(1, aux_expm1_inv[pos_not_converged]) - \
                                  np.divide(s_n[pos_not_converged], aux_expnm1_inv[pos_not_converged]) + \
                                                 s_a[pos_not_converged] + t * s_u[pos_not_converged] - \
                                                 s_x[pos_not_converged]

                        der[neg_not_converged] = np.divide((aux_expm1[neg_not_converged] + 1) *
                                                           (aux_expnm1[neg_not_converged] ** 2) -
                                            (aux_expnm1[neg_not_converged] + 1) * ((s_n[neg_not_converged] *
                                                                                    aux_expm1[neg_not_converged]) ** 2)
                                            , ((aux_expm1[neg_not_converged] * aux_expnm1[neg_not_converged]) ** 2)) + t

                        der[pos_not_converged] = np.divide(
                            (aux_expm1_inv[pos_not_converged] + 1) * (aux_expnm1_inv[pos_not_converged] ** 2) -
                            (aux_expnm1_inv[pos_not_converged] + 1) * ((s_n[pos_not_converged] *
                                                                        aux_expm1_inv[pos_not_converged]) ** 2),
                            ((aux_expm1_inv[pos_not_converged] * aux_expnm1_inv[pos_not_converged]) ** 2)) + t

                        return val, der

                    if self.warm_start_on:
                        try:
                            breg_prox.ref_counter += 1
                        except AttributeError:
                            breg_prox.ref_counter = 0

                    lb = (x - np.maximum(x, mean_val)) / t
                    ub = (x - np.minimum(x, mean_val)) / t

                    if self.warm_start_on and (breg_prox.ref_counter > 0):
                        initial_point = breg_prox.ref_root[
                            np.minimum(np.searchsorted(breg_prox.ref_var, x.reshape(-1)),
                                       self.warm_start_num - 1)]
                    else:
                        initial_point = (lb + ub) / 2

                    res = find_root(func_val_der, lower_bound=lb, upper_bound=ub, initial_point=initial_point,
                                      max_iter=self.root_max_iter, tol=self.root_tol)

                    effective_warm_start_num = min(self.warm_start_num, x.size)

                    if (self.warm_start_on and (breg_prox.ref_counter % self.warm_start_interval == 0)
                            and (breg_prox.ref_counter <= self.warm_start_max)):
                        breg_prox.ref_var = np.linspace(x.min(), x.max(), effective_warm_start_num)
                        idx_srt = np.argsort(x.view().reshape(-1))
                        breg_prox.ref_root = res.reshape(-1)[
                            idx_srt[np.searchsorted(x.view().reshape(-1)[idx_srt], breg_prox.ref_var)]]

                    return x - t * res

            case 'Poisson':
                super()._validateArguments(b, 'b', 'positive')
                def breg_prox(x, t):
                    x = np.atleast_1d(x)

                    def func_val_der(u, index_not_converged):

                        s_u = u[index_not_converged]
                        s_x = x[index_not_converged]
                        s_n = n[index_not_converged]
                        s_a = a[index_not_converged]
                        s_b = b[index_not_converged]
                        s_mean_val = mean_val[index_not_converged]

                        pos_not_converged = s_u > 0
                        neg_not_converged = s_u < 0

                        aux_expm1 = np.empty_like(s_n, dtype=float)
                        aux_expm1_inv = np.empty_like(s_n, dtype=float)
                        aux_expnm1 = np.empty_like(s_n, dtype=float)
                        aux_expnm1_inv = np.empty_like(s_n, dtype=float)
                        val = np.empty_like(s_n, dtype=float)
                        der = np.empty_like(s_n, dtype=float)

                        aux_expm1[neg_not_converged] = np.expm1(s_u[neg_not_converged])
                        aux_expm1[np.isnan(aux_expm1)] = -1
                        aux_expnm1[neg_not_converged] = np.expm1(s_n[neg_not_converged] * s_u[neg_not_converged])
                        aux_expnm1[np.isnan(aux_expnm1)] = -1
                        aux_expm1_inv[pos_not_converged] = np.expm1(- s_u[pos_not_converged])
                        aux_expm1_inv[np.isnan(aux_expm1_inv)] = -1
                        aux_expnm1_inv[pos_not_converged] = np.expm1(- s_n[pos_not_converged] * s_u[pos_not_converged])
                        aux_expnm1_inv[np.isnan(aux_expnm1_inv)] = -1

                        zer_not_converged = np.logical_or.reduce((s_u == 0, aux_expm1 == 0, aux_expm1_inv == 0,
                                                                  aux_expnm1 == 0, aux_expnm1_inv == 0))
                        pos_not_converged = np.logical_and(s_u > 0, np.logical_not(zer_not_converged))
                        neg_not_converged = np.logical_and(s_u < 0, np.logical_not(zer_not_converged))

                        val[zer_not_converged] = s_mean_val[zer_not_converged] - s_x[zer_not_converged]
                        der[zer_not_converged] = np.nan

                        val[neg_not_converged] = np.divide(s_n[neg_not_converged] * (aux_expnm1[neg_not_converged] + 1),
                                                           aux_expnm1[neg_not_converged]) - \
                                  np.divide(aux_expm1[neg_not_converged] + 1, aux_expnm1[neg_not_converged]) + \
                                                 s_a[neg_not_converged]  - np.exp(np.log(s_x[neg_not_converged]) -
                                                                                  t * s_u[neg_not_converged])

                        val[pos_not_converged] = np.divide(1, aux_expm1_inv[pos_not_converged]) - \
                                  np.divide(s_n[pos_not_converged], aux_expnm1_inv[pos_not_converged]) + \
                                                 s_a[pos_not_converged] - np.exp(np.log(s_x[pos_not_converged]) -
                                                                                 t * s_u[pos_not_converged])

                        der[neg_not_converged] = np.divide((aux_expm1[neg_not_converged] + 1) *
                                                           (aux_expnm1[neg_not_converged] ** 2) -
                                            (aux_expnm1[neg_not_converged] + 1) * ((s_n[neg_not_converged] *
                                                                                    aux_expm1[neg_not_converged]) ** 2)
                                            , ((aux_expm1[neg_not_converged] * aux_expnm1[neg_not_converged]) ** 2)) + \
                                                 t * np.exp(np.log(s_x[neg_not_converged]) - t * s_u[neg_not_converged])

                        der[pos_not_converged] = np.divide(
                            (aux_expm1_inv[pos_not_converged] + 1) * (aux_expnm1_inv[pos_not_converged] ** 2) -
                            (aux_expnm1_inv[pos_not_converged] + 1) * ((s_n[pos_not_converged] *
                                                                        aux_expm1_inv[pos_not_converged]) ** 2),
                            ((aux_expm1_inv[pos_not_converged] * aux_expnm1_inv[pos_not_converged]) ** 2)) + t * np.exp(
                            np.log(s_x[pos_not_converged]) - t * s_u[pos_not_converged])

                        return val, der

                    if self.warm_start_on:
                        try:
                            breg_prox.ref_counter += 1
                        except AttributeError:
                            breg_prox.ref_counter = 0

                    lb = (np.log(x) - np.log(np.maximum(x, mean_val))) / t
                    ub = (np.log(x) - np.log(np.maximum(np.minimum(x, mean_val), self.root_tol))) / t

                    if self.warm_start_on and (breg_prox.ref_counter > 0):
                        initial_point = breg_prox.ref_root[
                            np.minimum(np.searchsorted(breg_prox.ref_var, x.reshape(-1)),
                                       self.warm_start_num - 1)]
                    else:
                        initial_point = (lb + ub) / 2

                    res = find_root(func_val_der, lower_bound=lb, upper_bound=ub, initial_point=initial_point,
                                    max_iter=self.root_max_iter, tol=self.root_tol)

                    effective_warm_start_num = min(self.warm_start_num, x.size)

                    if (self.warm_start_on and (breg_prox.ref_counter % self.warm_start_interval == 0)
                            and (breg_prox.ref_counter <= self.warm_start_max)):
                        breg_prox.ref_var = np.linspace(x.min(), x.max(), effective_warm_start_num)
                        idx_srt = np.argsort(x.view().reshape(-1))
                        breg_prox.ref_root = res.reshape(-1)[
                            idx_srt[np.searchsorted(x.view().reshape(-1)[idx_srt], breg_prox.ref_var)]]

                    return x * np.exp(-t * res)

            case 'Gamma':  # x is assumed to be positive due to domain assumption.
                super()._validateArguments(b, 'b', 'positive')
                def breg_prox(x, t):
                    x = np.atleast_1d(x)

                    def func_val_der(u, index_not_converged):

                        s_u = u[index_not_converged]
                        s_x = x[index_not_converged]
                        s_n = n[index_not_converged]
                        s_a = a[index_not_converged]
                        s_mean_val = mean_val[index_not_converged]

                        pos_not_converged = s_u > 0
                        neg_not_converged = s_u < 0

                        aux_expm1 = np.empty_like(s_n, dtype=float)
                        aux_expm1_inv = np.empty_like(s_n, dtype=float)
                        aux_expnm1 = np.empty_like(s_n, dtype=float)
                        aux_expnm1_inv = np.empty_like(s_n, dtype=float)

                        aux_expm1[neg_not_converged] = np.expm1(s_u[neg_not_converged])
                        aux_expm1[np.isnan(aux_expm1)] = -1
                        aux_expnm1[neg_not_converged] = np.expm1(s_n[neg_not_converged] * s_u[neg_not_converged])
                        aux_expnm1[np.isnan(aux_expnm1)] = -1
                        aux_expm1_inv[pos_not_converged] = np.expm1(- s_u[pos_not_converged])
                        aux_expm1_inv[np.isnan(aux_expm1_inv)] = -1
                        aux_expnm1_inv[pos_not_converged] = np.expm1(- s_n[pos_not_converged] * s_u[pos_not_converged])
                        aux_expnm1_inv[np.isnan(aux_expnm1_inv)] = -1

                        zer_not_converged = np.logical_or.reduce((s_u == 0, aux_expm1 == 0, aux_expm1_inv == 0,
                                                                  aux_expnm1 == 0, aux_expnm1_inv == 0))
                        pos_not_converged = np.logical_and(s_u > 0, np.logical_not(zer_not_converged))
                        neg_not_converged = np.logical_and(s_u < 0, np.logical_not(zer_not_converged))

                        val = np.empty_like(s_n, dtype=float)
                        der = np.empty_like(s_n, dtype=float)

                        val[zer_not_converged] = s_mean_val[zer_not_converged] - s_x[zer_not_converged]
                        der[zer_not_converged] = np.nan

                        val[neg_not_converged] = np.divide(s_n[neg_not_converged] * (aux_expnm1[neg_not_converged] + 1),
                                                           aux_expnm1[neg_not_converged]) - \
                                  np.divide(aux_expm1[neg_not_converged] + 1, aux_expnm1[neg_not_converged]) + \
                                                 s_a[neg_not_converged] - np.divide(s_x[neg_not_converged],
                                                                                    s_x[neg_not_converged] *
                                                                                    t * s_u[neg_not_converged] + 1)

                        val[pos_not_converged] = np.divide(1 , aux_expm1_inv[pos_not_converged]) - \
                                  np.divide(s_n[pos_not_converged], aux_expnm1_inv[pos_not_converged]) + \
                                                 s_a[pos_not_converged] - np.divide(s_x[pos_not_converged],
                                                                                    s_x[pos_not_converged] *
                                                                                    t * s_u[pos_not_converged] + 1)

                        der[neg_not_converged] = np.divide((aux_expm1[neg_not_converged] + 1) *
                                                           (aux_expnm1[neg_not_converged] ** 2) -
                                            (aux_expnm1[neg_not_converged] + 1) * ((s_n[neg_not_converged] *
                                                                                    aux_expm1[neg_not_converged]) ** 2)
                                            , ((aux_expm1[neg_not_converged] * aux_expnm1[neg_not_converged]) ** 2)) + \
                                  np.divide(t * (s_x[neg_not_converged] ** 2), ((s_x[neg_not_converged] *
                                                                                 t * s_u[neg_not_converged] + 1) ** 2))

                        der[pos_not_converged] = np.divide(
                            (aux_expm1_inv[pos_not_converged] + 1) * (aux_expnm1_inv[pos_not_converged] ** 2) -
                            (aux_expnm1_inv[pos_not_converged] + 1) * ((s_n[pos_not_converged] *
                                                                        aux_expm1_inv[pos_not_converged]) ** 2),
                            ((aux_expm1_inv[pos_not_converged] * aux_expnm1_inv[pos_not_converged]) ** 2)) + \
                                                 np.divide(t * s_x[pos_not_converged] ** 2,
                                                           ((s_x[pos_not_converged] * t *
                                                             s_u[pos_not_converged] + 1) ** 2))

                        return val, der

                    if self.warm_start_on:
                        try:
                            breg_prox.ref_counter += 1
                        except AttributeError:
                            breg_prox.ref_counter = 0

                    x_mean_val_max = np.maximum(x, mean_val)
                    x_mean_val_min = np.maximum(np.minimum(x, mean_val), self.root_tol)
                    lb = np.maximum(- 1 / (x * t), (x - x_mean_val_max) / (x_mean_val_max * x * t))
                    ub = (x - x_mean_val_min) / (x_mean_val_min * x * t)

                    if self.warm_start_on and (breg_prox.ref_counter > 0):
                        initial_point = breg_prox.ref_root[
                            np.minimum(np.searchsorted(breg_prox.ref_var, x.reshape(-1)),
                                       self.warm_start_num - 1)]
                    else:
                        initial_point = (lb + ub) / 2

                    res = find_root(func_val_der, lower_bound=lb, upper_bound=ub, initial_point=initial_point,
                                      max_iter=self.root_max_iter, tol=self.root_tol)

                    effective_warm_start_num = min(self.warm_start_num, x.size)

                    if (self.warm_start_on and (breg_prox.ref_counter % self.warm_start_interval == 0)
                            and (breg_prox.ref_counter <= self.warm_start_max)):
                        breg_prox.ref_var = np.linspace(x.min(), x.max(), effective_warm_start_num)
                        idx_srt = np.argsort(x.view().reshape(-1))
                        breg_prox.ref_root = res.reshape(-1)[
                            idx_srt[np.searchsorted(x.view().reshape(-1)[idx_srt], breg_prox.ref_var)]]

                    return x / (x * t * res + 1)

            case _:
                return NotImplementedError

        return breg_prox

unifdsc = unifdsc_gen()


class unifcnt_gen(generic_dist):
    """
    Class representing a continuous uniform distribution.

    The continuous uniform distribution is parameterized by its lower and upper bounds.
    """

    def __init__(self, a=None, b=None):
        """
        Constructs a continuous uniform distribution with specified lower and upper bounds.

        Parameters
        ----------
        a : array_like, optional
            Lower bound of the distribution. If None, distribution parameters are not specified.
        b : array_like, optional
            Upper bound of the distribution. If None, distribution parameters are not specified.
        """

        a, b = super()._verifyPars((a, b))

        if a is not None:
            self.__validateArguments(a, b)

        self.a = a
        self.b = b
        super().__init__()

    def __validateArguments(self, a, b):
        """
        Validates that the distribution parameters are valid.

        Parameters
        ----------
        a : array_like
            Lower bound of the distribution.
        b : array_like
            Upper bound of the distribution.

        Raises
        ------
        ValueError
            If any of the distribution parameters is not valid.
        """

        super()._validateArguments(a, 'a')
        super()._validateArguments(b, 'b')
        super()._validateArguments(a, 'a', 'dimensions', b, 'b')
        super()._validateArguments(a, 'a', 'ordered', b, 'b')

    def __verifyPars(self, a, b):
        """
        Verifies that the distribution parameters are valid.

        Parameters
        ----------
        a : array_like, optional
            Lower bound of the distribution. If None, the current mean is used.
        b : array_like, optional
            Upper bound of the distribution. If None, the current standard deviation is used.

        Returns
        -------
        tuple
            Tuple of the verified lower and upper bounds.

        Raises
        ------
        ValueError
            If both a and b are None.

        Notes
        -----
        The function validates that either all None or all are not None. Furthermore, if the parameters all None, then
        the function retrieves the current parameters.
        """

        a, b = super()._verifyPars((a, b))

        if a is None:
            if self.a is not None:
                a = self.a
                b = self.b
            else:
                raise ValueError('Distribution parameters were not specified.')

        self.__validateArguments(a, b)

        return a, b

    def freeze(self, a=None, b=None):
        """
        Freezes the distribution by fixing its lower and upper bounds.

        Parameters
        ----------
        a : array_like, optional
            Lower bound of the distribution. If None, the current mean is used.
        b : array_like, optional
            Upper bound of the distribution. If None, the current standard deviation is used.

        Returns
        -------
        self
            Returns the instance itself with frozen parameters.

        Raises
        ------
        ValueError
            If both a and b are None.
        """

        a, b = self.__verifyPars(a, b)
        self.a = a
        self.b = b
        return self

    def cramer(self, x, a=None, b=None, entry_wise=False):
        """
        Evaluates the Cramer rate function at the given point(s).

        Parameters
        ----------
        x : array_like
            Point(s) at which to evaluate the Cramer rate function.
        a : array_like, optional
            Lower bound of the distribution. If None, the current mean is used.
        b : array_like, optional
            Upper bound of the distribution. If None, the current standard deviation is used.
        entry_wise : bool, optional
                    If True, the function returns an array where each element corresponds to the
                    Cramer rate evaluated at the corresponding element of `x`. If False (default),
                    the sum of the Cramer rate over all elements in `x` is returned.


        Returns
        -------
        float or ndarray
            The value of the Cramer rate function at the given point.
        """

        a, b = self.__verifyPars(a, b)
        x = np.atleast_1d(x)

        if not entry_wise:
            if np.any(x <= a) or np.any(x >= b):
                return np.inf

        val = np.empty_like(x, dtype=float)
        out_of_domain = np.logical_or(x <= a, x >= b)
        val[out_of_domain] = np.inf

        # if np.any(x <= a) or np.any(x >= b):
        #     return np.inf

        mean_val = (a + b) / 2
        n = b - a

        def func_val_der(u, index_not_converged):
            s_u = u[index_not_converged]
            s_x = x[index_not_converged]
            s_n = n[index_not_converged]
            s_a = a[index_not_converged]
            s_b = b[index_not_converged]
            s_mean_val = mean_val[index_not_converged]

            pos_not_converged = s_u > 0
            neg_not_converged = s_u < 0

            aux_expnm1 = np.ones_like(s_n, dtype=float)
            aux_expnm1_inv = np.ones_like(s_n, dtype=float)

            aux_expnm1[neg_not_converged] = np.expm1(s_n[neg_not_converged] * s_u[neg_not_converged])
            aux_expnm1[np.isnan(aux_expnm1)] = -1
            aux_expnm1_inv[pos_not_converged] = np.expm1(- s_n * s_u, where=pos_not_converged)[pos_not_converged]
            aux_expnm1_inv[np.isnan(aux_expnm1_inv)] = -1

            zer_not_converged = np.logical_or.reduce((s_u == 0, aux_expnm1 == 0, aux_expnm1_inv == 0))

            pos_not_converged = np.logical_and(s_u > 0, np.logical_not(zer_not_converged))
            neg_not_converged = np.logical_and(s_u < 0, np.logical_not(zer_not_converged))

            val = np.empty_like(s_n, dtype=float)
            val[zer_not_converged] = s_mean_val[zer_not_converged] - s_x[zer_not_converged]

            val[pos_not_converged] = np.divide((s_a[pos_not_converged] * s_u[pos_not_converged] - 1) *
                                               aux_expnm1_inv[pos_not_converged] - s_n[pos_not_converged] *
                                               s_u[pos_not_converged], s_u[pos_not_converged] *
                                               aux_expnm1_inv[pos_not_converged]) - s_x[pos_not_converged]

            val[neg_not_converged] = np.divide((s_b[neg_not_converged] * s_u[neg_not_converged] - 1) *
                                               aux_expnm1[neg_not_converged] + s_n[neg_not_converged] *
                                               s_u[neg_not_converged], s_u[neg_not_converged] *
                                               aux_expnm1[neg_not_converged]) - s_x[neg_not_converged]

            der = np.empty_like(s_n, dtype=float)
            der[zer_not_converged] = np.nan

            der[pos_not_converged] = np.divide((aux_expnm1_inv[pos_not_converged] ** 2) -
                                     (aux_expnm1_inv[pos_not_converged] + 1) * ((s_n[pos_not_converged] *
                                                                                         s_u[pos_not_converged]) ** 2),
                                               ((s_u[pos_not_converged] * aux_expnm1_inv[pos_not_converged]) ** 2))

            der[neg_not_converged] = np.divide((aux_expnm1[neg_not_converged] ** 2) -
                                     (aux_expnm1[neg_not_converged] + 1) * ((s_n[neg_not_converged] *
                                                                                     s_u[neg_not_converged]) ** 2),
                                               ((s_u[neg_not_converged] * aux_expnm1[neg_not_converged]) ** 2))
            return val, der

        # trivial cases (to avoid numerical instability)
        res = np.empty_like(n, dtype=float)

        pos_theta = np.logical_and(b > x, x >= mean_val)
        neg_theta = np.logical_and(mean_val > x, x > a)
        lb = np.zeros_like(x, dtype=float)
        ub = np.zeros_like(x, dtype=float)
        lb[neg_theta] = - 2 / (x[neg_theta] - a[neg_theta] + TOLERANCE)
        ub[pos_theta] = 2 / (b[pos_theta] - x[pos_theta] + TOLERANCE)

        initial_point = np.zeros_like(x, dtype=float)
        initial_point[pos_theta] = 1 / n[pos_theta]
        initial_point[neg_theta] = - 1 / n[neg_theta]

        theta = find_root(func_val_der, lower_bound=lb, upper_bound=ub, initial_point= initial_point,
                          max_iter=self.root_max_iter, tol=self.root_tol)

        nnz_theta = theta != 0
        pos_theta = np.logical_and(pos_theta, nnz_theta)
        neg_theta = np.logical_and(neg_theta, nnz_theta)

        res[np.logical_not(nnz_theta)] = 0

        res[pos_theta] = (x[pos_theta] - b[pos_theta]) * theta[pos_theta] - \
                         np.log(np.divide(-np.expm1(- n[pos_theta] * theta[pos_theta]), n[pos_theta]
                                          * theta[pos_theta]))

        pos_theta = np.logical_and(pos_theta, np.isnan(res))
        res[pos_theta] = (x[pos_theta] - b[pos_theta]) * theta[pos_theta] - \
                         np.log(np.divide(1, n[pos_theta] * theta[pos_theta]))

        res[neg_theta] = (x[neg_theta] - a[neg_theta]) * theta[neg_theta] - \
                         np.log(np.divide(np.expm1(n[neg_theta] * theta[neg_theta]), n[neg_theta] * theta[neg_theta]))

        neg_theta = np.logical_and(neg_theta, np.isnan(res))
        res[neg_theta] = (x[neg_theta] - a[neg_theta]) * theta[neg_theta] - \
                         np.log(np.divide(-1, n[neg_theta] * theta[neg_theta]))

        if entry_wise:
            return res

        return np.sum(res)

    def bregman_prox_gen(self, kernel, a=None, b=None):
        """
        Generate the Bregman proximal operator of the Cramer rate function corresponding to the specified Bregman
        divergence (kernel).

        Parameters
        ----------
        kernel : {'Normal', 'Poisson', 'Gamma'}
           The name of the Bregman divergence kernel.
        a : array_like, optional
           The lower bound of the distribution. Default is None.
        b : array_like, optional
           The upper bound of the distribution. Default is None.

        Returns
        -------
        breg_prox : callable
           A callable function that computes the Bregman proximal operator.

        Raises
        ------
        ValueError
           If the distribution parameters are not specified.

        NotImplementedError
           If the specified Bregman divergence kernel is not implemented.

        Notes
        -----
        This function generates the proximal operator corresponding to the specified Bregman divergence kernel.
        """

        a, b = self.__verifyPars(a, b)

        mean_val = (a + b) / 2
        n = b - a

        match kernel:
            case 'Normal':
                def breg_prox(x, t):
                    x = np.atleast_1d(x)

                    def func_val_der(u, index_not_converged):
                        s_u = u[index_not_converged]
                        s_x = x[index_not_converged]
                        s_n = n[index_not_converged]
                        s_a = a[index_not_converged]
                        s_b = b[index_not_converged]
                        s_mean_val = mean_val[index_not_converged]

                        pos_not_converged = s_u > 0
                        neg_not_converged = s_u < 0

                        aux_expnm1 = np.empty_like(s_n, dtype=float)
                        aux_expnm1_inv = np.empty_like(s_n, dtype=float)

                        aux_expnm1_inv[pos_not_converged] = np.expm1(- s_n[pos_not_converged] * s_u[pos_not_converged])
                        aux_expnm1_inv[np.isnan(aux_expnm1_inv)] = -1
                        aux_expnm1[neg_not_converged] = np.expm1(s_n[neg_not_converged] * s_u[neg_not_converged])
                        aux_expnm1[np.isnan(aux_expnm1)] = -1

                        zer_not_converged = np.logical_or.reduce((s_u == 0, aux_expnm1 == 0, aux_expnm1_inv == 0))
                        pos_not_converged = np.logical_and(s_u > 0, np.logical_not(zer_not_converged))
                        neg_not_converged = np.logical_and(s_u < 0, np.logical_not(zer_not_converged))

                        val = np.empty_like(s_n, dtype=float)
                        val[zer_not_converged] = s_mean_val[zer_not_converged] - s_x[zer_not_converged]

                        val[pos_not_converged] = np.divide((s_a[pos_not_converged] * s_u[pos_not_converged] - 1) *
                                                           aux_expnm1_inv[pos_not_converged]
                                                           - s_n[pos_not_converged] * s_u[pos_not_converged],
                                                           s_u[pos_not_converged] *
                                                           aux_expnm1_inv[pos_not_converged]) + \
                                                 t * s_u[pos_not_converged] - s_x[pos_not_converged]

                        val[neg_not_converged] = np.divide((s_b[neg_not_converged] * s_u[neg_not_converged] - 1) *
                                                           aux_expnm1[neg_not_converged] +
                                                           s_n[neg_not_converged] * s_u[neg_not_converged],
                                                           s_u[neg_not_converged] * aux_expnm1[neg_not_converged]) + \
                                                 t * s_u[neg_not_converged] - s_x[neg_not_converged]

                        der = np.empty_like(s_n, dtype=float)
                        der[zer_not_converged] = np.nan

                        der[pos_not_converged] = np.divide((aux_expnm1_inv[pos_not_converged] ** 2) -
                                                 (aux_expnm1_inv[pos_not_converged] + 1) *
                                                           ((s_n[pos_not_converged] * s_u[pos_not_converged]) ** 2),
                                                           ((s_u[pos_not_converged] *
                                                             aux_expnm1_inv[pos_not_converged]) ** 2)) + t

                        der[neg_not_converged] = np.divide((aux_expnm1[neg_not_converged] ** 2) -
                                            (aux_expnm1[neg_not_converged] + 1) * ((s_n[neg_not_converged] *
                                                                                    s_u[neg_not_converged]) ** 2),
                                                           ((s_u[neg_not_converged] *
                                                             aux_expnm1[neg_not_converged]) ** 2)) + t

                        return val, der

                    if self.warm_start_on:
                        try:
                            breg_prox.ref_counter += 1
                        except AttributeError:
                            breg_prox.ref_counter = 0

                    lb = (x - np.maximum(x, mean_val)) / t
                    ub = (x - np.minimum(x, mean_val)) / t

                    if self.warm_start_on and (breg_prox.ref_counter > 0):
                        initial_point = breg_prox.ref_root[
                            np.minimum(np.searchsorted(breg_prox.ref_var, x.reshape(-1)),
                                       self.warm_start_num - 1)]
                    else:
                        initial_point = (lb + ub) / 2

                    res = find_root(func_val_der, lower_bound=lb, upper_bound=ub, initial_point=initial_point,
                                      max_iter=self.root_max_iter, tol=self.root_tol)

                    effective_warm_start_num = min(self.warm_start_num, x.size)

                    if (self.warm_start_on and (breg_prox.ref_counter % self.warm_start_interval == 0)
                            and (breg_prox.ref_counter <= self.warm_start_max)):
                        breg_prox.ref_var = np.linspace(x.min(), x.max(), effective_warm_start_num)
                        idx_srt = np.argsort(x.view().reshape(-1))
                        breg_prox.ref_root = res.reshape(-1)[
                            idx_srt[np.searchsorted(x.view().reshape(-1)[idx_srt], breg_prox.ref_var)]]

                    return x - t * res

            case 'Poisson':
                super()._validateArguments(b, 'b', 'positive')

                def breg_prox(x, t):
                    x = np.atleast_1d(x)

                    def func_val_der(u, index_not_converged):
                        s_u = u[index_not_converged]
                        s_x = x[index_not_converged]
                        s_n = n[index_not_converged]
                        s_a = a[index_not_converged]
                        s_b = b[index_not_converged]
                        s_mean_val = mean_val[index_not_converged]

                        pos_not_converged = s_u > 0
                        neg_not_converged = s_u < 0

                        aux_expnm1 = np.empty_like(s_n, dtype=float)
                        aux_expnm1_inv = np.empty_like(s_n, dtype=float)

                        aux_expnm1_inv[pos_not_converged] = np.expm1(- s_n[pos_not_converged] * s_u[pos_not_converged])
                        aux_expnm1_inv[np.isnan(aux_expnm1_inv)] = -1
                        aux_expnm1[neg_not_converged] = np.expm1(s_n[neg_not_converged] * s_u[neg_not_converged])
                        aux_expnm1[np.isnan(aux_expnm1)] = -1

                        zer_not_converged = np.logical_or.reduce((s_u == 0, aux_expnm1 == 0, aux_expnm1_inv == 0))
                        pos_not_converged = np.logical_and(s_u > 0, np.logical_not(zer_not_converged))
                        neg_not_converged = np.logical_and(s_u < 0, np.logical_not(zer_not_converged))

                        val = np.empty_like(s_n, dtype=float)
                        val[zer_not_converged] = s_mean_val[zer_not_converged] - s_x[zer_not_converged]

                        val[pos_not_converged] = np.divide((s_a[pos_not_converged] * s_u[pos_not_converged] - 1) *
                                                           aux_expnm1_inv[pos_not_converged] -
                                                           s_n[pos_not_converged] * s_u[pos_not_converged],
                                                           s_u[pos_not_converged] *
                                                           aux_expnm1_inv[pos_not_converged]) - \
                                                 np.exp(np.log(s_x[pos_not_converged]) - t * s_u[pos_not_converged])

                        val[neg_not_converged] = np.divide((s_b[neg_not_converged] * s_u[neg_not_converged] - 1) *
                                                           aux_expnm1[neg_not_converged] +
                                                           s_n[neg_not_converged] * s_u[neg_not_converged],
                                                           s_u[neg_not_converged] *
                                                           aux_expnm1[neg_not_converged]) - \
                                                 np.exp(np.log(s_x[neg_not_converged]) - t * s_u[neg_not_converged])

                        der = np.empty_like(s_n, dtype=float)
                        der[zer_not_converged] = np.nan

                        der[pos_not_converged] = np.divide((aux_expnm1_inv[pos_not_converged] ** 2) -
                                                 (aux_expnm1_inv[pos_not_converged] + 1) *
                                                           ((s_n[pos_not_converged] * s_u[pos_not_converged]) ** 2),
                                                           ((s_u[pos_not_converged] *
                                                             aux_expnm1_inv[pos_not_converged]) ** 2)) + \
                                                 t * np.exp(np.log(s_x[pos_not_converged]) - t * s_u[pos_not_converged])

                        der[neg_not_converged] = np.divide((aux_expnm1[neg_not_converged] ** 2) -
                                            (aux_expnm1[neg_not_converged] + 1) * ((s_n[neg_not_converged] *
                                                                                    s_u[neg_not_converged]) ** 2),
                                                           ((s_u[neg_not_converged] *
                                                             aux_expnm1[neg_not_converged]) ** 2)) + \
                                                 t * np.exp(np.log(s_x[neg_not_converged]) - t * s_u[neg_not_converged])

                        return val, der

                    if self.warm_start_on:
                        try:
                            breg_prox.ref_counter += 1
                        except AttributeError:
                            breg_prox.ref_counter = 0

                    lb = (np.log(x) - np.log(np.maximum(x, mean_val))) / t
                    ub = (np.log(x) - np.log(np.maximum(np.minimum(x, mean_val), self.root_tol))) / t


                    if self.warm_start_on and (breg_prox.ref_counter > 0):
                        initial_point = breg_prox.ref_root[
                            np.minimum(np.searchsorted(breg_prox.ref_var, x.reshape(-1)),
                                       self.warm_start_num - 1)]
                    else:
                        initial_point = (lb + ub) / 2

                    res = find_root(func_val_der, lower_bound=lb, upper_bound=ub, initial_point=initial_point,
                                      max_iter=self.root_max_iter, tol=self.root_tol)

                    effective_warm_start_num = min(self.warm_start_num, x.size)

                    if (self.warm_start_on and (breg_prox.ref_counter % self.warm_start_interval == 0)
                            and (breg_prox.ref_counter <= self.warm_start_max)):
                        breg_prox.ref_var = np.linspace(x.min(), x.max(), effective_warm_start_num)
                        idx_srt = np.argsort(x.view().reshape(-1))
                        breg_prox.ref_root = res.reshape(-1)[
                            idx_srt[np.searchsorted(x.view().reshape(-1)[idx_srt], breg_prox.ref_var)]]

                    return x * np.exp(-t * res)
            case 'Gamma':  # x is assumed to be positive due to domain assumption.
                super()._validateArguments(b, 'b', 'positive')

                def breg_prox(x, t):
                    x = np.atleast_1d(x)

                    def func_val_der(u, index_not_converged):
                        s_u = u[index_not_converged]
                        s_x = x[index_not_converged]
                        s_n = n[index_not_converged]
                        s_a = a[index_not_converged]
                        s_b = b[index_not_converged]
                        s_mean_val = mean_val[index_not_converged]

                        pos_not_converged = s_u > 0
                        neg_not_converged = s_u < 0

                        aux_expnm1 = np.empty_like(s_n, dtype=float)
                        aux_expnm1_inv = np.empty_like(s_n, dtype=float)

                        aux_expnm1_inv[pos_not_converged] = np.expm1(- s_n[pos_not_converged] * s_u[pos_not_converged])
                        aux_expnm1_inv[np.isnan(aux_expnm1_inv)] = -1
                        aux_expnm1[neg_not_converged] = np.expm1(s_n[neg_not_converged] * s_u[neg_not_converged])
                        aux_expnm1[np.isnan(aux_expnm1)] = -1

                        zer_not_converged = np.logical_or.reduce((s_u == 0, aux_expnm1 == 0, aux_expnm1_inv == 0))
                        pos_not_converged = np.logical_and(s_u > 0, np.logical_not(zer_not_converged))
                        neg_not_converged = np.logical_and(s_u < 0, np.logical_not(zer_not_converged))

                        val = np.empty_like(s_n, dtype=float)
                        val[zer_not_converged] = s_mean_val[zer_not_converged] - s_x[zer_not_converged]


                        val[pos_not_converged] = np.divide((s_a[pos_not_converged] * s_u[pos_not_converged] - 1) *
                                                           aux_expnm1_inv[pos_not_converged] - s_n[pos_not_converged] *
                                                           s_u[pos_not_converged], s_u[pos_not_converged] *
                                                           aux_expnm1_inv[pos_not_converged]) - \
                                                 s_x[pos_not_converged] / (s_x[pos_not_converged] *
                                                                           t * s_u[pos_not_converged] + 1)

                        val[neg_not_converged] = np.divide((s_b[neg_not_converged] * s_u[neg_not_converged] - 1) *
                                                           aux_expnm1[neg_not_converged] + s_n[neg_not_converged] *
                                                           s_u[neg_not_converged], s_u[neg_not_converged] *
                                                           aux_expnm1[neg_not_converged]) - s_x[neg_not_converged] / \
                                                 (s_x[neg_not_converged] * t * s_u[neg_not_converged] + 1)

                        der = np.empty_like(s_n, dtype=float)
                        der[zer_not_converged] = np.nan

                        der[pos_not_converged] = np.divide((aux_expnm1_inv[pos_not_converged] ** 2) -
                                                 (aux_expnm1_inv[pos_not_converged] + 1) *
                                                           ((s_n[pos_not_converged] * s_u[pos_not_converged]) ** 2),
                                                           ((s_u[pos_not_converged] *
                                                             aux_expnm1_inv[pos_not_converged]) ** 2)) \
                                  + np.divide(t * (s_x[pos_not_converged] ** 2), ((s_x[pos_not_converged] * t *
                                                                                   s_u[pos_not_converged] + 1) ** 2))

                        der[neg_not_converged] = np.divide((aux_expnm1[neg_not_converged] ** 2) -
                                            (aux_expnm1[neg_not_converged] + 1) *
                                                           ((s_n[neg_not_converged] * s_u[neg_not_converged]) ** 2),
                                                           ((s_u[neg_not_converged] *
                                                             aux_expnm1[neg_not_converged]) ** 2)) + \
                                                 t * np.divide(s_x[neg_not_converged] ** 2,
                                                               ((s_x[neg_not_converged] * t *
                                                                 s_u[neg_not_converged] + 1) ** 2))

                        return val, der

                    if self.warm_start_on:
                        try:
                            breg_prox.ref_counter += 1
                        except AttributeError:
                            breg_prox.ref_counter = 0

                    x_mean_val_max = np.maximum(x, mean_val)
                    x_mean_val_min = np.maximum(np.minimum(x, mean_val), self.root_tol)
                    lb = np.maximum(- 1 / (x * t), (x - x_mean_val_max) / (x_mean_val_max * x * t))
                    ub = (x - x_mean_val_min) / (x_mean_val_min * x * t)

                    if self.warm_start_on and (breg_prox.ref_counter > 0):
                        initial_point = breg_prox.ref_root[
                            np.minimum(np.searchsorted(breg_prox.ref_var, x.reshape(-1)),
                                       self.warm_start_num - 1)]
                    else:
                        initial_point = (lb + ub) / 2

                    res = find_root(func_val_der, lower_bound=lb, upper_bound=ub, initial_point=initial_point,
                                      max_iter=self.root_max_iter, tol=self.root_tol)

                    effective_warm_start_num = min(self.warm_start_num, x.size)

                    if (self.warm_start_on and (breg_prox.ref_counter % self.warm_start_interval == 0)
                            and (breg_prox.ref_counter <= self.warm_start_max)):
                        breg_prox.ref_var = np.linspace(x.min(), x.max(), effective_warm_start_num)
                        idx_srt = np.argsort(x.view().reshape(-1))
                        breg_prox.ref_root = res.reshape(-1)[
                            idx_srt[np.searchsorted(x.view().reshape(-1)[idx_srt], breg_prox.ref_var)]]

                    return x / (x * t * res + 1)

            case _:
                return NotImplementedError

        return breg_prox

unifcnt = unifcnt_gen()
