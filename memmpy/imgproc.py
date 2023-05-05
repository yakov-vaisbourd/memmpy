"""
Copyright (c) 2023, Yakov Vaisbourd
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.


This module is designed as an image processing toolbox that supplements the memmpy package. It includes a range of basic
operators for image deblurring, with the implementation based on the book:

    Deblurring Images: Matrices, Spectra, and Filtering by P.C. Hansen, J.G. Nagy, and D.P. O'Leary.

To facilitate its use, the book is accompanied by a MatLab toolbox (HNO package), which can be accessed at
    http://www2.imm.dtu.dk/~pch/HNO.

For clarity and consistency with the reference, this module provides a Python implementation of some functions
described in the book, such as psfGauss, psfDefocus, and padPSF. For further details please refer to the original
implementation provided in the reference.

"""

import numpy as np
from scipy import fftpack as fp


def psfGauss(dim, s=None):
    """
    Create a point spread function for Gaussian blur.

    Parameters
    ----------
    dim : array-like
        Dimension of the psf array. Two-dimensional vector indicating the row and column dimensions, respectively.
        Alternatively, a scalar indicating the row and column dimensions in case that they are the same.
    s : array-like, optional
        Standard deviation of the Gaussian. Two-dimensional vector indicating the standard deviation along the rows
        and columns, respectively. Alternatively, a scalar indicating the standard deviation along the rows and columns
        in case that they are the same (default: None).

    Returns
    -------
    psf : ndarray
        Array containing the psf.
    center : tuple
        Two-dimensional vector containing the index of the center of psf.

    """

    dim = np.atleast_1d(np.asarray(dim))
    if dim.size == 1:
        m = dim
        n = dim
    else:
        m = dim[0]
        n = dim[1]

    s = np.atleast_1d(np.asarray(s))
    if s.size == 1:
        s = np.array([s, s])

    xx = np.arange(-(n // 2), - (-n // 2))
    yy = np.arange(-(m // 2), - (-m // 2))
    x, y = np.meshgrid(xx, yy)

    psf = np.exp(-(x ** 2) / (2 * s[0] ** 2) - (y ** 2) / (2 * s[1] ** 2))
    psf = psf / np.sum(psf)

    center = np.unravel_index(psf.argmax(), psf.shape)

    return psf, center


def psfDefocus(dim, r=None):
    """
    Create a point spread function for out-of-focus blur.

    The point spread function for out-of-focus blur is defined as 1/(pi*r*r) inside a circle of radius r, and zero
    otherwise.

    Parameters
    ----------
    dim : int or tuple of ints
        Dimension of the psf array. Two-dimensional vector indicating the row and column dimensions, respectively.
        Alternatively, a scalar indicating the row and column dimensions in case that they are the same.
    r : float or None, optional
        Radius of out-of-focus. If None, the radius will be set to the minimum value between the center of the psf
        and the boundary of the psf (Default: None).

    Returns
    -------
    psf : ndarray
        Array containing the point spread function.
    center : tuple of ints
        Two-dimensional vector containing the index of the center of the point spread function.
    """

    dim = np.atleast_1d(np.asarray(dim))
    if dim.size == 1:
        m = dim
        n = dim
    else:
        m = dim[0]
        n = dim[1]

    center = (np.array([m, n]) + 1) // 2 - 1
    if r is None:
        r = np.min(center)

    if r == 0:
        psf = np.zeros((m, n))
        psf[center] = 1
    else:
        psf = np.ones((m, n)) / (np.pi * r * r)

    x, y = np.meshgrid(np.arange(n), np.arange(m))
    idx = (x - center[1]) ** 2 + (y - center[0]) ** 2 > r ** 2
    psf[idx] = 0
    psf = psf / np.sum(psf)

    return psf, center


def padPSF(psf, m, n):
    """
    Pad psf with zeros to yield an m x n psf.

    Parameters
    ----------
    psf : ndarray
        Original point spread function.
    m : int
        Row dimension of the desired size.
    n : int
        Column dimension of the desired size.

    Returns
    -------
    ndarray
        Padded psf of dimension m x n.
    """

    PSF_big = np.zeros([m, n])
    PSF_big[0:psf.shape[0], 0:psf.shape[1]] = psf

    return PSF_big


def dctshift(psf, center):
    """
    Create the first column of the blurring matrix under reflexive boundary conditions.

    Parameters
    ----------
    psf : ndarray
        Point spread function.
    center : tuple of int
        Center of the point spread function.

    Returns
    -------
    ndarray
        A vector that contains the first column of the blurring matrix.
    """

    m, n = psf.shape
    i, j = center
    k = np.min((i, m - 1 - i, j, n - 1 - j))

    truncated_psf = psf[(i - k):(i + k + 1), (j - k):(j + k + 1)]

    extract_mat_1 = np.diag(np.ones(k + 1), k)
    extract_mat_2 = np.diag(np.ones(k), k + 1)

    psf_res = np.zeros_like(psf)
    psf_res[0:(2 * k + 1), 0:(2 * k + 1)] = (np.dot(extract_mat_1, np.dot(truncated_psf, extract_mat_1.transpose()))
                                             + np.dot(extract_mat_1, np.dot(truncated_psf, extract_mat_2.transpose()))
                                             + np.dot(extract_mat_2, np.dot(truncated_psf, extract_mat_1.transpose()))
                                             + np.dot(extract_mat_2, np.dot(truncated_psf, extract_mat_2.transpose())))

    return psf_res


def spectral_decomposition_gen(psf, psf_center, img_shape, boundary_conditions='periodic'):
    """
    Create the components of the spectral decomposition of the blurring matrix A based on the point spread function,
    desired image size and boundary conditions. Namely, find matrix B and S such that: A = conj(B)*S*B

    Parameters
    ----------
    psf : ndarray
        The point spread function.
    img_shape : tuple of int
        Desired image size.
    psf_center : tuple of int
        Center of the point spread function.
    boundary_conditions : str, optional
        Boundary conditions (Default: periodic).

    Returns
    -------
    spectra : ndarray
        The spectra matrix (S).
    basis_map : function
        An operator that yields the product of the basis matrix B with a given vector (image).
    basis_map_adj : function
        An operator that yields the product of the adjoint of the basis matrix B with a given vector
        (image in the frequency domain).

    Raises
    ------
    ValueError
        If the specified boundary conditions are not supported.

    """

    img_shape = np.atleast_1d(np.asarray(img_shape))
    if img_shape.size == 1:
        img_shape = np.array([img_shape, img_shape])

    psf_big = padPSF(psf, *img_shape)

    if boundary_conditions == 'periodic':
        spectra = fp.fft2(np.roll(np.roll(psf_big, -psf_center[1], axis=1), -psf_center[0], axis=0))
        trans = lambda x: fp.fft2(x) / np.sqrt(x.size)
        itrans = lambda x: np.real(fp.ifft2(x) * np.sqrt(x.size))

    elif boundary_conditions == 'reflexive':
        e1 = np.zeros_like(psf_big)
        e1[0, 0] = 1
        spectra = fp.dctn(dctshift(psf_big, psf_center), norm='ortho') / fp.dctn(e1, norm='ortho')
        trans = lambda x: fp.dctn(x, norm='ortho')
        itrans = lambda x: fp.idctn(x, norm='ortho')
    else:
        raise ValueError('Supported boundary conditions (boundary_conditions) are: periodic, reflexive.')

    return spectra, trans, itrans


def dif_map(x, boundary_conditions='periodic'):
    """
    Compute the discrete gradient of an input array in 2D.

    Parameters
    ----------
    x : numpy.ndarray
        The 2D input array.
    boundary_conditions : str, optional
        The boundary conditions to use for the gradient computation.
        Supported options are 'periodic' (default) and 'reflexive'.

    Returns
    -------
    numpy.ndarray
        The discrete gradient of the input array.

    Raises
    ------
    ValueError
        If the specified boundary conditions are not supported.

    """

    if boundary_conditions == 'periodic':
        return np.vstack((np.vstack((x[0:-1, :] - x[1:, :], x[-1:, :] - x[:1, :])),
                          np.column_stack((x[:, 0:-1] - x[:, 1:], x[:, -1:] - x[:, :1]))))
    elif boundary_conditions == 'reflexive':
        return np.vstack((np.vstack((x[0:-1, :] - x[1:, :], np.zeros(x.shape[1]))),
                          np.column_stack((x[:, 0:-1] - x[:, 1:], np.zeros(x.shape[0])))))
    else:
        raise ValueError('Supported boundary conditions are: periodic, reflexive.')


def dif_map_adj(y, boundary_conditions='periodic'):
    """
    Adjoint operator of the discrete gradient with reflexive or periodic boundary conditions.

    Parameters
    ----------
    y : ndarray
        The input gradient vector.
    boundary_conditions : str, optional
        The boundary conditions to use. Valid options are 'periodic' (default) or 'reflexive'.

    Returns
    -------
    ndarray
        The adjoint of the discrete gradient.

    Raises
    ------
    ValueError
        If the specified boundary conditions are not supported.

    """

    m = y.shape[0]//2
    if boundary_conditions == 'periodic':
        return (np.vstack((y[:1, :] - y[(m - 1):m, :], y[1:m, :] - y[0:(m - 1), :]))
                + np.column_stack((y[m:, :1] - y[m:, -1:], y[m:, 1:] - y[m:, 0:-1])))
    elif boundary_conditions == 'reflexive':
        return (np.vstack((y[0, :], y[1:(m - 1), :] - y[0:(m - 2), :], - y[(m-2):(m-1), :]))
                + np.column_stack((y[m:, 0], y[m:, 1:-1] - y[m:, 0:-2], - y[m:, -2:-1])))
    else:
        raise ValueError('Supported boundary conditions are: periodic, reflexive.')
