# Software Package for the Maximum Entropy on the Mean (MEM) Method

## Introduction
The maximum entropy on the mean (MEM) is an information driven statistical estimation method. In its core lies the MEM function defined as

$$ \kappa_P(y):=\inf \left\{\text{KL}(Q, P): E_Q=y, Q\ll P \right\},$$

where $P$ is a given reference distribution, $\text{KL}(Q,P)$ is the Kullback-Leibler divergence of a distribution $Q$ with respect to $P$, $E_Q$ stands for the mathematical expectation of $Q$ and finally $Q\ll P$ indicates that $Q$ is absolutely continuous with respect to $P$. Under some conditions, the function $\kappa_P$ coincides with the Cramér rate function which admits a closed form or computationally tractable expression for many popular choices of reference distribution $P$.

The goal of this notebook is to provide a documentation for memmpy, a python package supplementing the paper:
<br>
<center style="font-size:14px"> <em> Maximum Entropy on the Mean and the Cramér Rate Function in Statistical Estimation and Inverse Problems: Properties, Models and Algorithms
by Yakov Vaisbourd, Rustum Choksi, Ariel Goodwin, Tim Hoheisel, and Carola-Bibiane Schönlieb</em></center>


Additional details, proofs and examples can be found in the above reference and the supplemental material file that is available with this package.


This package contains an implementation of the Cramér rate function and Bregman proximal operators for many well known reference distributions. It supports the three linear models discussed in the paper, i.e., the linear models that correspond to the normal, Poisson and Gamma (with parameter $\beta=1$) distributions.
Furthermore, this package includes an implementation of some popular first-order optimization algorithms which employ the operators. Some examples for image processing applications are included as well.


The package is licensed under the BSD-style license found in the LICENSE file in the root directory of this source tree.


## Documentation

The memmpy package contains the following files:

- dist.py - implementation Cramér rate function and Bregman proximal operators for the supported reference distributions (see table below).
- methods.py - implementation of first order optimization algorithms.
- misc.py - miscellaneous utility routines that support the main operators implemented in this package.

Additional files distributed with the package are:
- imgproc.py - implementation of supplementary image processing routines used in order to demonstrate some practical use cases of the MEM framework.
- memmpy_sup.pdf - supplementary document that contains a detailed derivation of the Bregman proximal operators implemented in this package.

The package supports the following distributions:

<center style="font-size:14px">

|             Distribution             |    Class name    | Parameters                                                                                                                  |           Expected value           | Comments                                                                                                                        |
|:------------------------------------:|:----------------:|:----------------------------------------------------------------------------------------------------------------------------|:----------------------------------:|:--------------------------------------------------------------------------------------------------------------------------------|
|                Normal                |      normal      | $\mu$ - location <br/> $\sigma$ - scale                                                                                     |               $\mu$                | $\sigma>0$                                                                                                                      |
|         Multivariate Normal          |    multnormal    | $\mu$ - location <br/> $\Sigma$ - covariance matrix                                                                         |               $\mu$                | $\Sigma$ - positive definite                                                                                                    |
|              Bernoulli               |    bernoulli     | $p$ - success probability                                                                                                   |                $p$                 | $p\in[0, 1]$                                                                                                                    |
|       Normal-inverse Gaussian        |   norminvgauss   | $\mu$ - location <br/>  $\alpha$ - shape <br/> $\beta$ - skewness <br/> $\delta$ - scale                                    |     $\mu + \delta\beta/\gamma$     | $\gamma:=\sqrt{\alpha^2 - \beta^2}$, $\alpha>                                                                                   |\beta|$, $\delta>0$   |
| Multivariate Normal-inverse Gaussian | multnorminvgauss | $\mu$ - location <br/>  $\alpha$ - shape <br/> $\beta$ - skewness <br/> $\delta$ - scale <br/> $\Sigma$ - covariance matrix | $\mu + (\delta/\gamma)\Sigma\beta$ | $\gamma:=\sqrt{\alpha^2 - \beta^T\Sigma\beta}$, $\alpha^2 > \beta^T\Sigma\beta$, <br/> $\delta>0$, $\Sigma$ - positive definite |
|                Gamma                 |      gamma       | $\alpha$ - shape  <pr/>  $\beta$ - rate                                                                                     |           $\alpha/\beta$           | $\alpha > 0$, $\beta > 0$                                                                                                       |
|               Poisson                |     poisson      | $\lambda$ - rate                                                                                                            |             $\lambda$              | $\lambda>0$                                                                                                                     |
|           Discrete Uniform           |      unifdsc       | $a$ - lower bound  <pr/>  $b$ - upper bound                                                                                 |             $(a+b)/2$              | $a < b$                                                                                                                         |
|          Continuous Uniform          |     unifcnt      | $a$ - lower bound  <pr/>  $b$ - upper bound                                                                                                            |             $(a+b)/2$              | $a < b$                                                                                                                     |
</center>

Some remarks:
- Most parameters are assumed to be numpy arrays (valid inputs also include: scalars, tuples and lists). The parameters must be of compatible sizes, for example, for the normal distribution $\mu$ and $\sigma$ must have the same size (shape).
- Some distributions are defined by means of a covariance matrix. There is no need to provide the matrix itself, instead one should define the following operators (functions);
    - Multivariate normal distribution:
        - cov_mat(x) - an operator that yields the product $\Sigma x$.
        - cov_mat_inv(x) - an operator that yields the product $\Sigma^{-1}x$.
        - res_cov_mat(x, rho) - an operator that yields a product with the resolvent of the covariance matrix $(\rho I +\Sigma)^{-1}x$.
    - Multivariate normal-inverse Gaussian distribution:
        - cov_mat(x) - an operator that yields the product $\Sigma x$.
        - cov_mat_inv(x) - an operator that yields the product $\Sigma^{-1}x$.
        - res_cov_mat_inv(x, rho) - an operator that yields a product with the resolvent of the covariance inverse matrix $(\rho^{-1} I +\Sigma^{-1})^{-1}x$.

Each distribution supports the following functions
- freeze - freeze the distribution for the given parameters (arguments: distribution parameters).
- Cramer - retrieve the value of the Cramér rate function at a given point (arguments: distribution parameters and point for evaluation).
- bregman_prox_gen - retrieve a function defining the Bregman proximal operator (arguments: distribution parameters and reference distribution of the linear model). The resulting function computes the Bregman proximal operator at a given point under a predefined positive step size.

For the Bregman proximal operator, the package supports the following linear models: Normal, Gamma, and Poisson.
Multivariate Gaussian and Multivariate Normal-inverse Gaussian distributions currently support only the Normal linear model.

As an illustration, let us consider the normal distribution.

```python
import numpy as np
from memmpy.dist import normal

# The value of Cramer rate function with normal reference distribution with parameters (mu=0, sigma=1) at point x=2
print('Result: ', normal.cramer(2, mu=0, sigma=1))

# In order to provide additional details about each function we can use help. For example:
help(normal.cramer)

# We can use the freeze function to avoid repeated specification of the arguments
normal.freeze(mu=0, sigma=1)
print('Result (after freeze): ', normal.cramer(2))

# We can consider examples with higher dimensions
normal.freeze(mu=np.zeros(5), sigma=np.ones(5))
x = np.random.rand(5)
print('Result (higher dimension): ', normal.cramer(x))

# Retrieving the Bregman proximal operator that correspond to the Gamma linear model is as follows
breg_prox = normal.bregman_prox_gen(kernel='Gamma')

# Now we can use this function to compute the Bregman proximal operator with some step-size (t) at a given point
print('Result (Bregman proximal operator): ', breg_prox(x, t=1))

```

For the multivariate normal distribution we must provide the operators defining the mappings associated with the covariance matrix. This allows an efficient implementation when the covariance matrix admits some structure. Below we illustrate how to define these mappings on a simple small dimensional example.

```python
import numpy as np
from memmpy.dist import multnormal

N = 10
A = np.random.rand(2*N, N)
A = A.transpose().dot(A)

multnormal.freeze(np.zeros(N), lambda x: A.dot(x), lambda x: np.linalg.inv(A).dot(x), lambda x, rho: np.linalg.inv(rho+A).dot(x))

print('Result: ', multnormal.cramer(np.random.rand(N)))

```

One can generate the plot of the Cramér rate function as follows:

```python
import matplotlib.pyplot as plt
from memmpy.dist import bernoulli


# We will use the Bernoulli distribution with probability of success 1/3
bernoulli.freeze(p=1/3)

# Define plot parameters
N, plt_lb, plt_ub = 1000, 0, 1
x = np.linspace(plt_lb, plt_ub, N)

# Compute Cramer rate function
f = np.squeeze(np.array([bernoulli.cramer(xx, entry_wise=True) for xx in x]))

# Produce plot
plt.plot(x, f)
plt.title('Cramer function: Bernoulli distribution')
plt.xlim(plt_lb, plt_ub)
plt.show()
```

![img.png](images\cramer_bernoulli.png)

## Examples

We illustrate the use of the memmpy package for some classical image processing applications.

### Barcode Image Deblurring

We consider the problem of image deblurring of a QR code image. Due to its binary nature we will employ a MEM function induced by a Bernoulli reference distribution for the regularization term.

```python
import numpy as np
import matplotlib.pyplot as plt
import qrcode as qr
import memmpy.imgproc as ip
import memmpy.methods as mtd
from memmpy.dist import normal
from memmpy.dist import bernoulli


# Create a QR code image
qr_gen = qr.QRCode(error_correction=qr.constants.ERROR_CORRECT_H,  box_size=10)
qr_gen.add_data('www.google.com')
img = np.array(qr_gen.make_image().get_image(), dtype='float')

# Add blurring affect
psf, center = ip.psfGauss([8, 8], 4)
spectra, trans, itrans = ip.spectral_decomposition_gen(psf, center, img.shape, 'periodic')
img_obs = itrans(np.multiply(spectra, trans(img)))

# Add noise
noise = np.random.normal(0, 1, img.shape)
img_obs += 0.01 * np.linalg.norm(img_obs, 'fro') * noise/np.linalg.norm(noise)

# Define model functions - fidelity function - f
reg_par = 1e-5 # Set regularization parameter
observed_signal_freq = trans(img_obs)

def f_residual(x):
    return np.multiply(spectra, trans(x)) - observed_signal_freq

def f_val(x, is_residual=False):
    if is_residual:
        return 0.5 * np.linalg.norm(x) ** 2
    else:
        return 0.5 * np.linalg.norm(np.multiply(spectra, trans(x)) - observed_signal_freq) ** 2

def f_grad(x, is_residual=False):
    if is_residual:
        return itrans(np.multiply(spectra.conj(), x))
    else:
        return itrans(np.multiply(spectra.conj(), np.multiply(spectra, trans(x)) - observed_signal_freq))

f = mtd.ObjectiveFunction(val=f_val, grad=f_grad, residual=f_residual,
                          kernel=mtd.KernelFunction(normal.kernel_val, normal.kernel_grad, normal.kernel_grad_dual, is_normal=True),
                          smoothness_constant=np.max(np.max(np.abs(spectra) ** 2)))

# Define model functions - regularization function - g
bernoulli.freeze(np.full_like(img_obs, 0.5))
g = mtd.ObjectiveFunction(val=lambda x: reg_par * bernoulli.cramer(x),
                          proximal_operator=lambda x, t: bernoulli.bregman_prox_gen('Normal')(x, reg_par * t))

# Define model
data = mtd.ProblemData(f, g, img, img_obs)

# Solve
pars = mtd.Parameters(initial_point=img_obs, max_iter=50, is_fast=True)
res = mtd.bpg(data, pars)

# Plot the original, blurred and noisy, and reconstructed images
plt.gray()
fig = plt.figure(figsize=(20,6), facecolor="white")
ax1 = fig.add_subplot(131)
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133)

ax1.imshow(img)
ax1.axis('off')
ax1.title.set_text('Original image')
ax2.imshow(img_obs)
ax2.axis('off')
ax2.title.set_text('Blurred and noisy image')
ax3.imshow(res.opt_sol)
ax3.axis('off')
ax3.title.set_text('Reconstructed image')

fig.subplots_adjust(wspace=0.3)
plt.show()

print('Objective Value')
plt.plot(res.obj_val)
plt.show()

```
The barcode images:

![img_1.png](images\barcode_img.png)

Plot of objective value at each iteration:

![img_2.png](images\barcode_obj.png)

### Natural Image Deblurring

Similarly, we can consider the problem of image deblurring of a natural image. Here we will employ the normal inverse Gaussian reference distribution for the regularization term.

```python
import numpy as np
import memmpy.methods as mtd
import matplotlib.pyplot as plt
import memmpy.imgproc as ip
from scipy.misc import ascent
from memmpy.dist import norminvgauss

# Create an image
img = ascent()
img = img/np.max(img)

# Add blurring affect
psf, center = ip.psfGauss([6, 6], 3)
spectra, trans, itrans = ip.spectral_decomposition_gen(psf, center, img.shape, 'periodic')
img_obs = itrans(np.multiply(spectra, trans(img)))

# Add noise
noise = np.random.normal(0, 1, img.shape)
img_obs += 0.01 * np.linalg.norm(img_obs, 'fro') * noise / np.linalg.norm(noise)

# Define model functions - fidelity function - f
reg_par = 1e-12 # Set regularization parameter
Atrans_img_obs = itrans(np.multiply(spectra.conj(), trans(img_obs)))
observed_signal_freq = trans(img_obs)

def f_val(x):
    return (1 / 2) * np.linalg.norm(np.multiply(spectra, trans(x)) - observed_signal_freq) ** 2

def f_proximal_operator(x, step_size):
    return itrans(np.multiply(1 / (1 + step_size * np.multiply(spectra.conj(), spectra)),
                              trans(x + step_size * Atrans_img_obs)))

f = mtd.ObjectiveFunction(val=f_val, proximal_operator=f_proximal_operator)

# Define model functions - regularization function - g
par_shape = (2 * img.shape[0], img.shape[1])
norminvgauss.freeze(alpha=np.ones(par_shape), beta=np.zeros(par_shape),
                    mu=np.zeros(par_shape), delta=1e-10*np.ones(par_shape))

g = mtd.ObjectiveFunction(val=lambda x: reg_par * norminvgauss.cramer(x),
                          proximal_operator= lambda x, t: norminvgauss.bregman_prox_gen('Normal')(x, reg_par * t))

# Define the linear mapping - L
L = mtd.LinearMap(lambda x: ip.dif_map(x, 'periodic'), lambda y: ip.dif_map_adj(y, 'periodic'), 3) # 3/reg_par)

# Define model entity
data = mtd.ProblemData(f, g, L, img, img_obs)

# Solve
pars = mtd.Parameters(initial_point=img_obs, initial_point_dual=ip.dif_map(img_obs), max_iter=50, verbose=False)
res = mtd.cp(data, pars)

# Plot the original, blurred and noisy, and reconstructed images
plt.gray()
fig = plt.figure(figsize=(20,6), facecolor="white")
ax1 = fig.add_subplot(131)
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133)

ax1.imshow(img)
ax1.axis('off')
ax1.title.set_text('Original image')
ax2.imshow(img_obs)
ax2.axis('off')
ax2.title.set_text('Blurred and noisy image')
ax3.imshow(res.opt_sol)
ax3.axis('off')
ax3.title.set_text('Reconstructed image')

fig.subplots_adjust(wspace=-0.2)
plt.show()

print('Objective Value')
plt.plot(res.obj_val)
plt.show()
```
The images:

![img_3.png](images\natural_img.png)

Plot of objective value at each iteration:

![img_4.png](images\natural_obj.png)