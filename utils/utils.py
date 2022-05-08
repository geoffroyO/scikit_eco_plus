from scikit_eco_plus.utils.kernels import LinearKernel, GaussianKernel, PolynomialKernel, SigmoidKernel, LaplacianKernel


def get_kernel(name, **kwargs):
    if name == 'linear':
        kernel = LinearKernel()

    elif name == 'gaussian':
        l = kwargs.get('l', None)
        if l:
            kernel = GaussianKernel(l)
        else:
            kernel = GaussianKernel()

    elif name == 'laplacian':
        l = kwargs.get('l', None)
        if l:
            kernel = LaplacianKernel(l)
        else:
            kernel = LaplacianKernel()

    elif name == 'poly':
        degree = kwargs.get('degree', None)
        gamma = kwargs.get('gamma', None)
        coef0 = kwargs.get('coef0', None)
        if degree and gamma and coef0:
            kernel = PolynomialKernel(degree, gamma, coef0)
        elif degree and gamma:
            kernel = PolynomialKernel(degree, gamma)
        elif degree:
            kernel = PolynomialKernel(degree)
        elif degree and coef0:
            kernel = PolynomialKernel(degree, coef0=coef0)
        elif gamma and coef0:
            kernel = PolynomialKernel(gamma=gamma, coef0=coef0)
        else:
            kernel = PolynomialKernel()

    elif name == 'sigmoid':
        gamma = kwargs.get('gamma', None)
        coef0 = kwargs.get('coef0', None)
        if gamma and coef0:
            kernel = SigmoidKernel(gamma=gamma, coef0=coef0)
        elif gamma:
            kernel = SigmoidKernel(gamma)
        elif coef0:
            kernel = SigmoidKernel(coef0=coef0)
        else:
            kernel = SigmoidKernel()

    else:
        raise NotImplementedError(
            f"Kernel {name} non implémenté"
        )
    return kernel.kernel
