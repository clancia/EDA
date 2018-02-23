import numpy as np
from math import exp, log, sin, cos, pi
from scipy.special import factorial
from scipy.optimize import fsolve


def piPois_stupid(n, rho):
    if n == 0:
        return 1 - rho
    elif n == 1:
        return (1 - rho) * (exp(rho) - 1)
    else:
        f = md1aux(n, rho)
        foo = np.array(list(map(f, range(1, n))))
        alt = (-np.ones(n-1))**np.arange(n-1, 0, -1)
        bar = np.exp(rho * np.arange(1, n))
        return (1 - rho) * (exp(n*rho) + (bar*alt*foo).sum())


def md1terma(k, n, landa):
    return (k * landa)**(n-k) / factorial(n-k)


def md1termb(k, n, landa):
    return (k * landa)**(n-k-1) / factorial(n-k-1)


def md1aux(n, landa):
    def f(k):
        return md1terma(k, n, landa) + md1termb(k, n, landa)
    return f


def piPois(n, rho):
    # http://kashiwa.nagaokaut.ac.jp/members/nakagawa/ronbun/029.pdf
    pass


def equations(landa):
    def inner(p):
        x, y = p
        fx = x - exp(landa * (x-1)) * cos(landa * y)
        fy = y - exp(landa * (x-1)) * sin(landa * y)
        return (fx, fy)
    return inner


def basisroot(k, landa):
    return (2 * pi * k + pi / 2) / landa


def inity(k, landa):
    return basisroot(k, landa)


def initx(k, landa):
    return 1 / landa * log(basisroot(k, landa)) + 1


def get_first_n_roots(n, landa):
    def foo(x):
        return exp(landa * (x - 1)) - x
    i = 1
    x = 1
    while abs(x-1) < 10**-3:
        x, = fsolve(foo, i)
        i += 1
    zeta = [(x, 0.)]
    f = equations(landa)
    for i in range(1, n+1):
        ix = initx(i, landa)
        iy = inity(i, landa)
        x, y = fsolve(f, (ix, iy))
        zeta.append((x, y))
    return zeta
