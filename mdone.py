import numpy as np
from math import exp, log, sin, cos, pi
from scipy.special import factorial
from scipy.optimize import fsolve


class MD1(object):
    """
    """
    def __init__(self, rho):
        self._rho = rho
        self._fnag = np.vectorize(lambda z: (z - 1)/(self.rho * z - 1))

    @property
    def rho(self):
        return self._rho

    @rho.setter
    def rho(self, value):
        if value <= 0 or value >= 1:
            raise ValueError('rho must be between 0.0 and 1.0')
        self._rho = value

    def pi_n_stupid(self, n):
        return piPois_stupid(n, self.rho)

    def pi_stupid(self, maxn):
        pi = np.array([self.pi_n_stupid(i) for i in range(maxn+1)])
        return pi

    def set_roots(self, nroots):
        self.zeta = get_first_n_roots(nroots, self.rho)

    def nakagawa(self, n):
        # http://kashiwa.nagaokaut.ac.jp/members/nakagawa/ronbun/029.pdf
        # terms2sum = (self._fnag(self.zeta) * self.zeta**(-n)).real
        terms2sum = (self._fnag(self.zeta) *
                     np.exp(-n * np.log(self.zeta))).real
        return (1 - self.rho) * (terms2sum[0] + 2 * sum(terms2sum[1:]))

    def pi_n(self, n):
        if n == 0:
            return 1 - self.rho
        elif n == 1:
            return (1 - self.rho) * (exp(self.rho) - 1)
        else:
            return self.nakagawa(n)

    def pi(self, maxn):
        pi = np.array([self.pi_n(i) for i in range(maxn+1)])
        return pi


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
    j = 1j  # sqrt(-1)

    def foo(x):
        return exp(landa * (x - 1)) - x

    i = 1
    x = 1
    while abs(x-1) < 10**-3:
        x, = fsolve(foo, i)
        i += 1
    zeta = [x + 0j]
    f = equations(landa)
    for i in range(1, n+1):
        ix = initx(i, landa)
        iy = inity(i, landa)
        x, y = fsolve(f, (ix, iy))
        zeta.append(x + j*y)
    return np.array(zeta)


def test():
    import matplotlib.pyplot as plt
    landa = 0.50
    n = 50
    queue = MD1(landa)
    queue.set_roots(100)
    brute = queue.pi_stupid(n)
    smart = queue.pi(n)
    plt.plot(brute, '.', label="Taylor's expansion")
    plt.plot(smart, '.', label="Nakagawa")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    test()
