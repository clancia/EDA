import numpy as np
import numpy.random as npr
import scipy.linalg as linalg
import scipy.sparse as sparse
import scipy.stats as sps
import logging
from math import floor, ceil, log, sqrt

logger = logging.getLogger('eda')
logger.setLevel(logging.DEBUG)
# create file handler which logs even debug messages
fh = logging.FileHandler('eda.log')
fh.setLevel(logging.DEBUG)
# create console handler with a higher log level
# ch = logging.StreamHandler()
# ch.setLevel(logging.WARNING)
# create formatter and add it to the handlers
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
# ch.setFormatter(formatter)
# add the handlers to the logger
logger.addHandler(fh)
# logger.addHandler(ch)
logger.debug('Set up')


def isSquare(m):
    # https://stackoverflow.com/questions/22870734/check-if-a-matrix-is-square-python
    return all(len(row) == len(m) for row in m)


def kct(k):
        """
        Return {k+1 \choose 2}
        """
        return k * (k + 1) // 2


def maxtriangno(k):
    """
    Return \max\{j \geq 0 such that {j+1 \choose 2} \leq k\}

    maxtriangno is the inverse of kct
    """
    return int(floor(.5 * (-1.0 + sqrt(1.0 + 8.0 * k))))


def indextonl(k):
        """
        Map an integer k onto a couple (n,l) such that n + l = k + 1
        """
        t = int(maxtriangno(k))
        s = k - kct(t)
        return (t-s, s)


def nltoindex(n, l):
        """
        Map a couple (n,l) onto an integer alpha in such a way
        that n + l = alpha + 1
        """
        k = n+l
        return int(kct(k) + l)


def optimal_trunc(rho, q, tol=10**-9, logger=logger):
    opt = max(10, ceil((log(tol) - log(rho))/log(q)))
    if opt > 100:
        logger.warning('With rho %.2f & q %.2f, ' +
                       'optimal truncation very high %d', rho, q, opt)
    return opt


def computeMatrix(trunc, rho, q, prec=np.float64, tol=10**-9, logger=logger):
        """
        Return the {trunc+2 \choose 2} x {trunc+2 \choose 2} matrix
        that corresponds to the NW truncation of the linear system for the Pnl
        """
        # p = 1.0 - q
        opt = ceil((log(tol) - log(rho))/log(q))
        if trunc < opt:
            logger.warning('Sub-optimal truncation %d | should be %d',
                           trunc, opt)
        sysdim = kct(trunc+1)

        # A = -1 * np.identity(sysdim, dtype=prec)
        logger.debug('Starting to compute transition matrix')
        A = np.zeros((sysdim, sysdim), dtype=prec)
        for nn in range(trunc+1):
            for ll in range(trunc+1-nn):
                hh = nltoindex(nn, ll)
                if nn > 0:
                    # if ll != trunc-nn:
                    rv = sps.binom(ll+1, 1-q)
                    ks = np.array(
                        [nltoindex(nn+a-1, ll-a+1) for a in range(ll+2)])
                    A[hh, ks] += rho * rv.pmf(np.arange(ll+2))
                    rv = sps.binom(ll, 1-q)
                    ks = np.array(
                        [nltoindex(nn+a-1, ll-a) for a in range(ll+1)])
                    A[hh, ks] += (1-rho) * rv.pmf(np.arange(ll+1))
                else:
                    if ll != trunc-nn:
                        rv = sps.binom(ll+1, 1-q)
                        ks = np.array(
                            [nltoindex(a, ll-a+1) for a in range(ll+2)])
                        A[hh, ks] += rho * rv.pmf(np.arange(ll+2))
                    rv = sps.binom(ll, 1-q)
                    ks = np.array(
                        [nltoindex(a, ll-a) for a in range(ll+1)])
                    A[hh, ks] += (1-rho) * rv.pmf(np.arange(ll+1))
        # A[-1] = np.ones(sysdim, dtype=prec)
        logger.debug('Matrix computed')
        return A


def computeMatrix_sp(trunc, rho, q,
                     tol=10**-9, prec=np.float64, logger=logger):
        """
        Return the {trunc+2 \choose 2} x {trunc+2 \choose 2} matrix
        that corresponds to the NW truncation of the linear system for the Pnl
        """
        # p = 1.0 - q
        opt = ceil((log(tol) - log(rho))/log(q))
        if trunc < opt:
            logger.warning('Sub-optimal truncation %d | should be %d',
                           trunc, opt)
        sysdim = kct(trunc+1)

        # A = -1 * np.identity(sysdim, dtype=prec)
        logger.debug('Starting to compute sparse transition matrix')
        A = sparse.dok_matrix((sysdim, sysdim), dtype=prec)
        for nn in range(trunc+1):
            for ll in range(trunc+1-nn):
                hh = nltoindex(nn, ll)
                if nn > 0:
                    rv = sps.binom(ll+1, 1-q)
                    ks = np.array(
                        [nltoindex(nn+a-1, ll-a+1) for a in range(ll+2)])
                    A[hh, ks] += rho * rv.pmf(np.arange(ll+2))
                    rv = sps.binom(ll, 1-q)
                    ks = np.array(
                        [nltoindex(nn+a-1, ll-a) for a in range(ll+1)])
                    A[hh, ks] += (1-rho) * rv.pmf(np.arange(ll+1))
                else:
                    if ll != trunc-nn:
                        # this is to avoid to compute transitions
                        # from (0, trunc) to outside
                        rv = sps.binom(ll+1, 1-q)
                        ks = np.array(
                            [nltoindex(a, ll-a+1) for a in range(ll+2)])
                        A[hh, ks] += rho * rv.pmf(np.arange(ll+2))
                    rv = sps.binom(ll, 1-q)
                    ks = np.array(
                        [nltoindex(a, ll-a) for a in range(ll+1)])
                    A[hh, ks] += (1-rho) * rv.pmf(np.arange(ll+1))
        # A[-1] = np.ones(sysdim, dtype=prec)
        logger.debug('Matrix computed')
        return A


def rq_truncated_system(trunc):
        # Useful wrapper for generating the linear system with fixed truncation
        def sysMatrix(rho, q):
            return computeMatrix(trunc, rho, q)
        return sysMatrix


# def solveforPnl(trunc, rho, q):
#         sysdim = kct(trunc+1)
#         prec = np.float128
#
#         A = computeMatrix(trunc, rho, q)
#         b = np.zeros(sysdim, dtype=prec)
#         b[-1] = 1.0
#
#         #########################################################
#         # Only for debug purpouses
#         # det = np.linalg.det(A)
#         # if det < 0.1**5:
#         #     print("WARNING :: Linear system almost singular")
#         #     print("WARNING :: det(A) = {:.6f}".format(det))
#         #
#         #########################################################
#         x = linalg.solve(A, b)
#
#         JointP = np.zeros((trunc+1, trunc+1), dtype=prec)
#         for i in range(len(x)):
#                 nn, ll = indextonl(i)
#                 JointP[nn, ll] = x[i]
#         P0 = sum(JointP[0])
#         print("Probability of empty queue :: P0 = {:.4f}".format(P0))
#         if abs(P0 - 1 + rho)/(1-rho) > 0.1**8:
#                 print("WARNING :: Little's Law is not satisfied")
#                 print("WARNING :: P0 = {:.4f} and 1-rho = {:.4f}"
#                       .format(P0, 1-rho))
#         return JointP


def solveforPnl(A):
        prec = A.dtype
        sysdim = A.shape[0]
        b = np.zeros(sysdim, dtype=prec)
        b[-1] = 1.0

        x = linalg.solve(A, b)
        k = maxtriangno(sysdim)
        JointP = np.zeros((k, k), dtype=prec)
        for i in range(len(x)):
                nn, ll = indextonl(i)
                JointP[nn, ll] = x[i]
        return JointP


def pad_2sqr_mtrx(mtrxa, mtrxb):
    """
    mtrxa: array of dimensions (n1, m1)
    mtrxb: array of dimensions (n2, m2)
    return mtrxa and mtrxb zero-padded to (N, N)
    where N = max(n1, m1, n2, m2)
    """
    shpa = mtrxa.shape
    shpb = mtrxb.shape
    size = max(*shpa, *shpb)
    pada = np.pad(mtrxa, pad_width=[(0, size-shpa[0]), (0, size-shpa[1])],
                  mode='constant', constant_values=0)
    padb = np.pad(mtrxb, pad_width=[(0, size-shpb[0]), (0, size-shpb[1])],
                  mode='constant', constant_values=0)
    return pada, padb


class EdaD1Queue(object):
    """
    """
    def __init__(self, rho, q):
        """
        """
        self.rho = rho
        self.qqq = q

    def load_apprx_pnl(self, path):
        """
        Load a matrix with approximate stationary distribution
        """
        pass

    def load_simul_pnl(self, path):
        """
        Load a matrix with simulated stationary distribution
        """
        pass


class EdaD1Solver(object):
    """
    """
    def __init__(self, trunc):
        """
        """
        self._alp = trunc
        # self.rqmtrx = rq_truncated_system(trunc)
        self._pnl = {}
        self._AAA = {}
        self.logger = logging.getLogger('eda.solver')
        self.logger.info('Instance created')
        self.rqmtrx = rq_truncated_system(trunc)

    @property
    def alp(self):
        return self._alp

    @alp.setter
    def alp(self, value):
        if value <= 0:
            raise ValueError('truncation must be positive')
        self._alp = value

    def set_pnl(self, rho, qqq, pnl=None):
        if rho <= 0 or rho >= 1:
            raise ValueError('rho must be between 0.0 and 1.0')
        if qqq < 0 or qqq >= 1:
            raise ValueError('q must be between 0.0 and 1.0')
        if pnl is None:
            pnl, r, q = self.calc_pnl(rho, qqq)
        else:
            if not (isSquare(pnl) and type(pnl) is np.ndarray):
                raise ValueError('pnl must be a square 2-D array')
        self._pnl[(rho, qqq)] = pnl

    def get_pnl(self, rho, qqq):
        return self._pnl.get((rho, qqq))

    def calc_pnl(self, rho, qqq, loglevel='INFO'):
        curlev = self.logger.getEffectiveLevel()
        self.logger.setLevel(loglevel)
        if rho <= 0 or rho >= 1:
            raise ValueError('rho must be between 0.0 and 1.0')
        if qqq < 0 or qqq >= 1:
            raise ValueError('q must be between 0.0 and 1.0')
        A = self.get_mtrx(rho, qqq)
        if A is None:
            self.set_mtrx(rho, qqq)
            A = self.get_mtrx(rho, qqq)
        pnl = solveforPnl(A)
        if (pnl < 0).any():
            self.logger.warning('Pnl has negative elements (biggest abs val ' +
                                '%.5e)', np.abs(pnl[pnl < 0]).max())
        p0 = sum(pnl[0])
        if abs(p0 - 1 + rho)/(1-rho) > 0.1**8:
            self.logger.warning("Check on p0 failed :: %.4f | %.5e", 1-rho, p0)
        else:
            self.logger.info('Probability of empty queue :: %.4f', p0)
        self.logger.setLevel(curlev)
        return pnl, rho, qqq

    def dump_pnl(self, path, rho, q):
        """
        Dump a matrix with approximate stationary distribution
        """
        pnl = self._pnl.get((rho, q))
        if pnl is None:
            self.logger.error('The matrix is not computed yet, cannot dump')
        else:
            pnl.dump(path)

    def load_pnl(self, path, rho, q):
        """
        Load a matrix with approximate stationary distribution
        """
        # check if path exists
        try:
            loaded = np.load(path)
        except FileNotFoundError:
            print('The file path does not exist')
        else:
            self._pnl[(rho, q)] = loaded

    def set_mtrx(self, rho, qqq, A=None):
        if rho <= 0 or rho >= 1:
            raise ValueError('rho must be between 0.0 and 1.0')
        if qqq < 0 or qqq >= 1:
            raise ValueError('q must be between 0.0 and 1.0')
        if A is None:
            A = self.calc_mtrx(rho, qqq)
        else:
            if not (isSquare(A) and type(A) is np.ndarray):
                raise ValueError('A must be a square 2-D array')
        self._AAA[(rho, qqq)] = A

    def get_mtrx(self, rho, qqq):
        return self._AAA.get((rho, qqq))

    def calc_mtrx(self, rho, qqq, loglevel='INFO'):
        curlev = self.logger.getEffectiveLevel()
        self.logger.setLevel(loglevel)
        A, r, q = self.rqmtrx(rho, qqq)
        det = linalg.det(A)
        if det < 0.1**5:
            self.logger.warning('Nearly singular system :: det(A) = %.5e', det)
        self.logger.setLevel(curlev)
        return A

    def dump_mtrx(self, path, rho, q):
        """
        Dump a matrix with approximate stationary distribution
        """
        mtrx = self._mtrx.get((rho, q))
        if mtrx is None:
            self.logger.error('The matrix is not computed yet, cannot dump')
        else:
            mtrx.dump(path)

    def load_mtrx(self, path, rho, q):
        """
        Load a matrix with approximate stationary distribution
        """
        # check if path exists
        try:
            loaded = np.load(path)
        except FileNotFoundError:
            print('The file path does not exist')
        else:
            self._mtrx[(rho, q)] = loaded


class EdaD1Simulator(object):
    """
    Simulation of EDA/D/1 queue model
    Parameters:
    rho    - thinning intensity
    q      - probability of a customer being late
    tmax   - number of steps to simulate
    warmup - warmup time (during which evolution is not tracked)
    """
    def __init__(self, rho, q, tmax, warmup=0):
        """
        """
        self._rho = rho
        self._qqq = q
        self._ttt = tmax
        self._tel = 0
        self.trj = -1 * np.ones((2, self.ttt + 1))
        ############################
        # This is for debug purposes
        # npr.seed(2018)
        npr.seed()
        ############################
        self.set_ipos(0, 0)
        if type(warmup) is int:
            self.warmup(warmup)

    @property
    def rho(self):
        return self._rho

    @rho.setter
    def rho(self, value):
        if value <= 0 or value >= 1:
            raise ValueError('rho must be between 0.0 and 1.0')
        self._rho = value

    @property
    def qqq(self):
        return self._qqq

    @qqq.setter
    def qqq(self, value):
        if value < 0 or value >= 1:
            raise ValueError('q must be between 0.0 and 1.0')
        self._qqq = value

    @property
    def ttt(self):
        return self._tmax

    @ttt.setter
    def ttt(self, value):
        if value <= 0:
            raise ValueError('tmax must be positive')
        self._ttt = value

    @property
    def tel(self):
        return self._tel

    @tel.setter
    def tel(self, value):
        if value < 0:
            raise ValueError('Time cannot be negative')
        self._tel = value

    @property
    def pnl(self):
        return self._pnl/self._pnl.sum()

    @pnl.setter
    def pnl(self, mtrx):
        if (mtrx < 0).any():
            raise ValueError('Pnl cannot be negative')
        if np.modf(mtrx)[0].any():
            raise ValueError('Float MATRIX with non zero fractional part')
        # if not isSquare(mtrx):
        #     raise ValueError('Matrix must be square')
        try:
            self.pnl.shape[0]
        except AttributeError:
            self._pnl = mtrx
        else:
            padp, padm = pad_2sqr_mtrx(self._pnl, mtrx)
            self._pnl = padp + padm

    def set_ipos(self, n0, l0):
        """
        Set initial position of the chain
        """
        self.trj[:, 0] = (n0, l0)

    def set_tpos(self, time, queue, toarrive):
        """
        Get position of the chain at a given time
        """
        check = self.timecheck(time)
        if check:
            raise ValueError('non-zero time check: {}'.format(check))
        else:
            self.trj[:, time] = (queue, toarrive)

    def get_tpos(self, time):
        """
        Get position of the chain at a given time
        """
        check = self.timecheck(time)
        if check:
            raise ValueError('non-zero time check: {}'.format(check))
        else:
            return self.trj[:, time]

    def timecheck(self, time):
        """
        Check that time is non-negative integer and that time <= elapsed time
        """
        check = 0
        if type(time) is not int:
            check = -1
        if time < 0:
            check = -2
        if time > self.tel:
            check = -3
        return check

    def get_lpos(self):
        """
        Get last simulated position of the chain
        """
        return self.get_tpos(self.tel)

    def load_pnl(self, path):
        """
        Load a matrix with occupation frequencies in the quarter plane
        This might be useful to split large simulations
        """
        # check if path exists
        try:
            loaded = np.load(path)
        except FileNotFoundError:
            print('The file path does not exist')
        else:
            self.pnl = loaded

    def dump_pnl(self, path):
        """
        Dump a matrix with occupation frequencies in the quarter plane
        """
        try:
            self.pnl.shape
        except AttributeError:
            print('Attempting to dump Pnl but it is not created yet')
        else:
            self._pnl.dump(path)

    def bin_trajectory(self):
        """
        Make frequency count of occupation in the quarter plane
        out of the simulated trajectory
        """
        bins = self.trj.max() + 1  # number of bins needed
        mtrx, foo, bar = np.histogram2d(self.trj[0, :],
                                        self.trj[1, :],
                                        bins=np.arange(bins + 1))
        # bins + 1 is needed to obtain the correct bin edges
        return mtrx

    def do_step(self, state=None):
        """
        state must be a tuple
        """
        if state is None:
            queue, toarrive = self.get_lpos()
        else:
            queue, toarrive = state
        if npr.random() < self.rho:
            arrivals = npr.binomial(toarrive + 1, 1.0 - self.qqq)
            toarrive = toarrive + 1 - arrivals
        else:
            if toarrive == 0:
                arrivals = 0
            else:
                arrivals = npr.binomial(toarrive, 1.0 - self.qqq)
            toarrive = toarrive - arrivals
        queue = queue + arrivals - (1 if queue > 0 else 0)
        return (queue, toarrive)

    def simulate_chain(self):
        """
        Simulate the chain recording the trajectory in the quarter plane
        """
        for t in range(self.ttt):
            queue, toarrive = self.do_step()
            self.tel = t + 1
            self.set_tpos(t + 1, queue, toarrive)

    def warmup(self, twu):
        """
        Simulate the chain without recording the trajectory
        in the quarter plane
        Set the initial state to the last position of the warm-pu phase
        """
        queue, toarrive = self.get_tpos(0)
        for t in range(twu):
            queue, toarrive = self.do_step((queue, toarrive))
        self.set_ipos(queue, toarrive)
