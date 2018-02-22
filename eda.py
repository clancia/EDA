import numpy as np
import numpy.random as npr


def isSquare(m):
    # https://stackoverflow.com/questions/22870734/check-if-a-matrix-is-square-python
    return all(len(row) == len(m) for row in m)


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
    def __init__(self, rho, q, trunc):
        """
        """
        self.rho = rho
        self.qqq = q
        self.trc = trunc
        # self.alp = max alpha

    # def load_apprx_pnl(self, path):
    #     """
    #     Load a matrix with approximate stationary distribution
    #     """
    #     pass

    def dump_apprx_pnl(self, path):
        """
        Dump a matrix with approximate stationary distribution
        """
        pass


class EdaD1Simulator(object):
    """
    """
    def __init__(self, rho, q, tmax, warmup=0):
        """
        """
        self.rho = rho
        self.qqq = q
        self.ttt = tmax
        self.trj = -1 * np.ones((2, self.ttt + 1))
        self.time_elapsed = 0
        ############################
        # This is for debug purposes
        npr.seed(2018)
        ############################
        self.set_ipos(0, 0)
        self.warmup(warmup)

    @property
    def time_elapsed(self):
        return self._time_elapsed

    @time_elapsed.setter
    def time_elapsed(self, value):
        if value < 0:
            raise ValueError('Time cannot be negative')
        self._time_elapsed = value

    @property
    def pnl(self):
        return self._pnl/self._pnl.sum()

    @pnl.setter
    def pnl(self, mtrx):
        if (mtrx < 0).any():
            raise ValueError('Pnl cannot be negative')
        if np.modf(mtrx).any():
            raise ValueError('Float MATRIX with non zero fractional part')
        if not isSquare(mtrx):
            raise ValueError('Matrix must be square')
        sizem = mtrx.shape[0]
        try:
            sizep = self.pnl.shape[0]
        except AttributeError:
            self._pnl = mtrx
        else:
            size = max(sizep, sizem)
            padp = np.pad(self._pnl, (0, size-sizep), 'constant',
                          constant_value=0)
            padm = np.pad(self._pnl, (0, size-sizem), 'constant',
                          constant_value=0)
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
        if time > self.time_elapsed:
            check = -3
        return check

    def get_lpos(self):
        """
        Get last simulated position of the chain
        """
        return self.get_tpos(self.time_elapsed)

    def load_simul_pnl(self, path):
        """
        Load a matrix with occupation frequencies in the quarter plane
        This might be useful to split large simulations
        """
        pass

    def dump_simul_pnl(self, path):
        """
        Dump a matrix with occupation frequencies in the quarter plane
        """
        pass

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
            self.time_elapsed = t + 1
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
