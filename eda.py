import numpy as np
import numpy.random as npr


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
    def __init__(self, rho, q, tmax):
        """
        """
        self.rho = rho
        self.qqq = q
        self.ttt = tmax
        self.trj = -1 * np.ones((2, self.ttt))
        self.time_elapsed = 0
        ############################
        # This is for debug purposes
        npr.seed()
        ############################

    @property
    def time_elapsed(self):
        return self._time_elapsed

    @time_elapsed.setter
    def time_elapsed(self, value):
        if value < 0:
            raise ValueError('Time cannot be negative')
        self._time_elapsed = value

    def set_ipos(self, n0, l0):
        """
        Set initial position of the chain
        """
        self.trj[:, 0] = (n0, l0)

    def get_tpos(self, time):
        """
        Get position of the chain at a given time
        """
        # check that time is integer and that time <= elapsed time
        return self.trj[:, time]

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

    def normalize_pnl(self):
        """
        Normalize occupation frequencies in the quarter plane and return
        simulated stationary probabilitites
        """
        pass

    # def single_step(self, position, rho, q):
    #     if npr.random() < rho:
    #         arrivals = npr.binomial(position[1]+1, 1.0-q)
    #         toarrive = position[1] + 1 - arrivals
    #     else:
    #         if position[1] == 0:
    #             arrivals = 0
    #         else:
    #             arrivals = npr.binomial(position[1], 1.0-q)
    #         toarrive = position[1] - arrivals
    #     newqueue = position[0] + arrivals
    #     if position[0] != 0:
    #         newqueue -= 1
    #     return (newqueue, toarrive)
