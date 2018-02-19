#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  pnl.py
#
#  Copyright 2014 Carlo Lancia <clancia@g6-laptop>
#
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.
#
#

import numpy as np

import numpy.random as npr
#################################
import scipy.linalg as linalg
# If float128 precision is needed
#################################
import matplotlib.pyplot as plt
from matplotlib import cm
from math import sqrt


def binomial(n, k):
        """
        Compute n factorial by an additive method.
        """
        if k > n:
                return 0
        else:
                if k > n-k:
                    k = n-k  # Use symmetry of Pascal's triangle
                thediag = [i+1 for i in range(k+1)]
                for i in range(n-k-1):
                        for j in range(1, k+1):
                                thediag[j] += thediag[j-1]
                return thediag[k]


def bcoeff(j, l, q):
        """
        Return the probability of j successes over l trials
        """
        return binomial(l, j) * (1.0-q)**j * q**(l-j)


def maxtriangno(k):
        """
        Return \max\{j \geq 0 such that {j+1 \choose 2} \leq k\}
        """
        return np.floor(.5 * (-1.0 + np.sqrt(1.0 + 8.0 * k)))


def kct(k):
        """
        Return {k+1 \choose 2}
        """
        return k * (k + 1) / 2


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


def computeMatrix(trunc, rho, q):
        """
        Return the {trunc+2 \choose 2} x {trunc+2 \choose 2} matrix
        that corresponds to the NW truncation of the linear system for the Pnl
        """
        p = 1.0 - q
        sysdim = kct(trunc+1)
        prec = np.float64

        A = -1 * np.identity(sysdim, dtype=prec)
        for h in range(sysdim):
                n, l = indextonl(h)
                for j in range(n+1):
                        k = nltoindex(j+1, l+n-j)
                        if k < sysdim:
                                A[h, k] += (1-rho)*bcoeff(n-j, l+n-j, q)
                        k = nltoindex(j+1, n+l-j-1)
                        if n+l-j > 0 and k < sysdim:
                                A[h, k] += rho*bcoeff(n-j, l+n-j, q)
                k = nltoindex(0, l+n)
                if k < sysdim:
                        A[h, k] += (1-rho)*bcoeff(n, l+n, q)
                k = nltoindex(0, l+n-1)
                if l+n > 0 and k < sysdim:
                        A[h, k] += rho*bcoeff(n, l+n, q)
        A[-1] = np.ones(sysdim, dtype=prec)
        return A


def systemFactory(trunc):
        # Useful wrapper for generating the linear system with fixed truncation
        def sysMatrix(rho, q):
            return computeMatrix(trunc, rho, q)
        return sysMatrix


def solveforPnl(trunc, rho, q):
        sysdim = kct(trunc+1)
        prec = np.float128

        A = computeMatrix(trunc, rho, q)
        b = np.zeros(sysdim, dtype=prec)
        b[-1] = 1.0

        #########################################################
        # Only for debug purpouses
        # det = np.linalg.det(A)
        # if det < 0.1**5:
        #     print("WARNING :: Linear system almost singular")
        #     print("WARNING :: det(A) = {:.6f}".format(det))
        #
        #########################################################
        x = linalg.solve(A, b)

        JointP = np.zeros((trunc+1, trunc+1), dtype=prec)
        for i in range(len(x)):
                n, l = indextonl(i)
                JointP[n, l] = x[i]
        P0 = sum(JointP[0])
        print("Probability of empty queue :: P0 = {:.4f}".format(P0))
        if abs(P0 - 1 + rho)/(1-rho) > 0.1**8:
                print("WARNING :: Little's Law is not satisfied")
                print("WARNING :: P0 = {:.4f} and 1-rho = {:.4f}"
                      .format(P0, 1-rho))
        return JointP


def single_step(position, rho, q):
    if npr.random() < rho:
        arrivals = npr.binomial(position[1]+1, 1.0-q)
        toarrive = position[1] + 1 - arrivals
    else:
        if position[1] == 0:
            arrivals = 0
        else:
            arrivals = npr.binomial(position[1], 1.0-q)
        toarrive = position[1] - arrivals
    newqueue = position[0] + arrivals
    if position[0] != 0:
        newqueue -= 1
    return (newqueue, toarrive)


def distances(series1, series2):
        s1 = series1.shape
        s2 = series2.shape
        maxshape = (max(s1[0], s1[0]), max(s1[1], s2[1]))
        landa = np.zeros(maxshape)
        mu = np.zeros(maxshape)
        landa[:s1[0], :s1[1]] = np.absolute(series1)
        mu[:s2[0], :s2[1]] = np.absolute(series2)
        return (0.5 * np.linalg.norm(landa - mu, 1),
                np.linalg.norm(np.sqrt(landa) - np.sqrt(mu)) / np.sqrt(2))


def simulforPnl(rho, q, tmax):
        maxsize = 0
        freqcounter = {}
        position = (0, 0)
        # trajectory = np.empty((tmax,2), dtype=np.int8)
        for i in range(1, tmax+1):
            position = single_step(position, rho, q)
            f = freqcounter.setdefault(position, 0)
            freqcounter[position] = f+1
            if max(position) > maxsize:
                maxsize = max(position)
                # trajectory[i][0], trajectory[i][1] = position
        jointP = np.zeros((maxsize+1, maxsize+1), dtype=np.float64)
        for (k, v) in freqcounter.iteritems():
            jointP[k[0]][k[1]] = float(v)/tmax
        return jointP

# def plot_as_imgs(series1, series2, rho, q):
#         fig = plt.figure()

#         s1 = series1.shape
#         s2 = series2.shape
#         interp = 'nearest' #nearest, bilinear, bicubic

#         maxshape = (max(s1[0],s2[0]), max(s1[1], s2[1]))

#         ax = fig.add_subplot(121)
#         Z = np.zeros(maxshape)
#         Z[:s1[0], :s1[1]] = series1[:,:]
#         ax.imshow(Z.transpose(), interpolation=interp,
#                   cmap=cm.coolwarm,
#                   origin='lower'
#                   )
#         ax.set_title("Theoretical")
#         ax.set_xlabel("Queue length")
#         ax.set_ylabel("No. of late customers")

#         ax = fig.add_subplot(122)
#         Z = np.zeros(maxshape)
#         Z[:s2[0], :s2[1]] = series2[:,:]
#         ax.imshow(Z.transpose(), interpolation=interp,
#                   cmap=cm.coolwarm,
#                   origin='lower'
#                   )
#         ax.set_title("Empirical")
#         ax.set_xlabel("Queue length")
#         ax.set_ylabel("No. of late customers")

#         #fig.suptitle("Theoretical vs. Empirical equilibrium measure\nrho = %.2f, q = %.2f" % (rho, q), fontsize=18)
#         figname = "./rho%.3f_q%.3f_jointps.png" % (rho, q)
#         plt.tight_layout()
#         plt.savefig(figname)

#         logger.info("Joint measure comparison plotted and saved to file.")


# def comparemarginals(series1, series2, rho, q):
#         s1 = series1.shape
#         s2 = series2.shape
#         s1n = sum(series1.transpose())
#         s1l = sum(series1)
#         s2n = sum(series2.transpose())
#         s2l = sum(series2)
#         maxlenn = max(len(s1n),len(s2n))
#         maxlenl = max(len(s1l),len(s2l))

#         width = 0.25
#         fig = plt.figure()
#         ax = fig.add_subplot(211)
#         ind = np.arange(maxlenn)
#         rects = []
#         z = np.zeros(maxlenn)
#         z[:len(s1n)] = s1n[:]
#         rects.append(ax.bar(ind+width, z, width, color='r'))
#         z = np.zeros(maxlenn)
#         z[:len(s2n)] = s2n[:]
#         rects.append(ax.bar(ind+2*width, z, width, color='y'))
#         ax.set_title("Marginal of the queue length")
#         ax.set_xlabel("Queue length")
#         ax.set_xticks(ind+2*width)
#         ax.set_xticklabels(range(maxlenl))
#         ax.legend([rects[0], rects[1]], ["Theoretical", "Empirical"])

#         ax = fig.add_subplot(212)
#         ind = np.arange(maxlenl)
#         rects = []
#         z = np.zeros(maxlenl)
#         z[:len(s1l)] = s1l[:]
#         rects.append(ax.bar(ind+width, z, width, color='r'))
#         z = np.zeros(maxlenl)
#         z[:len(s2l)] = s2l[:]
#         rects.append(ax.bar(ind+2*width, z, width, color='y'))
#         ax.set_title("Marginal of the queue length")
#         ax.set_xlabel("Queue length")
#         ax.set_xticks(ind+2*width)
#         ax.set_xticklabels(range(maxlenl))
#         ax.legend([rects[0], rects[1]], ["Theoretical", "Empirical"])

#         figname = './rho%.3f_q%.3f_marignals.png' % (rho, q)
#         plt.tight_layout()
#         #fig.suptitle("Comparison between marignals :: rho = %.3f, q = %.3f" % (rho, q), fontsize=18)
#         plt.savefig(figname)

#         logger.info("Marginal measures comparison plotted and saved to file.")

# def main():
#         args = parser.parse_args()
#         rho = args.rho
#         q = args.q
#         trunc = args.trunc
#         tmax = args.tmax
#         logger.info("Arguments parsed :: trunc = %d, rho = %.3f, q = %.3f"
#                     % (trunc, rho, q))
#         Pnl = solveforPnl(trunc, rho, q)
#         sPnl = simulforPnl(rho, q, tmax)
#         plot_as_imgs(Pnl, sPnl, rho, q)
#         comparemarginals(Pnl, sPnl, rho, q)

#         print "\n" + "-"*33
#         print "Theoretical vs. Empirical EDA\n"
#         dist = distances(Pnl, sPnl)
#         print "TotalVar. distance :: %.6f\nHellinger distance :: %.6f" % (dist[0], dist[1])
#         print "-"*33

#         return 0

# if __name__ == '__main__':
# 	main()
