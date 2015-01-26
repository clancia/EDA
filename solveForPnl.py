#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  solveForPnl.py
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

from mpmath import *
from Pnl import *
import numpy as np
import argparse

prec = np.float64
 
parser = argparse.ArgumentParser(description='Approximate Solver of Pnl Balance Equations')
parser.add_argument('--rho', default=.95, type=float)
parser.add_argument('--q', default=.95, type=float)
parser.add_argument('--amax', default=50, type=int)

args = parser.parse_args()

print 'This is solveForPnl'
print 'rho ::', args.rho
print 'q ::', args.q
print 'alpha_max ::', args.amax, '\n'

print 'Generating the system of linear equations...',
A = computeMatrix(args.amax, args.rho, args.q)
print 'done.'

print 'The norm-1 condition number of the matrix is', np.linalg.cond(A,1), '\n'

b = np.zeros(A.shape[0], dtype=prec)
b[-1] = 1.

print 'Solving the linear system...',
x = np.linalg.solve(A, b)
print 'done.'

mp.dps = 25
mp.pretty = True
aPrioriError = args.amax * exp(log(qp(-args.q))
							- log(qp(args.q))
							+ (args.amax)*log(args.rho) 
							+ kct(args.amax)*log(args.q))

print 'The a priori error on the solution is', aPrioriError, '\n'

JointP = np.zeros((args.amax+1, args.amax+1), dtype=prec)
ofilename = 'approx_Pnl_r%.2f_q%.2f.txt' % (args.rho, args.q)
ofile = open(ofilename, 'w')
ofile.write('# Approximate solution of EDA/D/1 bivariate stationary measure\n')
ofile.write('# P[n][l] = probability of having n customers in queue and l customers late\n\n')
for i in range(len(x)):
    n,l = indextonl(i)
    JointP[n,l] = x[i]
    ofile.write('P[%3d][%3d] :: %.12f\n' % (n, l, x[i]))
print 'The sum of the vector P0l is', sum(JointP[0]), 'and should be', 1.-args.rho, '\n'

print 'The approximate Pnl have been written to output file', ofilename
print 'Completed.'


