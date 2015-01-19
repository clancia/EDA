#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  condNumberAlpha.py
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
from Pnl import computeMatrix, binomial
import matplotlib.pyplot as plt

rho = [.95, .95, .95]
qqq = [.05, .5, .95]
col = ['r', 'g', 'b']
linestyles = ['-', '--', ':']
alphas = np.arange(0, 101, 10)

condNo = np.empty((3, len(alphas)))

for i, r, q in zip(range(3), rho, qqq):
	for j in range(len(alphas)):
		A = computeMatrix(alphas[j], r, q)
		condNo[i,j] = np.linalg.cond(A, 1)

plt.figure()
for i, l, c, r, q in zip(range(3), linestyles, col, rho, qqq):
	plt.plot(alphas, condNo[i, :], l, color=c, label='rho=%.2f, q=%.2f' % (r, q))

#xd,yd = np.log(alphas[1:]), np.log(condNo[0, 1:])
#slope, intercept = np.polyfit(xd, yd, 1)
#yfit = np.e**( slope*xd + intercept )
#plt.plot(alphas[1:], yfit, 'k:', label='e^[%.3f*log(alpha) + %.2f]' % (slope, intercept))

plt.title('EDA linear system for Pnl')
plt.xlabel('Truncation (alpha_max)')
#plt.xlabel('Log of truncation (log alpha_max)')
#plt.xscale('log')
plt.ylabel('Log of condition number')
plt.yscale('log')
plt.grid(True)
plt.legend(loc='center right')
plt.show()