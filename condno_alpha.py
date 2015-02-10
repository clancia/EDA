#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  condno_alpha.py
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
import itertools
from pnl import computeMatrix
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

N = 4
R = .95
rho = np.tile([R], N)
qqq = np.linspace(.0, .95, N)
colors = itertools.cycle(('c', 'y', 'm', 'k')) 
markers = itertools.cycle('<h>*')
alphas = np.arange(10, 101, 5)

condNo = np.empty((N, len(alphas)))

for i, r, q in zip(range(N), rho, qqq):
	for j in range(len(alphas)):
		A = Pnl.computeMatrix(alphas[j], r, q)
		condNo[i,j] = np.linalg.cond(A, 1)

### SemiLog-y Figure ###

plt.figure()
for i, m, c, q in zip(range(N), markers, colors, qqq):
	plt.plot(alphas, condNo[i, :], linestyle='', marker=m, color=c, markersize=10, label='q=%.2f' % (q))

plt.title('EDA linear system for Pnl')
plt.xlabel('Truncation (alpha_max)')
plt.ylabel('Condition number')
plt.yscale('log')
plt.grid(True)
plt.legend(loc='lower right')
plt.savefig('./figures/CondNo.semilogy.alpha100.png')

### Log-Log Figure ###

plt.figure()
for i, m, c, q in zip(range(N), markers, colors, qqq):
	plt.plot(alphas, condNo[i, :], linestyle='', marker=m, color=c, markersize=10, label='q=%.2f' % (q))

xd,yd = np.log(alphas), np.log(condNo[0, :])
slope, intercept = np.polyfit(xd, yd, 1)
yfit = np.e**( slope*xd + intercept )
plt.plot(alphas, yfit, 'k:', label='e^[%.2f*log(alpha) + %.2f]' % (slope, intercept))

#plt.title('EDA linear system for Pnl')
plt.xlabel('Truncation (alpha_max)')
plt.xscale('log')
plt.ylabel('Condition number')
plt.yscale('log')
#plt.grid(True)
plt.legend(loc='lower right')
plt.savefig('./figures/CondNo.loglog.alpha100.png')
