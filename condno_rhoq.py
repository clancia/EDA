#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  condno_rhoq.py
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
from pnl import systemFactory, binomial
import matplotlib.pyplot as plt
#from matplotlib.ticker import MaxNLocator
from matplotlib import colors, ticker, cm
from datetime import datetime

ALPHA = 100
mtxComputer = systemFactory(ALPHA)

dr, dq = 0.04125, 0.04125
steps = np.linspace(0, 0.99, 25)

q, r = np.meshgrid(np.array(steps), np.array(steps))
z = np.empty(r.size).reshape(r.shape)

ofile = open('./figures/eda.log', 'w', 0)
ofile.write('%s :: Computation has started\n' % (datetime.now()))


for i in range(r.shape[0]):
	for j in range(r.shape[1]):
		#print 'Using rho =', r[i,j], 'and q =', q[i,j]
		z[i,j] = np.linalg.cond(mtxComputer(r[i,j], q[i,j]),1)
		ofile.write('%s :: Completed rho=%.5f, q=%.5f\n' % (datetime.now(), r[i,j], q[i,j]))

z = np.ma.masked_where(z<= 0, z)

levs = np.logspace(np.floor(np.log10(z.min())),
                       np.ceil(np.log10(z.max())), 20)
lev_exp = np.log10(levs)
plt.contourf(r + dr / 2., q + dq / 2., z, levs, norm=colors.LogNorm(), cmap=cm.Greys_r)
cbar = plt.colorbar(ticks=levs)
cbar.ax.set_yticklabels(['%.2f' % (l) for l in lev_exp])
plt.xlabel('rho')
plt.ylabel('q')
plt.title('Log10 of condition number (alpha_max = %d)' % (ALPHA))

plt.savefig('CondNo.alpha.100.png')
plt.savefig('/home/clancia/Dropbox/EDA/CondNo.alpha.100.png')
ofile.write('%s :: Computation completed\n' % (datetime.now()))

