#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  condNumberRhoQ.py
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
from Pnl import systemFactory, binomial
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

ALPHA = 100
mtxComputer = systemFactory(ALPHA)

dr, dq = 0.05, 0.05

q, r = np.mgrid[slice(0.05, 1, dq),
                slice(0.05, 1, dr)]
z = np.empty(r.size).reshape(r.shape)


for i in range(r.shape[0]):
	for j in range(r.shape[1]):
		print 'Using rho =', r[i,j], 'and q =', q[i,j]
		z[i,j] = np.linalg.cond(mtxComputer(r[i,j], q[i,j]),1)

cmap = plt.get_cmap('PiYG')
levels = MaxNLocator(nbins=25).tick_values(z.min(), z.max())

plt.figure()

plt.contourf(r + dr / 2., q + dq / 2., z, levels=levels, cmap=cmap)
plt.colorbar()
plt.xlabel('rho')
plt.ylabel('q')
plt.title('Conditioning number (alpha_max = %d)' % (ALPHA))


plt.show()

