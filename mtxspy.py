#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  mtxspy.py
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

# import numpy as np
import matplotlib.pyplot as plt
from Pnl import computeMatrix

ALPHA = 10
rho = .5
q = .5

A = computeMatrix(ALPHA, rho, q)

plt.figure()
# cmap = mcolors.ListedColormap(['c','r'])
plt.spy(A, cmap=plt.get_cmap('YlOrRd'), alpha=0.9)
plt.title('Non-zero entries of the matrix A (alpha_max = 10)')
plt.show()
