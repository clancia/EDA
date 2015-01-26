#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  plotBound.py
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
import numpy as np
import matplotlib.pyplot as plt

mp.dps = 25
mp.pretty = True
ALPHA = 40

qq = np.linspace(0.5, 0.95, 45)

numerator = np.array(map(qp, -qq), dtype=np.float128)
denominator = np.array(map(qp, qq), dtype=np.float128)
asympt = np.array(map(lambda q: q**(.5*ALPHA*(ALPHA-1)), qq), dtype=np.float128)

values = np.exp(np.log(numerator) + np.log(asympt) - np.log(denominator))

plt.figure()
plt.plot(qq, values, color='r', marker='h')
plt.xlabel('Value of q')
plt.ylabel('(-q;q)_\infty / (q;q)_\infty * q^{alpha \choose 2}')
plt.yscale('log')
plt.title('A priori error on Pnl as a function of q (alpha_max = 25)')
plt.grid(True)
plt.show()

