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

#mp.dps = 25
mp.pretty = True
ALPHA = 100

qq = np.linspace(0.9, 0.99, 10)
qq_mpf = [mpf(str(q)) for q in qq]
mqq_mpf = [mpf('-'+str(q)) for q in qq]
exponent = mpf(str(eval('.5*ALPHA*(ALPHA+1)')))

numerator = map(qp, mqq_mpf)
denominator = map(qp, qq_mpf)
asympt = [q**exponent for q in qq_mpf]

yy = [2*ALPHA*a/b*c for (a,b,c) in zip(numerator, denominator, asympt)]

for q, y in zip(qq_mpf, yy):
	print q, y

def bound(q):
	return log(qp(-q)/qp(q)*q**(.5*ALPHA*(ALPHA+1)))

plot(bound, xlim=[0,1], singularities=[0,1])

