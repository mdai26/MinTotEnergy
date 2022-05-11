# -*- coding: utf-8 -*-
"""
Created on Mon May  9 14:17:26 2022

@author: daimi
"""

import numpy as np
import sympy as sp
from sympy.utilities.lambdify import lambdify
from scipy.optimize import brute
import pandas as pd

# specify the parameters
# elastic modulus
c11, c12, c44 = sp.symbols('c11, c12, c44')
# anisotropy constant
K1, K2 = sp.symbols('K1, K2')
# magnetoelastic constant
B1, B2 = sp.symbols('B1, B2')
# strain value
e11, e22 = sp.symbols('e11, e22')
# magnetization vector
theta, phi = sp.symbols('theta, phi')

# get m1, m2 and m3
m1 = sp.cos(phi/sp.Rational(180)*sp.pi)*sp.sin(theta/sp.Rational(180)*sp.pi)
m2 = sp.sin(phi/sp.Rational(180)*sp.pi)*sp.sin(theta/sp.Rational(180)*sp.pi)
m3 = sp.cos(theta/sp.Rational(180)*sp.pi)

# energy
ftot = B1*(e11*m1**2 + e22*m2**2) \
     + (B1**2/(3*c11) - B1*c12/c11*(e11+e22))*m3**2 \
     - B2**2/(2*c44)*(m1**2+m2**2)*m3**2 \
     - B1**2/c11*m3**4 \
     - K1*m3**2 \
     - K2*m3**4

# get value
ftot = ftot.subs({
    c11: 2.181e11, # unit: Pa
    c12: 9.35e10,  # unit: Pa
    c44: 6.23e10,  # unit: Pa
    K1: 6.5118e3, # unit: J/m3
    K2: -2.2318e4, # unit: J/m3
    B1: -6.92e6, # unit: J/m3
    B2: -6.92e6, # unit: J/m3
    })

# consider apply strain along [100]  and [01-1] direction
e11vallist = np.arange(-0.01, 0.001, 0.0001)
e22vallist = np.arange(-0.001, 0.0001, 0.00001)
# specify theta list
thetalist = np.zeros((e11vallist.shape[0], e22vallist.shape[0]))
# specify theta and phi range
bnds = ((0,90),(0,180))
# go through strain value
for i in range(e11vallist.shape[0]):
    for j in range(e22vallist.shape[0]):
        e11val = e11vallist[i]
        e22val = e22vallist[j]
        ftot_temp = ftot.subs({
            e11: e11val,
            e22: e22val / np.sqrt(2)
        })
        # define function then
        # from link: https://stackoverflow.com/questions/34115233/python-optimization-using-sympy-lambdify-and-scipy
        lam_ftot = lambdify((theta, phi), ftot_temp,'numpy')
        def lam_ftot_v(x):
            return lam_ftot(*tuple(x))
        
        # get results
        result = brute(lam_ftot_v, ranges=bnds)
        # append data to list
        thetalist[i][j] = result[0]


# change numpy data to data framework and save it in csv file
df = pd.DataFrame(data=thetalist, index=e11vallist.astype(str), columns = e22vallist.astype(str)) 
df.to_csv('thetalist.csv', index = True)
 