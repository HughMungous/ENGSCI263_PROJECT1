from nicer_code import *
import numpy as np
from numpy.linalg import norm
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

# define an error tolerance
tol = 1.e-6
# Defines some global variables 
Model_Fit()

# Testing the improved_euler_step function
if False:
    # Testing for a improved euler step approximation to
    # derivative functions f_i = -(\Sigma a_i)ty^2, y(0)=1, a_i = (2,4,6)

    def dydt(t, y, a):
        return -1.*a*t*y*y
    
    def dydt2(t, y, a,b):
        return -1.*a*t*y*y*b

    def dydt3(t, y, a,b,c):
        return -1.*a*t*y*y*b*c

    # Improved Euler
    yk1 = improved_euler_step(dydt, 0., 1., 0.5, pars=[2.])
    yk2 = improved_euler_step(dydt2, 0., 1., 0.25, pars=[2.,4.])
    yk3 = improved_euler_step(dydt3, 0., 1., 0.125, pars=[2.,4.,6.])

    # Assert correct result
    assert(np.abs(yk1-0.75) < tol)
    assert(np.abs(yk2-0.75) < tol)
    assert(np.abs(yk3-0.625) < tol)

    # With negative values
    yk4 = improved_euler_step(dydt, -0., -1., -1, pars=[-2.])
    yk5 = improved_euler_step(dydt2, -0., -1., -0.25, pars=[-2.,-4.])
    yk6 = improved_euler_step(dydt3, -0., -1., -0.125, pars=[-2.,-4.,-6.])

    # Although in theory we should not need negative values, it's good to know the program won't crash.
    assert(np.abs(yk4-0) < tol)
    assert(np.abs(yk5+1.25) < tol)
    assert(np.abs(yk6+0.625) < tol)
    
# Testing PressureModel
if False:
    # Testing pressuremodel for both extrapolate and not, for known values of the parameters.

    # Extrapolate = false, for positive, negative, and zero (calculated from the given values of np.interp)
    pk1 = PressureModel(1,1,1,1,1)
    pk2 = PressureModel(0,0,0,0,0)
    pk3 = PressureModel(-1,-1,-1,-1,-1)

    # Concurring with Wolfram is good enough for me
    assert(np.abs(pk1 + 68.21367346938665) < tol)
    assert(np.abs(pk2-0) < tol)
    assert(np.abs(pk3-66.21367346938665) < tol)

# Testing SoluteModel
if False:

    #Parameter values depend on global variable declared here


    # Again, for extrapolate = false, +/- values of parameters, comparing with Wolfram
    # Breaks for solutemodel(0), returns NaN. Raise typeError instead.
    sk1 = SoluteModel(1,1,1,1)
    sk3 = SoluteModel(-1,-1,-1,-1)

    # Correct values
    assert(np.abs(sk1 + 0.9702) < tol)
    assert(np.abs(sk3+1.029) < tol)

# Testing SolvePressureODE
if True:
    # Just an absolute mess of global variables that I can't overwrite, but the PressureBenchmark function can be used to test various inputs
    pass

# Testing SolveSoluteODE
if True:
    # Similarly all global variables that I can't overwrite, but the SoluteBenchmark function can be used to test various inputs
    pass