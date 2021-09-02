import numpy as np
from numpy.core.numeric import NaN
from matplotlib import pyplot as plt
from numpy.lib.function_base import interp
from scipy.interpolate import interp1d
import itertools
from scipy.optimize import curve_fit
import statistics

time, prod, P, injec, C = np.genfromtxt('output.csv', delimiter = ',', skip_header= 1, missing_values= 0, usecols = [1,2,3,4,5]).T
injec[np.isnan(injec)] = 0
C[np.isnan(C)] = 0.03 # natural state
P[0] = P[1] # there is only one missing value
net = prod - injec
dqdt = net*0.
dqdt[0] = (net[1] - net[0])/(time[1] - time[0])
dqdt[-1] = (net[-1] - net[-2])/(time[-1] - time[-2])
dqdt[1:-1] = (net[2:] - net[:-2])/(time[2:] - net[:-2])
dt = 0.5

def main():
    pars = [0.001900432889906382,0.13953127401141918,0.0012590419811124832]
    Model_Fit(SolvePressureODE, pars)
    return
    
def Model_Fit(f, pars):
    if f is SolvePressureODE:
        bestfit_pars = curve_fit(SolvePressureODE, time, P, pars, bounds = (0,[1.,1.,1.]))
        time_fit = np.arange(time[0], time[-1], dt)
        P_SOL = SolvePressureODE(time_fit, *bestfit_pars[0])
        f, ax = plt.subplots(1, 1)
        ax.plot(time_fit,P_SOL, 'b', label = 'ODE')
        ax.plot(time,P, 'r', label = 'DATA')
        ax.legend()
        ax.set_title("Pressure flow in the Ohaaki geothermal field.")
        plt.show()
    else:
        print(1)
    return

def SolvePressureODE(t, *pars):
    ys = 0.*t
    ys[0] = P[0]
    for k in range(len(t)- 1):
        # need to change what paramters passing in
        ys[k+1] = improved_euler_step(PressureModel, t[k], ys[k], dt, pars)
    return ys

def PressureModel(t, Pk, a, b, c):
    q = np.interp(t, time, net)
    dqdti = np.interp(t, time, dqdt)
    return -a*q - b*(Pk-P[0]) - c*dqdti

def improved_euler_step(f, tk, yk, h, pars):
	""" Compute a single Improved Euler step.
	
		Parameters
		----------
		f : callable
			Derivative function.
		tk : float
			Independent variable at beginning of step.
		yk : float
			Solution at beginning of step.
		h : float
			Step size.
		pars : iterable
			Optional parameters to pass to derivative function.
			
		Returns
		-------
		yk1 : float
			Solution at end of the Improved Euler step.
	"""
	# print(pars)
	f0 = f(tk, yk, *pars) # calculates f0 using function
	f1 = f(tk + h, yk + h*f0, *pars) # calculates f1 using fuctions
	yk1 = yk + h*(f0*0.5 + f1*0.5) # calculates the new y value
	return yk1

if __name__ == "__main__":
	 main()