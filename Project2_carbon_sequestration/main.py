# The start of the Modelling stuff I guess
import numpy as np
from numpy.core.numeric import NaN
from matplotlib import pyplot as plt
from numpy.lib.function_base import interp
from scipy.interpolate import interp1d
import itertools



def main():
	time, Pressure ,netFlow = getPressureData()
	pars = [netFlow,0.0012653061224489797,0.09836734693877551,0.0032244897959183673,1]


	# q is variable so need to increment the different flows 
	# a,b,c are some constants we define
	# dqdt I assume is something we solve for depending on the change in flow rates
	# this will solve the ODE with the different net flow values
	dt = 0.5
	sol_time, sol_pressure = solve_ode(pressure_model, time[0], time[-1], dt , Pressure[0], pars)

	f, ax = plt.subplots(1, 1)
	ax.plot(sol_time,sol_pressure, 'b', label = 'ODE')
	ax.plot(time,Pressure, 'r', label = 'DATA')
	ax.legend()
	ax.set_title("Pressure flow in the Orakei geothermal field.")
	plt.show()
	return

def MSPE_A():
	'''
	Using MSPE as metric for brute-force calculating coefficients of the pressure ODE.

	Parameters : 
	------------
	None

	Returns : 
	---------
	A : float
		Best parameter one for ODE model
	B : float
		Best parameter two for ODE model
	C : float
		Best parameter three for ODE model

	Generates plots of various ODE models, best ODE model, and MSPE wrt. A    
	
	'''

	# Experimental Data, defining testing range for coefficient, constants
	time, Pressure ,netFlow = getPressureData()
	A = np.linspace(0.001,0.0015,50)
	# A = 9.81/(0.15*A)
	B = np.linspace(0.08,0.11,50)
	C = np.linspace(0.002,0.006,50)
	dt = 0.5
	MSPE_best = float('inf')
	best_A = 1000
	best_B = 1000
	best_C = 1000


	# Modelling ODE for each combination of A,B,C
	for A_i,B_i,C_i in itertools.product(A,B,C):
		pars = [netFlow,A_i,B_i,C_i,1]
		sol_time, sol_pressure = solve_ode(pressure_model, time[0], time[-1], dt , Pressure[0], pars)

		# Interpolating for comparison of MSE
		f = interp1d(sol_time,sol_pressure)
		analytic_pressure = f(time)
		diff_array = np.subtract(analytic_pressure,Pressure)
		squared_array = np.square(diff_array)
		MSPE = squared_array.mean()

		print(A_i)

		if (MSPE < MSPE_best):
			MSPE_best = MSPE
			best_A = A_i
			best_B = B_i
			best_C = C_i


	
	# Plotting best fit ODE
	pars = [netFlow,best_A,best_B,best_C,1]
	sol_time, sol_pressure = solve_ode(pressure_model, time[0], time[-1], dt , Pressure[0], pars)

	# Printout of results
	txt = "Best coefficient {} is {}"
	print(txt.format("A",best_A))
	print(txt.format("B",best_B))
	print(txt.format("C",best_C))
	print("Mean Squared Error is {}".format(MSPE_best))

	
	f, ax2 = plt.subplots(1, 1)
	ax2.plot(sol_time,sol_pressure, 'b', label = 'ODE')
	ax2.plot(time,Pressure, 'r', label = 'DATA')
	ax2.set_title("Best fit A coefficient")
	ax2.legend()
	plt.show()
		

	return best_A,best_B,best_C

def pressure_model(t, P, q, a, b, c, dqdt, P0):
	''' Return the derivative dx/dt at time, t, for given parameters.

		Parameters:
		-----------
		P : float
			Dependent variable.
		q : float
			Source/sink rate.
		a : float
			Source/sink strength parameter.
		b : float
			Recharge strength parameter.
		P0 : float
			Ambient value of dependent variable.
		c  : float
			Recharge strength parameter
		dqdt : float
			Rate of change of flow rate
		Returns:
		--------
		dPdt : float
			Derivative of Pressure variable with respect to independent variable.
	'''
	dPdt =  -a*q - b*(P-P0) - c*dqdt
	return dPdt

def solve_ode(f, t0, t1, dt, x0, pars):
	''' Solve an ODE numerically.

		Parameters:
		-----------
		f : callable
			Function that returns dxdt given variable and parameter inputs.
		t0 : float
			Initial time of solution.
		t1 : float
			Final time of solution.
		dt : float
			Time step length.
		x0 : float
			Initial value of solution.
		pars : array-like
			List of parameters passed to ODE function f.

		Returns:
		--------
		t : array-like
			Independent variable solution vector.
		x : array-like
			Dependent variable solution vector.

		Notes:
		------
		ODE should be solved using the Improved Euler Method. 

		Function q(t) should be hard coded within this method. Create duplicates of 
		solve_ode for models with different q(t).

		Assume that ODE function f takes the following inputs, in order:
			1. independent variable
			2. dependent variable
			3. forcing term, q
			4. all other parameters
	'''
	nt = int(np.ceil((t1-t0)/dt))
	ts = t0+np.arange(nt+1)*dt
	ys = 0.*ts
	ys[0] = x0
	netFlow = pars[0]
	for k in range(nt):
		pars[0] = netFlow[k]
		pars[4] = (netFlow[k+1] - netFlow[k])/(ts[k+1]-ts[k])
		ys[k + 1] = improved_euler_step(f, ts[k], ys[k], dt, x0, pars)
	return ts,ys


def improved_euler_step(f, tk, yk, h, x0, pars):
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
	f0 = f(tk, yk, *pars, x0)
	f1 = f(tk + h, yk + h*f0, *pars,x0)
	yk1 = yk + h*(f0*0.5 + f1*0.5)
	return yk1

def getPressureData():
	# reads the files' values
	vals = np.genfromtxt('output.csv', delimiter = ',', skip_header= 1, missing_values= 0)
	# extracts the relevant data
	t = vals[:,1]
	prod = vals[:, 2]
	P = vals[:,3]
	injec = vals[:,4]
	# cleans data
	# for CO2 injection if no data is present then a rate of 0 is given for Pressure 
	# it is given the most recent value
	injec[np.isnan(injec)] = 0
	P[0] = P[1] # there is only one missing value
	net = []
	for i in range(len(prod)):
		net.append(prod[i] - injec[i]) # getting net amount 
	return t, P, net

if __name__ == "__main__":
	 main()
	# MSPE_A()
	
