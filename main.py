# The start of the Modelling stuff I guess
import numpy as np
from numpy.core.numeric import NaN
from matplotlib import pyplot as plt
from numpy.lib.function_base import interp
from scipy.interpolate import interp1d
import itertools
from scipy.optimize import curve_fit as curve
import glob as glob

net = []

def main():
	time, Pressure, netFlow = getPressureData()
	pars = [Pressure[0],0.0012653061224489797,0.09836734693877551,0.0032244897959183673,1]
	# a,b,c are some constants we define
	# dqdt I assume is something we solve for depending on the change in flow rates
	# this will solve the ODE with the different net flow values

	# TODO:
	# Set net/q_sink as global (CHECK)
	# Create helper function(Not Check)





	dt = 0.5
	sol_time, sol_pressure = solve_Pressure_ode(pressure_model, time[0], time[-1], dt , Pressure[0], pars)

	f, ax = plt.subplots(1, 1)
	ax.plot(sol_time,sol_pressure, 'b', label = 'ODE')
	ax.plot(time,Pressure, 'r', label = 'DATA')
	plt.axvline(2004, color = 'black', linestyle = '--', label = 'Calibration point')
	ax.legend()
	ax.set_title("Pressure flow in the Orakei geothermal field.")
	plt.show()

	time, Pressure, CO2_injec, conc = getConcentrationData()
	# in order for pars
	# qCO2, a, b, d, P, P0, M0
	pars = [CO2_injec,1123412341351354,1,.3,sol_pressure,sol_pressure[0],8555.23459874256]
	dt = 0.5
	sol_time, sol_conc = solve_Solute_ode(SoluteModel, time[0], time[-1], dt , conc[0], pars)
	f, ax = plt.subplots(1, 1)	
	ax.plot(sol_time,sol_conc, 'b', label = 'ODE')
	ax.plot(time,conc, 'r', label = 'DATA')
	plt.axvline(2004, color = 'black', linestyle = '--', label = 'Calibration point')
	ax.legend()
	ax.set_title("Concentration of CO2 in the Orakei geothermal field.")
	plt.show()
	PlotBenchmark_pressure(sol_pressure,sol_time)
	return

def PlotBenchmark_pressure(sol_pressure,sol_time):
	analytical_soln = []
	time, Pressure ,netFlow = getPressureData()
	a = 0.0012653061224489797
	b = 0.09836734693877551
	for i in range(len(time)):
		analytical_soln.append(analytical_pressure(Pressure[0],a,b,netFlow[i], time[i]))

	f, ax = plt.subplots(1, 1)	
	ax.plot(time,analytical_soln, 'k', label = 'Analytical Solution')
	ax.plot(sol_time,sol_pressure, 'r', label = 'Numerical Solution')
	ax.legend()
	ax.set_title("Analytical vs Numerical Solution for Pressure")
	plt.show()
	return

def analytical_pressure(P0, a, b, q, t):
	return P0 - ((a*q)/b)*(1-np.exp(-b*t))

def PlotBenchmark_Solute():
	return

def analytical_solute():
	return

def getConcentrationData():
	'''
	Reads all relevant data from output.csv file
	Parameters : 
	------------
	None

	Returns : 
	---------
	t : np.array
		Time data that matches with other relevant quantities 
	P : np.array
		Relevant Pressure data in MPa
	injec : np.array
		Relevant CO2 injection data in kg/s
	CO2_conc : np.array
		Relevant concentrations of CO2 in wt %

	Notes :
	------
	CO2 concentration before injection is assumed to be natural state
	of 3 wt %. 
	'''

	# reads all the data from excel file
	vals = np.genfromtxt('output.csv', delimiter = ',', skip_header= 1, missing_values= 0)
	# extracts the relevant data
	t = vals[:,1] # time values
	P = vals[:,3] # Pressure values
	injec = vals[:,4] # CO2 injection values 
	CO2_conc = vals[:,5] # CO2 concentration values
	CO2_conc[np.isnan(CO2_conc)] = 0.03 # inputting natural state 
	injec[np.isnan(injec)] = 0 # absence of injection values is 0
	P[0] = P[1]

	return t, P, injec, CO2_conc

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
		sol_time, sol_pressure = solve_Pressure_ode(pressure_model, time[0], time[-1], dt , Pressure[0], pars)

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
	sol_time, sol_pressure = solve_Pressure_ode(pressure_model, time[0], time[-1], dt , Pressure[0], pars)

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
	''' Return the Pressure derivative dP/dt at time, t, for given parameters.

		Parameters:
		-----------
		t : float
			Independent variable.
		P : float
			Dependent variable.
		q : float
			Source/sink rate.
		a : float
			Source/sink strength parameter.
		b : float
			Recharge strength parameter.
		c  : float
			Recharge strength parameter
		dqdt : float
			Rate of change of flow rate
		P0 : float
			Ambient value of dependent variable.
		Returns:
		--------
		dPdt : float
			Derivative of Pressure variable with respect to independent variable.
	'''
	dPdt =  -a*q - b*(P-P0) - c*dqdt
	return dPdt

def SoluteModel(t, C, qC02, a, b, d, P, P0, M0, C0):
	''' Return the Solute derivative dC/dt at time, t, for given parameters.

		Parameters:
		-----------
		t : float
			Independent variable.
		C : float
			Dependent variable.
		qCO2 : float
			Source/sink rate.
		a : float
			Source/sink strength parameter.
		b : float
			Recharge strength parameter.
		d  : float
			Recharge strength parameter
		P : float
			Pressure at time point t
		P0 : float
			Ambient value of Pressure within the system.
		M0 : float
			Ambient value of Mass of the system
		C0 : float
			Ambient value of the dependent variable.

		Returns:
		--------
		dCdt : float
			Derivative of Pressure variable with respect to independent variable.
	'''
	# performing calculating C' for ODE
	if (P > P0):
		Cdash = C
	else:
		Cdash = C0

	qloss = (b/a)*(P-P0)*Cdash*t # calculating CO2 loss to groundwater

	qC02 = qC02 - qloss # qCO2 after the loss

	dCdt = ((1 - C)*qC02)/M0 - (b/(a*M0))*(P-P0)*(Cdash-C) - d*(C-C0) # calculates the derivative
	return dCdt

def solve_Solute_ode(f, t0, t1, dt, x0, pars):
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
		ys : array-like
			Dependent variable solution vector.

		Notes:
		------
		ODE is solved using improved Euler
	'''
	nt = int(np.ceil((t1-t0)/dt)) # gets size of the array needing to solve
	ts = t0+np.arange(nt+1)*dt # creates time array
	ys = 0.*ts # creates array to put solutions in
	ys[0] = x0 # inputs initial value
	qC02 = pars[0] # extracts sink rate
	Pressure = pars[4] # extracts pressure values
	for k in range(nt):
		# improved euler needs different values of pressure and sink rate
		# for different time values
		pars[0] = qC02[k]
		pars[4] = Pressure[k]
		ys[k + 1] = improved_euler_step(f, ts[k], ys[k], dt, x0, pars)
	return ts,ys

def NEW_solve_solute_ode(timeSpace, y0: float, a: float, b: float, d: float, P: float, P0: float, M0: float, dt: float = 0.5)->List[float]:
	pass


def solve_Pressure_ode(f, t0, t1, dt, x0, pars):
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
		ys : array-like
			Dependent variable solution vector.

		Notes:
		------
		ODE is solved using improved Euler
	'''

	nt = int(np.ceil((t1-t0)/dt)) # gets size of the array needing to solve
	ts = t0+np.arange(nt+1)*dt # creates time array
	ys = 0.*ts # creates array to put solutions in
	ys[0] = x0 # inputs initial value
	netFlow = pars[0] # extracts the sink values 
	for k in range(nt):
		pars[0] = netFlow[k] # ODE needs sink at different time points
		pars[4] = (netFlow[k+1] - netFlow[k])/(ts[k+1]-ts[k])# calculates dqdt using forward differentiation
		ys[k + 1] = improved_euler_step(f, ts[k], ys[k], dt, x0, pars)
	return ts,ys

def IMPROVED_solve_pressure_ode(t0, t1, dt, q_sink, pars, optimised=False, pressure = None):

	time = t0 + np.arange(int(np.ceil((t1-t0)/dt))+1)*dt
	# TODO: check dimensions are the same for data 2 and 3
	q_sink = interp(time, t0 + np.arange(int(np.ceil((t1-t0)/0.5))+1)*0.5, net)
	
	def subFunc(timeSpace, y0: float, a: float, b: float, c: float)->List[float]:
		''' Solve an ODE numerically.
			Parameters:
			-----------
			timeSpace : array-like
				Time values used, the difference between values must be linear
				It must have the same shape as net, unless net is interpolated
			
			y0 : float
				Initial value of solution.

			a : float
				q_sink coefficient

			b : float
				pressure coefficient

			c : float
				dqdt coefficient

			Returns:
			--------
			ans : array-like
				Dependent variable solution vector
				Same shape as input array

			Notes:
			------
			ODE is solved using improved Euler
			Uses constant time step of 0.5
			assumes net is readable withen the scope
		'''
		# The step size cannot be variable as it would be adjusted by curve_fit()
		# this is also dictated by qsink - unless we interpolate
		nt = len(timeSpace)

		ans = 0.*timeSpace
		ans[0] = y0

		pars = [0, a, b, c, 0]
		for k in range(1, nt):
			# setting the value for q sink and dqdt
			pars[0] = q_sink[k]

			pars[-1] = (q_sink[k] - q_sink[k-1])/dt
			ans[k] = improved_euler_step(pressure_model, timeSpace[k], ans[k-1], dt, y0, pars)

		return ans
	
	if optimised and pressure != None and len(pressure) == int(np.ceil((t1-t0)/0.5))+1:
		pars = curve(subFunc, timeSpace, interp(time, t0 + np.arange(int(np.ceil((t1-t0)/0.5))+1)*0.5, pressure), p0=pars)[0]
		
	return time, subFunc(time, *pars)

def NEW_solve_pressure_ode(timeSpace, y0: float, a: float, b: float, c: float)->List[float]:
	''' Solve an ODE numerically.
		Parameters:
		-----------
		timeSpace : array-like
			Time values used, the difference between values must be linear
			It must have the same shape as net, unless net is interpolated
		
		y0 : float
			Initial value of solution.

		a : float
			q_sink coefficient

		b : float
			pressure coefficient

		c : float
			dqdt coefficient

		Returns:
		--------
		ans : array-like
			Dependent variable solution vector
			Same shape as input array

		Notes:
		------
		ODE is solved using improved Euler
		Uses constant time step of 0.5
		assumes net is readable withen the scope
	'''
	# The step size cannot be variable as it would be adjusted by curve_fit()
	# this is also dictated by qsink - unless we interpolate
	nt = len(timeSpace)
	dt = 0.5

	ans = 0.*timeSpace
	ans[0] = y0

	pars = [0, a, b, c, 0]
	for k in range(1, nt):
		# setting the value for q sink and dqdt
		pars[0] = net[k]

		pars[-1] = (net[k] - net[k-1])/dt
		ans[k] = improved_euler_step(pressure_model, timeSpace[k], ans[k-1], dt, y0, pars)

	return ans

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
	f0 = f(tk, yk, *pars, x0) # calculates f0 using function
	f1 = f(tk + h, yk + h*f0, *pars,x0) # calculates f1 using fuctions
	yk1 = yk + h*(f0*0.5 + f1*0.5) # calculates the new y value
	return yk1

def getPressureData():
	'''
	Reads all relevant data from output.csv file
	Parameters : 
	------------
	None

	Returns : 
	---------
	t : np.array
		Time data that matches with other relevant quantities 
	P : np.array
		Relevant Pressure data in MPa
	net : np.array
		Overall net flow for the system in kg/s
	'''
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
	global net

	for i in range(len(prod)):
		net.append(prod[i] - injec[i]) # getting net amount 
	return t, P, net

if __name__ == "__main__":
	main()
	#MSPE_A()
	
