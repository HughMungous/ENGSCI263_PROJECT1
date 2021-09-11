from glob import glob
from os import sep as sysSep

## numpy and math
import numpy as np
from numpy.core.numeric import NaN
from numpy.lib.function_base import interp

import statistics
import itertools

from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

## plotting
from matplotlib import pyplot as plt

## type handling
from typing import List

## helper functions
from helper import *

## global variable declaration
## raw data
time 		= []
pressure 	= []
injection 	= []
CO2_conc 	= []
finalProduction 	= 0
dt 				    = 0.5

##parameters
# static
basePressure 		= 6.1777
baseConcentration 	= 0.03

# optimised
baseMass = 9900.495 # might need to change this to a parameter
  
a = 0.00192137
b = 0.14009364
c = 0.00063577
d = 0.24652261

covariance = []

## derived data 
net 	= [] # left in for now
dqdt 	= []

## analytical solutions
analyticalPressure 	= []
analyticalSolute	= []
analyticalQLoss		= []

# extrapolated data
extrapolatedTimespace 		= []
extrapolatedPressure 		= []
extrapolatedConcentration 	= []

# used for parsing the correct pressure data for the solute ode
extrapolationIndices 		= [] 

"""
DONE:
1. read data
2. interpolate
3. plot naive version
4. optimise pars
5. plot new version
6. extrapolate
7. plot
8. Uncertainty (posterior paramater distribution)
9. Misfit
10. Benchmark

TODO:

11. Uncertainty (confidence intervals)
12. confint on prediction???
13. comments & docstrings
"""

def getMeasurementData(interpolated: bool):
	"""Reads in data from either the interpolated CSV or the original files

		Parameters:
		----------
		interpolated : bool
			which data set to use, the interpolated one matches the most frequent data (half yearly)

		Returns:
		--------
		originalData: Dict[Union[List[float],List[float]]]
			A dictionary with keys: "pressure"; "production"; "injection"; "concentration".
			The timespace and corresponding measurements are stored as the zeroth and first
			values for each key.

		Notes:
		------
		The function only returns originalData when interpolated == False
	"""
	if interpolated:
		global time, pressure, injection, CO2_conc, basePressure, net, dqdt, finalProduction

		fileAddress = glob("output.csv")[0]
		vals = np.genfromtxt(fileAddress, delimiter = ',', skip_header= 1, missing_values= 0)

		## data extraction
		time 		= vals[:,1]
		production 	= vals[:,2]
		pressure 	= vals[:,3]
		injection 	= vals[:,4]
		CO2_conc 	= vals[:,5]

		## data cleaning
		injection[np.isnan(injection)] = 0
		CO2_conc[np.isnan(CO2_conc)] = 0.03

		# first value missing
		pressure[0] = pressure[1] 

		basePressure = pressure[0]

		# necessary
		finalProduction = production[-1]
		

		for i in range(len(production)):
			net.append(production[i] - injection[i]) # getting net amount 
		
		net = np.array(net)

		dqdt 		= 0.* net
		dqdt[1:-1] 	= (net[2:]-net[:-2]) / dt # central differences
		dqdt[0] 	= (net[1]-net[0]) / dt    # forward difference
		dqdt[-1] 	= (net[-1]-net[-2]) / dt  # backward difference

		
		return 

	# gets the original data - uninterpolated
	originalData = {
		"pressure": np.genfromtxt('data/cs_p.txt', delimiter = ',', skip_header=1).T,
		"production": np.genfromtxt('data/cs_q.txt', delimiter = ',', skip_header=1).T,
		"injection": np.genfromtxt('data/cs_c.txt', delimiter = ',', skip_header=1).T,
		"concentration": np.genfromtxt('data/cs_cc.txt', delimiter = ',', skip_header=1).T
	}

	return originalData

def interpolate(dtNew: float)->None:
	"""Function to interpolate all of the data for some new timestep.
	
		Parameters:
		-----------
		dtNew : float
			the new timestep

		Returns:
		--------
		None :
			This function alters specific global variables

		Notes:
		------
		It is assumed that a call togetMeasurementData(True) is made
		prior to any call of this function, as the data to interpolate
		will not be properly instantiated and the outcome is undefined.
	"""
	global time, dt, pressure, injection, CO2_conc, net, dqdt
	# creating a temporary timespace defined by the new dt
	
	temp = np.arange(time[0], time[-1] + dtNew, dtNew)

	# interpolating the data
	pressure	= interp(temp, time, pressure)
	injection 	= interp(temp, time, injection)
	CO2_conc 	= interp(temp, time, CO2_conc)
	net 		= interp(temp, time, net)
	dqdt 		= interp(temp, time, dqdt)

	# updating the timespace and timestep
	time 	= temp
	dt 		= dtNew
	return

def solve(t: List[float], a: float, b: float, c: float, d: float, M0: float, func: str, finalDataPoint = 0, extraP: List[float] = [], extrapolate = None)->List[float]:
	"""Function to solve a model..."""
	nt = len(t)
	result = 0.*t

	if func == "pressure":
		result[0] = basePressure
		params = [basePressure, 0, 0, a, b, c]

		if extrapolate != None:
			# assuming that the production stays constant - need to verify
			params[1] = finalProduction - extrapolate*injection[-1]
			# result[0] = analyticalPressure[-1]
			result[0] = finalDataPoint

			for k in range(nt-1):				
				result[k+1] = improved_euler_step(pressureModel, t[k], result[k], dt, params)
			
			return result

		for k in range(nt-1):
			# setting the value for q sink and dqdt
			params[1] = net[k]								# net sink rate, q
			params[2] = dqdt[k]
			
			result[k+1] = improved_euler_step(pressureModel, t[k], result[k], dt, params)

		return result

	elif func == "solute":
		result[0] = baseConcentration
		#dt, c0, P, P0, injection, M0, a, b, d
		params = [baseConcentration, 0, basePressure, 0, M0, a, b, d]
		
		if extrapolate != None:
			# assuming that the production stays constant - need to verify
			params[3] = injection[-1]*extrapolate

			# result[0] = analyticalSolute[-1]
			result[0] = finalDataPoint

			for k in range(nt-1):	
				params[1] = extraP[k]			
				result[k+1] = improved_euler_step(soluteModel, t[k], result[k], dt, params)
			
			return result

		for k in range(nt-1):
			params[1] = pressure[k]
			params[3] = injection[k]
			result[k+1] = improved_euler_step(soluteModel, t[k], result[k], dt, params)

		return result

	else:
		raise("implementation for qLoss missing")
		pass

	pass

def optimise(calibrationPoint = -1)->None:
	"""This function optimises all of the parameters for both models simultaneously.
	
		Parameters:
		-----------
		calibrationPoint: Optional, int
			Only the data prior to this index will be used to optimise the paramters

		Returns:
		--------
		None:
			global variables are adjusted

		Notes:
		------
		It is assumed that a call togetMeasurementData(True) is made
		prior to any call of this function, as the outcome is undefined 
		otherwise.
	"""
	global a, b, c, d, baseMass, covariance
	# global a, b, c, covariance
	nt = len(time[:calibrationPoint])

	def subFunc(t: List[float], ta: float, tb: float, tc: float, td: float, tM0: float)->List[float]:
		pressureSolution = solve(t[:nt], ta, tb, tc, td, tM0, "pressure")
		soluteSolution = solve(t[nt:], ta, tb, tc, td, tM0, "solute")
		return np.append(pressureSolution, soluteSolution)
	# def subFunc(t: List[float], ta: float, tb: float, tc: float)->List[float]:
	# 	pressureSolution = solve(t[:nt], ta, tb, tc, d, baseMass, "pressure")
	# 	soluteSolution = solve(t[nt:], ta, tb, tc, d, baseMass, "solute")
	# 	return np.append(pressureSolution, soluteSolution)
	

	pars, covariance = curve_fit(subFunc, np.append(time[:calibrationPoint], time[:calibrationPoint]), np.append(pressure[:calibrationPoint], CO2_conc[:calibrationPoint]), [a, b, c, d, baseMass])
	# pars, covariance = curve_fit(subFunc, np.append(time, time), np.append(pressure, CO2_conc), [a, b, c], method="lm")

	a, b, c, d, baseMass = pars
	# a, b, c = pars
	return

def extrapolate(endPoint: float, proposedRates: List[float], pars: List[float], finalDataPoint: List[float], uncert = False):
	"""	
	This function creates projections for each of the provided rates from the endpoint
	of the analytical solution to the declared endpoint for the projection.
	
	"""
	if not uncert:
		global extrapolatedTimespace, extrapolationIndices, extrapolatedPressure, extrapolatedConcentration

		extrapolationIndices = proposedRates
		extrapolatedTimespace = np.arange(time[-1],endPoint + dt, dt)

		for rate in proposedRates:
			extrapolatedPressure.append(solve(extrapolatedTimespace, *pars, "pressure", finalDataPoint[0], extrapolate=rate))
			extrapolatedConcentration.append(solve(extrapolatedTimespace, *pars, "solute", finalDataPoint[1], extraP = extrapolatedPressure[-1], extrapolate=rate))
		return
		
	pressureSol, concentrationSol = [], []

	for rate in proposedRates:
		pressureSol.append(solve(extrapolatedTimespace, *pars, "pressure", finalDataPoint[0], extrapolate=rate))
		concentrationSol.append(solve(extrapolatedTimespace, *pars, "solute", finalDataPoint[1], extraP= pressureSol[-1],extrapolate=rate))
	
	return pressureSol, concentrationSol

def uncertainty(n: int, nPars: int):
	pars = np.random.default_rng().multivariate_normal([a, b, c, d, baseMass][:nPars], [l[:nPars] for l in covariance[:nPars]], n)
	# pars2 = np.random.default_rng().multivariate_normal([a, b, c, d, baseMass][nPars:], [l[nPars:] for l in covariance[nPars:]], n)
	# pars = np.random.default_rng().multivariate_normal([a, b, c], covariance, n, method="svd")

	pressurePosterior, solutePosterior = [], []
	pressurePosteriorExtrap = {rate: [] for rate in extrapolationIndices}
	solutePosteriorExtrap = {rate: [] for rate in extrapolationIndices}

	for par in pars:
	# for i in range(n):
		pressurePosterior.append(solve(time, *par, *[a, b, c, d, baseMass][nPars:], "pressure"))
		solutePosterior.append(solve(time, *par, *[a, b, c, d, baseMass][nPars:], "solute"))
		# pressurePosterior.append(solve(time, *pars[i], *pars2[i], "pressure"))
		# solutePosterior.append(solve(time, *pars[i], *pars2[i], "solute"))
		
		temp = extrapolate(2050, extrapolationIndices, [*par, *[a, b, c, d, baseMass][nPars:]], [pressurePosterior[-1][-1], solutePosterior[-1][-1]], True)
		# temp = extrapolate(2050, extrapolationIndices, [*pars[i], *pars2[i]], [pressurePosterior[-1][-1], solutePosterior[-1][-1]], True)

		for j in range(len(extrapolationIndices)):
			pressurePosteriorExtrap[extrapolationIndices[j]].append(temp[0][j])
			solutePosteriorExtrap[extrapolationIndices[j]].append(temp[1][j])

	return pressurePosterior, solutePosterior, pressurePosteriorExtrap, solutePosteriorExtrap

def misfit(pressureTime: List[float], pressure: List[float], concentrationTime: List[float], concentration: List[float]):
	pRes = interp(pressureTime, time, analyticalPressure)
	cRes = interp(concentrationTime,time, analyticalSolute)
	return np.array([pressure[i]-pRes[i] for i in range(len(pRes))]), np.array([concentration[i]-cRes[i] for i in range(len(cRes))]) 
	
def benchmark(t: List[float], newdt: float, C0, P0, q0, q, a, b, c, d, M0, func: str):
	numerical, analytical = 0.*t, 0.*t
	if func == "pressure":
		analytical[0] = P0 + ((-a*q0)/b)*(1-np.exp(-b*t[0]))
		numerical[0] = P0
		steadyState = P0 - a * q0 / b

		for i in range(len(t)-1):
			analytical[i+1] = P0 + ((-a*q0)/b)*(1-np.exp(-b*t[i+1]))
			numerical[i+1] = improved_euler_step(pressureModel, t[i], numerical[i], newdt, [P0, q, 0, a, b, c])
			
	else:
		k = q / M0
		L = (k*C0 - k)/(k + d)

		analytical[0] = (k + (d * C0))/(k + d) + L/(np.exp(t[0] * (k + d)))
		numerical[0] = C0
		steadyState = ((q / M0) + d*C0)/((q / M0) + d) 
		
		for i in range(len(t)-1):
			analytical[i+1] = (k + (d * C0))/(k + d) + L/(np.exp(t[0] * (k + d)))
			numerical[i+1] = improved_euler_step(soluteModel, t[i], numerical[i], newdt, [C0, basePressure, basePressure, q, M0, a, b, d])
		
	return numerical, analytical, steadyState

## TESTING
# ------------------------------------------
# 
# ------------------------------------------

def main(interpoRate: float, calibrationPoint: int, nPars: int = 3, nPredicts: int = 50, plotting: List[bool] = [False]*5):
	colours = {
		0: "c",
		0.5:"m",
		1: "b",
		2: "y",
		4: "k"
	}
	## original data
	originalData = getMeasurementData(False)

	if plotting[0]:
		f1, ax1a = plt.subplots(1,1)
		f2, ax2a = plt.subplots(1,1)

		ax1b = ax1a.twinx()
		ax2b = ax2a.twinx()
		
		ax1a.plot(*originalData["pressure"], "b", label = "pressure")
		ax1b.plot(*originalData["injection"], "y", label = "injection")
		ax1b.plot(*originalData["production"], "r", label = "production")

		ax2a.plot(*originalData["concentration"], "b", label = "concentration")
		ax2b.plot(*originalData["injection"], "y", label = "injection")
		ax2b.plot(*originalData["production"], "r", label = "production")

		ax1a.legend(loc=2)
		ax1b.legend(loc=1)

		ax2a.legend(loc=2)
		ax2b.legend(loc=1)

		ax1a.set_title("Original data.")
		ax2a.set_title("Original data.")
		plt.show()

	## part 2 - electric boogaloo
	getMeasurementData(True)
	interpolate(interpoRate)
	i, = np.where(np.isclose(time,calibrationPoint))[0]
	optimise(i)
	
	global analyticalPressure, analyticalSolute
	analyticalPressure = solve(time, a, b, c, d, baseMass, "pressure")
	analyticalSolute = solve(time, a, b, c, d, baseMass, "solute")

	## extrapolation
	extrapolate(2050, [0,0.5,1,2,4], [a,b,c,d,baseMass], [analyticalPressure[-1], analyticalSolute[-1]])

	if plotting[1]:
		f1, ax1 = plt.subplots(1,1)
		f2, ax2 = plt.subplots(1,1)
		# plt.subplots()
		
		ax1.plot(originalData["pressure"][0], originalData["pressure"][1], 'r.', label = "measurements")
		ax1.plot(time, analyticalPressure, "b", label = "analytical sol")
		
		ax2.plot(originalData["concentration"][0], originalData["concentration"][1], 'r.', label = "measurements")
		ax2.plot(time, analyticalSolute, "b", label = "analytical sol")

		for i, x in enumerate(extrapolationIndices):
			ax1.plot(extrapolatedTimespace, extrapolatedPressure[i], colours[x], label = f"{x*injection[-1]} kg/s")
			ax2.plot(extrapolatedTimespace, extrapolatedConcentration[i], colours[x], label = f"{x*injection[-1]} kg/s")

		ax1.legend()
		ax1.set_title("Pressure solution")

		ax2.legend()
		ax2.set_title("Solute solution")
		
		plt.show()

	if plotting[2]:
		pPos, sPos, pPosEx, sPosEx = uncertainty(nPredicts, nPars)

		f1, ax1 = plt.subplots(1,1)
		f2, ax2 = plt.subplots(1,1)
		
		for i in range(nPredicts):
			ax1.plot(time, pPos[i], "b")
			ax2.plot(time, sPos[i], "b")
			
			for k in pPosEx:
				ax1.plot(extrapolatedTimespace, pPosEx[k][i], colours[k])
				ax2.plot(extrapolatedTimespace, sPosEx[k][i], colours[k])

		ax1.set_title("Pressure solution")
		ax2.set_title("Solute solution")

		plt.show()

	if plotting[3]:
		misfitPressure, misfitConcentration = misfit(*originalData["pressure"], *originalData["concentration"])

		f1, ax1 = plt.subplots(1, 1)
		f2, ax2 = plt.subplots(1, 1)
		
		ax1.plot(originalData["pressure"][0],misfitPressure, 'rx')
		ax1.axhline(0, color = 'black', linestyle = '--')
		ax1.set_ylabel('Pressure [MPa]')
		ax1.set_xlabel('Time [years]')
		ax1.set_title("Best Fit Pressure LPM Model")

		ax2.plot(originalData["concentration"][0],misfitConcentration, 'rx')
		ax2.axhline(0, color = 'black', linestyle = '--')
		ax2.set_ylabel('CO2 [wt %]')
		ax2.set_title("Best Fit Solute LPM Model")

		plt.show()
	
	if plotting[4]:
		dt = 0.1
		tempTime = np.arange(0, 10 + dt, dt)

		numerical, analytical, steadyState = benchmark(tempTime, dt, 0, basePressure, 4, 4, 1, 2, 0, 0, 0, "pressure")

		# plot 1
		f, ax = plt.subplots(1, 1) # plotting numerical vs analytical solutions

		ax.plot(tempTime,analytical, 'b', label = 'Analytical')
		ax.plot(tempTime, numerical, 'kx', label = 'Numerical')
		ax.set_xlabel("Time [seconds]")
		ax.set_ylabel("Pressure [MPa]")

		ax.axhline(steadyState, linestyle = '--', color = 'red', label = 'steady state')

		ax.legend()
		ax.set_title("Analytical vs Numerical Solution Benchmark for Pressure ODE")

		plt.show()

		# plot 2
		dt = 1.1 # changing time step for instability analysis
		tempTime = np.arange(0, 10+dt, dt)
		numerical, analytical, steadyState = benchmark(tempTime, dt, 0, basePressure, 4, 4, 1, 2, 0, 0, 0, "pressure")

		f, ax = plt.subplots(1, 1) # plotting numerical vs analytical solutions
		ax.plot(tempTime,analytical, 'b', label = 'Analytical')
		ax.plot(tempTime,numerical, 'kx', label = 'Numerical')
		ax.set_xlabel("Time [seconds]")
		ax.set_ylabel("Pressure [MPa]")
		ax.axhline(steadyState, linestyle = '--', color = 'red', label = 'steady state')

		ax.legend()
		ax.set_title("Instability at a large time step for Pressure ODE")
		plt.show()

		# plot 3
		dt = 0.25 # performing same process as above except for Solute ODE
		tempTime = np.arange(0, 10 + dt, dt)
		numerical, analytical, steadyState = benchmark(tempTime, dt, baseConcentration, basePressure, 4, 1, 1, 2, 0, 3, 1, "solute")
		
		# currently broken :()
		f, ax = plt.subplots(1, 1) # plotting analytical vs numerical results
		ax.plot(tempTime,analytical, 'b', label = 'Analytical')
		ax.plot(tempTime, numerical, 'kx', label = 'Numerical')
		ax.axhline(steadyState, linestyle = '--', color = 'red', label = 'steady state')
		ax.set_xlabel("Time [seconds]")
		ax.set_ylabel("CO2 Concentration [wt %]")
		ax.legend()
		ax.set_title("Analytical vs Numerical Solution Benchmark for Solute ODE")
		plt.show()

		# plot 4 - this should be looped if possible
		dt = 1.1
		tempTime = np.arange(0, 10 + dt, dt)
		numerical, analytical, steadyState = benchmark(tempTime, dt, baseConcentration, basePressure, 4, 1, 1, 2, 0, 3, 1, "solute")

		f, ax = plt.subplots(1, 1)
		ax.plot(tempTime,analytical, 'b', label = 'Analytical')
		ax.plot(tempTime, numerical, 'kx', label = 'Numerical')
		ax.axhline(steadyState, linestyle = '--', color = 'red', label = 'steady state')
		ax.set_xlabel("Time [seconds]")
		ax.set_ylabel("CO2 Concentration [wt %]")
		ax.legend()
		ax.set_title("Instability at a large time step for Solute ODE")
		plt.show()
		pass
	# print(a,b,c,d,baseMass)
	# print(covariance[3][3:])
	# print(covariance[4][3:])
	# print(np.random.multivariate_normal([d, baseMass], [covariance[3][3:],covariance[4][3:]], n))
	return


if __name__ == "__main__":
	main(0.25, 2010, 3, plotting=[True,True,True,True,True])