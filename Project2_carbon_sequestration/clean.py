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

TODO:
8. Uncertainty
9. Missfit
10. Benchmark
"""

def getMeasurementData(interpolated: bool):
	"""Reads in data from the interpolated csv
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
	"""Function to interpolate the data """
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

def solve(t: List[float], a: float, b: float, c: float, d: float, M0: float, func: str, extraP: List[float] = [], extrapolate = None)->List[float]:
	"""Function to solve a model..."""
	nt = len(t)
	result = 0.*t

	if func == "pressure":
		result[0] = basePressure
		params = [basePressure, 0, 0, a, b, c]
		if extrapolate != None:
			# assuming that the production stays constant - need to verify
			params[1] = finalProduction - extrapolate*injection[-1]
			result[0] = analyticalPressure[-1]

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
		params = [dt, baseConcentration, 0, basePressure, 0, M0, a, b, d]
		
		if extrapolate != None:
			# assuming that the production stays constant - need to verify
			params[4] = injection[-1]*extrapolate

			result[0] = analyticalSolute[-1]

			for k in range(nt-1):	
				params[2] = extraP[k]			
				result[k+1] = improved_euler_step(soluteModel, t[k], result[k], dt, params)
			
			return result


		for k in range(nt-1):
			params[2] = pressure[k]
			params[4] = injection[k]
			result[k+1] = improved_euler_step(soluteModel, t[k], result[k], dt, params)

		return result
	else:
		raise("implementation for qLoss missing")
		pass

	pass

def optimise()->None:
	"""This function optimises all of the parameters"""
	global a, b, c, d, baseMass, covariance
	# global a, b, c, covariance
	nt = len(time)

	def subFunc(t: List[float], ta: float, tb: float, tc: float, td: float, tM0: float)->List[float]:
		pressureSolution = solve(t[:nt], ta, tb, tc, td, tM0, "pressure")
		soluteSolution = solve(t[nt:], ta, tb, tc, td, tM0, "solute")
		return np.append(pressureSolution, soluteSolution)
	# def subFunc(t: List[float], ta: float, tb: float, tc: float)->List[float]:
	# 	pressureSolution = solve(t[:nt], ta, tb, tc, d, baseMass, "pressure")
	# 	soluteSolution = solve(t[nt:], ta, tb, tc, d, baseMass, "solute")
	# 	return np.append(pressureSolution, soluteSolution)
	

	pars, covariance = curve_fit(subFunc, np.append(time, time), np.append(pressure, CO2_conc), [a, b, c, d, baseMass], method="lm")
	# pars, covariance = curve_fit(subFunc, np.append(time, time), np.append(pressure, CO2_conc), [a, b, c], method="lm")

	a, b, c, d, baseMass = pars
	# a, b, c = pars
	return

def extrapolate(endPoint: float, proposedRates: List[float], pars: List[float], uncert = False):
	"""	
	This function creates projections for each of the provided rates from the endpoint
	of the analytical solution to the declared endpoint for the projection.
	
	"""
	if not uncert:
		global extrapolatedTimespace, extrapolationIndices, extrapolatedPressure, extrapolatedConcentration

		extrapolationIndices = proposedRates
		extrapolatedTimespace = np.arange(time[-1],endPoint + dt, dt)

		for rate in proposedRates:
			extrapolatedPressure.append(solve(extrapolatedTimespace, *pars, "pressure", extrapolate=rate))
			extrapolatedConcentration.append(solve(extrapolatedTimespace, *pars, "solute", extraP = extrapolatedPressure[-1], extrapolate=rate))

	pressureSol, concentrationSol = [], []

	for rate in proposedRates:
		pressureSol.append(solve(extrapolatedTimespace, *pars, "pressure", extrapolate=rate))
		concentrationSol.append(solve(extrapolatedTimespace, *pars, "solute", extraP= pressureSol[-1],extrapolate=rate))
	
	return pressureSol, concentrationSol

def uncertainty(n: int):
	pars = np.random.multivariate_normal([a, b, c], [l[:3] for l in covariance[:3]], n)
	# pars = np.random.default_rng().multivariate_normal([a, b, c], covariance, n, method="svd")

	pressurePosterior, solutePosterior = [], []
	pressurePosteriorExtrap = {rate: [] for rate in extrapolationIndices}
	solutePosteriorExtrap = {rate: [] for rate in extrapolationIndices}

	for par in pars:
		pressurePosterior.append(solve(time, *par, d, baseMass, "pressure"))
		solutePosterior.append(solve(time, *par, d, baseMass, "solute"))

		temp = extrapolate(2050, extrapolationIndices, [*par, d, baseMass], True)

		for i in range(len(extrapolationIndices)):
			pressurePosteriorExtrap[extrapolationIndices[i]].append(temp[0][i])
			solutePosteriorExtrap[extrapolationIndices[i]].append(temp[1][i])

	return pressurePosterior, solutePosterior, pressurePosteriorExtrap, solutePosteriorExtrap

## TESTING
# ------------------------------------------
# 
# ------------------------------------------

def main():
	colours = {
		0: "c",
		0.5:"m",
		1: "b",
		2: "y",
		4: "k"
	}
	## original data
	originalData = getMeasurementData(False)

	plotting1 = False
	if plotting1:
		f, ax = plt.subplots(1,1)
		for k in temp:
			ax.plot(originalData[k][0], originalData[k][1], label = k)

		ax.legend()
		ax.set_title("Original data.")
		plt.show()

	## part 2 - electric boogaloo
	getMeasurementData(True)
	interpolate(0.1)
	optimise()
	
	global analyticalPressure, analyticalSolute
	analyticalPressure = solve(time, a, b, c, d, baseMass, "pressure")
	analyticalSolute = solve(time, a, b, c, d, baseMass, "solute")

	## extrapolation
	extrapolate(2050, [0,0.5,1,2,4], [a,b,c,d,baseMass])

	plotting2 = False
	if plotting2:
		f, ax = plt.subplots(1,1)
		f1, ax2 = plt.subplots(1,1)
		# plt.subplots()
		
		ax.plot(originalData["pressure"][0], originalData["pressure"][1], 'r.', label = "measurements")
		ax.plot(time, analyticalPressure, "b", label = "analytical sol")
		
		ax2.plot(originalData["concentration"][0], originalData["concentration"][1], 'r.', label = "measurements")
		ax2.plot(time, analyticalSolute, "b", label = "analytical sol")

		for i, x in enumerate(extrapolationIndices):
			ax.plot(extrapolatedTimespace, extrapolatedPressure[i], colours[x], label = f"{x*injection[-1]} kg/s")
			ax2.plot(extrapolatedTimespace, extrapolatedConcentration[i], colours[x], label = f"{x*injection[-1]} kg/s")

		ax.legend()
		ax.set_title("Pressure solution")

		ax2.legend()
		ax2.set_title("Solute solution")
		
		plt.show()

	n = 50
	pPos, sPos, pPosEx, sPosEx = uncertainty(n)

	plotting3 = True
	if plotting3:
		f1, ax = plt.subplots(1,1)
		f2, ax2 = plt.subplots(1,1)
		
		for i in range(n):
			ax.plot(time, pPos[i], "b")
			ax2.plot(time, sPos[i], "b")
			
			for k in pPosEx:
				ax.plot(extrapolatedTimespace, pPosEx[k][i], colours[k])
				ax2.plot(extrapolatedTimespace, sPosEx[k][i], colours[k])

		plt.show()
		

	# print(a,b,c,d,baseMass)
	# print(covariance)
	return


if __name__ == "__main__":
	main()