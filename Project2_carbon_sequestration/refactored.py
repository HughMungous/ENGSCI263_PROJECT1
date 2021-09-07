## data handling
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

"""
Notes:
	Currently M0 is not being altered as a paramter for the solute model - it is hardcoded as ~9900
		This means that the coefficients a and b used in SoluteModel() difer from the ones in 
		PressureModel() - is this important?
"""

class Helper:
	"""Class containing helper functions 

	Methods:
		COMPLETE
			improved_euler_step(...)
				DESCRIPTION HERE

		TODO: INCOMPLETE
			benchmark - compares analytical solution vs data

			grid_search

			construct_samples

			model_ensemble

			!!!assosciated error functions^
		
	
	"""
	@staticmethod
	def improved_euler_step(self, f, tk: float, yk: float, h: float, pars: List[float])->float:
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
		f0 = f(tk, yk, *pars)
		f1 = f(tk + h, yk + h*f0, *pars)
		yk1 = yk + h*0.5*(f0 + f1)

		return yk1

# class LumpedModel:
# 	def __init__(self, sharedPars = [1, 1], pressurePars = [1], solutePars = [1,1,1]):
# 		# raw data
# 		self.time 		= []
# 		self.pressure 	= []
# 		self.production = []
# 		self.injection 	= []
# 		self.CO2_conc 	= []

# 		self.dt 				= 0.5
# 		self.basePressure 		= 6.1777
# 		self.baseConcentration 	= 0.03
# 		self.baseMass 			= 9900.495 # might need to change this to a parameter

# 		# derived data 
# 		self.net 	= [] # left in for now
# 		self.dqdt 	= []

# 		# model paramaters
# 		self.sharedPars 	= sharedPars
# 		self.pressurePars 	= pressurePars
# 		self.solutePars 	= solutePars

# 		# extrapolated data
# 		self.extrapolatedTimespace 		= []
# 		self.extrapolatedPressure 		= []
# 		self.extrapolatedConcentration 	= []

# 	def getMeasurementData(self, interpolated: bool = True)->None:
# 		"""Reads in data from the interpolated csv
# 		"""
# 		if interpolated:
# 			fileAddress = glob("output.csv")[0]
# 			vals = np.genfromtxt(fileAddress)

# 			## data extraction
# 			self.time 		= vals[:,1]
# 			self.production = vals[:,2]
# 			self.pressure 	= vals[:,3]
# 			self.injection 	= vals[:,4]
# 			self.CO2_conc 	= vals[:,5]

# 			## data cleaning
# 			self.injection[np.isnan(self.injection)] = 0
# 			self.CO2_conc[np.isnan(self.CO2_conc)] = 0.03

# 			# first value missing
# 			self.pressure[0] = self.pressure[1] 

# 			self.basePressure = self.pressure[0]

# 			# check if necessary
# 			self.finalProduction 	= self.production[-1]
# 			self.finalInjection 	= self.injection[-1]
			

# 			for i in range(len(prod)):
# 				self.net.append(self.production[i] - self.injection[i]) # getting net amount 
			
# 			self.net = np.array(self.net)

# 			self.dqdt 		= 0.* self.net
# 			self.dqdt[1:-1] = (self.net[2:]-self.net[:-2]) / self.dt # central differences
# 			self.dqdt[0] 	= (self.net[1]-self.net[0]) / self.dt    # forward difference
# 			self.dqdt[-1] 	= (self.net[-1]-self.net[-2]) / self.dt  # backward difference

			
# 			return 

# 		raise("currently this functionality is not implemented")
# 		return

# 	def pressureModel(self, ):
# 		pass


class PressureModel:
	"""Class containing pressure model methods, instantiated using: newVar = PressureModel()

		Data :
			...

		Methods :
			IMPLEMENTED/COMPLETE
			run :
				runs everything ...

			getPresureData : 
				None -> None
				reads in the data from output.csv adding it to the class data

			model : 
				float -> float
				returns the first derivative of pressure with respect to time

			solve :
				...

			interpolate :
				interpolates the data for the new timestep
				the analytical solution must be recomputed 

			optimise :
				...

			plot :
				...

			extrapolate : 
				extrapolates the data
			
			TODO/INCOMPLETE 
			!!! add docstrings !!!

	TODO:

		add other functions - ensemble

	"""
	def __init__(self, pars = [1,1,1]):
		self.time 		= []	# data for time
		self.pressure 	= []	# data for pressure
		self.net 		= []	# net sink rate 
		self.dqdt 		= []
		self.analytical = []	# analytical solution for pressure

		self.pars 		= pars	# variable parameters, default = 1,1,1
		self.cov		= [] 	# parameter covariance
		self.dt 		= 0.5	# time-step
		self.basePressure = 0	# P0

		self.finalProduction 		= 0
		self.finalInjection 		= 0
		self.extrapolatedTimespace 	= []
		self.extrapolatedSolutions 	= []

		return

	def getPressureData(self)->None:
		''' Reads all relevant data from output.csv file and adds them to the class
			in order for the other methods to function correctly.
			
			Parameters : 
			------------
			None

			Returns : 
			---------
			None (instantiates class data)
		'''
		# reads the files' values
		vals = np.genfromtxt('output.csv', delimiter = ',', skip_header= 1, missing_values= 0)

		# extracts the relevant data
		self.time 			= vals[:,1]
		self.pressure 		= vals[:,3]
		self.pressure[0] 	= self.pressure[1] # there is only one missing value

		self.basePressure 	= self.pressure[0] # P0

		prod 	= vals[:,2]	
		injec 	= vals[:,4]

		# relecant for the extrapolation
		self.finalProduction 	= prod[-1] 
		self.finalInjection 	= injec[-1]

		
		## cleaning data
		# for CO2 injection if no data is present then a rate of 0 is given for Pressure 
		# it is given the most recent value
		injec[np.isnan(injec)] = 0
		
		# calculating net flux
		for i in range(len(prod)):
			self.net.append(prod[i] - injec[i]) # getting net amount 
		
		# converting to numpy array
		self.net = np.array(self.net)

		# calculating dqdt
		self.dqdt 		= 0.* self.net
		self.dqdt[1:-1] = (self.net[2:]-self.net[:-2]) / self.dt	# central differences
		self.dqdt[0] 	= (self.net[1]-self.net[0]) / self.dt   	# forward difference
		self.dqdt[-1] 	= (self.net[-1]-self.net[-2]) / self.dt		# backward difference

		return 

	def model(self, t: float, P: float, q: float, dqdt: float, a: float, b: float, c: float)->float:
		"""Returns the first derivative of pressure with respect to time.

			parameters: (all floats)
			-----------
			t : time, seconds

			P : pressure, MPA

			q : net sink rate, kg/s

			dqdt : first derivative of net sink rate, kg/s^2
			
			a, b, c : arbitrary coefficient for each term 

			returns:
			--------
			dPdt : first derivative of pressure with respect to time, MPA/s
		"""
		return -a*q - b*(P-self.basePressure) - c*dqdt

	def solve(self, t: List[float], a: float, b: float, c: float, extrapolate: float = None)->List[float]:
		"""	
			Solves the pressure ODE using the improved euler method. This should not be used as
			a static method due to requiring certain class data.

			parameters: 
			----------
			t : time, seconds
				time points at which to evaluate pressure

			a, b, c : 
				arbitrary coefficients for PressureModel.model()

			extrapolate : 
				an optional parameter, containing the proposed rate, for when extrapolation is intended

			returns:
			--------
			analyticalPressure :
				the analytical solution for the pressure ODE
		"""
		nt = len(t)
		result = 0.*t	# creating output array 
		result[0] = self.basePressure	# setting the initial value

		params = [0, 0, a, b, c]

		if extrapolate != None:
			# assuming that the production stays constant - need to verify
			params[0] = self.finalProduction - extrapolate*self.finalInjection
			result[0] = self.analytical[-1]

			for k in range(nt-1):				
				result[k+1] = Helper.improved_euler_step(self, self.model, t[k], result[k], self.dt, params)
			
			return result

		for k in range(nt-1):
			# setting the value for q sink and dqdt
			params[0] = self.net[k]								# net sink rate, q
			params[1] = self.dqdt[k]
			
			result[k+1] = Helper.improved_euler_step(self, self.model, t[k], result[k], self.dt, params)

		return result

	def benchmark(self, t: List[float], q0: float, a: float, b: float):
		# need to check this
		nt = len(t)
		analyticalBenchmark, numericalBenchmark = 0.*t, self.solve(t, a, b, c)

		for i in range(nt):
			analyticalBenchmark[i] = self.basePressure + ((-a*q0)/b)*(1-np.exp(-b * t[i]))

		return analyticalBenchmark, numericalBenchmark
	
	def optimise(self)->None:
		"""Function which uses curve_fit() to optimise the paramaters for the ode
		"""
		self.pars, self.cov = curve_fit(self.solve, self.time, self.pressure, self.pars)
		return  

	def interpolate(self, dtNew: float)->None:
		"""This interpolates all of the data at the same time if a different time step is desired from the given one
		"""
		# creating a temporary timespace defined by the new dt
		temp = np.arange(self.time[0], self.time[-1] + dtNew, dtNew)

		# interpolating the data
		self.pressure 	= interp(temp, self.time, self.pressure)
		self.net 		= interp(temp, self.time, self.net)
		self.dqdt 		= interp(temp, self.time, self.dqdt)

		# updating the timespace and timestep
		self.time 	= temp
		self.dt 	= dtNew
		return
	
	def extrapolate(self, endPoint: float, proposedRates: List[float]):
		"""	
		This function creates projections for each of the provided rates from the endpoint
		of the analytical solution to the declared endpoint for the projection.
		
		"""
		self.extrapolatedTimespace = np.arange(self.time[-1],endPoint + self.dt, self.dt)
		for rate in proposedRates:
			self.extrapolatedSolutions.append(self.solve(self.extrapolatedTimespace, *self.pars, extrapolate=rate))
		
		return

	def plot(self, c1: str = 'r.', c2: str = 'b', extraColours = ["c","m","b","y","k"], extraLabels = ["0","24","48","96","192"])->None:
		f, ax = plt.subplots(1,1)

		ax.plot(self.time,self.pressure, c1, label = "Measurements")
		ax.plot(self.time,self.analytical, c2, label = "Analyitical Solution")

		for i in range(len(self.extrapolatedSolutions)):
			ax.plot(self.extrapolatedTimespace, self.extrapolatedSolutions[i], extraColours[i], label = extraLabels[i])

		ax.legend()
		ax.set_title("Pressure in the Ohaaki geothermal field.")
		plt.show()
		
		return

	def run(self, plotArgs: List[str] = [])->None:
		"""This function runs everything and produces a plot of the analytical solution

		TODO: 
			-	include optional paramaters for extrapolation, uncertainty
			- 	include the graph we used to validate our ode
		
		"""
		self.getPressureData()
		self.interpolate(0.1)
		self.optimise()
		self.analytical = self.solve(self.time, *self.pars)
		self.extrapolate(2050, [0, 0.5, 1, 2, 4])
		self.plot(*plotArgs)

		return

class SoluteModel:
	"""Class containing solute model methods, instantiated using: newVar = SoluteModel()

		Data : 
			- ...

		Methods :
			- ...

	TODO:
	
		- ...
	"""
	def __init__(self, pars = [1, 1, 0.228646653, 9900.495]):
		self.time 		= []	# timespace
		self.pressure 	= []	# pressure
		self.qCO2 		= []	# CO2 injection rate
		self.CO2_conc 	= []	# CO2 concentration
		self.analytical = []
		
		self.pars 	= pars
		self.cov 	= []

		self.extrapolatedPressure 	= []
		self.extrapolatedTimespace 	= []
		self.extrapolatedSolutions 	= []
		self.extrapolationIndices 	= []

		self.dt 				= 0.5
		self.basePressure 		= 6.1777
		self.baseConcentration 	= 0.03
		self.baseMass 			= 9900.495 	# hardcoded M0 # might need to change this to a parameter
		
	def getConcentrationData(self)->None:
		''' Reads all relevant data from output.csv file and adds them to the class
			in order for the other methods to function correctly.
			
			Parameters : 
			------------
			None

			Returns : 
			---------
			None (instantiates class data)

			Notes :
			------
				CO2 concentration before injection is assumed to be natural state
				of 3 wt %. 
		'''
		# reads all the data from excel file
		vals = np.genfromtxt('output.csv', delimiter = ',', skip_header= 1, missing_values= 0)
		
		## extracting the relevant data
		self.time 		= vals[:,1] # time values
		self.pressure 	= vals[:,3] # Pressure values
		self.qCO2 		= vals[:,4] # CO2 injection values 
		self.CO2_conc 	= vals[:,5]	# CO2 concentration values
		
		## Cleaning the data
		self.qCO2[np.isnan(self.qCO2)] 			= 0 	# absence of injection values is 0
		self.CO2_conc[np.isnan(self.CO2_conc)] 	= 0.03 	# inputting natural state 

		self.pressure[0] = self.pressure[1]			# missing initial pressure data point
		
		self.basePressure = self.pressure[0]

		# for ar in [self.time, self.pressure, self.qCO2, self.CO2_conc]:
		# 	if np.NaN in ar: raise("deez nuts")
		return 

	def model(self, t: float, C: float, qCO2: float, P: float, a: float, b: float, d: float, M0: float = 9900.495)->float:
		''' Return the Solute derivative dC/dt at time, t, for given parameters.
			Parameters:
			-----------
			t : time, seconds
				Independent variable.
			C : CO2 concentration, weight %
				Dependent variable. (Current CO2 concentration)
			qCO2 : CO2 injection rate, kg/s
				Source/sink rate. (injection rate of CO2)
			a : float
				Source/sink strength parameter.
			b : float
				Recharge strength parameter.
			d  : float
				Recharge strength parameter
			P : float
				Pressure at time point t
			
			Returns:
			--------
			dCdt : float
				Derivative of Pressure variable with respect to independent variable.
		'''
		qLoss, cPrime = 0, self.baseConcentration

		if P > self.basePressure:
			cPrime = C
			# the loss due to a higher than baseline pressure during the space between  injection periods
			qLoss = (b / a) * (P - self.basePressure) * cPrime * self.dt	
		
		qCO2 -= qLoss

		# return ((1 - C) * (qCO2 / self.baseMass)) - ((b / (a * self.baseMass)) * (P - self.basePressure) * (cPrime - C)) - (d * (C - self.baseConcentration))
		return ((1 - C) * (qCO2 / M0)) - ((b / (a * M0)) * (P - self.basePressure) * (cPrime - C)) - (d * (C - self.baseConcentration))

	def solve(self, t: List[float], a: float, b: float, d: float, M0: float = 9900.495, extrapolate = None)->List[float]:
		nt = len(t)
		result = 0.*t
		result[0] = self.baseConcentration

		params = [0, 0, a, b, d]
		
		if extrapolate != None:
			# assuming that the production stays constant - need to verify
			params[0] = self.qCO2[-1]*extrapolate

			result[0] = self.analytical[-1]

			for k in range(nt-1):	
				params[1] = self.extrapolatedPressure[self.extrapolationIndices.index(extrapolate)][k]			
				result[k+1] = Helper.improved_euler_step(self, self.model, t[k], result[k], self.dt, params)
			
			return result


		for k in range(nt-1):
			params[0] = self.qCO2[k]
			params[1] = self.pressure[k]
			result[k+1] = Helper.improved_euler_step(self, self.model, t[k], result[k], self.dt, params)

		return result

	def benchmark(self):
		pass
	
	def optimise(self)->None:
		self.pars[:-1], self.cov = curve_fit(self.solve, self.time, self.CO2_conc, self.pars[:-1], method="lm")
		return  

	def interpolate(self, dtNew: float)->None:
		# creating a temporary timespace defined by the new dt
		temp = np.arange(self.time[0], self.time[-1] + dtNew, dtNew)

		# interpolating the data
		self.pressure = interp(temp, self.time, self.pressure)
		self.qCO2 = interp(temp, self.time, self.qCO2)
		self.CO2_conc = interp(temp, self.time, self.CO2_conc)

		# updating the timespace and timestep
		self.time = temp
		self.dt = dtNew
		return

	def extrapolate(self, endPoint: float, proposedRates: List[float])->None:
		"""	
		This function creates projections for each of the provided rates from the endpoint
		of the analytical solution to the declared endpoint for the projection.
		
		"""
		self.extrapolationIndices = proposedRates
		self.extrapolatedTimespace = np.arange(self.time[-1],endPoint + self.dt, self.dt)
		
		for rate in proposedRates:
			self.extrapolatedSolutions.append(self.solve(self.extrapolatedTimespace, *self.pars, extrapolate=rate))
		
		return

	def plot(self, c1: str = 'r.', c2: str = 'b', extraColours = ["c","m","b","y","k"], extraLabels = ["0","24","48","96","192"])->None:
		f, ax = plt.subplots(1,1)

		ax.plot(self.time,self.CO2_conc, c1, label = "Measurements")
		ax.plot(self.time,self.analytical, c2, label = "Analyitical Solution")

		for i in range(len(self.extrapolatedSolutions)):
			ax.plot(self.extrapolatedTimespace, self.extrapolatedSolutions[i], extraColours[i], label = extraLabels[i])

		ax.legend()
		ax.set_title("Pressure in the Ohaaki geothermal field.")
		plt.show()
		
		return
	
	def run(self, plotArgs = [])->None:
		self.getConcentrationData()
		self.interpolate(0.1)
		self.optimise()
		self.analytical = self.solve(self.time, *self.pars)
		self.extrapolate(2050, [0, 0.5, 1, 2, 4])
		self.plot(*plotArgs)

		return
## ---------------------------------------------------------
## ---------------------------------------------------------
## ---------------------------------------------------------
def main():
	pass

if __name__ == "__main__":
	# t = LumpedModel()
	# t.getMeasurementData()

	
	if input("Y/N? ") in "yY":
		pressureModel = PressureModel()
		pressureModel.run()
		
		# print(pressureModel.pars)
		soluteModel = SoluteModel()

		soluteModel.pars[0] = pressureModel.pars[0]	# copying the value for a
		soluteModel.pars[1] = pressureModel.pars[1] # copying the value for b
		soluteModel.extrapolatedPressure = pressureModel.extrapolatedSolutions.copy() # this needs to be reworked
		

		soluteModel.run()
		print(soluteModel.cov)
		print(soluteModel.pars)
		# print(soluteModel.pars)
	
	
	pass