## data handling
from glob import glob

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
		self.time = []			# data for time
		self.pressure = []		# data for pressure
		self.net = []			# net sink rate 
		self.dqdt = []
		self.analytical = []	# analytical solution for pressure
		self.pars = pars		# variable parameters, default = 1,1,1
		self.dt = 0.5			# time-step
		self.basePressure = 0

		self.finalProduction = 0
		self.finalInjection = 0
		self.extrapolatedTimespace = []
		self.extrapolatedSolutions = []

		## TODO: current method uses interpolated data as the original values and thus plots that
		# self.originalTime = []
		# self.originalPressure = []
		# self.run()
		return

	def getPressureData(self)->None:
		'''
			Reads all relevant data from output.csv file
			
			Parameters : 
			------------
			None

			Returns : 
			---------
			None (instantiates class data)

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
		self.time = vals[:,1]
		self.pressure = vals[:,3]
		self.pressure[0] = self.pressure[1] # there is only one missing value
		self.pars[-1] = self.pressure[0]

		self.basePressure = self.pressure[0]

		prod = vals[:, 2]
		injec = vals[:,4]

		self.finalProduction = prod[-1]
		self.finalInjection = injec[-1]
		# print(statistics.mean(prod[71:]/injec[71:]),statistics.variance(prod[71:]/injec[71:]))
		# print(statistics.mean(prod[71:]-injec[71:]),statistics.variance(prod[71:]-injec[71:]))
		
		# cleans data
		# for CO2 injection if no data is present then a rate of 0 is given for Pressure 
		# it is given the most recent value
		injec[np.isnan(injec)] = 0
		

		for i in range(len(prod)):
			self.net.append(prod[i] - injec[i]) # getting net amount 
		
		self.net = np.array(self.net)

		self.dqdt = 0.* self.net
		self.dqdt[1:-1] = (self.net[2:]-self.net[:-2]) / self.dt # central differences
		self.dqdt[0] = (self.net[1]-self.net[0]) / self.dt       # forward difference
		self.dqdt[-1] = (self.net[-1]-self.net[-2]) / self.dt    # backward difference

		
		return 

	def model(self, t: float, P: float, q: float, dqdt: float, a: float, b: float, c: float)->float:
		"""Returns the first derivative of pressure with respect to time
		"""
		dPdt = -a*q - b*(P-self.basePressure) - c*dqdt
		return dPdt

	def solve(self, t: List[float], a: float, b: float, c: float, extrapolate = None)->List[float]:
		""" Solves ode ...
		
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

	def optimise(self)->None:
		"""Function which uses curve_fit() to optimise the paramaters for the ode
		"""
		self.pars = curve_fit(self.solve, self.time, self.pressure, self.pars)[0]
		return  

	def interpolate(self, dtNew: float)->None:
		"""This function reformats the data if we wish to change the timestep
		
		...

		"""
		# creating a temporary timespace defined by the new dt
		temp = np.arange(self.time[0], self.time[-1] + dtNew, dtNew)

		# interpolating the data
		self.pressure = interp(temp, self.time, self.pressure)
		self.net = interp(temp, self.time, self.net)
		self.dqdt = interp(temp, self.time, self.dqdt)

		# updating the timespace and timestep
		self.time = temp
		self.dt = dtNew
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

	def plot(self, c1: str = 'r.', c2: str = 'b', extraColours = ["b","c","m","y","k"], extraLabels = ["0","24","48","96","192"])->None:
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
	def __init__(self, pars = [1, 1, 0.228646653]):
		self.time = []		# timespace
		self.pressure = []	# pressure
		self.qCO2 = []		# CO2 injection rate
		self.CO2_conc = []	# CO2 concentration
		self.analytical = []
		self.pars = pars

		# self.a = 0
		# self.b = 0

		self.dt = 0.5
		self.basePressure = 6.1777
		self.baseConcentration = 0.03
		self.baseMass = 9900.495 # need to change this
		

	def getConcentrationData(self)->None:
		'''	Reads all relevant data from output.csv file

			Parameters : 
			------------
				None

			Returns : 
			---------
				None, modifies class data

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
		
		## extracting the relevant data
		self.time = vals[:,1] 		# time values
		self.pressure = vals[:,3] 	# Pressure values
		self.qCO2 = vals[:,4] 		# CO2 injection values 
		self.CO2_conc = vals[:,5]	# CO2 concentration values
		
		## Cleaning the data
		self.qCO2[np.isnan(self.qCO2)] = 0 				# absence of injection values is 0
		self.CO2_conc[np.isnan(self.CO2_conc)] = 0.03 	# inputting natural state 

		self.pressure[0] = self.pressure[1]			# missing initial pressure data point
		
		self.basePressure = self.pressure[0]

		# for ar in [self.time, self.pressure, self.qCO2, self.CO2_conc]:
		# 	if np.NaN in ar: raise("deez nuts")
		return 

	def model(self, t: float, C: float, qCO2: float, P: float, a: float, b: float, d: float)->float:
		''' Return the Solute derivative dC/dt at time, t, for given parameters.
			Parameters:
			-----------
			t : float
				Independent variable.
			C : float
				Dependent variable. (Current CO2 concentration)
			qCO2 : float
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

		if (P > self.basePressure):
			cPrime = C
			# the loss due to a higher than baseline pressure during the space between  injection periods
			qLoss = (b / a) * (P - self.basePressure) * cPrime * self.dt	
		
		qCO2 -= qLoss

		return ((1 - C) * (qCO2 / self.baseMass)) - ((b / (a * self.baseMass)) * (P - self.basePressure) * (cPrime - C)) - (d * (C - self.baseConcentration))

	def solve(self, t: List[float], a: float, b: float, d: float, extrapolate = None)->List[float]:
		nt = len(t)
		result = 0.*t
		result[0] = self.baseConcentration
		
		# self.baseMass = M0
		params = [0, 0, a, b, d]

		for k in range(nt-1):
			params[0] = self.qCO2[k]
			params[1] = self.pressure[k]
			result[k+1] = Helper.improved_euler_step(self, self.model, t[k], result[k], self.dt, params)

		return result

	def optimise(self)->None:
		self.pars = curve_fit(self.solve, self.time, self.CO2_conc, self.pars)[0]
		return  

	def interpolate(self, dtNew: float):
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

	def extrapolate(self, endPoint: float, proposedRates: List[float]):
		"""	
		This function creates projections for each of the provided rates from the endpoint
		of the analytical solution to the declared endpoint for the projection.
		
		"""
		self.extrapolatedTimespace = np.arange(self.time[-1],endPoint + self.dt, self.dt)

		for rate in proposedRates:
			self.extrapolatedSolutions.append(self.solve(self.extrapolatedTimespace, *self.pars, extrapolate=rate))
		
		return

	def plot(self, c1: str = 'r.', c2: str = 'b', extraColours = ["b","c","m","y","k"], extraLabels = ["0","24","48","96","192"])->None:
		f, ax = plt.subplots(1,1)

		ax.plot(self.time,self.CO2_conc, c1, label = "Measurements")
		ax.plot(self.time[:-20],self.analytical[:-20], c2, label = "Analyitical Solution")

		# for i in range(len(self.extrapolatedSolutions)):
		# 	ax.plot(self.extrapolatedTimespace, self.extrapolatedSolutions[i], extraColours[i], label = extraLabels[i])

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


if __name__ == "__main__":
	pressureModel = PressureModel()
	pressureModel.getPressureData()
	pressureModel.optimise()

	# print(pressureModel.pars)
	# raise("deez nuts")

	soluteModel = SoluteModel()
	
	soluteModel.pars[0] = pressureModel.pars[0]	# copying the value for a
	soluteModel.pars[1] = pressureModel.pars[1] # copying the value for b

	

	soluteModel.getConcentrationData()
	
	soluteModel.optimise()
	soluteModel.analytical = soluteModel.solve(soluteModel.time,*soluteModel.pars)

	if 1:
	# if input() == "plot":
		f, ax = plt.subplots(1,1)

		ax.plot(soluteModel.time[:70], soluteModel.CO2_conc[:70], "r.", label= "Measurements")
		ax.plot(soluteModel.time[:70], soluteModel.analytical[:70], "b-", label= "Analytical")

		ax.legend()
		ax.set_title("CO2 weight percentage")
		plt.show()
	pass