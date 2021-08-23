
import ntpath
import numpy as np
from numpy.core.numeric import NaN
from matplotlib import pyplot as plt
from numpy.lib.function_base import interp
from scipy.interpolate import interp1d
import itertools
from scipy.optimize import curve_fit
import statistics

"""
Do you think we should generate a base class containing the basic ode? 
	probably unnescessary from my pov.

"""

class Helper:
	"""Class containing helper functions 
	
	TODO:

	- Euler step - semi done? could include a bool for sebs code
	- OPTIONAL: 
		pressure ode
		solute ode?
	"""
	def improved_euler_step(self, f, tk: float, yk: float, h: float, x0: float, pars)->float:
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
		yk1 = yk + h*0.5*(f0 + f1)
		
		return yk1

class Plotting:
	"""Class contianing plotting functions
	
	TODO:
		plot pressure
		plot solute
		plot más?

	"""
	pass

class DataInput:
	

	def getConcentrationData(self):
		pass

class PressureModel:
	"""Class containing pressure model methods, instantiated using: newVar = PressureModel()

		Data :
			...

		Functions :
			getPresureData : 
				None -> None
				reads in the data from output.csv adding it to the class data

			model : 
				float -> float
				returns the first derivative of pressure with respect to time
	TODO:

		add other functions

	"""
	def __init__(self):
		self.time = []
		self.pressure = []
		self.net = []
		self.pars = [1, 1, 0.0012653061224489797, 0.09836734693877551, 0.0032244897959183673] # maybe not under __init__?
		self.dt = 0.5

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
		self.net = []

		prod = vals[:, 2]
		injec = vals[:,4]

		# cleans data
		# for CO2 injection if no data is present then a rate of 0 is given for Pressure 
		# it is given the most recent value
		injec[np.isnan(injec)] = 0
		
		for i in range(len(prod)):
			self.net.append(prod[i] - injec[i]) # getting net amount 

		return 

	def model(self, t: float, P: float, q: float, dqdt: float, a: float, b: float, c: float, P0: float)->float:
		"""Returns the first derivative of pressure with respect to time
		"""
		dPdt = -a*q - b*(P-P0) - c*dqdt
		return dPdt

	def solve(self, t, y0: float, a: float, b: float, c: float):
		""" Solves ode ...
		
		"""
		nt = len(t)
		if nt != len(self.net):
			raise ValueError("If the original time data is not used then the net sink must be interpolated to fit")

		result = 0.*t
		result[0] = y0

		params = [0, a, b, c, 0]

		for k in range(1, nt):
			# setting the value for q sink and dqdt
			params[0] = self.net[k]
			params[-1] = (self.net[k] - self.net[k-1]) / self.dt
			
			result[k] = Helper.improved_euler_step(self, self.model, t[k], result[k-1], self.dt, y0, params)

		return result

	def optimise(self)->None:
		"""Function which uses curve_fit() to optimise the paramaters for the ode
		"""
		
		pass

class SoluteModel:
	pass

## ---------------------------------------------------------
## ---------------------------------------------------------
## ---------------------------------------------------------

def main():

	## 	TEST 1
	model1 = PressureModel()
	model1.getPressureData()
	print(model1.solve(model1.time, model1.pressure[0], *model1.pars[2:]))
	pass

if __name__ == "__main__":
	main()