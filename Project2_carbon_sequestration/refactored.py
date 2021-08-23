
import numpy as np

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
	def getPressureData(self)->None:
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

		TODO:
			thinking of making this return nothing and only store as class data	
		'''
			# reads the files' values
		vals = np.genfromtxt('output.csv', delimiter = ',', skip_header= 1, missing_values= 0)

		# extracts the relevant data
		self.time = vals[:,1]
		self.pressure = vals[:,3]
		self.pressure[0] = self.pressure[1] # there is only one missing value

		prod = vals[:, 2]
		injec = vals[:,4]

		# cleans data
		# for CO2 injection if no data is present then a rate of 0 is given for Pressure 
		# it is given the most recent value
		injec[np.isnan(injec)] = 0
		
		self.net = []

		for i in range(len(prod)):
			self.net.append(prod[i] - injec[i]) # getting net amount 

		return 


	pass

class SoluteModel:
	pass

## ---------------------------------------------------------
## ---------------------------------------------------------
## ---------------------------------------------------------

def main():
	pass

if __name__ == "__main__":
	pass