
import numpy as np


class helper:
	"""Class containing helper functions 
	
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