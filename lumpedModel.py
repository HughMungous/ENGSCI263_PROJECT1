
"""
def pressureModel(time, P, q, dqdt, a, b, c, P0):
    return -a*q - b*(P-P0) - c*dqdt

def soluteModel(time, C, P, q, ... a, b, d, M0, P0, C0):
    qLoss, cPrime = 0, self.baseConcentration

    if P > self.basePressure:
        cPrime = C
        # the loss due to a higher than baseline pressure during the space between  injection periods
        qLoss = (b / a) * (P - self.basePressure) * cPrime * self.dt	
    
    qCO2 -= qLoss

    injection   = ((1 - C) * (qCO2 / self.baseMass)) 
    recharge    = ((b / (a * self.baseMass)) * (P - self.basePressure) * (cPrime - C))
    reaction    = (d * (C - self.baseConcentration))

    return injection - recharge - reaction

def lumpedModel(time, C, P, q, dqdt,... a, b, c, d, M0, ... P0, C0):
    qLoss, cPrime = 0, C0

    if P > P0:
        cPrime = C

        qLoss = (b / a) * (P - P0) * cPrime * self.dt	

    q -= qLoss
    dqdt += q
"""

class LumpedModel:
	def __init__(self, sharedPars = [1, 1], pressurePars = [1], solutePars = [1,1,1]):
		# raw data
		self.time 		= []
		self.pressure 	= []
		self.production = []
		self.injection 	= []
		self.CO2_conc 	= []

		self.dt 				= 0.5
		self.basePressure 		= 6.1777
		self.baseConcentration 	= 0.03
		self.baseMass 			= 9900.495 # might need to change this to a parameter

		# derived data 
		self.net 	= [] # left in for now
		self.dqdt 	= []

		# model paramaters
		self.sharedPars 	= sharedPars
		self.pressurePars 	= pressurePars
		self.solutePars 	= solutePars

		# extrapolated data
		self.extrapolatedTimespace 		= []
		self.extrapolatedPressure 		= []
		self.extrapolatedConcentration 	= []

	def getMeasurementData(self, interpolated: bool = True)->None:
		"""Reads in data from the interpolated csv
		"""
		if interpolated:
			fileAddress = glob("output.csv")[0]
			vals = np.genfromtxt(fileAddress)

			## data extraction
			self.time 		= vals[:,1]
			self.production = vals[:,2]
			self.pressure 	= vals[:,3]
			self.injection 	= vals[:,4]
			self.CO2_conc 	= vals[:,5]

			## data cleaning
			self.injection[np.isnan(self.injection)] = 0
			self.CO2_conc[np.isnan(self.CO2_conc)] = 0.03

			# first value missing
			self.pressure[0] = self.pressure[1] 

			self.basePressure = self.pressure[0]

			# check if necessary
			self.finalProduction 	= self.production[-1]
			self.finalInjection 	= self.injection[-1]
			

			for i in range(len(prod)):
				self.net.append(self.production[i] - self.injection[i]) # getting net amount 
			
			self.net = np.array(self.net)

			self.dqdt 		= 0.* self.net
			self.dqdt[1:-1] = (self.net[2:]-self.net[:-2]) / self.dt # central differences
			self.dqdt[0] 	= (self.net[1]-self.net[0]) / self.dt    # forward difference
			self.dqdt[-1] 	= (self.net[-1]-self.net[-2]) / self.dt  # backward difference

			
			return 

		raise("currently this functionality is not implemented")
		return

	def pressureModel(self, time: float, P: float, net: float, dqdt: float, a: float, b: float, c: float):
		return -a*q - b*(P-self.basePressure) - c*dqdt

    def soluteModel(self, time, P, C, cPrime):
        pass