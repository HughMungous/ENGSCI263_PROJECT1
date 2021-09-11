import numpy as np
from numpy.core.numeric import NaN
from matplotlib import pyplot as plt
from numpy.lib.function_base import interp
from scipy.interpolate import interp1d
import itertools
from scipy.optimize import curve_fit

# getting the data from the files suplied
tp, pp = np.genfromtxt('data/cs_p.txt', delimiter = ',', skip_header=1).T
tqc, qc = np.genfromtxt('data/cs_c.txt', delimiter = ',', skip_header=1).T
tq, q = np.genfromtxt('data/cs_q.txt', delimiter = ',', skip_header=1).T
tcc, cc = np.genfromtxt('data/cs_cc.txt', delimiter = ',', skip_header=1).T


net = np.append(q[0:33],(q[33::]-qc)) # creating net flow array
dqdt = net*0. # initialising dqdt array
dqdt[0] = (net[1] - net[0])/(tq[1] - tq[0]) # forwards differnce
dqdt[-1] = (net[-1] - net[-2])/(tq[-1] - tq[-2]) # backwards differences
dqdt[1:-1] = (net[2:] - net[:-2])/(tq[2:] - tq[:-2]) # central difference

qc = np.append(np.zeros(33), qc) # making sure qc is same length as net array

# initialising global variables
P_SOL = []
C_SOL = []
extrapolation = False
d = 0
M0 = 0
extraPressure = []
k = 0
dt = 0.4


def main():

    PlotOriginal() # plotting the original data

    global a,b,c
    a,b,c = MSE() # getting initial model fit 

    Model_Fit() # fitting the model 

    Extrapolate(2050) # extrapolating data into the future

    PlotMisfit() # getting the misfit plots

    BenchMark() # plotting the benchmarks for both ODEs

    Uncertainty() # performing uncertainty analysis

    return

def PlotOriginal():
    """ Plots the given data before any changes are made
	
		Parameters
		----------
		None
			
		Returns
		-------
		None
	"""
    # reading in the data
    tm, CO2_inj = np.genfromtxt('data/cs_c.txt',delimiter=',',skip_header=1).T
    tq, CO2_perc = np.genfromtxt('data/cs_cc.txt',delimiter=',',skip_header=1).T
    ty, pres = np.genfromtxt('data/cs_p.txt',delimiter=',',skip_header=1).T
    tz, prod = np.genfromtxt('data/cs_q.txt',delimiter=',',skip_header=1).T
    # plotting the data
    f, ax1 = plt.subplots(nrows=1,ncols=1)
    ax2 = ax1.twinx()
    ax1.plot(tm, CO2_inj, 'b-', label= 'CO2 Injection')
    ax1.plot(tz, prod, 'r-', label= 'Extraction')
    ax2.plot(tq, CO2_perc*100, 'y-', label= 'CO2 Percentage')
    ax1.legend(loc=2)
    ax2.legend(loc = 1)
    ax1.set_ylabel('production and injection rate [kg/s]')
    ax2.set_ylabel('CO2 concentration [wt %]')
    ax1.set_xlabel('time [yr]')
    ax2.set_title('Extraction and Injection rates with CO2 Concentration at Ohaaki')
    plt.show()

    f,ax1 = plt.subplots(nrows=1,ncols=1)
    ax2 = ax1.twinx()
    ax1.plot(tm, CO2_inj, 'b-', label= 'CO2 Injection')
    ax1.plot(tz, prod, 'r-', label= 'Extraction')
    ax2.plot(ty, pres, 'g-', label= 'Pressure')
    ax1.legend(loc=2)
    ax2.legend(loc = 1)
    ax1.set_ylabel('production and injection rate [kg/s]')
    ax2.set_ylabel('pressure [MPa]')
    ax1.set_xlabel('time [yr]')
    ax2.set_title('Extraction and Injection rates with Pressure Change at Ohaaki')        
    plt.show()
    return



def BenchMark():
    """ Plotting the benchmark solution for both ODE. This includes an instability plot.
	
		Parameters
		----------
		None
			
		Returns
		-------
		None
	"""
    dt = 0.1 # sets up time step

    time = np.arange(0, 10, dt) # creates time array to solve over

    global net # setting up parameters and global variables
    net = 4
    a = 1
    b = 2
    c = 0
    q0 = 4
    # benchmarking for Pressure ODE
    ys, analytical = PressureBenchmark(pp[0], a, b, c, q0, time, dt) # getting benchmark solution for analytical and numerical solution
    steady_state = pp[0] - (a*q0)/b # getting the steady state value

    f, ax = plt.subplots(1, 1) # plotting numerical vs analytical solutions
    ax.plot(time,analytical, 'b', label = 'Analytical')
    ax.plot(time, ys, 'kx', label = 'Numerical')
    ax.set_xlabel("Time [seconds]")
    ax.set_ylabel("Pressure [MPa]")
    ax.axhline(steady_state, linestyle = '--', color = 'red', label = 'steady state')
    ax.legend()
    ax.set_title("Analytical vs Numerical Solution Benchmark for Pressure ODE")
    plt.show()

    dt = 1.1 # changing time step for instability analysis
    time = np.arange(0,10, dt)
    ys, analytical = PressureBenchmark(pp[0], 1, 2, 0, 4, time, dt) # getting benchmark solutions

    f, ax = plt.subplots(1, 1) # plotting numerical vs analytical solutions
    ax.plot(time,analytical, 'b', label = 'Analytical')
    ax.plot(time,ys, 'kx', label = 'Numerical')
    ax.set_xlabel("Time [seconds]")
    ax.set_ylabel("Pressure [MPa]")
    ax.axhline(steady_state, linestyle = '--', color = 'red', label = 'steady state')
    ax.legend()
    ax.set_title("Instability at a large time step for Pressure ODE")
    plt.show()
    
    # Benchmarking for Solute ODE
    dt = 0.25 # performing same process as above except for Solute ODE

    time = np.arange(0, 10, dt)

    global injec # setting up parameters and global variables necessary
    injec = 1
    a = 1
    b = 2
    d = 3
    M0 = 1

    ys, analytical = SoluteBenchmark(cc[0], injec, d, M0, time, dt) # getting numerical and analytical results

    steady_state = ((injec/M0) + d*cc[0])/((injec/M0) + d) # getting steady state equation

    f, ax = plt.subplots(1, 1) # plotting analytical vs numerical results
    ax.plot(time,analytical, 'b', label = 'Analytical')
    ax.plot(time, ys, 'kx', label = 'Numerical')
    ax.axhline(steady_state, linestyle = '--', color = 'red', label = 'steady state')
    ax.set_xlabel("Time [seconds]")
    ax.set_ylabel("CO2 Concentration [wt %]")
    ax.legend()
    ax.set_title("Analytical vs Numerical Solution Benchmark for Solute ODE")
    plt.show()
    # performing instability analysis
    dt = 1.1
    time = np.arange(0,10, dt)
    ys, analytical = SoluteBenchmark(cc[0], injec, d, M0, time, dt)

    f, ax = plt.subplots(1, 1)
    ax.plot(time,analytical, 'b', label = 'Analytical')
    ax.plot(time,ys, 'kx', label = 'Numerical')
    ax.axhline(steady_state, linestyle = '--', color = 'red', label = 'steady state')
    ax.set_xlabel("Time [seconds]")
    ax.set_ylabel("CO2 Concentration [wt %]")
    ax.legend()
    ax.set_title("Instability at a large time step for Solute ODE")
    plt.show()

    return

def SoluteBenchmark(C0, qCO2, d, M0, time, dt):
    """ Solves Solute ODE analytically and numerically
	
		Parameters
		----------
		C0   : flota
			Initial CO2 concentration of the reservoir (wt %)
	    qCO2 : float
			Injection rate of CO2 into the reservoir (kg/s)
		d    : float
			Parameter strength of CO2 reaction
		M0   : float
			Initial mass of the reservoir (kg)
		time : np.array
			Time range to be solved over
        dt   : float
			Step size.
		Returns
		-------
		ys   : np.array
			Numerical Solution of the Solute ODE.
        analytical : np.array
            Analytical Solution of the Solute ODE.
	"""
    ys, analytical = 0.*time, 0.*time # initialsiing the numerical and analytical arrays
    
    # calculating terms used in the analytical solution
    k = qCO2/M0  
    L = (k*C0 - k)/(k + d)

    analytical[0] = (k + (d * C0))/(k + d) + L/(np.exp(time[0]*(k+d)))
    ys[0] = cc[0]

    pars = [d,M0] # parameters used by the Solute ODE

    for i in range(len(time)-1):
        analytical[i+1] = (k + (d * C0))/(k + d) + L/(np.exp(time[i+1]*(k+d))) # calculating the analytical solution at differnt time steps
        ys[i+1] = improved_euler_step(SoluteModel, time[i], ys[i], dt, pars)   # solving the Solute ODE numerically using improved euler
    
    return ys, analytical # returning numerical and analytical solutions

def PressureBenchmark(P0, a, b , c, q0, time, dt):
    """ Solves Pressure ODE analytically and numerically
	
		Parameters
		----------
		P0   : float
			Initial Pressure of the reservoir (MPa)
	    a : float
	    	Parameter strength of injection/extraction 
		b    : float
			Parameter strength of recharge
		c    : float
			Parameter strength of slow drainage
		q0   : float
			Net extraction/injection rate
        time : np.array
			Time range to solve over
        dt   : float
            Step size.
		Returns
		-------
		ys   : np.array
			Numerical Solution of the Pressure ODE.
        analytical : np.array
            Analytical Solution of the Pressure ODE.
	"""
    # initialsiing the numerical and analytical arrays
    ys, analytical = 0.*time, 0.*time

    # setting initial values
    analytical[0] = P0 + ((-a*q0)/b)*(1-np.exp(-b*time[0]))
    ys[0] = P0 

    pars = [a,b,c] # parameters to pass into the model
    
    for i in range(len(time)-1):
        analytical[i+1] = P0 + ((-a*q0)/b)*(1-np.exp(-b*time[i+1])) # solves the analytical solution at different time points
        ys[i+1] = improved_euler_step(PressureModel, time[i], ys[i], dt, pars) # solves numerical ODE using improved euler method 
    return ys, analytical # retunrs numerical and analytical solutions

def PlotMisfit():
    """ Plots the misfit of the data
	
		Parameters
		----------
		None

		Returns
		-------
		None

	"""
    # reads in the given data
    pressure_time = np.genfromtxt('data/cs_p.txt', skip_header = 1,delimiter = ',', usecols = 0)
    pressure = np.genfromtxt('data/cs_p.txt', skip_header = 1,delimiter = ',', usecols = 1)

    P_Result = [] # initialises the result array
    # since solution is solved over different time period, need to get Pressure data at same points as pressure_time
    for i in range(len(pressure_time)):
        P_Result.append(np.interp(pressure_time[i], time_fit, P_SOL))

    misfit_P = [] # initialises misfit array
    # calculates the misfit for all the points
    for i in range(len(P_Result)):
        misfit_P.append(pressure[i] - P_Result[i])
    # plots the misfit for the Presssure Solution
    f, ax = plt.subplots(1, 1)
    ax.plot(pressure_time,misfit_P, 'rx')
    ax.axhline(0, color = 'black', linestyle = '--')
    ax.set_ylabel('Pressure [MPa]')
    ax.set_xlabel('Time [years]')
    ax.set_title("Best Fit Pressure LPM Model")
    plt.show()

    # reads in the relevant solute data
    solute_time = np.genfromtxt('data/cs_cc.txt', skip_header = 1, delimiter = ',', usecols = 0)
    solute = np.genfromtxt('data/cs_cc.txt', skip_header = 1, delimiter = ',', usecols = 1)
    # performs the same process as the pressure data
    C_Result = []
    for i in range(len(solute_time)):
        C_Result.append(np.interp(solute_time[i], time_fit, C_SOL))
    misfit_C = []
    for i in range(len(C_Result)):
        misfit_C.append(solute[i] - C_Result[i])
    # graphs the misfit for solute ODE
    f, ax = plt.subplots(1, 1)
    ax.plot(solute_time,misfit_C, 'rx')
    ax.axhline(0, color = 'black', linestyle = '--')
    ax.set_ylabel('CO2 [wt %]')
    ax.set_title("Best Fit Solute LPM Model")
    plt.show()
    return
    
def Model_Fit():
    """ Gets the best fit for both models using curve_fit function 
    
		Parameters
		----------
		None

		Returns
		-------
		None

	"""
    global time_fit, a, b, c # setting up values for parameters

    time_fit = np.arange(tp[0], tp[-1], dt) # time range to solve over

    pars = [a,b,c] # parameters for Pressure ODE

    global pressurecov
    # first output is the optimal parameters of Pressure ODE
    # second output is the covariance matrix, saved as a global variable to use for uncertainty
    bestfit_pars, pressurecov = curve_fit(SolvePressureODE, tp, pp, pars)
    
    global P_SOL # sets up solved pressure values
    a = bestfit_pars[0]
    b = bestfit_pars[1]
    c = bestfit_pars[2]
    # prints the best values for the Pressure ODE
    print("Best parameters found : \na = " + str(a) + "\n" + "b = " + str(b) + "\n" + "c = " + str(c))

    P_SOL = SolvePressureODE(time_fit, *bestfit_pars) # solves the Pressure ODE using new parameters
    # plotting the data vs the solution found
    f, ax = plt.subplots(1, 1)
    ax.plot(time_fit,P_SOL, 'b', label = 'ODE')
    ax.plot(tp,pp, 'r.', label = 'DATA')
    #plt.axvline(tp[34], color = 'black', linestyle = '--', label = 'Calibration point')
    ax.set_xlabel("Time [years]")
    ax.set_ylabel("Pressure [MPa]")
    ax.legend()
    ax.set_title("Pressure flow in the Ohaaki geothermal field")
    plt.show()
    # initial guess for d and M0
    pars = [0.1, 999999999999]
    global solutecov
    # same process as the pressure curve_fit
    bestfit_pars, solutecov = curve_fit(SolveSoluteODE, tcc, cc, pars)

    global d, M0, C_SOL # sets up global variables
    d = bestfit_pars[0]
    M0 = bestfit_pars[1]

    print("d = " + str(d) + "\n" + "M0 = " + str(M0) + "\n") # prints the best variables for d and M0

    C_SOL = SolveSoluteODE(time_fit, *bestfit_pars) # solves with new d and M0 parameters
    # plots the solved solute values compared to the data
    f, ax = plt.subplots(1, 1)
    ax.plot(time_fit,C_SOL, 'b', label = 'ODE')
    ax.plot(tcc,cc, 'r.', label = 'DATA')
    #plt.axvline(tcc[15], color = 'black', linestyle = '--', label = 'Calibration point')
    ax.set_xlabel("Time [years]")
    ax.set_ylabel("CO2 Concentration [wt %]")
    ax.legend()
    ax.set_title("CO2 concentration in the Ohaaki geothermal field.")
    plt.show()

    pars = [a,b] # qloss needs a and b as parameter values
    global Q_SOL
    # can't use curve fit as no data is given for qloss
    Q_SOL = SolveQLoss(time_fit, *pars) # solves the qloss over the time range
    # plots qloss over time
    f, ax = plt.subplots(1, 1)
    ax.plot(time_fit,Q_SOL, 'b', label = 'ODE')
    ax.set_xlabel("Time [years]")
    ax.set_ylabel("CO2 lost [kg]")
    ax.legend()
    ax.set_title("CO2 Lost from Reservoir")
    plt.show()
    return

def SolveQLoss(t, *pars):
    """ Solves Q loss using Improved Euler Method

		Parameters
		----------
		t     : np.array
            time range to solve over
        *pars : np.array
            usually contains parameters a and b
		Returns
		-------
		np.array
            cumulative sum of CO2 lost to groundwater
	"""
    global step # sets up rolling dt so time steps can be more accurate
    if (extrapolation is False): 
        ys = 0.*tp # sets up solution array
        ys[0] = 0 # initial value is 0 as P-P0 is 0
        # iteratively solves for qloss using improved euler method
        for k in range(len(tp)- 1):
            step = tp[k+1] - tp[k] # changes step size accordingly
            ys[k+1] = improved_euler_step(QLossModel, tp[k], ys[k], tp[k+1] - tp[k], pars)
        return np.cumsum(np.interp(t, tp, ys)) # returns the cumulative sum of qloss
    if extrapolation is True:
        ys = 0.*prediction
        ys[0] = Q_SOL[-1] # if extrapolating then the first value is the last value solved for
        for k in range(len(prediction)- 1):
            step = prediction[k+1] - prediction[k] # changing step size
            ys[k+1] = improved_euler_step(QLossModel, prediction[k], ys[k], prediction[k+1] - prediction[k], pars)
        return np.cumsum(ys) # returns cumulative sum of qloss

def QLossModel(t,Q,*pars):
    """ Solves Q loss at different time points
    
		Parameters
		----------
		t     : float
            time to solve equation with (years)
        Q     : float
            dependent variable
        *pars : np.array
            usually contains parameters a and b
		Returns
		-------
		float
            qloss over time step
	"""
    if extrapolation is False:
        P = np.interp(t, time_fit, P_SOL) # need to get Pressure values if inside data range
        # finds C' to use
        if (P > pp[0]):
            C_1 = np.interp(t, time_fit, C_SOL)
        else:
            C_1 = 0
    else:
        P = extraPressure[k] # gets relevant pressure values
        # finds correct C' to use
        if (P > pp[0]):
            C_1 = extraSolute[k]
        else:
            C_1 = 0
    # returns qloss
    return (pars[1]/pars[0])*(P-pp[0])*C_1*step

def Extrapolate(t):
    """ Extrapolates the data to a time point, t, in the future.

		Parameters
		----------
		t     : float
            time to solve up to (years)
		
        Returns
		-------
		None
        
        Notes
        -----
        t needs to be greater than 2019.5
	"""
    if (t < tp[-1]):
        raise ValueError # raises error if t is not allowed

    inject = qc[-1] # gets the most recent injection value

    global prediction # sets up prediction array for time to solve over
    prediction = np.arange(tp[-1],t, dt)
    
    stakeholder = [0,0.5,1,2,4] # injection multipliers to solve with
    colours = ['g', 'r','b','y','k'] # colours to use with the graphing

    global extrapolation
    extrapolation = True # setting extrapolation to true to initialise correct solvers
    
    f1, ax = plt.subplots(1, 1) # initialising plots
    f2, ax2 = plt.subplots(1,1)
    f3, ax3 = plt.subplots(1,1)

    for i in range(len(stakeholder)): # looping over injection mulitipliers
        global net
        net = q[-1] - stakeholder[i]*inject # changing global net amount
        
        global injec
        injec = inject*stakeholder[i] # changing global injection amount
        
        pars = [a,b,c] # using optimal parameters
        
        global extraPressure # extraPressure will be used for SoluteODE
        extraPressure = SolvePressureODE(prediction, *pars)
        # plots relevant data
        ax.plot(np.append(time_fit, prediction), np.append(P_SOL,extraPressure), colours[i], label = 'Prediction' + ' for ' + str(injec) + ' kg/s')
        
        pars = [d, M0] # using optimal parameters
        
        global extraSolute # this will be used for qloss ODE
        extraSolute = SolveSoluteODE(prediction, *pars)
        # plots solved Solute values
        ax2.plot(np.append(time_fit, prediction), np.append(C_SOL,extraSolute), colours[i], label = 'Prediction' + ' for ' + str(injec) + ' kg/s')
        
        pars = [a,b] # using a and b for qloss
        
        qloss = SolveQLoss(prediction, *pars)
        # plotting qloss
        ax3.plot(np.append(time_fit, prediction), np.append(Q_SOL, qloss), colours[i], label = 'Prediction' + ' for ' + str(injec) + ' kg/s')
    
    # adding informative lines and labels
    ax.axhline(pp[0], color = 'cyan', linestyle = '--', label = 'Ambient Value')
    ax2.axhline(.1, color = 'cyan', linestyle = '--', label = 'Corrosive Point')

    ax.legend()
    ax2.legend()
    ax3.legend()

    ax2.set_title("Weight Percentage of CO2 in Ohaaki geothermal field")
    ax2.set_xlabel("Time [years]")
    ax2.set_ylabel("Weight Percent CO2 [wt %]")

    ax.set_title("Pressure in the Ohaaki geothermal field.")
    ax.set_ylabel("Pressure [MPa]")
    ax.set_xlabel("Time [years]")

    ax3.set_title("CO2 Lost due to seepage")
    ax3.set_ylabel("CO2 lost [kg]")
    ax3.set_xlabel("Time [years]")


    plt.show()
    plt.close(f1)
    plt.show()
    plt.close(f2)
    plt.show()
    return

def SolvePressureODE(t, *pars):
    """ Solves Pressure ODE using Improved Euler Method

		Parameters
		----------
		t     : np.array
            time range to solve over
        *pars : np.array
            usually contains parameters a, b and c
		Returns
		-------
		np.array
            pressure values which correlate to t array 
	"""
    global step
    if not extrapolation:
        ys = 0.*tp
        ys[0] = pp[0] # sets inital value of pressure
        for k in range(len(tp)- 1):
            step = tp[k+1] - tp[k] # chaning time step because some data has variable measurements
            ys[k+1] = improved_euler_step(PressureModel, tp[k], ys[k], tp[k+1] - tp[k], pars)
        return np.interp(t, tp, ys) # returns interpolated data
    else:
        ys = 0.*prediction
        ys[0] = P_SOL[-1] # if extrapolating the first point is the last point of the data
        for k in range(len(prediction)- 1):
            step = prediction[k+1] - prediction[k]
            ys[k+1] = improved_euler_step(PressureModel, prediction[k], ys[k], prediction[k+1] - prediction[k], pars)
        return ys # returns pressure values

def SolveSoluteODE(t, *pars):
    """ Solves Solute ODE using Improved Euler Method

		Parameters
		----------
		t     : np.array
            time range to solve over
        *pars : np.array
            usually contains parameters d, M0
		Returns
		-------
		np.array
            solute values which correlate to t array 
	"""
    global k, step # k helps to pick out extrapolated pressure values
    # follows same processa as SolvePressureODE
    if not extrapolation:
        ys = 0.*tcc
        ys[0] = cc[0]
        for k in range(len(tcc)- 1):
            step = tp[k+1] - tp[k] # varying time step
            ys[k+1] = improved_euler_step(SoluteModel, tcc[k], ys[k], tcc[k+1] - tcc[k], pars)
        return np.interp(t, tcc, ys) # returns interpolated data
    else:
        ys = 0.*prediction
        ys[0] = C_SOL[-1] # if extrapolating then first value is last value 
        for k in range(len(prediction)- 1):
            step = prediction[k+1] - prediction[k]
            ys[k+1] = improved_euler_step(SoluteModel, prediction[k], ys[k], prediction[k+1] - prediction[k], pars)
        return ys # returns solved data

def SoluteModel(t, conc, d, M0):
    """ Solves Solute ODE using Improved Euler Method

		Parameters
		----------
		t     : float
            time to solve ODE with, independent variable (year)
        conc  : float
            dependent variable, concentration at time t (wt %)
        d     : float
            strength parameter of CO2 reaction
        M0    : float
            parameter which is initial mass of the reservoir
		Returns
		------
        float
            dCdt at time point t
	"""
    if not extrapolation: # gets relevant qCO2 values and pressure values depending on extrapolating or not
        qCO2 = np.interp(t, tq, qc)
        pressure = np.interp(t, time_fit, P_SOL)
    else:
        qCO2 = injec
        pressure = extraPressure[k]
    
    C1 = conc if pressure > pp[0] else cc[0] # gets correct C' value

    return (((1 - conc)*qCO2)/ M0) - (b/(a * M0))*(pressure - pp[0])*(C1 - conc) - d*(conc - cc[0])

def PressureModel(t, Pk, a, b, c):
    """ Solves Pressure ODE using Improved Euler Method

		Parameters
		----------
		t     : float
            time to solve ODE with, independent variable (year)
        Pk  : float
            dependent variable, pressure at time t (MPa)
        a     : float
            strength parameter of net flow
        b    : float
            strength parameter of recharge
        c    : float
            strength parameter of slow drainage
		Returns
		------
        float
            dPdt at time point t
	"""
    if not extrapolation: # gets correct values for q and dqdt
        q = np.interp(t, tq, net)
        dqdti = np.interp(t, tq, dqdt)
    else:
        dqdti = 0
        q = net

    return -a*q - b*(Pk-pp[0]) - c*dqdti # calculates dPdt

def improved_euler_step(f, tk, yk, h, pars):
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
	f0 = f(tk, yk, *pars) # calculates f0 using function
	f1 = f(tk + h, yk + h*f0, *pars) # calculates f1 using fuctions
	yk1 = yk + h*(f0*0.5 + f1*0.5) # calculates the new y value
	return yk1

def Uncertainty():
    # sets up global variables
    global a,b,c,d,M0
    global net, injec
    # initialises net array again
    net = np.append(q[0:33],(q[33::]-qc[33::]))
    # initliasing parameters
    pressure_pars = [a,b,c]
    solute_pars = [d,M0]
    # initlialising arrays to store the results
    pressures0 = []
    pressures1 = []
    pressures2 = []
    pressures3 = []
    pressures4 = []

    psol = []

    concs0 = []
    concs1 = []
    concs2 = []
    concs3 = []
    concs4 = []

    csol = []

    last1 = []
    last2 = []
    last3 = []
    last4 = []
    last5 = []
    last6 = []
    last7 = []
    last8 = []
    last9 = []
    last10 = []
    # sets random seed so everyones results are the same
    np.random.seed(seed = 111592442)
    # getting random distribution of parameters using covariance matrix generated earlier
    p_pars = np.random.multivariate_normal(pressure_pars, pressurecov, 500)
    c_pars = np.random.multivariate_normal(solute_pars, solutecov, 500)
    # stakeholder based multipliers
    flows = [0,0.5,1,2,4]
    global prediction
    global P_SOL
    global C_SOL
    ogPSOL = P_SOL
    # this follows same process as the extrapolation
    # goes through all random parameter values generated and produces different extrapolations
    for pprams in p_pars:
        a = pprams[0]
        b = pprams[1]
        c = pprams[2]
        global extrapolation
        extrapolation = False
        net = np.append(q[0:33],(q[33::]-qc[33::]))
        P_SOL = SolvePressureODE(time_fit, *[a,b,c])
        psol.append(P_SOL)
        global extraploation
        extrapolation = True
        net = q[-1] - flows[1]*qc[-1]
        press = SolvePressureODE(prediction, *[a,b,c])
        pressures0.append(press)
        last1.append(press[-1])
        net = q[-1] - flows[2]*qc[-1]
        press = SolvePressureODE(prediction, *[a,b,c])
        pressures1.append(press)
        last2.append(press[-1])
        net = q[-1] - flows[3]*qc[-1]
        press = SolvePressureODE(prediction, *[a,b,c])
        pressures2.append(press)
        last3.append(press[-1])
        net = q[-1] - flows[4]*qc[-1]
        press = SolvePressureODE(prediction, *[a,b,c])
        pressures3.append(press)
        last4.append(press[-1])
        net = q[-1]
        press = SolvePressureODE(prediction, *[a,b,c])
        pressures4.append(press)
        last5.append(press[-1])
    # sets a, b, c back to normal values for solute ODE
    a, b, c = pressure_pars
    P_SOL = ogPSOL
    for ccprams in c_pars:
        d = ccprams[0]
        M0 = ccprams[1]
        extrapolation = False
        C_SOL = SolveSoluteODE(time_fit, *[d,M0])
        csol.append(C_SOL)
        extrapolation = True
        injec = flows[1]*qc[-1]
        solu = SolveSoluteODE(prediction, *[d,M0])
        concs0.append(solu)
        last6.append(solu[-1])
        injec = flows[2]*qc[-1]
        solu = SolveSoluteODE(prediction, *[d,M0])
        concs1.append(solu)
        last7.append(solu[-1])
        injec = flows[3]*qc[-1]
        solu = SolveSoluteODE(prediction, *[d,M0])
        concs2.append(solu)
        last8.append(solu[-1])
        injec = flows[4]*qc[-1]
        solu = SolveSoluteODE(prediction, *[d,M0])
        last9.append(solu[-1])
        concs3.append(solu)
        injec = 0
        solu = SolveSoluteODE(prediction, *[d,M0])
        last10.append(solu[-1])
        concs4.append(solu)

    # plots the results
    f, ax  = plt.subplots(1,1)
    for i in range(len(pressures0)):
        ax.plot(time_fit, psol[i], color = 'k', alpha = 0.1, lw = 0.5)
        ax.plot(prediction, pressures0[i], color = 'r', alpha = 0.1, lw = 0.4)
        ax.plot(prediction, pressures1[i], color = 'b', alpha = 0.1, lw = 0.4)
        ax.plot(prediction, pressures2[i], color = 'g', alpha = 0.1, lw = 0.4)
        ax.plot(prediction, pressures3[i], color = 'y', alpha = 0.1, lw = 0.4)
        ax.plot(prediction, pressures4[i], color = 'c', alpha = 0.1, lw = 0.4)
    # adds information to the legend 
    ax.plot([],[], color = 'k', label = 'Model Ensemble')
    ax.plot([],[], color = 'r', label = 'Injection = ' + str(flows[1]*qc[-1]) + ' kg/s')
    ax.plot([],[], color = 'b', label = 'Injection = ' + str(flows[2]*qc[-1]) + ' kg/s')
    ax.plot([],[], color = 'g', label = 'Injection = ' + str(flows[3]*qc[-1]) + ' kg/s')
    ax.plot([],[], color = 'y', label = 'Injection = ' + str(flows[4]*qc[-1]) + ' kg/s')
    ax.plot([],[], color = 'c', label = 'Injection = 0 kg/s')
    # adding information 
    ax.plot(tp, pp, 'r.', label = 'Observations')
    ax.set_xlabel("Time [years]")
    ax.axhline(pp[0], color = 'orange', linestyle = '--', label = 'Ambient Value')
    ax.set_ylabel("Pressure [MPa]")
    ax.set_title("Pressure Flow in Ohaaki")
    ax.legend()
    plt.show()
    # plotting results
    f, ax  = plt.subplots(1,1)
    for i in range(len(concs0)):
        ax.plot(time_fit, csol[i], color = 'k', alpha = 0.1, lw = 0.5)
        ax.plot(prediction, concs0[i], color = 'r', alpha = 0.1, lw = 0.4)
        ax.plot(prediction, concs1[i], color = 'b', alpha = 0.1, lw = 0.4)
        ax.plot(prediction, concs2[i], color = 'g', alpha = 0.1, lw = 0.4)
        ax.plot(prediction, concs3[i], color = 'y', alpha = 0.1, lw = 0.4)
        ax.plot(prediction, concs4[i], color = 'c', alpha = 0.1, lw = 0.4)
    # adding legend labels
    ax.plot([],[], color = 'k', label = 'Model Ensemble')
    ax.plot([],[], color = 'r', label = 'Injection = ' + str(flows[1]*qc[-1]) + ' kg/s')
    ax.plot([],[], color = 'b', label = 'Injection = ' + str(flows[2]*qc[-1]) + ' kg/s')
    ax.plot([],[], color = 'g', label = 'Injection = ' + str(flows[3]*qc[-1]) + ' kg/s')
    ax.plot([],[], color = 'y', label = 'Injection = ' + str(flows[4]*qc[-1]) + ' kg/s')
    ax.plot([],[], color = 'c', label = 'Injection = 0 kg/s')

    ax.plot(tcc, cc, 'r.', label = 'Observations')
    ax.set_xlabel("Time [years]")
    ax.set_ylabel("CO2 Concentration [wt %]")
    ax.set_title("Concentration of CO2 in Ohaaki")
    ax.axhline(0.1, color = 'orange', linestyle = '--', label = "Corrosive Point")
    ax.legend()
    plt.show()
    # performing confidence interval calculations
    # using 2.5 and 97.5 percentile creates 95% confidence interval
    five = np.percentile(last9,2.5)
    ninefive = np.percentile(last9,97.5)
    plt.hist(last9, bins = 'auto')
    plt.axvline(five , color = 'r', linestyle = '--')
    plt.axvline(ninefive , color = 'r', linestyle = '--')
    plt.title("CO2 Concentration Histogram at 2050")
    plt.ylabel("Count")
    plt.xlabel("CO2 Concentration [wt %]")
    plt.show()
    print("Concentration Confidence Interval for quadruple Injection = [ "+ str(five) + ", " + str(ninefive) + "] \n")
    five = np.percentile(last6,2.5)
    ninefive = np.percentile(last6,97.5)
    print("Concentration Confidence Interval for half Injection = [ "+ str(five) + ", " + str(ninefive) + "] \n")
    five = np.percentile(last7,2.5)
    ninefive = np.percentile(last7,97.5)
    print("Concentration Confidence Interval for same Injection = [ "+ str(five) + ", " + str(ninefive) + "] \n")
    five = np.percentile(last8,2.5)
    ninefive = np.percentile(last8,97.5)
    print("Concentration Confidence Interval for two times Injection = [ "+ str(five) + ", " + str(ninefive) + "] \n")
    five = np.percentile(last10,2.5)
    ninefive = np.percentile(last10,97.5)
    print("Concentration Confidence Interval for no Injection = [ "+ str(five) + ", " + str(ninefive) + "] \n")

    five = np.percentile(last4,2.5)
    ninefive = np.percentile(last4,97.5)
    plt.hist(last4, bins = 'auto')
    plt.axvline(five , color = 'r', linestyle = '--')
    plt.axvline(ninefive , color = 'r', linestyle = '--')
    plt.title("Pressure Histogram at 2050")
    plt.ylabel("Count")
    plt.xlabel("Pressure [MPa]")
    plt.show()
    print("Pressure Confidence Interval for quadruple Injection = [ "+ str(five) + ", " + str(ninefive) + "] \n")
    five = np.percentile(last1,2.5)
    ninefive = np.percentile(last1,97.5)
    print("Pressure Confidence Interval for half Injection = [ "+ str(five) + ", " + str(ninefive) +"] \n")
    five = np.percentile(last2,2.5)
    ninefive = np.percentile(last2,97.5)
    print("Pressure Confidence Interval for same Injection = [ "+ str(five) + ", " + str(ninefive) + "] \n")
    five = np.percentile(last3,2.5)
    ninefive = np.percentile(last3,97.5)
    print("Pressure Confidence Interval for two times Injection = [ "+ str(five) + ", " + str(ninefive) +"] \n")
    five = np.percentile(last5,2.5)
    ninefive = np.percentile(last5,97.5)
    print("Pressure Confidence Interval for no Injection = [ "+ str(five) + ", " + str(ninefive) + "] \n")
    return

def MSE():
    time = tp
    Pressure = pp
    A = np.linspace(0.001,0.0015,15)
    B = np.linspace(0.08,0.11,15)
    C = np.linspace(0.002,0.006,15)
    dt = 0.5
    MSPE_best = float('inf')
    best_A = 1000
    best_B = 1000
    best_C = 1000
    time_range = np.arange(time[0], time[-1], dt)

	# Modelling ODE for each combination of A,B,C
    for A_i,B_i,C_i in itertools.product(A,B,C):
        pars = [A_i,B_i,C_i]
        sol_pressure = SolvePressureODE(time_range, *pars)

		# Interpolating for comparison of MSE
        f = interp1d(time_range,sol_pressure)
        analytic_pressure = f(time_range)
        diff_array = np.subtract(analytic_pressure,sol_pressure)
        squared_array = np.square(diff_array)
        MSPE = squared_array.mean()
        if (MSPE < MSPE_best):
            MSPE_best = MSPE
            best_A = A_i
            best_B = B_i
            best_C = C_i


	
	# Plotting best fit ODE
    pars = [best_A,best_B,best_C]
    sol_pressure = SolvePressureODE(time_range, *pars)

	# Printout of results
    txt = "Best coefficient {} is {}"
    print(txt.format("A",best_A))
    print(txt.format("B",best_B))
    print(txt.format("C",best_C))
    print("Mean Squared Error is {}".format(MSPE_best))
    f, ax2 = plt.subplots(1, 1)
    ax2.plot(time_range,sol_pressure, 'b', label = 'a = ' + str(best_A) + '\n' + 'b = ' + str(best_B) + '\n' + 'c = ' + str(best_C))
    ax2.plot(time,Pressure, 'r', label = 'DATA')
    ax2.set_title("Best Initial Guess for Parameters. Mean Squared Error = " + str(MSPE_best))
    ax2.legend()
    plt.show()
    return best_A,best_B,best_C

if __name__ == "__main__":
    main()