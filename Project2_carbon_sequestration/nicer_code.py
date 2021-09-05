import numpy as np
from numpy.core.numeric import NaN, _move_axis_to_0
from matplotlib import pyplot as plt
from numpy.lib.function_base import interp
from scipy.interpolate import interp1d
import itertools
from scipy.optimize import curve_fit
import statistics

time, prod, P, injec, C = np.genfromtxt('output.csv', delimiter = ',', skip_header= 1, missing_values= 0, usecols = [1,2,3,4,5]).T
injec[np.isnan(injec)] = 0
C[np.isnan(C)] = 0.03 # natural state
P[0] = P[1] # there is only one missing value
net = prod - injec
dqdt = net*0.
dqdt[0] = (net[1] - net[0])/(time[1] - time[0])
dqdt[-1] = (net[-1] - net[-2])/(time[-1] - time[-2])
dqdt[1:-1] = (net[2:] - net[:-2])/(time[2:] - net[:-2])
P_SOL = []
extrapolation = False
a = 0
b = 0
c = 0
d = 0
M0 = 0
extraPressure = []
k = 0
dt = 0.5
C_SOL = []

def main():

    Model_Fit()

    Extrapolate(2050)

    PlotMisfit()

    BenchMark()
    return

def BenchMark():
    dt = 0.25
    time = np.arange(0, 10, dt)
    global net
    net = 4
    a = 1
    b = 2
    c = 0
    q0 = 4
    ys, analytical = PressureBenchmark(P[0], a, b, c, q0, time, dt)
    steady_state = P[0] - (a*q0)/b
    f, ax = plt.subplots(1, 1)
    ax.plot(time,analytical, 'b', label = 'Analtyical')
    ax.plot(time, ys, 'kx', label = 'Numerical')
    ax.axhline(steady_state, linestyle = '--', color = 'red', label = 'steady state')
    ax.legend()
    ax.set_title("Analytcial vs Numerical Solution Benchmark for Pressure ODE")
    plt.show()
    dt = 1.1
    time = np.arange(0,10, dt)
    ys, analytical = PressureBenchmark(P[0], 1, 2, 0, 4, time, dt)
    f, ax = plt.subplots(1, 1)
    ax.plot(time,analytical, 'b', label = 'Analtyical')
    ax.plot(time,ys, 'kx', label = 'Numerical')
    ax.axhline(steady_state, linestyle = '--', color = 'red', label = 'steady state')
    ax.legend()
    ax.set_title("Instability at a large time step for Pressure ODE")
    plt.show()
    dt = 0.25
    time = np.arange(0, 10, dt)
    global injec
    injec = 1
    a = 1
    b = 2
    d = 3
    M0 = 1
    ys, analytical = SoluteBenchmark(C[0], injec, a, b, d, M0, P[0], time, dt)
    steady_state = ((injec/M0) + d*C[0])/((injec/M0) + d)
    f, ax = plt.subplots(1, 1)
    ax.plot(time,analytical, 'b', label = 'Analtyical')
    ax.plot(time, ys, 'kx', label = 'Numerical')
    ax.axhline(steady_state, linestyle = '--', color = 'red', label = 'steady state')
    ax.legend()
    ax.set_title("Analytcial vs Numerical Solution Benchmark for Solute ODE")
    plt.show()
    dt = 0.9
    time = np.arange(0,10, dt)
    ys, analytical = SoluteBenchmark(C[0], injec, a, b, d, M0, P[0], time, dt)
    f, ax = plt.subplots(1, 1)
    ax.plot(time,analytical, 'b', label = 'Analtyical')
    ax.plot(time,ys, 'kx', label = 'Numerical')
    ax.axhline(steady_state, linestyle = '--', color = 'red', label = 'steady state')
    ax.legend()
    ax.set_title("Instability at a large time step for Solute ODE")
    plt.show()
    return

def SoluteBenchmark(C0, qCO2, a, b, d, M0, P0, time, dt):
    analytical = []
    for i in range(len(time)):
        k = qCO2/M0
        L = (k*C0 - k)/(k + d)
        anaC = (k + (d * C0))/(k + d) + L/(np.exp(k*time[i]+d*time[i]))
        analytical.append(anaC)
    nt = int(np.ceil((time[-1]-time[0])/dt))
    ts = time[0]+np.arange(nt+1)*dt
    ys = ts*0.
    ys[0] = C[0]
    pars = [d,M0,P0]
    for i in range(nt):
        ys[i+1] = improved_euler_step(SoluteModel, ts[i], ys[i], dt, pars)
    return ys, analytical

def PressureBenchmark(P0, a, b , c, q0, time, dt):
    analytical = []
    for i in range(len(time)):
        P = P0 + ((-a*q0)/b)*(1-np.exp(-b*time[i]))
        analytical.append(P)
    nt = int(np.ceil((time[-1]-time[0])/dt))
    ts = time[0]+np.arange(nt+1)*dt
    ys = ts*0.
    ys[0] = P0
    pars = [a,b,c]
    for i in range(nt):
        ys[i+1] = improved_euler_step(PressureModel, ts[i], ys[i], dt, pars)
    return ys, analytical

def PlotMisfit():
    pressure_time = np.genfromtxt('data/cs_p.txt', skip_header = 1,delimiter = ',', usecols = 0)
    pressure = np.genfromtxt('data/cs_p.txt', skip_header = 1,delimiter = ',', usecols = 1)
    # we dont have data that matches with time values in pressure array
    # so we interpret the SOL_P stuff
    P_Result = []
    for i in range(len(pressure_time)):
        P_Result.append(np.interp(pressure_time[i], time_fit, P_SOL))
    misfit_P = []
    for i in range(len(P_Result)):
        misfit_P.append(pressure[i] - P_Result[i])

    f, ax = plt.subplots(1, 1)
    ax.plot(pressure_time,misfit_P, 'rx')
    ax.axhline(0, color = 'black', linestyle = '--')
    ax.set_ylabel('Pressure [MPa]')
    ax.set_xlabel('Time [years]')
    ax.set_title("Best Fit Pressure LPM Model")
    plt.show()

    solute_time = np.genfromtxt('data/cs_cc.txt', skip_header = 1, delimiter = ',', usecols = 0)
    solute = np.genfromtxt('data/cs_cc.txt', skip_header = 1, delimiter = ',', usecols = 1)
    C_Result = []
    for i in range(len(solute_time)):
        C_Result.append(np.interp(solute_time[i], time_fit, C_SOL))
    misfit_C = []
    for i in range(len(C_Result)):
        misfit_C.append(solute[i] - C_Result[i])

    f, ax = plt.subplots(1, 1)
    ax.plot(solute_time,misfit_C, 'rx')
    ax.axhline(0, color = 'black', linestyle = '--')
    ax.set_ylabel('CO2 [wt %]')
    ax.set_title("Best Fit Solute LPM Model")
    plt.show()
    return
    
def Model_Fit():
    pars = [0.00187,0.153,0.00265]
    bestfit_pars = curve_fit(SolvePressureODE, time[0:92], P[0:92], pars, bounds = (0, [1,1,1]))
    
    global a, b, c, time_fit, P_SOL
    a = bestfit_pars[0][0]
    b = bestfit_pars[0][1]
    c = bestfit_pars[0][2]

    time_fit = np.arange(time[0], time[-1], dt)
    
    P_SOL = SolvePressureODE(time_fit, *bestfit_pars[0])
    
    f, ax = plt.subplots(1, 1)
    ax.plot(time_fit,P_SOL, 'b', label = 'ODE')
    ax.plot(time,P, 'r', label = 'DATA')
    plt.axvline(time[91], color = 'black', linestyle = '--', label = 'Calibration point')
    ax.legend()
    ax.set_title("Pressure flow in the Ohaaki geothermal field.")
    plt.show()

    pars = [0.001,8012346, 6.17]

    bestfit_pars = curve_fit(SolveSoluteODE, time[0:92], C[0:92], pars, bounds = ([0,0,5], [10,10000000,10]))

    global d, M0, C_SOL, P0
    d = bestfit_pars[0][0]
    M0 = bestfit_pars[0][1]
    P0 = bestfit_pars[0][2]

    C_SOL = SolveSoluteODE(time_fit, *bestfit_pars[0])

    f, ax = plt.subplots(1, 1)
    ax.plot(time_fit,C_SOL, 'b', label = 'ODE')
    ax.plot(time,C, 'r', label = 'DATA')
    plt.axvline(time[91], color = 'black', linestyle = '--', label = 'Calibration point')
    ax.legend()
    ax.set_title("CO2 concentration in the Ohaaki geothermal field.")
    plt.show()
    return

def Extrapolate(t):

    inject = np.genfromtxt('output.csv', delimiter = ',', skip_header= 1, missing_values= 0, usecols = 4)
    inject = inject[-1]
        
    prediction = np.arange(time_fit[-1],t, dt)
    
    stakeholder = [0.5,1,2,4]
    amount = ['half injection', 'same amount', 'double the rate', 'CEL proposed']
    colours = ['r','b','y','k']

    global extrapolation
    extrapolation = True
    
    f1, ax = plt.subplots(1, 1)
    f2, ax2 = plt.subplots(1,1)

    for i in range(len(stakeholder)):
        global net
        net = prod[-1] - stakeholder[i]*inject
       # net = statistics.mean(net)
        global injec
        injec = inject*stakeholder[i]
        pars = [a,b,c]
        global extraPressure
        extraPressure = SolvePressureODE(prediction, *pars)
        ax.plot(np.append(time_fit, prediction), np.append(P_SOL,extraPressure), colours[i], label = 'Prediction' + ' for ' + amount[i])
        pars = [d,M0, P0]
        extraSolute = SolveSoluteODE(prediction, *pars)
        ax2.plot(np.append(time_fit, prediction), np.append(C_SOL,extraSolute), colours[i], label = 'Prediction' + ' for ' + amount[i])

    ax.axvline(time[91], color = 'black', linestyle = '--', label = 'Calibration point')
    ax2.axvline(time[91], color = 'black', linestyle = '--', label = 'Calibration point')
    ax2.axhline(.1, color = 'cyan', linestyle = '--', label = 'Corrosive Point')

    ax.legend()
    ax2.legend()

    ax2.set_title("Weight Percentage of CO2 in Ohaaki geothermal field")
    ax2.set_xlabel("Time [years]")
    ax2.set_ylabel("Weight Percent CO2 [wt %]")

    ax.set_title("Pressure in the Ohaaki geothermal field.")
    ax.set_ylabel("Pressure [MPa]")
    ax.set_xlabel("Time [years]")

    plt.show()
    plt.close(f1)
    plt.show()
    return

def SolvePressureODE(t, *pars):
    ys = 0.*t
    if extrapolation is False:
        ys[0] = P[0]
    else:
        ys[0] = P_SOL[-1]
    for k in range(len(t)- 1):
        ys[k+1] = improved_euler_step(PressureModel, t[k], ys[k], dt, pars)
    return ys

def SolveSoluteODE(t, *pars):
    ys = 0.*t
    if extrapolation is False:
        ys[0] = C[0]
    else:
        ys[0] = C_SOL[-1]
    global k
    for k in range(len(t)- 1):
        ys[k+1] = improved_euler_step(SoluteModel, t[k], ys[k], dt, pars)
    return ys

def SoluteModel(t, conc, d, M0, P0):
    if extrapolation is False:
        qCO2 = np.interp(t, time, injec)
        pressure = np.interp(t, time_fit, P_SOL)
    else:
        qCO2 = injec
        pressure = extraPressure[k]
    
    if (pressure > P[0]):
        C1 = conc
        C2 = conc
    else:
        C1 = C[0]
        C2 = 0

    #qloss = ((b/a)*(pressure - P0)*C2)

    #qCO2 -= qloss

    return (((1 - conc)*qCO2)/ M0) - (b/(a * M0))*(pressure - P0)*(C1 - conc) - d*(conc - C[0])

def PressureModel(t, Pk, a, b, c):
    if extrapolation is False:
        q = np.interp(t, time, net)
        dqdti = np.interp(t, time, dqdt)
    else:
        dqdti = 0
        q = net

    if (Pk - P[0] > 0):
        C_1 = np.interp(t,time,C)
    else:
        C_1 = 0
    
   # qloss = (b/a)*Pk*t*C_1

    #q -= qloss

    return -a*q - b*(Pk-P[0]) - c*dqdti

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
	# print(pars)
	f0 = f(tk, yk, *pars) # calculates f0 using function
	f1 = f(tk + h, yk + h*f0, *pars) # calculates f1 using fuctions
	yk1 = yk + h*(f0*0.5 + f1*0.5) # calculates the new y value
	return yk1

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

if __name__ == "__main__":
	 main()