import numpy as np
from numpy.core.numeric import NaN
from matplotlib import pyplot as plt
from numpy.lib.function_base import interp
from scipy.interpolate import interp1d
import itertools
from scipy.optimize import curve_fit
import statistics

tp, pp = np.genfromtxt('data/cs_p.txt', delimiter = ',', skip_header=1).T
tqc, qc = np.genfromtxt('data/cs_c.txt', delimiter = ',', skip_header=1).T
tq, q = np.genfromtxt('data/cs_q.txt', delimiter = ',', skip_header=1).T
tcc, cc = np.genfromtxt('data/cs_cc.txt', delimiter = ',', skip_header=1).T

net = np.append(q[0:33],(q[33::]-qc))
dqdt = net*0.
dqdt[0] = (net[1] - net[0])/(tq[1] - tq[0])
dqdt[-1] = (net[-1] - net[-2])/(tq[-1] - tq[-2])
dqdt[1:-1] = (net[2:] - net[:-2])/(tq[2:] - tq[:-2])
qc = np.append(np.zeros(33), qc)

P_SOL = []
extrapolation = False
other_extrapolation = False
a = 0
b = 0
c = 0
d = 0
M0 = 0
extraPressure = []
k = 0
dt = 0.1
C_SOL = []

def main():

    Model_Fit()

    Extrapolate(2050)

    PlotMisfit()

    BenchMark()

    Uncertainty()
    return



def BenchMark():
    dt = 0.1
    time = np.arange(0, 10, dt)
    global net
    net = 4
    a = 1
    b = 2
    c = 0
    q0 = 4
    ys, analytical = PressureBenchmark(pp[0], a, b, c, q0, time, dt)
    steady_state = pp[0] - (a*q0)/b
    f, ax = plt.subplots(1, 1)
    ax.plot(time,analytical, 'b', label = 'Analtyical')
    ax.plot(time, ys, 'kx', label = 'Numerical')
    ax.set_xlabel("Time [seconds]")
    ax.set_ylabel("Pressure [MPa]")
    ax.axhline(steady_state, linestyle = '--', color = 'red', label = 'steady state')
    ax.legend()
    ax.set_title("Analytcial vs Numerical Solution Benchmark for Pressure ODE")
    plt.show()
    dt = 1.1
    time = np.arange(0,10, dt)
    ys, analytical = PressureBenchmark(pp[0], 1, 2, 0, 4, time, dt)
    f, ax = plt.subplots(1, 1)
    ax.plot(time,analytical, 'b', label = 'Analtyical')
    ax.plot(time,ys, 'kx', label = 'Numerical')
    ax.set_xlabel("Time [seconds]")
    ax.set_ylabel("Pressure [MPa]")
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
    ys, analytical = SoluteBenchmark(cc[0], injec, a, b, d, M0, time, dt)
    steady_state = ((injec/M0) + d*cc[0])/((injec/M0) + d)
    f, ax = plt.subplots(1, 1)
    ax.plot(time,analytical, 'b', label = 'Analtyical')
    ax.plot(time, ys, 'kx', label = 'Numerical')
    ax.axhline(steady_state, linestyle = '--', color = 'red', label = 'steady state')
    ax.set_xlabel("Time [seconds]")
    ax.set_ylabel("CO2 Concentration [wt %]")
    ax.legend()
    ax.set_title("Analytcial vs Numerical Solution Benchmark for Solute ODE")
    plt.show()
    dt = 1.1
    time = np.arange(0,10, dt)
    ys, analytical = SoluteBenchmark(cc[0], injec, a, b, d, M0, time, dt)
    f, ax = plt.subplots(1, 1)
    ax.plot(time,analytical, 'b', label = 'Analtyical')
    ax.plot(time,ys, 'kx', label = 'Numerical')
    ax.axhline(steady_state, linestyle = '--', color = 'red', label = 'steady state')
    ax.set_xlabel("Time [seconds]")
    ax.set_ylabel("CO2 Concentration [wt %]")
    ax.legend()
    ax.set_title("Instability at a large time step for Solute ODE")
    plt.show()
    return

def SoluteBenchmark(C0, qCO2, a, b, d, M0, time, dt):
    analytical = []
    for i in range(len(time)):
        k = qCO2/M0
        L = (k*C0 - k)/(k + d)
        anaC = (k + (d * C0))/(k + d) + L/(np.exp(k*time[i]+d*time[i]))
        analytical.append(anaC)
    nt = int(np.ceil((time[-1]-time[0])/dt))
    ts = time[0]+np.arange(nt+1)*dt
    ys = ts*0.
    ys[0] = cc[0]
    pars = [d,M0]
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
    global pressurecov
    bestfit_pars, pressurecov = curve_fit(SolvePressureODE, tp[0:34], pp[0:34], pars)
    
    global a, b, c, time_fit, P_SOL
    a = bestfit_pars[0]
    b = bestfit_pars[1]
    c = bestfit_pars[2]

    time_fit = np.arange(tp[0], tp[-1], dt)
    
    P_SOL = SolvePressureODE(time_fit, *bestfit_pars)
    
    f, ax = plt.subplots(1, 1)
    ax.plot(time_fit,P_SOL, 'b', label = 'ODE')
    ax.plot(tp,pp, 'r.', label = 'DATA')
    plt.axvline(tp[34], color = 'black', linestyle = '--', label = 'Calibration point')
    ax.set_xlabel("Time [years]")
    ax.set_ylabel("Pressure [MPa]")
    ax.legend()
    ax.set_title("Pressure flow in the Ohaaki geothermal field.")
    plt.show()

    pars = [0.01,1000]
    global solutecov
    bestfit_pars, solutecov = curve_fit(SolveSoluteODE, tcc[0:16], cc[0:16], pars, bounds = (0, [10000000,100000000]))

    global d, M0, C_SOL
    d = bestfit_pars[0]
    M0 = bestfit_pars[1]


    C_SOL = SolveSoluteODE(time_fit, *bestfit_pars)
    f, ax = plt.subplots(1, 1)
    ax.plot(time_fit,C_SOL, 'b', label = 'ODE')
    ax.plot(tcc,cc, 'r.', label = 'DATA')
    plt.axvline(tcc[15], color = 'black', linestyle = '--', label = 'Calibration point')
    ax.set_xlabel("Time [years]")
    ax.set_ylabel("CO2 Concentration [wt %]")
    ax.legend()
    ax.set_title("CO2 concentration in the Ohaaki geothermal field.")
    plt.show()
    return

def Extrapolate(t):

    inject = qc[-1]
    global prediction
    prediction = np.arange(tp[-1],t, dt)
    
    stakeholder = [0.5,1,2,4]
    amount = ['half injection', 'same amount', 'double the rate', 'CEL proposed']
    colours = ['r','b','y','k']

    global extrapolation
    extrapolation = True
    
    f1, ax = plt.subplots(1, 1)
    f2, ax2 = plt.subplots(1,1)

    for i in range(len(stakeholder)):
        global net
        net = q[-1] - stakeholder[i]*inject
       # net = statistics.mean(net)
        global injec
        injec = inject*stakeholder[i]
        pars = [a,b,c]
        global extraPressure
        extraPressure = SolvePressureODE(prediction, *pars)
        ax.plot(np.append(time_fit, prediction), np.append(P_SOL,extraPressure), colours[i], label = 'Prediction' + ' for ' + amount[i])
        pars = [d, M0]
        extraSolute = SolveSoluteODE(prediction, *pars)
        ax2.plot(np.append(time_fit, prediction), np.append(C_SOL,extraSolute), colours[i], label = 'Prediction' + ' for ' + amount[i])

    ax.axvline(2002, color = 'black', linestyle = '--', label = 'Calibration point')
    ax2.axvline(2002, color = 'black', linestyle = '--', label = 'Calibration point')
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
    global step
    if (extrapolation is False):
        ys = 0.*tp
        ys[0] = pp[0]
        for k in range(len(tp)- 1):
            step = tp[k+1] - tp[k]
            ys[k+1] = improved_euler_step(PressureModel, tp[k], ys[k], tp[k+1] - tp[k], pars)
        return np.interp(t, tp, ys)
    if extrapolation is True:
        ys = 0.*prediction
        ys[0] = P_SOL[-1]
        for k in range(len(prediction)- 1):
            step = prediction[k+1] - prediction[k]
            ys[k+1] = improved_euler_step(PressureModel, prediction[k], ys[k], prediction[k+1] - prediction[k], pars)
        return ys

def SolveSoluteODE(t, *pars):
    global k, step
    if extrapolation is False:
        ys = 0.*tcc
        ys[0] = cc[0]
        for k in range(len(tcc)- 1):
            step = tp[k+1] - tp[k]
            ys[k+1] = improved_euler_step(SoluteModel, tcc[k], ys[k], tcc[k+1] - tcc[k], pars)
    else:
        ys = 0.*prediction
        ys[0] = C_SOL[-1]
        for k in range(len(prediction)- 1):
            step = prediction[k+1] - prediction[k]
            ys[k+1] = improved_euler_step(SoluteModel, prediction[k], ys[k], prediction[k+1] - prediction[k], pars)
        return ys
    return np.interp(t, tcc, ys)

def SoluteModel(t, conc, d, M0):
    if extrapolation is False:
        qCO2 = np.interp(t, tq, qc)
        pressure = np.interp(t, time_fit, P_SOL)
    else:
        qCO2 = injec
        pressure = extraPressure[k]
    
    if (pressure > pp[0]):
        C1 = conc
        qloss = (b/a)*(pressure - pp[0])*conc*step
        qCO2 -= qloss
    else:
        C1 = cc[0]

    return (((1 - conc)*qCO2)/ M0) - (b/(a * M0))*(pressure - pp[0])*(C1 - conc) - d*(conc - cc[0])

def PressureModel(t, Pk, a, b, c):
    if (extrapolation is False):
        q = np.interp(t, tq, net)
        dqdti = np.interp(t, tq, dqdt)
        conc = np.interp(t, tcc, cc)
    else:
        dqdti = 0
        q = net
        conc = np.interp(t, tcc, cc)
    if (Pk > pp[0]):
        q -= ((b/a)*(Pk-pp[0])*conc*step)
    return -a*q - b*(Pk-pp[0]) - c*dqdti

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

def Uncertainty():
    global a,b,c,d,M0
    global net, injec
    net = np.append(q[0:33],(q[33::]-qc[33::]))
    pressure_pars = [a,b,c]
    solute_pars = [d,M0]
    pressures0 = []
    pressures1 = []
    pressures2 = []
    pressures3 = []
    psol = []
    concs0 = []
    concs1 = []
    concs2 = []
    concs3 = []
    csol = []
    p_pars = np.random.multivariate_normal(pressure_pars, pressurecov, 250)
    flows = [0.5,1,2,4]
    c_pars = np.random.multivariate_normal(solute_pars, solutecov, 250)
    global prediction
    global P_SOL
    global C_SOL
    ogPSOL = P_SOL
    prediction = np.arange(tp[-1],2050, 0.1)
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
        net = q[-1] - flows[0]*qc[-1]
        press = SolvePressureODE(prediction, *[a,b,c])
        pressures0.append(press)
        net = q[-1] - flows[1]*qc[-1]
        press = SolvePressureODE(prediction, *[a,b,c])
        pressures1.append(press)
        net = q[-1] - flows[2]*qc[-1]
        press = SolvePressureODE(prediction, *[a,b,c])
        pressures2.append(press)
        net = q[-1] - flows[3]*qc[-1]
        press = SolvePressureODE(prediction, *[a,b,c])
        pressures3.append(press)

    a, b, c = pressure_pars
    P_SOL = ogPSOL
    for ccprams in c_pars:
        d = ccprams[0]
        M0 = ccprams[1]
        extrapolation = False
        C_SOL = SolveSoluteODE(time_fit, *[d,M0])
        csol.append(C_SOL)
        extrapolation = True
        injec = flows[0]*qc[-1]
        solu = SolveSoluteODE(prediction, *[d,M0])
        concs0.append(solu)
        injec = flows[1]*qc[-1]
        solu = SolveSoluteODE(prediction, *[d,M0])
        concs1.append(solu)
        injec = flows[2]*qc[-1]
        solu = SolveSoluteODE(prediction, *[d,M0])
        concs2.append(solu)
        injec = flows[3]*qc[-1]
        solu = SolveSoluteODE(prediction, *[d,M0])
        concs3.append(solu)

    f, ax  = plt.subplots(1,1)
    for i in range(len(pressures0)):
        ax.plot(time_fit, psol[i], color = 'k', alpha = 0.1, lw = 0.5)
        ax.plot(prediction, pressures0[i], color = 'r', alpha = 0.1, lw = 0.4)
        ax.plot(prediction, pressures1[i], color = 'b', alpha = 0.1, lw = 0.4)
        ax.plot(prediction, pressures2[i], color = 'g', alpha = 0.1, lw = 0.4)
        ax.plot(prediction, pressures3[i], color = 'y', alpha = 0.1, lw = 0.4)
    ax.plot([],[], color = 'r', label = 'Injection = ' + str(flows[0]*qc[-1]) + ' kg/s')
    ax.plot([],[], color = 'b', label = 'Injection = ' + str(flows[1]*qc[-1]) + ' kg/s')
    ax.plot([],[], color = 'g', label = 'Injection = ' + str(flows[2]*qc[-1]) + ' kg/s')
    ax.plot([],[], color = 'y', label = 'Injection = ' + str(flows[3]*qc[-1]) + ' kg/s')
    ax.plot(tp, pp, 'r.', label = 'Observations')
    ax.set_xlabel("Time [years]")
    ax.set_ylabel("Pressure [MPa]")
    ax.set_title("Pressure Flow in Ohaaki")
    ax.legend()
    plt.show()
    f, ax  = plt.subplots(1,1)
    for i in range(len(concs0)):
        ax.plot(time_fit, csol[i], color = 'k', alpha = 0.1, lw = 0.5)
        ax.plot(prediction, concs0[i], color = 'r', alpha = 0.1, lw = 0.4)
        ax.plot(prediction, concs1[i], color = 'b', alpha = 0.1, lw = 0.4)
        ax.plot(prediction, concs2[i], color = 'g', alpha = 0.1, lw = 0.4)
        ax.plot(prediction, concs3[i], color = 'y', alpha = 0.1, lw = 0.4)
    ax.plot([],[], color = 'r', label = 'Injection = ' + str(flows[0]*qc[-1]) + ' kg/s')
    ax.plot([],[], color = 'b', label = 'Injection = ' + str(flows[1]*qc[-1]) + ' kg/s')
    ax.plot([],[], color = 'g', label = 'Injection = ' + str(flows[2]*qc[-1]) + ' kg/s')
    ax.plot([],[], color = 'y', label = 'Injection = ' + str(flows[3]*qc[-1]) + ' kg/s')
    ax.plot(tcc, cc, 'r.', label = 'Observations')
    ax.set_xlabel("Time [years]")
    ax.set_ylabel("CO2 Concentration [wt %]")
    ax.set_title("Concentration of CO2 in Ohaaki")
    ax.axhline(0.1, color = 'r', linestyle = '--', label = "Corrosive Point")
    ax.legend()
    plt.show()

    barp = []
    barc = []
    for i in range(len(concs0)):
        barp.append(pressures3[i][-1])
        barc.append(concs3[i][-1])
    
    five = np.percentile(barc,2.5)
    ninefive = np.percentile(barc,97.5)

    plt.hist(barc, bins = 'auto')
    plt.axvline(five , color = 'r', linestyle = '--')
    plt.axvline(ninefive , color = 'r', linestyle = '--')
    plt.show()

    return

if __name__ == "__main__":
	 main()