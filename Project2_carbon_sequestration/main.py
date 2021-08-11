# The start of the Modelling stuff I guess
import numpy as np
from numpy.core.numeric import NaN


def main():
    time, netFlow ,Pressure = getPressureData()
    # test comment
    # pars = [q,a,b,c,dqdt]
    # q is variable so need to increment the different flows 
    # a,b,c are some constants we define
    # dqdt I assume is something we solve for depending on the change in flow rates

    # this will solve the ODE with the different net flow values
    #for i in range(len(netFlow)):
        #solve_ode(pressure_model, t0 = time[0], t1=time[-1], dt = 1, x0 = Pressure[0], )
    return

def pressure_model(t, P, q, a, b, c, dqdt, P0):
    ''' Return the derivative dx/dt at time, t, for given parameters.

        Parameters:
        -----------
        P : float
            Dependent variable.
        q : float
            Source/sink rate.
        a : float
            Source/sink strength parameter.
        b : float
            Recharge strength parameter.
        P0 : float
            Ambient value of dependent variable.
        c  : float
            Recharge strength parameter
        dqdt : float
            Rate of change of flow rate
        Returns:
        --------
        dPdt : float
            Derivative of Pressure variable with respect to independent variable.
    '''
    dPdt =  -a*q - b*(P-P0) - c*dqdt
    return dPdt

def solve_ode(f, t0, t1, dt, x0, pars):
    ''' Solve an ODE numerically.

        Parameters:
        -----------
        f : callable
            Function that returns dxdt given variable and parameter inputs.
        t0 : float
            Initial time of solution.
        t1 : float
            Final time of solution.
        dt : float
            Time step length.
        x0 : float
            Initial value of solution.
        pars : array-like
            List of parameters passed to ODE function f.

        Returns:
        --------
        t : array-like
            Independent variable solution vector.
        x : array-like
            Dependent variable solution vector.

        Notes:
        ------
        ODE should be solved using the Improved Euler Method. 

        Function q(t) should be hard coded within this method. Create duplicates of 
        solve_ode for models with different q(t).

        Assume that ODE function f takes the following inputs, in order:
            1. independent variable
            2. dependent variable
            3. forcing term, q
            4. all other parameters
    '''
    nt = int(np.ceil((t1-t0)/dt))
    ts = t0+np.arange(nt+1)*dt
    ys = 0.*ts
    ys[0] = x0
    for k in range(nt):
        ys[k + 1] = improved_euler_step(f, ts[k], ys[k], dt, x0, pars)
    return ts,ys


def improved_euler_step(f, tk, yk, h, x0, pars):
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
    yk1 = yk + h*(f0*0.5 + f1*0.5)
    return yk1

def getPressureData():
    # reads the files' values
    vals = np.genfromtxt('output.csv', delimiter = ',', skip_header= 1, missing_values= 0)
    # extracts the relevant data
    t = vals[:,1]
    prod = vals[:, 2]
    P = vals[:,3]
    injec = vals[:,4]
    # cleans data
    # for CO2 injection if no data is present then a rate of 0 is given for Pressure 
    # it is given the most recent value
    injec[np.isnan(injec)] = 0
    P[0] = P[1] # there is only one missing value
    net = []
    for i in range(len(prod)):
        net.append(prod[i] - injec[i]) # getting net amount 
    return t, P, net

if __name__ == "__main__":
    main()