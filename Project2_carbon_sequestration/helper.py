from typing import List

def improved_euler_step(f, tk: float, yk: float, h: float, pars: List[float])->float:
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

def pressureModel(t: float, P: float, P0: float, q: float, dqdt: float, a: float, b: float, c: float)->float:
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
    return -a*q - b*(P-P0) - c*dqdt

def soluteModel(t: float, C: float, dt: float, C0: float, P: float, P0: float, q: float, M0: float, a: float, b: float, d: float)->float:
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
    # q -= qLossModel(dt, C, P, P0, a, b)
    
    if P > P0:
        # the equation simplifies in the case of P > P0
        return ((1 - C) * (q / M0)) - (d * (C - C0))

    return ((1 - C) * (q / M0)) - ((b / (a * M0)) * (P - P0) * (C0 - C)) - (d * (C - C0))

def qLossModel(dt: float, C: float, P: float, P0: float, a: float, b: float)->float:
    """DOCSTRING NEEDED
    """
    if P > P0:
        return (b / a) * (P - P0) * C * dt

    return 0
    