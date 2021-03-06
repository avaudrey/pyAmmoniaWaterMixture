#!/usr/bin/python
# -*- coding: utf-8 -*-

#----------------------------------------------------------------------------
#   Copyright (C) 2017 <Alexandre Vaudrey>                                  |
#                                                                           |
#   This program is free software: you can redistribute it and/or modify    |
#   it under the terms of the GNU General Public License as published by    |
#   the Free Software Foundation, either version 3 of the License, or       |
#   (at your option) any later version.                                     |
#                                                                           |
#   This program is distributed in the hope that it will be useful,         |
#   but WITHOUT ANY WARRANTY; without even the implied warranty of          |
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the           |
#   GNU General Public License for more details.                            |
#                                                                           |
#   You should have received a copy of the GNU General Public License       |
#   along with this program.  If not, see <http://www.gnu.org/licenses/>.   |
#---------------------------------------------------------------------------|

import numpy as np
import scipy.optimize as sp

__docformat__ = "restructuredtext en"
__author__ = "Alexandre Vaudrey <alexandre.vaudrey@gmail.com>"
__date__ = "06/09/2017"

class SaturatedAmmonia:
    """
    Pure ammonia (NH3) in a saturated state.
    """
    def __init__(self):
        # If it's necessary to give a name to the state
        self.name = 'state1'
        # Temperature, in [°C]
        self._temperature = 20.0
        # Pressure, in [bar], by default equal to the equilibrium one at default
        # temperature.
        self._pressure = 8.534
    # Attributes defined as properties, with first of all, the temperature
    def _get_temperature(self):
        """ Temperature value, in [°C]. """
        return self._temperature
    def _set_temperature(self, t):
        """ New temperature value, in [°C]. """
        # Once we get the new value of temperature
        self._temperature = t
        # We immediately compute the corresponding value of the equilibrium
        # vapor pressure
        self._pressure = self.ammonia_equilibrium_vapor_pressure(t)
    temperature = property(_get_temperature, _set_temperature)
    def _get_pressure(self):
        """ Absolute pressure value, in [bar]. """
        return self._pressure
    def _set_pressure(self, p):
        """ New absolute pressure value, in [bar]. """
        # We check if this value is positive
        if p<0:
            raise ValueError("Absolute pressure must be positive!")
        self._pressure = p
        # An we calculate the corresponding value of the temperature, firstly in
        # defining the function to solve in order to find the temperature value
        def f_to_solve(temp):
            # We look for the root of the square of the function, and not of the
            # function itself, in order to go quicker to the solution and to
            # avoid any problem with the initial sign of the difference.
            return pow(self.ammonia_equilibrium_vapor_pressure(temp)-p,2)
        # Solving by a Newton method
        newt = sp.newton(f_to_solve, self._temperature)
        # And change of the temperature value.
        self._temperature = newt
    pressure = property(_get_pressure, _set_pressure)
    # Static methods 
    @staticmethod
    def ammonia_equilibrium_vapor_pressure(temperature):
        """
        Equilibrium vapor pressure of pure ammonia, in [bar] at a given temperature,
        in [°C], between -78°C and 133°C.
        """
        # Check of the temperature range
        if (temperature < -78.) or (temperature > 133.): 
            raise ValueError("Temperature must be such as -78°C < T < 70°C")
        # If it's OK
        tK = temperature + 273.15
        if (temperature > -78.) and (temperature < 60.):
            # Lower temperature range
            # Empirical coefficients used to compute the pressure value
            c = np.array([-1648.6068,9.584586,-1.638646e-2,2.403276e-5,-1.168708e-8])
            # Exponents
            e = np.arange(-1,4)
            logp = sum(c*pow(tK, e))
        else:
            # Higher temperature range, with a temperature in [K]
            # Critical temperature for ammonia
            tcrit = 406.
            # Empirical coefficients used to compute the pressure value
            c = np.array([2.9771,-1.492414e-3,1.36142e-5,-5.47917e-8])
            # Exponents
            e = np.arange(0,4)
            logp = 2.050418-(tcrit-tK)/tK*sum(c*pow(tcrit-tK,e))
        # The coefficient '1.01325' is here because the original formula considered
        # the pressure in atmospheres.
        return 1.01325*pow(10, logp)
    # And other methods
    def liquid_specific_volume(self):
        """
        Specific volume, in [m3/kg], of pure ammonia in its saturated liquid
        phase.
        """
        if (self._temperature < -70.) or (self._temperature > 130.):
            raise ValueError("Temperature T must be such as -70°C < T < 130°C")
        # Intermediate variables
        DT = 133.-self._temperature
        sDT = np.sqrt(DT)
        return 1e-3*(4.283+0.813055*sDT-0.0082861*DT)/(1+0.424805*sDT+0.015938*DT)
    def liquid_density(self):
        """
        Density, in [kg/m3], of pure ammonia in its saturated liquid phase.
        """
        return 1/self.liquid_specific_volume()
    def vapor_specific_volume(self):
        """
        Specific volume, in [m3/kg], of pure ammonia in its saturated vapor 
        phase.
        """
        if (self._temperature < -70.) or (self._temperature > 50.):
            raise ValueError("Temperature T must be such as -70°C < T < 50°C")
        # Experimental correlation
        logv = 1939.032/(self._temperature+273.1)-32.0661\
                +10.70409*np.log10(self._temperature+273.1)\
                +8.62366e-2*np.sqrt(133.-self._temperature)\
                +2.667e-3*(133.-self._temperature)
        # The result was initially in cm3/g, so 1e-3 factor
        return 1e-3*pow(10, logv)
    def vapor_density(self):
        """
        Density, in [kg/m3], of pure ammonia in its saturated vapor phase.
        """
        return 1/self.vapor_specific_volume()
    def vapor_enthalpy(self):
        """ Mass specific enthalpy of the vapor phase, in [kJ/kg]. """
        # Calculated from the NIST correlations :
        # http://webbook.nist.gov/cgi/inchi/InChI%3D1S/H3N/h1H3
        pass

class WaterAmmoniaMixture:
    """ Mixture of water and ammonia. """
    # TODO : Check if all the fractions of ammonia are actually calculated in
    # mass.
    def __init__(self):
        # Name of the mixture, if necessary
        self.name = 'mixture1'
        # Temperature, in [°C]
        self._temperature = 20.0
        # Pressure, in [bar] 
        self._pressure = 1.0
    # Attributes defined as properties, with first of all, the temperature
    def _get_temperature(self):
        """ Temperature value, in [°C]. """
        return self._temperature
    def _set_temperature(self, t):
        """ New temperature value, in [°C], that must be within a defined range
        in order to actually have a mixture of water and ammonia. """
        # ---- Pressure range check -------------------------------------------
        # Maximum and minimum values of the mixture temperature at the given
        # pressure are calculated using the saturated properties of each
        # component.
        tmin , tmax = self.temperature_range()
        # Check if the entered temperature is within the corresponding mixture
        # temperature range.
        if (t < tmin) or (t > tmax):
            # If not, we cannot have a mixture of water and ammonia
            raise ValueError("At this pressure, temperature it must be between %3.2f°C and %3.2f°C to have a mixture!" % (tmin, tmax))
        else:
            # If it's OK, we get the new value of the temperature
            self._temperature = t
    temperature = property(_get_temperature, _set_temperature)
    def _get_pressure(self):
        """ Absolute pressure value, in [bar]. """
        return self._pressure
    def _set_pressure(self, p):
        """ New absolute pressure value, in [bar], that must be within a limited 
        range in order to actually have a mixture of water and ammonia. """
        # We check if this value is positive
        if p<0:
            raise ValueError("Absolute pressure must be positive!")
        # ---- Temperature range check ----------------------------------------
        # Maximum and minimum bubble point temperatures at this new pressure
        pmin , pmax = self.pressure_range() 
        if (p < pmin) or (p > pmax):
            # If the entered pressure is lower than the water vapor equilibrium
            # one or higher than the ammonia equilibrium one, there is no
            # mixture.
            # Minimum and maximum temperature at this pressure correspond to the
            # equilibrium vapor ones of ammonia and water, respectively.
            raise ValueError("At this temperature, pressure must be between %3.3f bar and %3.3f bar to have a mixture!" % (pmin, pmax))
        else:
            # If it's OK, we get our new pressure
            self._pressure = p
    pressure = property(_get_pressure, _set_pressure)
    # ---- Saturated states of pure ammonia and water -------------------------
    @staticmethod
    def ammonia_equilibrium_vapor_pressure(temperature):
        """
        Equilibrium vapor pressure of pure ammonia, in [bar] at a given temperature,
        in [°C], between -78°C and 133°C.
        """
        # Check of the temperature range
        if (temperature < -78.) or (temperature > 133.): 
            raise ValueError("Temperature must be such as -78°C < T < 70°C")
        # If it's OK
        tK = temperature + 273.15
        if (temperature > -78.) and (temperature < 60.):
            # Lower temperature range
            # Empirical coefficients used to compute the pressure value
            c = np.array([-1648.6068,9.584586,-1.638646e-2,2.403276e-5,-1.168708e-8])
            # Exponents
            e = np.arange(-1,4)
            logp = sum(c*pow(tK, e))
        else:
            # Higher temperature range, with a temperature in [K]
            # Critical temperature for ammonia
            tcrit = 406.
            # Empirical coefficients used to compute the pressure value
            c = np.array([2.9771,-1.492414e-3,1.36142e-5,-5.47917e-8])
            # Exponents
            e = np.arange(0,4)
            logp = 2.050418-(tcrit-tK)/tK*sum(c*pow(tcrit-tK,e))
        # The coefficient '1.01325' is here because the original formula considered
        # the pressure in atmospheres.
        return 1.01325*pow(10, logp)
    def ammonia_equilibrium_vapor_temperature(self, pressure):
        """
        Equilibrium vapor temperature of pure ammonia, in [°C] at a given
        pressure, in [bar], between 0.11 bar and 108 bar.
        """
        # We calculate the corresponding value of the temperature, firstly in
        # defining the function to solve in order to find the temperature value.
        def f_to_solve(temp):
            # We look for the root of the square of the function, and not of the
            # function itself, in order to go quicker to the solution and to
            # avoid any problem with the initial sign of the difference.
            return self.ammonia_equilibrium_vapor_pressure(temp)-pressure
        # Solving by a Brent method within a temperature range corresponding to
        # the one going from the triple temperature to the critical one.
        teq = sp.brenth(f_to_solve, -77., 132.) 
        # And result
        return teq
    @staticmethod
    def water_equilibrium_vapor_pressure(temperature):
        """
        Equilibrium vapor pressure of pure water, in [bar] at a given temperature,
        in [°C], between 1°C and 374°C, using the Antoine's equation.
        """
        # Check of the temperature range
        if (temperature < 1.0) or (temperature > 374.): 
            raise ValueError("Temperature must be such as 0°C < T < 374°C")
        # If it's OK
        if (temperature > 1.0) and (temperature < 100.):
            # Lower temperature range
            # Empirical coefficients used to compute the pressure value
            c = np.array([8.07131, 1730.63, 233.426])
            logp = c[0] - c[1]/(c[2]+temperature)
        else:
            # Empirical coefficients used to compute the pressure value
            c = np.array([8.14019, 1810.94, 244.485])
            logp = c[0] - c[1]/(c[2]+temperature)
        # The coefficient '1.333224e-3' is here because the original formula
        # considered the pressure in Torricelli.
        return 1.333224e-3*pow(10, logp)
    def water_equilibrium_vapor_temperature(self, pressure):
        """
        Equilibrium vapor temperature of pure water, in [°C] at a given
        pressure, in [bar], between 0.008 bar and 217 bar.
        """
        # We calculate the corresponding value of the temperature, firstly in
        # defining the function to solve in order to find the temperature value.
        def f_to_solve(temp):
            # We look for the root of the square of the function, and not of the
            # function itself, in order to go quicker to the solution and to
            # avoid any problem with the initial sign of the difference.
            return self.water_equilibrium_vapor_pressure(temp)-pressure
        # Solving by a Brent method within a temperature range corresponding to
        # the one going from the triple temperature to the critical one.
        teq = sp.brenth(f_to_solve, 1.0, 374.) 
        # And result
        return teq
    @staticmethod
    def bubble_point_temperature(pressure, amm_liquid_mass_fraction):
        """
        Calculation of the bubble point temperature (in °C) of a Water+Ammonia
        mixture for given values of pressure (in bar) and ammonia liquid mass
        fraction xNH3. """
        # Experimental parameters used in the final calculation, that use the
        # formulation proposed by Pátek and Klomfar [1].
        # [1] http://dx.doi.org/10.1016/0140-7007(95)00006-W
        # Reference values of temperature and pressure
        T0 , p0 = 100., 20.
        # And corresponding empirical coefficients
        m = np.array([0]*5+[1]*3+[2]+[4]+[5]*2+[6]+[13])
        n = np.array([0,1,2,3,4,0,1,2,3,0,0,1,0,1])
        a = np.array([0.322302e+1,-0.384206e+0,0.460965e-1,-0.378945e-2,\
                      0.135610e-3,0.487755e+0,-0.120108e+0,0.106154e-1,\
                      -0.533589e-3,0.785041e+1,-0.115941e+2,-0.523150e-1,\
                      0.489596e+1,0.421059e-1])
        # And calculation of the bubble temperature
        Tb = T0*(np.array(a)*pow(1-amm_liquid_mass_fraction,np.array(m))*\
                 pow(np.log(p0/pressure),n)).sum()
        return Tb-273.15
    @staticmethod
    def dew_point_temperature(pressure, amm_vapour_mass_fraction):
        """
        Calculation of the dew point temperature (in °C) of a Water+Ammonia
        mixture for given values of pressure (in bar) and ammonia vapour mass
        fraction yNH3.
        """
        # Experimental parameters used in the final calculation, that use the
        # formulation proposed by Pátek and Klomfar [1].
        # [1] http://dx.doi.org/10.1016/0140-7007(95)00006-W
        # Reference values of temperature and pressure
        T0 , p0 = 100., 20.
        # And corresponding empirical coefficients
        m = np.array([0]*4+[1]*3+[2]*2+[3]*2+[4]*2+[5]*2+[6]+[7])
        n = np.array([0,1,2,3,0,1,2,0,1,0,1,0,2,0,2,0,2])
        a = np.array([0.324004e+1,-0.395920e+0,0.435624e-1,-0.218943e-2,\
                      -0.143526e+1,0.105256e+1,-0.719281e-1,0.122362e+2,\
                      -0.224368e+1,-0.201780e+2,0.110834e+1,0.145399e+2,\
                      0.644312e+0,-0.221246e+1,-0.756266e+0,-0.135529e+1,\
                      0.183541e+0])
        # And calculation of the bubble temperature
        Td = T0*(np.array(a)*pow(1-amm_vapour_mass_fraction,0.25*np.array(m))*\
                 pow(np.log(p0/pressure),n)).sum()
        return Td-273.15
    # ---- Mixture properties -------------------------------------------------
    def temperature_range(self):
        """ Range of temperature corresponding to a mixture of ammonia and
        water at the current pressure."""
        # Minimum and maximum temperature at this pressure, which correspond to
        # the equilibrium vapor ones of ammonia and water, respectively.
        teqmin = self.ammonia_equilibrium_vapor_temperature(self._pressure)
        teqmax = self.water_equilibrium_vapor_temperature(self._pressure)
        return [teqmin, teqmax]
    def pressure_range(self):
        """ Range of pressure corresponding to a mixture of ammonia and
        water at the current temperature. """
        # Minimum and maximum pressure at this temperature, which correspond to
        # the equilibrium vapor ones of water and ammonia, respectively.
        peqmin = self.water_equilibrium_vapor_pressure(self._temperature)
        peqmax = self.ammonia_equilibrium_vapor_pressure(self.temperature)
        return [peqmin, peqmax]
    def ammonia_liquid_mass_fraction(self):
        """ Mass fraction of ammonia xNH3 within the liquid phase, calculated
        from the temperature in [°C] and the pressure in [bar]. """
        def f_to_solve(xNH3):
            # Function created only for calculation purpose
            return self.bubble_point_temperature(self._pressure, xNH3)\
                    -self._temperature
        # And solving to obtain the solution, using the Brent method.
        return sp.brenth(f_to_solve,0.0,1.0)
    def ammonia_vapor_mass_fraction(self):
        """ Mass fraction of ammonia yNH3 within the vapor phase, calculated
        from the temperature in [°C] and the pressure in [bar]. """
        def f_to_solve(yNH3):
            # Function created only for calculation purpose
            return self.dew_point_temperature(self._pressure, yNH3)\
                    -self._temperature
        # And solving to obtain the solution, using the Brent method.
        return sp.brenth(f_to_solve,0.0,1.0)
    def liquid_enthalpy(self):
        """ Calculation of the enthalpy of the liquid phase, in kJ/kg. """
        # Experimental parameters used in the final calculation, that use the
        # formulation proposed by Pátek and Klomfar [1].
        # [1] http://dx.doi.org/10.1016/0140-7007(95)00006-W
        # Reference values of temperature and pressure
        T0 , h0 = 273.16 , 100.
        # And corresponding empirical coefficients
        m = np.array([0]*6+[1]*2+[2]+[3]+[5]*3+[6]*2+[8])
        n = np.array([1,4,8,9,12,14,0,1,1,3,3,4,5,2,4,0])
        a = np.array([-0.76108e+1,0.256905e+2,-0.247092e+3,0.325952e+3,\
                      -0.158854e+3,0.619084e+2,0.114314e+2,0.118157e+1,\
                      0.284179e+1,0.741609e+1,0.891844e+3,-0.161309e+4,\
                      0.622106e+3,-0.207588e+3,-0.687393e+1,0.350716e+1])
        # And calculation of the enthalpy 
        xNH3 = self.ammonia_liquid_mass_fraction()
        hL = h0*sum(np.array(a)*pow(xNH3, np.array(n))*pow((self._temperature+273.15)/T0-1.0,m))
        return hL
    def vapor_enthalpy(self):
        """ Calculation of the enthalpy of the vapor phase, in kJ/kg. """
        # Experimental parameters used in the final calculation, that use the
        # formulation proposed by Pátek and Klomfar [1].
        # [1] http://dx.doi.org/10.1016/0140-7007(95)00006-W
        # Reference values of temperature and pressure
        T0 , h0 = 324. , 1000.
        # And corresponding empirical coefficients
        m = np.array([0,1,2,3]*2+[0,1,2,0,1,0,4,2,1])
        n = np.array([0]*4+[2]*4+[3]*3+[4]*2+[5,6,7,10])
        a = np.array([0.128827e+1,0.125247e+0,-0.208748e+1,0.217696e+1,\
                      0.235687e+1,-0.886987e+1,0.102635e+2,-0.23744e+1,\
                      -0.670155e+1,0.164508e+2,-0.936849e+1,0.842254e+1,\
                      -0.858907e+1,-0.277049e+1,-0.961248e+0,0.988009e+0,\
                      0.308482e+0])
        # And calculation of the enthalpy 
        yNH3 = self.ammonia_vapor_mass_fraction()
        hG = h0*sum(np.array(a)*pow(1-yNH3,np.array(n)*0.25)*\
                    pow(1-(self._temperature+273.15)/T0,m))
        return hG

if __name__ == '__main__':
#    state1 = SaturatedAmmonia()
#    print(state1.__dict__)
#    state1.temperature = 50.
#    print(state1.__dict__)
    pass
