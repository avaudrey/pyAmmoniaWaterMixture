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
        """ Entry of a new temperature value, in [°C]. """
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
        """ Entry of a new absolute pressure value, in [bar]. """
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
    # Methods 
    @staticmethod
    def ammonia_equilibrium_vapor_pressure(temperature):
        """
        Equilibrium vapor pressure of pure ammonia, in [bar] at a given temperature,
        in [°C].
        """
        if (temperature < -78.) or (temperature > 70.):
            raise ValueError("Temperature T must be such as -78°C < T < 70°C")
        # Empirical coefficients used to compute the pressure value
        c = np.array([-1648.6068,9.584586,-1.638646e-2,2.403276e-5,-1.168708e-8])
        # Exponents
        e = np.arange(-1,4)
        logp = (c*pow(temperature+273.15, e)).sum()
        # The coefficient '1.01325' is here because the original formula considered
        # the pressure in atmospheres.
        return 1.01325*pow(10, logp)

if __name__ == '__main__':
    state1 = SaturatedAmmonia()
    print(state1.__dict__)
    state1.temperature = 50.
    print(state1.__dict__)
