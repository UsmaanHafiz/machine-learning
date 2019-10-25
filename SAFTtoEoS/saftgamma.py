#!/usr/bin/env python
import numpy as np 
import pandas as pd 
import time
from math import pi,tanh,log10
import matplotlib.pyplot as plt
import sys
from scipy.optimize import fsolve, least_squares

from dervarnp import *

import defconst as cst
import methods as mt 

class System(object):
    def __init__(self, temperature=293, pressure=101325, volume=1000, *args):
        '''
        System units definitions: temperature = K, pressure = Pa, volume = nm^3
        '''
        self.temperature = temperature
        self.pressure = pressure
        self.volume = volume
        self.moles = {}
        self.n_molecules = 0

        # storage variables for calculations
        self.molfrac = {}

    def n_den(self):
        return self.n_molecules / (self.volume * pow(cst.nmtom, 3))

    def set_molar_volume(self, molar_v):
        self.volume = self.n_molecules * molar_v / (cst.Na * pow(cst.nmtom,3))
        return self.volume

    def quick_set(self, *args):
        for arg in args:
            mt.checkerr(isinstance(arg, tuple) or isinstance(arg, list), "Use iterable tuple or list for quick_set")
            mt.checkerr(isinstance(arg[0], Component), "First element of iterable must be an instance of Component class")
            mt.checkerr(isinstance(arg[1], int), "Second element of iterable must be of int type")
            self.moles[arg[0]] = arg[1]
            self.molfrac[arg[0]] = 0

        self.n_molecules = sum(self.moles.values())
        nmoles = self.n_molecules
        for comp in self.molfrac:
            self.molfrac[comp] = self.moles[comp] / nmoles

        return self

    def add_comp(self, comp, number):
        mt.checkerr(isinstance(comp, Component), "Component entry must be an instance of Component class")
        mt.checkerr(isinstance(number, int), "Only integer amount of molecules can be added")

        if comp in self.moles:
            self.moles[comp] += number
        else:
            self.moles[comp] = number
            self.molfrac[comp] = 0

        self.n_molecules = sum(self.moles.values())
        nmoles = self.n_molecules
        for comp in self.molfrac:
            self.molfrac[comp] = self.moles[comp] / nmoles

        return self

    # Get derivatives #################

    def dT(self, a='both'):
        Var.set_order(1)
        T = self.temperature
        self.temperature = Var(T)

        fn = 0
        if a == 'res':
            fn = self.helmholtz_residual()
        elif a == 'ideal' or a == 'id':
            fn = self.helmholtz_ideal()
        else:
            fn = self.helmholtz()

        da = derivative(fn, self.temperature)
        self.temperature = T
        
        return da


    def dV(self, a='both'):
        Var.set_order(1)
        V = self.volume
        self.volume = Var(V)
        
        fn = 0
        if a == 'res':
            fn = self.helmholtz_residual()
        elif a == 'ideal' or a == 'id':
            fn = self.helmholtz_ideal()
        else:
            fn = self.helmholtz()

        da = derivative(fn, self.volume)
        da = da / cst.nmtom**3
        self.volume = V
        
        return da

    def dT2(self, a='both', dT=False):
        Var.set_order(2)
        T = self.temperature
        self.temperature = Var(T)

        fn = 0
        if a == 'res':
            fn = self.helmholtz_residual()
        elif a == 'ideal' or a == 'id':
            fn = self.helmholtz_ideal()
        else:
            fn = self.helmholtz()

        da, d2a = derivative(fn, self.temperature, order=2)
        self.temperature = T

        Var.set_order(1)
        return (da, d2a) if dT else d2a

    def dV2(self, a='both', dV=False):
        Var.set_order(2)
        V = self.volume
        self.volume = Var(V)
        
        fn = 0
        if a == 'res':
            fn = self.helmholtz_residual()
        elif a == 'ideal' or a == 'id':
            fn = self.helmholtz_ideal()
        else:
            fn = self.helmholtz()

        da, d2a = derivative(fn, self.volume, order=2)
        da = da / cst.nmtom**3
        d2a = d2a / cst.nmtom**6
        self.volume = V
        
        Var.set_order(1)
        return (da, d2a) if dV else d2a

    def dTdV(self, a='both', first_order=False):
        Var.set_order(2)
        V = self.volume
        T = self.temperature
        self.volume = Var(V)
        self.temperature = Var(T)
        
        fn = 0
        if a == 'res':
            fn = self.helmholtz_residual()
        elif a == 'ideal' or a == 'id':
            fn = self.helmholtz_ideal()
        else:
            fn = self.helmholtz()

        da, d2a = derivative(fn, self.temperature, self.volume, order=2)
        da[1] = da[1] / cst.nmtom**3
        d2a[1] = d2a[1] / cst.nmtom**3
        d2a[2] = d2a[2] / cst.nmtom**6
        self.volume = V
        self.temperature = T

        Var.set_order(1)
        return (da, d2a) if first_order else d2a

    # End derivatives #################

    def p_v_isotherm(self, volume, temperature=None, gibbs=False):
        '''
        Get pressure profile from volume inputs. Volume in m3 per mol
        '''
        if isinstance(temperature, float) or isinstance(temperature, int):
            self.temperature = temperature

        mt.checkerr(isinstance(volume,float) or isinstance(volume, np.ndarray), "Use floats or numpy array for volume")
        volume = self.n_molecules * volume / (cst.Na * pow(cst.nmtom,3))
        Var.set_order(1)
        # Case single volume value
        if isinstance(volume, float):
            self.volume = Var(volume)
            A = self.helmholtz()
            result = -derivative(A,self.volume,order=1) / pow(cst.nmtom,3)

            # Reset system back to float
            self.volume = volume
            return result

        # Case numpy array
        old_v = self.volume
        vlen = np.size(volume)
        P = np.zeros(vlen)
        if gibbs: G = np.zeros(vlen)

        self.volume = Var(volume)
        A = self.helmholtz()
        P = -derivative(A,self.volume) / pow(cst.nmtom,3)

        if gibbs: G = (A.value + P * self.volume.value*pow(cst.nmtom,3))/self.n_molecules

        # Reset system back to float
        self.volume = old_v
        return (P, G) if gibbs else P

    # Get direct properties ###########
    def get_pressure(self, temperature=None, volume=None, molar_volume=None):
        '''
        Calculates pressure in units Pa
        where Pressure = NkT/V - dA_res/dV
        '''
        if temperature == None or (not isinstance(temperature, (int, float))):
            temperature = self.temperature
        if molar_volume != None and isinstance(molar_volume, (int, float)):
            volume = self.set_molar_volume(molar_volume)
        elif volume == None or (not isinstance(volume, (int, float))):
            volume = self.volume

        self.temperature = temperature
        self.volume = volume

        dArdV = self.dV(a='res')
        P = (self.n_molecules / (volume * cst.nmtom**3) * cst.k * temperature - dArdV)

        return P
    
    def get_entropy(self, temperature=None, volume=None, molar_volume=None):
        '''
        Calculates entropy in units J/mol K
        where entropy = -dA/dT
        '''
        if temperature == None or (not isinstance(temperature, (int, float))):
            temperature = self.temperature
        if molar_volume != None and isinstance(molar_volume, (int, float)):
            volume = self.set_molar_volume(molar_volume)
        elif volume == None or (not isinstance(volume, (int, float))):
            volume = self.volume

        self.temperature = temperature
        self.volume = volume

        dAdT = self.dT()
        s = -dAdT * cst.Na / self.n_molecules

        return s

    def get_u(self, temperature=None, volume=None, molar_volume=None):
        '''
        Calculates internal energy in units J/mol
        where u = A + TS
        '''
        if temperature == None or (not isinstance(temperature, (int, float))):
            temperature = self.temperature
        if molar_volume != None and isinstance(molar_volume, (int, float)):
            volume = self.set_molar_volume(molar_volume)
        elif volume == None or (not isinstance(volume, (int, float))):
            volume = self.volume

        self.temperature = temperature
        self.volume = volume

        A = self.helmholtz()
        dAdT = self.dT()
        s = -dAdT
        u = (A + temperature * s) * cst.Na / self.n_molecules

        return u

    def get_enthalpy(self, temperature=None, volume=None, molar_volume=None):
        '''
        Calculates enthalpy in units J/mol
        where H = A + TS + PV
        '''
        if temperature == None or (not isinstance(temperature, (int, float))):
            temperature = self.temperature
        if molar_volume != None and isinstance(molar_volume, (int, float)):
            volume = self.set_molar_volume(molar_volume)
        elif volume == None or (not isinstance(volume, (int, float))):
            volume = self.volume

        self.temperature = temperature
        self.volume = volume

        A = self.helmholtz()
        dAdT = self.dT()
        dAdV = self.dV()
        s = -dAdT
        P = -dAdV
        h = (A + temperature * s + P * volume * cst.nmtom**3) * cst.Na / self.n_molecules

        return h

    def get_gibbs(self, temperature=None, volume=None, molar_volume=None):
        '''
        Calculates gibbs free energy in units J/mol
        where gibbs = A + PV
        '''
        if temperature == None or (not isinstance(temperature, (int, float))):
            temperature = self.temperature
        if molar_volume != None and isinstance(molar_volume, (int, float)):
            volume = self.set_molar_volume(molar_volume)
        elif volume == None or (not isinstance(volume, (int, float))):
            volume = self.volume

        self.temperature = temperature
        self.volume = volume

        A = self.helmholtz()
        dAdV = self.dV()

        P = -dAdV
        g = (A + P * volume * cst.nmtom**3) * cst.Na / self.n_molecules

        return g

    def get_cv(self, temperature=None, volume=None, molar_volume=None):
        '''
        Calculates isochoric heat capacity in units J/(mol K)
        where cv = (dU/dT)_V
        '''
        if temperature == None or (not isinstance(temperature, (int, float))):
            temperature = self.temperature
        if molar_volume != None and isinstance(molar_volume, (int, float)):
            volume = self.set_molar_volume(molar_volume)
        elif volume == None or (not isinstance(volume, (int, float))):
            volume = self.volume

        self.temperature = temperature
        self.volume = volume

        d2AdT2 = self.dT2()

        cv = -temperature * cst.Na / self.n_molecules * d2AdT2
        return cv

    def get_kappa(self, temperature=None, volume=None, molar_volume=None):
        '''
        Calculates isothermal compressibility in units m3 / J
        where kappa = -1/V (dV/dP)_T 
        '''
        if temperature == None or (not isinstance(temperature, (int, float))):
            temperature = self.temperature
        if molar_volume != None and isinstance(molar_volume, (int, float)):
            volume = self.set_molar_volume(molar_volume)
        elif volume == None or (not isinstance(volume, (int, float))):
            volume = self.volume

        self.temperature = temperature
        self.volume = volume

        d2adv2 = self.dV2(a='res')
        v = volume * cst.nmtom**3

        dpdv = -self.n_molecules * cst.k * temperature / pow(v, 2) - d2adv2
        kappa = -1 / (v * dpdv)
        return kappa

    def get_cp(self, temperature=None, volume=None, molar_volume=None):
        '''
        Calculates isobaric heat capacity in units J/(mol K)
        where cp = (dH/dT)_V
        '''
        if temperature == None or (not isinstance(temperature, (int, float))):
            temperature = self.temperature
        if molar_volume != None and isinstance(molar_volume, (int, float)):
            volume = self.set_molar_volume(molar_volume)
        elif volume == None or (not isinstance(volume, (int, float))):
            volume = self.volume

        self.temperature = temperature
        self.volume = volume

        d2r = self.dTdV(a='res')
        dt2_id = self.dT2(a='id')
        dTdV_r = d2r[1]
        dV2_r = d2r[2]
        dT2 = dt2_id + d2r[0]

        T = temperature
        V = volume * cst.nmtom**3
        N = self.n_molecules
        k = cst.k

        dhdt_v = -T * dT2 + N*k - V * dTdV_r
        dhdv_t = -T * dTdV_r - V * dV2_r
        dpdt_v = N*k/V - dTdV_r
        dpdv_t = -N*k*T/V**2 - dV2_r

        cp = dhdt_v - dhdv_t * dpdt_v / dpdv_t
        cp = cp / N * cst.Na
        return cp

    def get_w(self, temperature=None, volume=None, molar_volume=None):
        '''
        Calculates speed of sound in units m/s
        where w = sqrt(dP/drho) where rho is in kg/m3
        '''
        if temperature == None or (not isinstance(temperature, (int, float))):
            temperature = self.temperature
        if molar_volume != None and isinstance(molar_volume, (int, float)):
            volume = self.set_molar_volume(molar_volume)
        elif volume == None or (not isinstance(volume, (int, float))):
            volume = self.volume

        self.temperature = temperature
        self.volume = volume

        d2r = self.dTdV(a='res')
        dt2_id = self.dT2(a='id')
        dTdV_r = d2r[1]
        dV2_r = d2r[2]
        dT2 = dt2_id + d2r[0]

        T = temperature
        V = volume * cst.nmtom**3
        N = self.n_molecules
        k = cst.k

        dsdt_v = -dT2
        dsdv_t = N*k/V - dTdV_r
        dpdt_v = N*k/V - dTdV_r
        dpdv_t = -N*k*T/V**2 - dV2_r
        
        mass = 0.

        for c in self.moles:
            mass += c.mw * self.moles[c] / self.n_molecules
        mass = mass * 1e-3

        dpdv_s = dpdv_t - dpdt_v * dsdv_t / dsdt_v
        dpdv_s = dpdv_s * N / cst.Na # Pa/(m3/mol)
        vm = V * cst.Na / N
        w = -vm**2 * dpdv_s  # dp/d(rho) J / mol
        w = w / mass # J/kg = m2/s2
        w = sqrt(w)
        return w

    def get_jt(self, temperature=None, volume=None, molar_volume=None):
        '''
        Calculates Joule-Thomson coefficient in units K m3 / J
        where u_JT = (dT/dP)_h
        '''
        if temperature == None or (not isinstance(temperature, (int, float))):
            temperature = self.temperature
        if molar_volume != None and isinstance(molar_volume, (int, float)):
            volume = self.set_molar_volume(molar_volume)
        elif volume == None or (not isinstance(volume, (int, float))):
            volume = self.volume

        self.temperature = temperature
        self.volume = volume

        d2r = self.dTdV(a='res')
        dt2_id = self.dT2(a='id')
        dTdV_r = d2r[1]
        dV2_r = d2r[2]
        dT2 = dt2_id + d2r[0]

        T = temperature
        V = volume * cst.nmtom**3
        N = self.n_molecules
        k = cst.k

        dhdt_v = -T * dT2 + N*k - V * dTdV_r
        dhdv_t = -T * dTdV_r - V * dV2_r
        dpdt_v = N*k/V - dTdV_r
        dpdv_t = -N*k*T/V**2 - dV2_r

        dtdp_h = (dpdt_v - dpdv_t * dhdt_v / dhdv_t) ** -1
        ujt = dtdp_h
        return ujt
    # End direct properties ###########

    # Calculate critical point ########
    def critical_point(self, initial_t=300, v_nd=None, get_volume=False, get_density=False, print_results=True, solver=least_squares, solver_kwargs={'bounds': (0.1,10)}, xtol=1e-8, print_progress=False):
        tscale = initial_t
        x0 = np.array([1.])
        if not isinstance(solver_kwargs, dict): solver_kwargs = {}
        if not isinstance(v_nd, np.ndarray): v_nd = np.logspace(-4,-2, 50)
        Var.set_order(2)
        crit_pt = solver(self.__crit_t, x0, xtol=xtol, args=(v_nd, tscale, print_progress),**solver_kwargs)

        if print_progress: print()
        if print_results: print(crit_pt)
        T = crit_pt.x[0]*tscale

        parr = self.__crit_t(np.array([T]), v_nd, 1, False, True)
        v_guess = v_nd[parr==max(parr)]
        crit_v = solver(self.__crit_v, v_guess, xtol=xtol, args=(T,), bounds=(min(v_nd), max(v_nd)))
        v = crit_v.x[0]
        
        Var.set_order(1)
        self.volume = Var(mt.m3mol_to_nm(v, molecules=self.n_molecules))
        self.temperature = T

        A = self.helmholtz()
        P = -derivative(A, self.volume, order=1) / pow(cst.nmtom,3)

        self.volume = mt.m3mol_to_nm(v, molecules=self.n_molecules)
        results = (P, T)
        if get_volume: results += (v,)
        if get_density: results += (1/v,)

        return results

    def __crit_v(self, v, T):
        self.temperature = T
        v = v[0]
        self.volume = Var(mt.m3mol_to_nm(v, molecules=self.n_molecules))
        A = self.helmholtz()
        _, dpdv = derivative(A, self.volume, order=2)
        dpdv = -dpdv / (pow(cst.nmtom,6) * cst.Na) / (cst.R * T)

        return dpdv

    def __crit_t(self, x, v_nd, scale, print_, getarr=False):
        T = x[0] * scale

        # numpy version
        self.volume = Var(mt.m3mol_to_nm(v_nd, molecules=self.n_molecules))
        self.temperature = T
        A = self.helmholtz()
        _,dpdv = derivative(A, self.volume, order=2)
        dp = -dpdv / (pow(cst.nmtom,6) * cst.Na) / (cst.R * T)
        if print_: 
            print(f'Current: T = {T:7.3f}: max dP/dV = {max(dp):7.3e}', end='\r')
        return max(dp) if getarr == False else dp
    # Critical point end ##############

    # Calculate vapour pressure #######
    def __eqpg(self, x, print_, scaler):
        '''
        Takes in ndarray to find Pressure1 - Pressure2
        '''
        P = []
        G = []
        v_ = [x[0]*scaler[0], x[1]*scaler[1]]
        for i in v_:
            self.volume = Var(mt.m3mol_to_nm(i, molecules=self.n_molecules))
            Pi = -self.dV()
            P.append(Pi)
            G.append((A.value + Pi * self.volume.value*pow(cst.nmtom,3))*cst.Na)

        if print_: print(f'Current: v\'s = {v_[0]:7.3e}, {v_[1]:7.3e}; dP = {(P[0]-P[1]):7.3e}, dG = {(G[0]-G[1]):7.3e}')
        return np.array([(P[0]-P[1]), (G[0]-G[1])])

    def vapour_pressure(self, temperature=None, initial_guess=(1e-4,.1), get_volume=False, get_gibbs=False, print_results=True, solver=least_squares, solver_kwargs={'bounds': ((0.1,1e-2),(1e2,1e2))}, print_progress=False):
        if isinstance(temperature, float) or isinstance(temperature, int):
            self.temperature = temperature
        scaler = initial_guess
        x0 = np.array([1.,1.])
        if not isinstance(solver_kwargs, dict): solver_kwargs = {}
        vle = solver(self.__eqpg, x0, args=(print_progress, scaler), **solver_kwargs)
        if print_progress: print()
        if print_results: 
            print(f'Scaled wrt initial guess of ({scaler[0]:7.3e},{scaler[1]:7.3e})')
            print(vle)
        vlv = (vle.x[0]*scaler[0], vle.x[1]*scaler[1])
        v = vlv[1]
        self.volume = Var(mt.m3mol_to_nm(v, molecules=self.n_molecules))

        A = self.helmholtz()
        P = -derivative(A, self.volume, order=1) / pow(cst.nmtom,3)
        G = (A.value + P * self.volume.value*pow(cst.nmtom,3)) / self.n_molecules
        self.volume = mt.m3mol_to_nm(v, molecules=self.n_molecules)
        result = (P,)
        if get_volume: result += (vlv,)
        if get_gibbs: result += (G,)
        return result
    # Vapour pressure end #############

    # Get volume from set P ###########
    def __getp(self, x, targetP):
        '''
        Takes in single value to get P
        '''
        x = x[0]
        v = mt.m3mol_to_nm(x, molecules=self.n_molecules)
        self.volume = v
        P = -self.dV()

        return P - targetP

    def single_phase_v(self, P, T=None, print_results=True, get_density=False, use_jac=False, vle_ig=(1e-4, 1e-2), v_crit=None, v_init=None, supercritical=False, solver_kwargs={'bounds': (1e-4,1e-1)}):
        if isinstance(T, (int,float)):
            self.temperature = T

        if isinstance(P, (int,float)):
            if supercritical:
                mt.checkerr(v_crit is not None, "For supercritical, please provide v_crit")
                v_init = v_crit if v_init is None else v_init
                solver_kwargs['bounds'] = (0.2 * v_crit, 1000*v_crit)
            else:
                if v_crit is not None:
                    vle_ig = (0.25 * v_crit, 100 * v_crit)
                    solver_kwargs['bounds'] = (0.2 * v_crit, 1000*v_crit)
                Pv, vle = self.vapour_pressure(temperature=T, initial_guess=vle_ig, get_volume=True, print_results=False)
                v_init = vle[0] if P > Pv else vle[1]
            getv = least_squares(self.__getp, v_init, kwargs={"targetP": P}, **solver_kwargs)

            if print_results: print(getv)

            return getv.x[0] if get_density == False else (getv.x[0], 1/getv.x[0])

        mt.checkerr(isinstance(P, np.ndarray), "Use float/int or ndarray for pressure inputs")
        varr = np.zeros(np.size(P))
        for i in range(len(P)):
            Pv, vle = self.vapour_pressure(temperature=T, initial_guess=vle_ig, get_volume=True, print_results=False)
            v_init = vle[0] if P[i] > Pv else vle[1]
            getv = least_squares(self.__getp, v_init, kwargs={"targetP": P[i]}, **solver_kwargs)

            varr[i] = getv.x[0]

        return varr if get_density == False else (varr, 1./varr)
    # End v finding ###################

    def helmholtz(self):
        '''
        Function placeholder: this has to be altered for different versions of 
        equation-of-state and systems.
        '''
        return 42

    def helmholtz_ideal(self):
        '''
        Function placeholder: this has to be altered for different versions of 
        equation-of-state and systems.
        '''
        return 42

    def helmholtz_residual(self):
        '''
        Function placeholder: this has to be altered for different versions of 
        equation-of-state and systems.
        '''
        return 42

class SAFTVRSystem(System):
    '''
    SAFT-VR Mie System as defined by SAFT-VR Mie Equation-of-state
    given in ref: J. Chem. Phys. 139, 154504 (2013); https://doi.org/10.1063/1.4819786
    '''
    def __init__(self, temperature=293, pressure=101325, volume=1000):
        super().__init__(temperature, pressure, volume)

        # stored values
        self.hsd = np.array([[None]])
        self.segratio = 0.
        self.segden = 0.
        self.gcomb = np.array([[None]])

    def helmholtz(self):
        # calculate stored values
        nvr = VRMieComponent.n_total()
        self.hsd = np.zeros((nvr, nvr), dtype=object)
        self.gcomb = np.zeros((nvr, nvr), dtype=object)
        segratio = 0.
        for comp in self.moles:
            # molfrac
            self.molfrac[comp] = self.moles[comp] / self.n_molecules
            # hard-sphere diam
            ci = comp.index
            self.hsd[ci, ci] = comp.hsd(self.temperature) * 1e-10
            for comp2 in self.moles:
                ci2 = comp2.index
                if self.hsd[ci, ci2] == 0. and self.hsd[ci2, ci2] != 0.:
                    self.hsd[ci, ci2] = (self.hsd[ci,ci] + self.hsd[ci2, ci2]) / 2
                    self.hsd[ci2, ci] = (self.hsd[ci,ci] + self.hsd[ci2, ci2]) / 2
                #gcomb
                self.gcomb[ci, ci2] = comp + comp2
            # segden
            segratio += self.molfrac[comp] * comp.ms
        self.segratio = segratio
        self.segden = segratio * self.n_den()

        # calculate helmholtz through indiv contributions
        A = self.a_ideal() + self.a_res() * self.n_molecules * cst.k * self.temperature

        return A

    def helmholtz_residual(self):
        # calculate stored values
        nvr = VRMieComponent.n_total()
        self.hsd = np.zeros((nvr, nvr), dtype=object)
        self.gcomb = np.zeros((nvr, nvr), dtype=object)
        segratio = 0.
        for comp in self.moles:
            # molfrac
            self.molfrac[comp] = self.moles[comp] / self.n_molecules
            # hard-sphere diam
            ci = comp.index
            self.hsd[ci, ci] = comp.hsd(self.temperature) * 1e-10
            for comp2 in self.moles:
                ci2 = comp2.index
                if self.hsd[ci, ci2] == 0. and self.hsd[ci2, ci2] != 0.:
                    self.hsd[ci, ci2] = (self.hsd[ci,ci] + self.hsd[ci2, ci2]) / 2
                    self.hsd[ci2, ci] = (self.hsd[ci,ci] + self.hsd[ci2, ci2]) / 2
                #gcomb
                self.gcomb[ci, ci2] = comp + comp2
            # segden
            segratio += self.molfrac[comp] * comp.ms
        self.segratio = segratio
        self.segden = segratio * self.n_den()

        # calculate helmholtz through indiv contributions
        a = self.a_res()
        A = a * self.n_molecules * cst.k * self.temperature
        return A

    def helmholtz_ideal(self):
        # calculate stored values
        for comp in self.moles:
            # molfrac
            self.molfrac[comp] = self.moles[comp] / self.n_molecules

        # calculate helmholtz through indiv contributions
        A = self.a_ideal()
        return A

    def a_ideal(self):
        result = 0.
        mole_tol = self.n_molecules
        cpideal = True
        for comp in self.moles:
            cpint = comp.cp_int(self.temperature)
            if cpint is None:
                cpideal = False

        if cpideal == False:
            for comp in self.moles:
                debrogv = pow(comp.thdebroglie(self.temperature),3)
                molfrac = self.molfrac[comp]
                nden = molfrac * self.n_den()
                result = result + molfrac * log(nden * debrogv) 
            result = result - 1
            return result * self.n_molecules * cst.k * self.temperature
        
        for comp in self.moles:
            h0 = comp.ref[0]
            s0 = comp.ref[1]
            tref = comp.ref[2]
            pref = comp.ref[3]
            vref = cst.Na * cst.k * tref / pref
            t = self.temperature
            xi = self.molfrac[comp]
            n_mol = self.n_molecules / cst.Na
            V = self.volume * cst.nmtom**3
            cpint = comp.cp_int(t)
            cptint = comp.cp_t_int(t)
            
            si = cptint - cst.Na * cst.k * log(xi * t * vref * n_mol / (tref*V)) + s0

            ai_mol = cpint - t * si - cst.Na * cst.k * t + h0 # per mol
            result = result + xi * n_mol * ai_mol

        return result

    def a_res(self):
        return self.a_mono() + self.a_chain() + self.a_assoc()

    def a_mono(self):
        a_m = self.a_hs() + self.a_1() + self.a_2() + self.a_3()
        return a_m

    def a_chain(self):
        result = 0.
        for c in self.moles:
            molfrac = self.molfrac[c]
            ms = c.ms
            gmieii = self.__gmieij(c.index, c.index)
            result += molfrac * (ms - 1) * log(gmieii)
        result = -result
        return result

    def a_assoc(self):
        return 0 # placeholder self.volume ** 3 * 0.1 + exp(-self.volume)

    # for all
    def __xi_m(self, m):
        fac = pi * self.segden / 6
        gsum = 0.
        for c in self.moles:
            xsi = self.molfrac[c] * c.ms / self.segratio
            gsum += xsi * pow(self.hsd[c.index, c.index], m)
        return fac * gsum

    def __xi_x(self):
        fac = pi * self.segden / 6
        result = 0.
        for c1 in self.moles:
            c1i = c1.index
            xsi = self.molfrac[c1] * c1.ms / self.segratio
            for c2 in self.moles:
                c2i = c2.index
                xsj = self.molfrac[c2] * c2.ms / self.segratio
                dkl = self.hsd[c1i, c2i]
                result += xsi * xsj * pow(dkl, 3)
        return fac * result

    def __xi_sx(self):
        fac = pi * self.segden / 6
        result = 0.
        for c1 in self.moles:
            c1i = c1.index
            xsi = self.molfrac[c1] * c1.ms / self.segratio
            for c2 in self.moles:
                c2i = c2.index
                xsj = self.molfrac[c2] * c2.ms / self.segratio
                skl = self.gcomb[c1i, c2i].sigma * cst.atom
                result += xsi * xsj * pow(skl, 3)
        return fac * result

    def __bkl(self, c1i, c2i, lam):
        '''
        Use 2 groups as input
        set rep = True for lam_r, rep = False for lam_a
        2 * pi * segden * dkl^3 epsikl * bkl_sum(lambda_r or a, xi_x)
        '''
        comb = self.gcomb[c1i, c2i]
        segden = self.segden
        dkl = self.hsd[c1i, c2i]
        dkl_3 = pow(dkl, 3)
        epsikl = comb.epsilon * cst.k

        x0kl = comb.x0kl(dkl)
        xix = self.__xi_x()

        bsum = mt.bkl_sum(xix, x0kl, lam)
        result = 2*pi*segden*dkl_3*epsikl*bsum

        return result

    def __as1kl(self, c1i, c2i, lam):
        '''
        - 2 * pi * segden * (epsikl * dkl^3/ (lam - 3)) * (1 - xi_x_eff/2)/(1-xi_x_eff)^3
        '''
        comb = self.gcomb[c1i, c2i]
        segden = self.segden
        dkl = self.hsd[c1i, c2i]
        dkl_3 = pow(dkl, 3)
        epsikl = comb.epsilon * cst.k

        xix = self.__xi_x()

        xixeff = mt.xi_x_eff(xix, lam)

        num = 1 - xixeff/2
        den = pow(1 - xixeff, 3)

        result = -2 * pi * segden * (epsikl * dkl_3 / (lam - 3)) * (num/den)

        return result

    # for mono
    def a_hs(self):
        xi1 = self.__xi_m(1) # nm-2
        xi2 = self.__xi_m(2) # nm-1
        xi3 = self.__xi_m(3) # dimless
        xi0 = self.__xi_m(0) # nm-3

        if len(self.moles) == 1:
            t1 = 0
        else:
            t1 = (pow(xi2, 3) / pow(xi3, 2) - xi0) * log(1-xi3) # t1 is HUGE, and because of numerical discrepancy, even though
                                                                # if there is only 1 component, t1 approx 0,
                                                                # but because each term is ~ 10^27 so discrepancies could be millions even
        t2 = 3 * xi1 * xi2 / (1 - xi3)
        t3 = pow(xi2, 3) / (xi3 * pow((1-xi3),2))
        # print('t2 + t3 gives', t2+t3)
        # print('With pure imp:',(4*xi3 - 3*pow(xi3,2))/pow(1-xi3,2))

        xiterm = t1 + t2 + t3 # nm-3
        result = 6 * xiterm / (pi * self.n_den()) # uses n_den instead of segden
                                                  # because it's just gonna multiply segratio again
                                                  # so cancel it out
        return result

    def a_1(self):
        '''
        1/kT * cgss * sum of sum of _g1/cgss _g2/cgss * a1kl
        '''
        a1sum = 0.
        for c1 in self.moles:
            xsi = self.molfrac[c1] * c1.ms / self.segratio
            for c2 in self.moles:
                xsj = self.molfrac[c2] * c2.ms / self.segratio
                a1kl = self.__a_1kl(c1.index, c2.index)
                a1sum += xsi * xsj * a1kl 
        result = self.segratio / (cst.k * self.temperature) * a1sum 
        return result

    def __a_1kl(self, c1i, c2i):
        '''
        Ckl * [ x0kl^att(as1kl(att) + bkl(att)) - x0kl^rep(as1kl(rep) + bkl(rep)) ]
        '''
        comb = self.gcomb[c1i, c2i]
        hsd = self.hsd[c1i, c2i]

        x0kl = comb.x0kl(hsd)
        rep = comb.rep
        att = comb.att
        ckl = comb.premie()

        as1kl_a = self.__as1kl(c1i, c2i, att)
        bkl_a = self.__bkl(c1i, c2i, att)

        as1kl_r = self.__as1kl(c1i, c2i, rep)
        bkl_r = self.__bkl(c1i, c2i, rep)

        t1 = pow(x0kl, att) * (as1kl_a + bkl_a)
        t2 = pow(x0kl, rep) * (as1kl_r + bkl_r)
        result = ckl * (t1 - t2)

        return result

    def a_2(self):
        '''
        (1/kT)**2 * cgss * sum of sum of _g1/cgss _g2/cgss * a2kl
        '''
        a2sum = 0.
        for c1 in self.moles:
            xsi = self.molfrac[c1] * c1.ms / self.segratio
            for c2 in self.moles:
                xsj = self.molfrac[c2] * c2.ms / self.segratio
                a2kl = self.__a_2kl(c1.index, c2.index)
                a2sum += xsi * xsj * a2kl 
        result = self.segratio / pow((cst.k * self.temperature),2) * a2sum 
        return result

    def __a_2kl(self, c1i, c2i):
        '''
        1/2 * khs * (1-corrf) * epsikl * ckl^2 *
        { x0kl^(2att) * (as1kl(2*att) + bkl(2*att))
          - 2*x0kl^(att+rep) * (as1kl(att+rep) + bkl(att+rep))
          + x0kl^(2rep) * (as1kl(2rep) + bkl(2rep)) }
        '''
        comb = self.gcomb[c1i, c2i]
        hsd = self.hsd[c1i, c2i]
        khs = self.__khs()
        corrf = self.__corrf(c1i, c2i)
        epsikl = comb.epsilon * cst.k
        ckl = comb.premie()
        x0kl = comb.x0kl(hsd)
        rep = comb.rep
        att = comb.att

        t1 = pow(x0kl, 2*att) * (self.__as1kl(c1i, c2i, 2*att) + self.__bkl(c1i, c2i, 2*att))
        t2 = 2*pow(x0kl, att+rep) * (self.__as1kl(c1i, c2i, att+rep) + self.__bkl(c1i, c2i, att+rep))
        t3 = pow(x0kl, 2*rep) * (self.__as1kl(c1i, c2i, 2*rep) + self.__bkl(c1i, c2i, 2*rep))

        fac = 0.5 * khs * (1+corrf) * epsikl * pow(ckl, 2)
        res = t1 - t2 + t3

        result = fac * res
        return result

    def __khs(self):
        xix = self.__xi_x()
        num = pow(1-xix, 4)
        den = 1 + 4*xix + 4*pow(xix,2) - 4*pow(xix,3) + pow(xix,4)
        return num/den

    def __corrf(self, c1i, c2i):
        xisx = self.__xi_sx()
        alkl = self.__alphakl(c1i, c2i)

        t1 = mt.f_m(alkl, 1) * xisx
        t2 = mt.f_m(alkl, 2) * pow(xisx, 5)
        t3 = mt.f_m(alkl, 3) * pow(xisx, 8)
        result = t1 + t2 + t3
        return result

    def __alphakl(self, c1i, c2i):
        comb = self.gcomb[c1i, c2i]
        t1 = 1 / (comb.att - 3)
        t2 = 1 / (comb.rep - 3)
        result = comb.premie() * (t1 - t2) 

        return result
    
    def a_3(self):
        '''
        (1/kT)**3 * cgss * sum of sum of _g1/cgss _g2/cgss * a3kl
        '''
        a3sum = 0.
        for c1 in self.moles:
            xsi = self.molfrac[c1] * c1.ms / self.segratio
            for c2 in self.moles:
                xsj = self.molfrac[c2] * c2.ms / self.segratio
                a3kl = self.__a_3kl(c1.index, c2.index)
                a3sum += xsi * xsj * a3kl 
        result = self.segratio / pow((cst.k * self.temperature),3) * a3sum 
        return result
    
    def __a_3kl(self, c1i, c2i):
        xisx = self.__xi_sx()
        alkl = self.__alphakl(c1i, c2i)
        comb = self.gcomb[c1i, c2i]
        epsikl = comb.epsilon * cst.k

        preexp = - pow(epsikl,3) * mt.f_m(alkl, 4) * xisx
        expt1 = mt.f_m(alkl, 5) * xisx 
        expt2 = mt.f_m(alkl, 6) * pow(xisx,2)

        result = preexp * exp(expt1 + expt2)
        return result
    
    # for chain
    def __gmieij(self, c1i, c2i):
        gdhs = self.__gdhs(c1i, c2i)
        g1 = self.__g1(c1i, c2i)
        g2 = self.__g2(c1i, c2i)
        b = 1 / (cst.k * self.temperature)
        gcomp = self.gcomb[c1i, c2i]
        epsi = gcomp.epsilon * cst.k

        expt = b * epsi * g1 / gdhs + pow(b*epsi,2) * g2 / gdhs
        result = gdhs * exp(expt)
        return result

    def __der_xi_x(self):
        fac = pi / 6
        result = 0.
        for c1 in self.moles:
            c1i = c1.index
            xsi = self.molfrac[c1] * c1.ms / self.segratio
            for c2 in self.moles:
                c2i = c2.index
                xsj = self.molfrac[c2] * c2.ms / self.segratio
                dkl = self.hsd[c1i, c2i]
                result += xsi * xsj * pow(dkl, 3)
        return fac * result

    def __der_as1kl(self, c1i, c2i, lam):
        '''
        -2 * pi * (ep*d^3/(lam-3)) * (1/(1-xieff)^3) * (1 - xieff/2 + segden * derxieff * (3*(1-xieff/2)/(1-xieff) - 1/2))
        '''
        segden = self.segden
        dkl = self.hsd[c1i, c2i]
        d_3 = pow(dkl, 3)
        gcomp = self.gcomb[c1i, c2i]
        epsi = gcomp.epsilon * cst.k

        derxix = self.__der_xi_x()
        xix = self.__xi_x()

        derxixeff = mt.der_xi_x_eff(xix, derxix, lam)
        xixeff = mt.xi_x_eff(xix, lam)

        fac = 2 * pi * epsi * d_3 / (lam-3) / pow(1-xixeff,3)
        t1 = 1 - xixeff / 2
        t2 = segden * derxixeff
        t3 = 3 * (1-xixeff/2) / (1-xixeff)

        result = - fac * (t1 + t2 * (t3 - 1/2))

        return result

    def __der_bkl(self, c1i, c2i, lam):
        segden = self.segden
        dkl = self.hsd[c1i, c2i]
        d_3 = pow(dkl, 3)
        gcomp = self.gcomb[c1i, c2i]
        epsi = gcomp.epsilon * cst.k

        x0ii = gcomp.x0kl(dkl)
        xix = self.__xi_x()
        derxix = self.__der_xi_x()

        fac = 2 * pi * epsi * d_3 / pow(1-xix,3)

        t11 = 1 - xix/2
        t12 = segden * derxix
        t13 = 3 * (1-xix/2) / (1-xix) - 1/2
        t1 = t11 + t12 * t13

        t21 = 1 + pow(xix,2)
        t22 = segden*derxix
        t23 = 1 + 2 * xix + 3 * xix * (1+xix) / (1-xix) 
        t2 = 9/2 * (t21 + t22 * t23)

        result = fac * (t1 * mt.intI(x0ii,lam) - t2 * mt.intJ(x0ii,lam))
        return result

    def __der_a1kl(self, c1i, c2i):
        gcomp = self.gcomb[c1i, c2i]
        x0ii = gcomp.x0kl(self.hsd[c1i, c2i])
        rep = gcomp.rep
        att = gcomp.att
        ckl = gcomp.premie()

        tatt = self.__der_as1kl(c1i, c2i, att) + self.__der_bkl(c1i, c2i, att)
        trep = self.__der_as1kl(c1i, c2i, rep) + self.__der_bkl(c1i, c2i, rep)

        result = ckl * (pow(x0ii, att) * tatt - pow(x0ii, rep) * trep)
        return result

    def __gdhs(self, c1i, c2i):
        xi_x = self.__xi_x()
        comb = self.gcomb[c1i, c2i]
        x0ii = comb.x0kl(self.hsd[c1i, c2i])

        result = mt.gdhs(x0ii, xi_x)
        return result

    def __g1(self, c1i, c2i):
        '''
        1 / (2pi epsiii hsdii^3) * [ 3 da1kl/dp - premieii * attii * x0ii^attii * (as1kl(attii) + Bkl(attii))/segden 
                                     +  premieii * repii * x0ii^repii * (as1kl(repii) + Bkl(repii))/segden ]
        '''
        gcomp = self.gcomb[c1i, c2i]
        hsd = self.hsd[c1i, c2i]
        premie = gcomp.premie()
        epsi = gcomp.epsilon * cst.k
        rep = gcomp.rep
        att = gcomp.att
        x0ii = gcomp.x0kl(hsd)

        segden = self.segden

        t1 = 3* self.__der_a1kl(c1i, c2i)
        t2 = premie * att * pow(x0ii, att) * (self.__as1kl(c1i, c2i,att) + self.__bkl(c1i, c2i,att)) / segden
        t3 = premie * rep * pow(x0ii, rep) * (self.__as1kl(c1i, c2i,rep) + self.__bkl(c1i, c2i,rep)) / segden

        result = 1 / (2 * pi * epsi * pow(hsd,3)) * (t1 - t2 + t3)

        return result

    def __g2(self, c1i, c2i):
        '''
        (1+gammacii) * g2MCA(hsdii)
        '''
        gcomp = self.gcomb[c1i, c2i]
        xisx = self.__xi_sx()
        alii = self.__alphakl(c1i, c2i)
        theta = exp(gcomp.epsilon / self.temperature) -1

        gammacii = cst.phi[6,0] * (-tanh(cst.phi[6,1] * (cst.phi[6,2]-alii)) + 1) * xisx * theta * exp(cst.phi[6,3]*xisx + cst.phi[6,4] * pow(xisx, 2))
        g2mca = self.__g2mca(c1i, c2i)
        result = (1 + gammacii) * g2mca
        return result

    def __der_khs(self):
        xix = self.__xi_x()
        derxix = self.__der_xi_x()

        den = 1 + 4*xix + 4*pow(xix,2) - 4*pow(xix,3) + pow(xix,4)
        t1 = 4 * pow(1-xix,3) / den
        t2 = pow(1-xix,4) * (4 + 8*xix -12*pow(xix,2) + 4*pow(xix,3)) / pow(den,2)
        result = derxix * -(t1 + t2)

        return result

    def __der_a2kl(self, c1i, c2i):
        khs = self.__khs()
        derkhs = self.__der_khs()
        gcomp = self.gcomb[c1i, c2i]
        hsd = self.hsd[c1i, c2i]
        epsi = gcomp.epsilon * cst.k
        ckl = gcomp.premie()
        x0kl = gcomp.x0kl(hsd)
        rep = gcomp.rep
        att = gcomp.att

        t11 = pow(x0kl, 2*att) * (self.__as1kl(c1i, c2i, 2*att) + self.__bkl(c1i, c2i, 2*att))
        t12 = 2*pow(x0kl, att+rep) * (self.__as1kl(c1i, c2i, att+rep) + self.__bkl(c1i, c2i, att+rep))
        t13 = pow(x0kl, 2*rep) * (self.__as1kl(c1i, c2i, 2*rep) + self.__bkl(c1i, c2i, 2*rep))
        t1 =  derkhs * (t11 - t12 + t13)

        t21 = pow(x0kl, 2*att) * (self.__der_as1kl(c1i, c2i, 2*att) + self.__der_bkl(c1i, c2i, 2*att))
        t22 = 2*pow(x0kl, att+rep) * (self.__der_as1kl(c1i, c2i, att+rep) + self.__der_bkl(c1i, c2i, att+rep))
        t23 = pow(x0kl, 2*rep) * (self.__der_as1kl(c1i, c2i, 2*rep) + self.__der_bkl(c1i, c2i, 2*rep))
        t2 = khs * (t21 - t22 + t23)

        result = 0.5 * epsi * pow(ckl, 2) * (t1 + t2)
        return result

    def __g2mca(self, c1i, c2i):
        '''
        1 / (2pi epsiii^2 hsdii^3) *
        [
        3 * d/dp (a2ii/(1+chi))
        - epsiii * KHS * premie^2 * rep * x0ii^2rep * (as1kl(2rep) + Bkl(2rep))/segden 
        + epsiii * KHS * premie^2 * (rep+att) * x0ii^(rep+att) * (as1kl(rep+att) + Bkl(rep+att))/segden 
        - epsiii * KHS * premie^2 * att * x0ii^2att * (as1kl(2att) + Bkl(2att))/segden 
        ]
        '''
        gcomp = self.gcomb[c1i, c2i]
        hsd = self.hsd[c1i, c2i]
        premie = gcomp.premie()
        epsi = gcomp.epsilon * cst.k
        rep = gcomp.rep
        att = gcomp.att
        x0ii = gcomp.x0kl(hsd)
        khs = self.__khs()

        segden = self.segden

        t1 = 3 * self.__der_a2kl(c1i, c2i)
        t2 = epsi * khs * pow(premie,2) * rep * pow(x0ii, 2*rep) * (self.__as1kl(c1i, c2i,2*rep) + self.__bkl(c1i, c2i,2*rep)) / segden
        t3 = epsi * khs * pow(premie,2) * (rep+att) * pow(x0ii, rep+att) * (self.__as1kl(c1i, c2i,rep+att) + self.__bkl(c1i, c2i,rep+att)) / segden
        t4 = epsi * khs * pow(premie,2) * att * pow(x0ii, 2*att) * (self.__as1kl(c1i, c2i,2*att) + self.__bkl(c1i, c2i,2*att)) / segden
        
        result = 1 / (2 * pi * pow(epsi,2) * pow(hsd, 3)) * (t1 - t2 + t3 - t4)

        return result

    # for assoc

class SAFTgMieSystem(System):
    '''
    SAFT-gamma Mie System as defined by SAFT-gamma Mie Equation-of-state
    given in ref: J. Chem. Phys. 140, 054107 (2014); https://doi.org/10.1063/1.4851455
    '''
    def __init__(self, temperature=293, pressure=101325, volume=1000):
        super().__init__(temperature, pressure, volume)
        self.groups = {}

        # stored values
        self.ghsd = np.array([[None]])
        self.segratio = 0.
        self.segden = 0.
        self.gcomb = np.array([[None]])
        self.chsd = np.array([None])
        self.ccomb = np.array([None])
        
    def get_groups(self):
        self.groups = {}
        for comp in self.moles:
            for g in comp.groups:
                if g not in self.groups:
                    self.groups[g] = self.molfrac[comp] * comp.groups[g] 
                else:
                    self.groups[g] += self.molfrac[comp] * comp.groups[g]
        return self.groups

    def quick_set(self, *args):
        super().quick_set(*args)
        self.get_groups()
        return self

    def add_comp(self, comp, number):
        super().quick_set(*args)
        self.get_groups()
        return self

    def helmholtz(self):
        # calculate stored values
        nmg = GMieGroup.n_total()
        self.ghsd = np.zeros((nmg, nmg), dtype=object)
        self.gcomb = np.zeros((nmg, nmg), dtype=object)
        segratio = 0.

        for comp in self.moles:
            # molfrac
            self.molfrac[comp] = self.moles[comp] / self.n_molecules

        self.get_groups() # in case molfrac is Var

        for g in self.groups:
            # hard-sphere diam
            gi = g.index
            self.ghsd[gi, gi] = g.hsd(self.temperature) * 1e-10
            for g2 in self.groups:
                gi2 = g2.index
                if self.ghsd[gi, gi2] == 0. and self.ghsd[gi2, gi2] != 0.:
                    self.ghsd[gi, gi2] = (self.ghsd[gi,gi] + self.ghsd[gi2, gi2]) / 2
                    self.ghsd[gi2, gi] = (self.ghsd[gi,gi] + self.ghsd[gi2, gi2]) / 2
                #gcomb
                self.gcomb[gi, gi2] = g + g2
            # segden
            segratio += self.groups[g] * g.vk * g.sf

        self.segratio = segratio
        self.segden = segratio * self.n_den()

        # chain comb
        nc = GMieComponent.n_total()
        self.chsd = np.zeros((nc,), dtype=object)
        self.ccomb = np.zeros((nc,), dtype=object)
        for c in self.moles:
            ci = c.index
            gc, hsdii = c.get_gtypeii(hsd=self.ghsd)
            self.ccomb[ci] = gc
            self.chsd[ci] = hsdii

        # calculate helmholtz through indiv contributions
        A = self.a_ideal() + self.a_res() * self.n_molecules * cst.k * self.temperature

        return A

    def helmholtz_residual(self):
        # calculate stored values
        nmg = GMieGroup.n_total()
        self.ghsd = np.zeros((nmg, nmg), dtype=object)
        self.gcomb = np.zeros((nmg, nmg), dtype=object)
        segratio = 0.

        for comp in self.moles:
            # molfrac
            self.molfrac[comp] = self.moles[comp] / self.n_molecules

        self.get_groups() # in case molfrac is Var

        for g in self.groups:
            # hard-sphere diam
            gi = g.index
            self.ghsd[gi, gi] = g.hsd(self.temperature) * 1e-10
            for g2 in self.groups:
                gi2 = g2.index
                if self.ghsd[gi, gi2] == 0. and self.ghsd[gi2, gi2] != 0.:
                    self.ghsd[gi, gi2] = (self.ghsd[gi,gi] + self.ghsd[gi2, gi2]) / 2
                    self.ghsd[gi2, gi] = (self.ghsd[gi,gi] + self.ghsd[gi2, gi2]) / 2
                #gcomb
                self.gcomb[gi, gi2] = g + g2
            # segden
            segratio += self.groups[g] * g.vk * g.sf

        self.segratio = segratio
        self.segden = segratio * self.n_den()

        # chain comb
        nc = GMieComponent.n_total()
        self.chsd = np.zeros((nc,), dtype=object)
        self.ccomb = np.zeros((nc,), dtype=object)
        for c in self.moles:
            ci = c.index
            gc, hsdii = c.get_gtypeii(hsd=self.ghsd)
            self.ccomb[ci] = gc
            self.chsd[ci] = hsdii
            
        # calculate helmholtz through indiv contributions
        a = self.a_res()
        A = a * self.n_molecules * cst.k * self.temperature

        return A

    def helmholtz_ideal(self):
        # calculate stored values
        for comp in self.moles:
            # molfrac
            self.molfrac[comp] = self.moles[comp] / self.n_molecules

        self.get_groups() # in case molfrac is Var
            
        # calculate helmholtz through indiv contributions
        A = self.a_ideal()

        return A

    def a_ideal(self):
        result = 0.
        mole_tol = self.n_molecules
        cpideal = True
        for comp in self.moles:
            cpint = comp.cp_int(self.temperature)
            if cpint is None:
                cpideal = False

        if cpideal == False:
            for comp in self.moles:
                debrogv = pow(comp.thdebroglie(self.temperature),3)
                molfrac = self.molfrac[comp]
                nden = molfrac * self.n_den()
                result = result + molfrac * log(nden * debrogv) 
            result = result - 1
            return result * self.n_molecules * cst.k * self.temperature
        
        for comp in self.moles:
            h0 = comp.ref[0]
            s0 = comp.ref[1]
            tref = comp.ref[2]
            pref = comp.ref[3]
            vref = cst.Na * cst.k * tref / pref
            t = self.temperature
            xi = self.molfrac[comp]
            n_mol = self.n_molecules / cst.Na
            V = self.volume * cst.nmtom**3
            cpint = comp.cp_int(t)
            cptint = comp.cp_t_int(t)
            
            si = cptint - cst.Na * cst.k * log(xi * t * vref * n_mol / (tref*V)) + s0

            ai_mol = cpint - t * si - cst.Na * cst.k * t + h0 # per mol
            result = result + xi * n_mol * ai_mol

        return result

    def a_res(self):
        a_r = self.a_mono() + self.a_chain() + self.a_assoc()

        return a_r

    def a_mono(self):
        a_m = self.a_hs() + self.a_1() + self.a_2() + self.a_3()
        return a_m

    def a_chain(self):
        result = 0.
        for c in self.moles:
            molfrac = self.molfrac[c]
            gmieii = self.__gmieij(c.index)
            mseq = 0.
            for g in c.groups:
                mseq += c.groups[g] * g.vk * g.sf
            result += molfrac * (1 - mseq) * log(gmieii)
        return result

    def a_assoc(self):
        return 0 # placeholder self.volume ** 3 * 0.1 + exp(-self.volume)

    # for all
    def __xi_m(self, m):
        fac = pi * self.segden / 6
        gsum = 0.
        for g in self.groups:
            xsk = self.groups[g] * g.vk * g.sf / self.segratio
            gsum += xsk * pow(self.ghsd[g.index, g.index], m)
        return fac * gsum

    def __xi_x(self):
        fac = pi * self.segden / 6
        result = 0.

        for g1 in self.groups:
            g1i = g1.index
            xsk = self.groups[g1] * g1.vk * g1.sf / self.segratio
            for g2 in self.groups:
                g2i = g2.index
                xsl = self.groups[g2] * g2.vk * g2.sf / self.segratio
                dkl = self.ghsd[g1i, g2i]
                result += xsk * xsl * pow(dkl, 3)
        return fac * result

    def __xi_sx(self):
        fac = pi * self.segden / 6
        result = 0.

        for g1 in self.groups:
            g1i = g1.index
            xsk = self.groups[g1] * g1.vk * g1.sf / self.segratio
            for g2 in self.groups:
                g2i = g2.index
                xsl = self.groups[g2] * g2.vk * g2.sf / self.segratio
                skl = self.gcomb[g1i, g2i].sigma * cst.atom
                result += xsk * xsl * pow(skl, 3)
        return fac * result

    def __bkl(self, comb, dkl, lam):
        '''
        Use 2 groups as input
        set rep = True for lam_r, rep = False for lam_a
        2 * pi * segden * dkl^3 epsikl * bkl_sum(lambda_r or a, xi_x)
        '''
        segden = self.segden
        dkl_3 = pow(dkl, 3)
        epsikl = comb.epsilon * cst.k

        x0kl = comb.x0kl(dkl)
        xix = self.__xi_x()

        bsum = mt.bkl_sum(xix, x0kl, lam)
        result = 2*pi*segden*dkl_3*epsikl*bsum

        return result

    def __as1kl(self, comb, dkl, lam):
        '''
        - 2 * pi * segden * (epsikl * dkl^3/ (lam - 3)) * (1 - xi_x_eff/2)/(1-xi_x_eff)^3
        '''
        segden = self.segden
        dkl_3 = pow(dkl, 3)
        epsikl = comb.epsilon * cst.k

        xix = self.__xi_x()

        xixeff = mt.xi_x_eff(xix, lam)

        num = 1 - xixeff/2
        den = pow(1 - xixeff, 3)

        result = -2 * pi * segden * (epsikl * dkl_3 / (lam - 3)) * (num/den)

        return result

    # for mono
    def a_hs(self):
        xi1 = self.__xi_m(1) # nm-2
        xi2 = self.__xi_m(2) # nm-1
        xi3 = self.__xi_m(3) # dimless
        xi0 = self.__xi_m(0) # nm-3

        if len(self.moles) == 1:
            t1 = 0
        else:
            t1 = (pow(xi2, 3) / pow(xi3, 2) - xi0) * log(1-xi3) # t1 is HUGE, and because of numerical discrepancy, even though
                                                                # if there is only 1 component, t1 approx 0,
                                                                # but because each term is ~ 10^27 so discrepancies could be millions even
        t2 = 3 * xi1 * xi2 / (1 - xi3)
        t3 = pow(xi2, 3) / (xi3 * pow((1-xi3),2))
        # print('t2 + t3 gives', t2+t3)
        # print('With pure imp:',(4*xi3 - 3*pow(xi3,2))/pow(1-xi3,2))

        xiterm = t1 + t2 + t3 # nm-3
        result = 6 * xiterm / (pi * self.n_den()) # uses n_den instead of segden
                                                  # because it's just gonna multiply segratio again
                                                  # so cancel it out
        return result

    def a_1(self):
        '''
        1/kT * cgss * sum of sum of _g1/cgss _g2/cgss * a1kl
        '''
        a1sum = 0.
        for g1 in self.groups:
            xsk = self.groups[g1] * g1.vk * g1.sf / self.segratio
            for g2 in self.groups:
                xsl = self.groups[g2] * g2.vk * g2.sf  / self.segratio
                a1kl = self.__a_1kl(g1.index, g2.index)
                a1sum += xsk * xsl * a1kl
        result = self.segratio / (cst.k * self.temperature) * a1sum 
        return result

    def __a_1kl(self, g1i, g2i):
        '''
        Ckl * [ x0kl^att(as1kl(att) + bkl(att)) - x0kl^rep(as1kl(rep) + bkl(rep)) ]
        '''
        comb = self.gcomb[g1i, g2i]
        hsd = self.ghsd[g1i, g2i]

        x0kl = comb.x0kl(hsd)
        rep = comb.rep
        att = comb.att
        ckl = comb.premie()

        as1kl_a = self.__as1kl(comb, hsd, att)
        bkl_a = self.__bkl(comb, hsd, att)

        as1kl_r = self.__as1kl(comb, hsd, rep)
        bkl_r = self.__bkl(comb, hsd, rep)

        t1 = pow(x0kl, att) * (as1kl_a + bkl_a)
        t2 = pow(x0kl, rep) * (as1kl_r + bkl_r)
        result = ckl * (t1 - t2)

        return result

    def a_2(self):
        '''
        (1/kT)**2 * cgss * sum of sum of _g1/cgss _g2/cgss * a2kl
        '''
        a2sum = 0.
        for g1 in self.groups:
            xsk = self.groups[g1] * g1.vk * g1.sf / self.segratio
            for g2 in self.groups:
                xsl = self.groups[g2] * g2.vk * g2.sf / self.segratio
                a2kl = self.__a_2kl(g1.index, g2.index)
                a2sum += xsk * xsl * a2kl 
        result = self.segratio / pow((cst.k * self.temperature),2) * a2sum 
        return result

    def __a_2kl(self, g1i, g2i):
        '''
        1/2 * khs * (1-corrf) * epsikl * ckl^2 *
        { x0kl^(2att) * (as1kl(2*att) + bkl(2*att))
          - 2*x0kl^(att+rep) * (as1kl(att+rep) + bkl(att+rep))
          + x0kl^(2rep) * (as1kl(2rep) + bkl(2rep)) }
        '''
        comb = self.gcomb[g1i, g2i]
        hsd = self.ghsd[g1i, g2i]
        khs = self.__khs()
        corrf = self.__corrf(comb)
        epsikl = comb.epsilon * cst.k
        ckl = comb.premie()
        x0kl = comb.x0kl(hsd)
        rep = comb.rep
        att = comb.att

        t1 = pow(x0kl, 2*att) * (self.__as1kl(comb, hsd, 2*att) + self.__bkl(comb, hsd, 2*att))
        t2 = 2*pow(x0kl, att+rep) * (self.__as1kl(comb, hsd, att+rep) + self.__bkl(comb, hsd, att+rep))
        t3 = pow(x0kl, 2*rep) * (self.__as1kl(comb, hsd, 2*rep) + self.__bkl(comb, hsd, 2*rep))

        fac = 0.5 * khs * (1+corrf) * epsikl * pow(ckl, 2)
        res = t1 - t2 + t3

        result = fac * res
        return result

    def __khs(self):
        xix = self.__xi_x()
        num = pow(1-xix, 4)
        den = 1 + 4*xix + 4*pow(xix,2) - 4*pow(xix,3) + pow(xix,4)
        return num/den

    def __corrf(self, comb):
        xisx = self.__xi_sx()
        alkl = self.__alphakl(comb)

        t1 = mt.f_m(alkl, 1) * xisx
        t2 = mt.f_m(alkl, 2) * pow(xisx, 5)
        t3 = mt.f_m(alkl, 3) * pow(xisx, 8)
        result = t1 + t2 + t3
        return result

    def __alphakl(self, comb):
        t1 = 1 / (comb.att - 3)
        t2 = 1 / (comb.rep - 3)
        result = comb.premie() * (t1 - t2) 

        return result
    
    def a_3(self):
        '''
        (1/kT)**3 * cgss * sum of sum of _g1/cgss _g2/cgss * a3kl
        '''
        a3sum = 0.
        for g1 in self.groups:
            xsk = self.groups[g1] * g1.vk * g1.sf / self.segratio
            for g2 in self.groups:
                xsl = self.groups[g2] * g2.vk * g2.sf / self.segratio
                a3kl = self.__a_3kl(g1.index, g2.index)
                a3sum += xsk * xsl * a3kl 
        result = self.segratio / pow((cst.k * self.temperature),3) * a3sum 
        return result
    
    def __a_3kl(self, g1i, g2i):
        comb = self.gcomb[g1i, g2i]
        xisx = self.__xi_sx()
        alkl = self.__alphakl(comb)
        epsikl = comb.epsilon * cst.k

        preexp = - pow(epsikl,3) * mt.f_m(alkl, 4) * xisx
        expt1 = mt.f_m(alkl, 5) * xisx 
        expt2 = mt.f_m(alkl, 6) * pow(xisx,2)

        result = preexp * exp(expt1 + expt2)
        return result
    
    # for chain
    def __gmieij(self, ci):
        gdhs = self.__gdhs(ci)
        g1 = self.__g1(ci)
        g2 = self.__g2(ci)
        b = 1 / (cst.k * self.temperature)
        gcomp = self.ccomb[ci]
        epsi = gcomp.epsilon * cst.k

        expt = b * epsi * g1 / gdhs + pow(b*epsi,2) * g2 / gdhs
        result = gdhs * exp(expt)
        return result

    def __der_xi_x(self):
        fac = pi / 6
        result = 0.
        for g1 in self.groups:
            g1i = g1.index
            xsk = self.groups[g1] * g1.vk * g1.sf / self.segratio
            for g2 in self.groups:
                g2i = g2.index
                xsl = self.groups[g2] * g2.vk * g2.sf / self.segratio
                dkl = self.ghsd[g1i, g2i]
                result += xsk * xsl * pow(dkl, 3)
        return fac * result

    def __der_as1kl(self, comb, dkl, lam):
        '''
        -2 * pi * (ep*d^3/(lam-3)) * (1/(1-xieff)^3) * (1 - xieff/2 + segden * derxieff * (3*(1-xieff/2)/(1-xieff) - 1/2))
        '''
        segden = self.segden
        d_3 = pow(dkl, 3)
        epsi = comb.epsilon * cst.k

        derxix = self.__der_xi_x()
        xix = self.__xi_x()

        derxixeff = mt.der_xi_x_eff(xix, derxix, lam)
        xixeff = mt.xi_x_eff(xix, lam)

        fac = 2 * pi * epsi * d_3 / (lam-3) / pow(1-xixeff,3)
        t1 = 1 - xixeff / 2
        t2 = segden * derxixeff
        t3 = 3 * (1-xixeff/2) / (1-xixeff)

        result = - fac * (t1 + t2 * (t3 - 1/2))

        return result

    def __der_bkl(self, comb, dkl, lam):
        segden = self.segden
        d_3 = pow(dkl, 3)
        epsi = comb.epsilon * cst.k

        x0ii = comb.x0kl(dkl)
        xix = self.__xi_x()
        derxix = self.__der_xi_x()

        fac = 2 * pi * epsi * d_3 / pow(1-xix,3)

        t11 = 1 - xix/2
        t12 = segden * derxix
        t13 = 3 * (1-xix/2) / (1-xix) - 1/2
        t1 = t11 + t12 * t13

        t21 = 1 + pow(xix,2)
        t22 = segden*derxix
        t23 = 1 + 2 * xix + 3 * xix * (1+xix) / (1-xix) 
        t2 = 9/2 * (t21 + t22 * t23)

        result = fac * (t1 * mt.intI(x0ii,lam) - t2 * mt.intJ(x0ii,lam))
        return result

    def __der_a1kl(self, gcomp, hsdii):
        x0ii = gcomp.x0kl(hsdii)
        rep = gcomp.rep
        att = gcomp.att
        ckl = gcomp.premie()

        tatt = self.__der_as1kl(gcomp, hsdii, att) + self.__der_bkl(gcomp, hsdii, att)
        trep = self.__der_as1kl(gcomp, hsdii, rep) + self.__der_bkl(gcomp, hsdii, rep)

        result = ckl * (pow(x0ii, att) * tatt - pow(x0ii, rep) * trep)
        return result

    def __gdhs(self, ci):
        xi_x = self.__xi_x()
        comb = self.ccomb[ci]
        x0ii = comb.x0kl(self.chsd[ci])

        result = mt.gdhs(x0ii, xi_x)
        return result

    def __g1(self, ci):
        '''
        1 / (2pi epsiii hsdii^3) * [ 3 da1kl/dp - premieii * attii * x0ii^attii * (as1kl(attii) + Bkl(attii))/segden 
                                     +  premieii * repii * x0ii^repii * (as1kl(repii) + Bkl(repii))/segden ]
        '''
        gcomp = self.ccomb[ci]
        hsd = self.chsd[ci]
        premie = gcomp.premie()
        epsi = gcomp.epsilon * cst.k
        rep = gcomp.rep
        att = gcomp.att
        x0ii = gcomp.x0kl(hsd)

        segden = self.segden

        t1 = 3* self.__der_a1kl(gcomp, hsd)
        t2 = premie * att * pow(x0ii, att) * (self.__as1kl(gcomp, hsd, att) + self.__bkl(gcomp, hsd, att)) / segden
        t3 = premie * rep * pow(x0ii, rep) * (self.__as1kl(gcomp, hsd, rep) + self.__bkl(gcomp, hsd, rep)) / segden

        result = 1 / (2 * pi * epsi * pow(hsd,3)) * (t1 - t2 + t3)

        return result

    def __g2(self, ci):
        '''
        (1+gammacii) * g2MCA(hsdii)
        '''
        gcomp = self.ccomb[ci]
        xisx = self.__xi_sx()
        alii = self.__alphakl(gcomp)
        theta = exp(gcomp.epsilon / self.temperature) -1

        gammacii = cst.phi[6,0] * (-tanh(cst.phi[6,1] * (cst.phi[6,2]-alii)) + 1) * xisx * theta * exp(cst.phi[6,3]*xisx + cst.phi[6,4] * pow(xisx, 2))
        g2mca = self.__g2mca(ci)
        result = (1 + gammacii) * g2mca
        return result

    def __der_khs(self):
        xix = self.__xi_x()
        derxix = self.__der_xi_x()

        den = 1 + 4*xix + 4*pow(xix,2) - 4*pow(xix,3) + pow(xix,4)
        t1 = 4 * pow(1-xix,3) / den
        t2 = pow(1-xix,4) * (4 + 8*xix -12*pow(xix,2) + 4*pow(xix,3)) / pow(den,2)
        result = derxix * -(t1 + t2)

        return result

    def __der_a2kl(self, gcomp, hsd):
        khs = self.__khs()
        derkhs = self.__der_khs()
        epsi = gcomp.epsilon * cst.k
        ckl = gcomp.premie()
        x0kl = gcomp.x0kl(hsd)
        rep = gcomp.rep
        att = gcomp.att

        t11 = pow(x0kl, 2*att) * (self.__as1kl(gcomp, hsd, 2*att) + self.__bkl(gcomp, hsd, 2*att))
        t12 = 2*pow(x0kl, att+rep) * (self.__as1kl(gcomp, hsd, att+rep) + self.__bkl(gcomp, hsd, att+rep))
        t13 = pow(x0kl, 2*rep) * (self.__as1kl(gcomp, hsd, 2*rep) + self.__bkl(gcomp, hsd, 2*rep))
        t1 =  derkhs * (t11 - t12 + t13)

        t21 = pow(x0kl, 2*att) * (self.__der_as1kl(gcomp, hsd, 2*att) + self.__der_bkl(gcomp, hsd, 2*att))
        t22 = 2*pow(x0kl, att+rep) * (self.__der_as1kl(gcomp, hsd, att+rep) + self.__der_bkl(gcomp, hsd, att+rep))
        t23 = pow(x0kl, 2*rep) * (self.__der_as1kl(gcomp, hsd, 2*rep) + self.__der_bkl(gcomp, hsd, 2*rep))
        t2 = khs * (t21 - t22 + t23)

        result = 0.5 * epsi * pow(ckl, 2) * (t1 + t2)
        return result

    def __g2mca(self, ci):
        '''
        1 / (2pi epsiii^2 hsdii^3) *
        [
        3 * d/dp (a2ii/(1+chi))
        - epsiii * KHS * premie^2 * rep * x0ii^2rep * (as1kl(2rep) + Bkl(2rep))/segden 
        + epsiii * KHS * premie^2 * (rep+att) * x0ii^(rep+att) * (as1kl(rep+att) + Bkl(rep+att))/segden 
        - epsiii * KHS * premie^2 * att * x0ii^2att * (as1kl(2att) + Bkl(2att))/segden 
        ]
        '''
        gcomp = self.ccomb[ci]
        hsd = self.chsd[ci]
        premie = gcomp.premie()
        epsi = gcomp.epsilon * cst.k
        rep = gcomp.rep
        att = gcomp.att
        x0ii = gcomp.x0kl(hsd)
        khs = self.__khs()

        segden = self.segden

        t1 = 3 * self.__der_a2kl(gcomp, hsd)
        t2 = epsi * khs * pow(premie,2) * rep * pow(x0ii, 2*rep) * (self.__as1kl(gcomp, hsd,2*rep) + self.__bkl(gcomp, hsd,2*rep)) / segden
        t3 = epsi * khs * pow(premie,2) * (rep+att) * pow(x0ii, rep+att) * (self.__as1kl(gcomp, hsd,rep+att) + self.__bkl(gcomp, hsd,rep+att)) / segden
        t4 = epsi * khs * pow(premie,2) * att * pow(x0ii, 2*att) * (self.__as1kl(gcomp, hsd,2*att) + self.__bkl(gcomp, hsd,2*att)) / segden
        
        result = 1 / (2 * pi * pow(epsi,2) * pow(hsd, 3)) * (t1 - t2 + t3 - t4)

        return result

    # for assoc


class MieGroup(object):
    def __init__(self, lambda_r, lambda_a, sigma, epsilon):
        '''
        Init for basic Mie group with Mie parameters
        '''
        self.rep = lambda_r
        self.att = lambda_a
        self.sigma = sigma # units A
        self.epsilon = epsilon # units K

    def __add__(self, other):
        '''
        Basic combination rules for Mie groups in general
        '''
        mt.checkerr(isinstance(other, MieGroup), "MieGroup can only combine with other MieGroup")
        if other is self:
            return MieGroup(self.rep, self.att, self.sigma, self.epsilon)
        sigma = (self.sigma + other.sigma) / 2
        epsi = sqrt(self.sigma**3 * other.sigma**3) / sigma**3 * sqrt(self.epsilon * other.epsilon)
        rep = 3 + sqrt((self.rep - 3) * (other.rep - 3))
        att = 3 + sqrt((self.att - 3) * (other.att - 3))

        result = MieGroup(rep, att, sigma, epsi)

        return result

    def __repr__(self):
        return "<Mie-({:5.3f},{:5.3f}) at ({:4.3f} a, {:4.3f} K)>".format(self.rep, self.att, self.sigma, self.epsilon)

    def premie(self):
        return self.rep/(self.rep-self.att) * pow(self.rep/self.att, self.att/(self.rep-self.att))

    def hsd(self, temperature, x_inf=0.):
        '''
        Calculates hard sphere diameter using the method defined in methods.py
        Temperature takes in as units of K
        '''
        result = mt.hsdiam(temperature / self.epsilon, self.rep, self.att, x_inf) * self.sigma
        return result

    def x0kl(self, hsdval):
        return self.sigma * cst.atom / hsdval # sigma in angstrom, hsd given in SI

class Component(object):
    def __init__(self, molar_weight):
        self.mw = molar_weight
        self.cp = None
        self.ref = None

    def thdebroglie(self, temp):
        '''
        Takes in component mass (au) and temperature (K) and return thermal de broglie wavelength
        '''
        Lambda_sq = pow(cst.h,2) * 1e3 * cst.Na / (2 * pi * self.mw * cst.k * temp)
        return sqrt(Lambda_sq)

    def set_cp_ideal(self, cpparam, ref=(0.,0.,273.15,101325)):
        mt.checkerr(isinstance(cpparam, (np.ndarray, list, tuple)) and len(cpparam)==4,
            "Cp parameters must be iterable with length of 4")
        # Cp parameters in form [A, B, C, D]
        self.cp = cpparam
        self.ref = ref # ref: h0, s0, T0, P0

    def cp_int(self, T):
        # cp integral evaluated at T
        # A(T-Tref) + B/2(T^2-Tref^2) + C/3(T^3-Tref^3) + D/4(T^4-Tref^4)
        if self.cp is None: return None

        A = self.cp[0]
        B = self.cp[1]
        C = self.cp[2]
        D = self.cp[3]
        tref = self.ref[2]

        cpint = A*T + (B/2)*T**2 + (C/3)*T**3 + (D/4)*T**4 - (A*tref + (B/2)*tref**2 + (C/3)*tref**3 + (D/4)*tref**4)
        return cpint

    def cp_t_int(self, T):
        # cp/T integral evaluated at T
        # A (ln(T)-ln(Tref)) + B(T-Tref) + C/2(T^2-Tref^2) + D/3(T^3-Tref^3)
        if self.cp is None: return None

        A = self.cp[0]
        B = self.cp[1]
        C = self.cp[2]
        D = self.cp[3]
        tref = self.ref[2]

        cptint = A*log(T) + B*T + (C/2)*T**2 + (D/3)*T**3 - (A*log(tref) + B*tref + (C/2)*tref**2 + (D/3)*tref**3)
        return cptint

class VRMieComponent(MieGroup, Component):
    '''
    SAFT-VR Mie Component as defined by SAFT-VR Mie Equation-of-state
    given in ref: J. Chem. Phys. 139, 154504 (2013); https://doi.org/10.1063/1.4819786
    '''
    _total = 0
    def __init__(self, mspheres, mw, lambda_r, lambda_a, sigma, epsilon):
        MieGroup.__init__(self, lambda_r, lambda_a, sigma, epsilon)
        Component.__init__(self, mw)
        self.ms = mspheres
        self.index = VRMieComponent._total
        VRMieComponent._total += 1
        VRMieComponent.add_elem()
        
    def __repr__(self):
        return "VR-Mie: {:5.3f} spheres of ".format(self.ms) + super().__repr__()

    # Combining rules with exceptions build-in
    _ktable = np.array([[0.]])
    _gtable = np.array([[0.]])

    @classmethod
    def n_total(cls):
        return cls._total

    @classmethod
    def add_elem(cls):
        t1 = cls._ktable
        t2 = cls._gtable
        if cls._total > 1:
            t12 = np.append(t1, np.zeros((1, t1.shape[1])), axis=0)
            cls._ktable = np.append(t12, np.zeros((t12.shape[0], 1)), axis=1)
            t22 = np.append(t2, np.zeros((1, t2.shape[1])), axis=0)
            cls._gtable = np.append(t22, np.zeros((t22.shape[0], 1)), axis=1)

    def __add__(self, other):
        mg = super().__add__(other)
        kij = VRMieComponent._ktable[self.index, other.index]
        mg.epsilon = (1 - kij) * mg.epsilon
        gij = VRMieComponent._gtable[self.index, other.index]
        mg.rep = 3 + (1 - gij) * (mg.rep - 3)
        return mg

    @classmethod
    def print_table(cls):
        print("{:30s}{:d}".format("Total VR Components: ", cls._total))
        print("{:40s}".format("Combination k values for Epsilon:"))
        print(cls._ktable)
        print("{:40s}".format("Combination gamma values for Lambda_r:"))
        print(cls._gtable)

    @classmethod
    def combining_e_kij(cls, g1, g2, val):
        mt.checkerr(isinstance(g1, VRMieComponent) and isinstance(g2, VRMieComponent), "Set cross interactions between same class only")
        i1 = g1.index
        i2 = g2.index
        cls._ktable[i1,i2] = val
        cls._ktable[i2,i1] = val

    @classmethod
    def combining_e_val(cls, g1, g2, val):
        mt.checkerr(isinstance(g1, VRMieComponent) and isinstance(g2, VRMieComponent), "Set cross interactions between same class only")
        i1 = g1.index
        i2 = g2.index
        s1 = g1.sigma
        s2 = g2.sigma
        e1 = g1.epsilon
        e2 = g2.epsilon

        actl = sqrt(pow(s1,3) * pow(s2,3)) / pow((s1+s2)/2, 3) * sqrt(e1 * e2)
        ratio = val / actl
        kij = 1 - ratio
        cls._ktable[i1,i2] = kij
        cls._ktable[i2,i1] = kij

    @classmethod
    def combining_lr_gij(cls, g1, g2, val):
        mt.checkerr(isinstance(g1, VRMieComponent) and isinstance(g2, VRMieComponent), "Set cross interactions between same class only")
        i1 = g1.index
        i2 = g2.index
        cls._gtable[i1,i2] = val
        cls._gtable[i2,i1] = val

    @classmethod
    def combining_lr_val(cls, g1, g2, val):
        mt.checkerr(isinstance(g1, VRMieComponent) and isinstance(g2, VRMieComponent), "Set cross interactions between same class only")
        i1 = g1.index
        i2 = g2.index
        lr1 = g1.rep
        lr2 = g2.rep

        intm = val - 3
        ratio = intm / sqrt((lr1 - 3) * (lr2 - 3))
        gij = 1 - ratio
        cls._gtable[i1,i2] = gij
        cls._gtable[i2,i1] = gij

    def set_cross_epsilon(self, other, val):
        mt.checkerr(isinstance(other, VRMieComponent), "Set cross interactions between same class only")
        i1 = self.index
        i2 = other.index
        s1 = self.sigma
        s2 = other.sigma
        e1 = self.epsilon
        e2 = other.epsilon

        actl = sqrt(pow(s1,3) * pow(s2,3)) / pow((s1+s2)/2, 3) * sqrt(e1 * e2)
        ratio = val / actl
        kij = 1 - ratio
        VRMieComponent._ktable[i1,i2] = kij
        VRMieComponent._ktable[i2,i1] = kij

        return kij

    def set_cross_e_kij(self, other, val):
        mt.checkerr(isinstance(other, VRMieComponent), "Set cross interactions between same class only")
        i1 = self.index
        i2 = other.index
        kij = val
        VRMieComponent._ktable[i1,i2] = kij
        VRMieComponent._ktable[i2,i1] = kij

        return kij

    def set_cross_rep(self, other, val):
        mt.checkerr(isinstance(other, VRMieComponent), "Set cross interactions between same class only")
        i1 = self.index
        i2 = other.index
        lr1 = self.rep
        lr2 = other.rep

        intm = val - 3
        ratio = intm / sqrt((lr1 - 3) * (lr2 - 3))
        gij = 1 - ratio
        VRMieComponent._gtable[i1,i2] = gij
        VRMieComponent._gtable[i2,i1] = gij

        return gij

    def set_cross_r_gij(self, other, val):
        mt.checkerr(isinstance(other, VRMieComponent), "Set cross interactions between same class only")
        i1 = self.index
        i2 = other.index
        gij = val
        VRMieComponent._gtable[i1,i2] = val
        VRMieComponent._gtable[i2,i1] = val

        return gij

# class AssocSite(object):
#     _rc_table = np.array([[0.]])
#     def __init__(self):

class GMieGroup(MieGroup):
    '''
    SAFT-gamma Mie Group as defined by SAFT-gamma Mie Equation-of-state
    given in ref: J. Chem. Phys. 140, 054107 (2014); https://doi.org/10.1063/1.4851455
    '''
    _total = 0
    def __init__(self, lambda_r, lambda_a, sigma, epsilon, molar_weight=None, shape_factor=1., id_seg=1, name=None):
        super().__init__(lambda_r, lambda_a, sigma, epsilon)
        self.sf = shape_factor
        self.vk = id_seg
        self.mw = molar_weight

        self.index = GMieGroup._total
        self.name = name
        GMieGroup._total += 1
        GMieGroup.add_elem()

        self.cp = None
        self.ref = None

    def __repr__(self):
        if isinstance(self.name, str):
            return "G-Mie {:s} group".format(self.name)
        return "G-Mie: sf={:4.3f}, id_seg={:4.3f} ".format(self.sf, self.vk) + super().__repr__()

    def set_cp_ideal(self, cpparam, ref=(0.,0.,273.15,101325)):
        mt.checkerr(isinstance(cpparam, (np.ndarray, list, tuple)) and len(cpparam)==4,
            "Cp parameters must be iterable with length of 4")
        # Cp parameters in form [A, B, C, D]
        self.cp = cpparam
        self.ref = ref # ref: h0, s0, T0, P0

    # Combining rules with exceptions build-in
    _ktable = np.array([[0.]])
    _gtable = np.array([[0.]])

    @classmethod
    def n_total(cls):
        return cls._total

    @classmethod
    def add_elem(cls):
        t1 = cls._ktable
        t2 = cls._gtable
        if cls._total > 1:
            t12 = np.append(t1, np.zeros((1, t1.shape[1])), axis=0)
            cls._ktable = np.append(t12, np.zeros((t12.shape[0], 1)), axis=1)
            t22 = np.append(t2, np.zeros((1, t2.shape[1])), axis=0)
            cls._gtable = np.append(t22, np.zeros((t22.shape[0], 1)), axis=1)

    def __add__(self, other):
        mg = super().__add__(other)
        if other is self:
            return mg
        kij = GMieGroup._ktable[self.index, other.index]
        mg.epsilon = (1 - kij) * mg.epsilon
        gij = GMieGroup._gtable[self.index, other.index]
        mg.rep = 3 + (1 - gij) * (mg.rep - 3)
        return mg

    @classmethod
    def print_table(cls):
        print("{:30s}{:d}".format("Total Gamma Mie Groups: ", cls._total))
        print("{:40s}".format("Combination k values for Epsilon:"))
        print(cls._ktable)
        print("{:40s}".format("Combination gamma values for Lambda_r:"))
        print(cls._gtable)

    @classmethod
    def combining_e_kij(cls, g1, g2, val):
        mt.checkerr(isinstance(g1, GMieGroup) and isinstance(g2, GMieGroup), "Set cross interactions between same class only")
        i1 = g1.index
        i2 = g2.index
        cls._ktable[i1,i2] = val
        cls._ktable[i2,i1] = val

    @classmethod
    def combining_e_val(cls, g1, g2, val):
        mt.checkerr(isinstance(g1, GMieGroup) and isinstance(g2, GMieGroup), "Set cross interactions between same class only")
        i1 = g1.index
        i2 = g2.index
        s1 = g1.sigma
        s2 = g2.sigma
        e1 = g1.epsilon
        e2 = g2.epsilon

        actl = sqrt(pow(s1,3) * pow(s2,3)) / pow((s1+s2)/2, 3) * sqrt(e1 * e2)
        ratio = val / actl
        kij = 1 - ratio
        cls._ktable[i1,i2] = kij
        cls._ktable[i2,i1] = kij

    @classmethod
    def combining_lr_gij(cls, g1, g2, val):
        mt.checkerr(isinstance(g1, GMieGroup) and isinstance(g2, GMieGroup), "Set cross interactions between same class only")
        i1 = g1.index
        i2 = g2.index
        cls._gtable[i1,i2] = val
        cls._gtable[i2,i1] = val

    @classmethod
    def combining_lr_val(cls, g1, g2, val):
        mt.checkerr(isinstance(g1, GMieGroup) and isinstance(g2, GMieGroup), "Set cross interactions between same class only")
        i1 = g1.index
        i2 = g2.index
        lr1 = g1.rep
        lr2 = g2.rep

        intm = val - 3
        ratio = intm / sqrt((lr1 - 3) * (lr2 - 3))
        gij = 1 - ratio
        cls._gtable[i1,i2] = gij
        cls._gtable[i2,i1] = gij

    def set_cross_epsilon(self, other, val):
        mt.checkerr(isinstance(other, GMieGroup), "Set cross interactions between same class only")
        i1 = self.index
        i2 = other.index
        s1 = self.sigma
        s2 = other.sigma
        e1 = self.epsilon
        e2 = other.epsilon

        actl = sqrt(pow(s1,3) * pow(s2,3)) / pow((s1+s2)/2, 3) * sqrt(e1 * e2)
        ratio = val / actl
        kij = 1 - ratio
        GMieGroup._ktable[i1,i2] = kij
        GMieGroup._ktable[i2,i1] = kij

        return kij

    def set_cross_e_kij(self, other, val):
        mt.checkerr(isinstance(other, GMieGroup), "Set cross interactions between same class only")
        i1 = self.index
        i2 = other.index
        kij = val
        GMieGroup._ktable[i1,i2] = kij
        GMieGroup._ktable[i2,i1] = kij

        return kij

    def set_cross_rep(self, other, val):
        mt.checkerr(isinstance(other, GMieGroup), "Set cross interactions between same class only")
        i1 = self.index
        i2 = other.index
        lr1 = self.rep
        lr2 = other.rep

        intm = val - 3
        ratio = intm / sqrt((lr1 - 3) * (lr2 - 3))
        gij = 1 - ratio
        GMieGroup._gtable[i1,i2] = gij
        GMieGroup._gtable[i2,i1] = gij

        return gij

    def set_cross_r_gij(self, other, val):
        mt.checkerr(isinstance(other, GMieGroup), "Set cross interactions between same class only")
        i1 = self.index
        i2 = other.index
        gij = val
        GMieGroup._gtable[i1,i2] = val
        GMieGroup._gtable[i2,i1] = val

        return gij

class GMieComponent(Component):
    '''
    SAFT-gamma Mie Component as defined by SAFT-gamma Mie Equation-of-state
    given in ref: J. Chem. Phys. 140, 054107 (2014); https://doi.org/10.1063/1.4851455
    '''
    _total = 0
    def __init__(self, molar_weight=None):
        if molar_weight == None:
            super().__init__(0.)
        else:
            if isinstance(molar_weight,(int,float)):
                super().__init__(molar_weight)
        self.groups = {}
        self.ngtype = 0
        self.ngroups = 0
        self.index = GMieComponent._total
        GMieComponent._total += 1

    def __repr__(self):
        s = 'G-Mie Component with {:d} groups\n'.format(self.ngroups)
        for g in self.groups:
            s += '{:d} x {:s}\n'.format(self.groups[g], repr(g))
        return s

    @classmethod
    def n_total(cls):
        return cls._total

    def quick_set(self, *args):
        for arg in args:
            mt.checkerr(isinstance(arg, tuple) or isinstance(arg, list), "Use iterable tuple or list for quick_set")
            mt.checkerr(isinstance(arg[0], GMieGroup), "First element of iterable must be an instance of GMieGroup class")
            mt.checkerr(isinstance(arg[1], int), "Second element of iterable must be of int type")
            mt.checkerr(arg[1] > 0, "Second element of iterable must be of integer > 0")
            self.groups[arg[0]] = arg[1]

        self.ngroups = sum(self.groups.values())
        self.ngtype = len(self.groups)

        # Update molar weight
        for g in self.groups:
            if g.mw is not None:
                self.mw += g.mw * self.groups[g]

        return self

    def add_group(self, group, number):
        mt.checkerr(isinstance(group, GMieGroup), "Group entry must be an instance of GMieGroup class")
        mt.checkerr(isinstance(number, int), "Only integer amount of molecules can be added")

        if group in self.groups:
            self.groups[group] += number
            self.ngroups += number
        else:
            self.groups[group] = number
            self.ngroups += number
            self.ngtype += 1

        if group.mw is not None:
            self.mw += g.mw * number

        return self

    def get_gtypeii(self, hsd=None):
        sig3 = 0.
        epsi = 0.
        rep = 0.
        att = 0.
        hsdii = 0.

        den = 0.
        for g in self.groups:
            den += self.groups[g] * g.vk * g.sf

        for g1 in self.groups:
            num1 = self.groups[g1] * g1.vk * g1.sf # v_ki * v*_k * Sk
            zki = num1 / den
            for g2 in self.groups: 
                num2 = self.groups[g2] * g2.vk * g2.sf # v_ki * v*_k * Sk
                zli = num2 / den
                gc = g1+g2
                sig3 += zki * zli * gc.sigma**3
                epsi += zki * zli * gc.epsilon
                rep += zki * zli * gc.rep
                att += zki * zli * gc.att
                if isinstance(hsd, np.ndarray):
                    g1i = g1.index
                    g2i = g2.index
                    hsdii += zki * zli * hsd[g1i, g2i]

        mg = MieGroup(rep, att, sig3**(1/3), epsi)

        if not isinstance(hsd, np.ndarray):
            return mg
        return (mg, hsdii)

    def cp_int(self, T):
        if self.cp is None:
            self.cp = [0] * 4
            for g in self.groups:
                if g.cp is None:
                    self.cp = None
                    return None
                for i in range(4):
                    self.cp[i] += self.groups[g] * g.cp[i]
            self.ref = (0.,0.,273.15,101325) # default reference state

        return super().cp_int(T)

    def cp_t_int(self, T):
        if self.cp is None:
            self.cp = [0] * 4
            for g in self.groups:
                if g.cp is None:
                    self.cp = None
                    return None
                for i in range(4):
                    self.cp[i] += self.groups[g] * g.cp[i]
            self.ref = (0.,0.,273.15,101325) # default reference state

        return super().cp_t_int(T)

def main():
    CH4 = MieGroup(12.504, 6., 3.737, 152.575)
    CH3 = MieGroup(15.04982, 6., 4.077257, 256.7662)
    CH2 = MieGroup(19.87107, 6., 4.880081, 473.3893)
    COO = MieGroup(31.189, 6., 3.9939, 868.92)
    CO2 = MieGroup(26.408, 5.055, 3.05, 207.891)
    CH = MieGroup(8.0, 6.0, 5.295, 95.621)
    C2H6 = MieGroup(10.16, 6., 3.488, 165.513)
    LJ = MieGroup(12,6,1,1)


    vrc = {}

    # J. Chem. Phys. 139, 154504 (2013); https://doi.org/10.1063/1.4819786
    vrc['methane'] =                VRMieComponent(1.0000,  16.04, 12.650,      6, 3.7412, 153.36)
    vrc['ethane'] =                 VRMieComponent(1.4373,  30.07, 12.400,      6, 3.7257, 206.12)
    vrc['propane'] =                VRMieComponent(1.6845,  44.10, 13.006,      6, 3.9056, 239.89)
    vrc['n-butane'] =               VRMieComponent(1.8514,  58.12, 13.650,      6, 4.0887, 273.64)
    vrc['n-pentane'] =              VRMieComponent(1.9606,  72.15, 15.847,      6, 4.2928, 321.94)
    vrc['n-hexane'] =               VRMieComponent(2.1097,  86.18, 17.203,      6, 4.4230, 354.38)
    vrc['n-heptane'] =              VRMieComponent(2.3949, 100.21, 17.092,      6, 4.4282, 358.51)
    vrc['n-octane'] =               VRMieComponent(2.6253, 114.23, 17.378,      6, 4.4696, 369.18)
    vrc['n-nonane'] =               VRMieComponent(2.8099, 128.20, 18.324,      6, 4.5334, 387.55)
    vrc['n-decane'] =               VRMieComponent(2.9976, 142.29, 18.885,      6, 4.5890, 400.79)
    vrc['n-dodecane'] =             VRMieComponent(3.2519, 170.33, 20.862,      6, 4.7484, 437.72)
    vrc['n-pentadecane'] =          VRMieComponent(3.9325, 212.41, 20.822,      6, 4.7738, 444.51)
    vrc['n-eicosane'] =             VRMieComponent(4.8794, 282.55, 22.926,      6, 4.8788, 475.76)
    vrc['perfluoromethane'] =       VRMieComponent(1.0000,  88.00, 42.553, 5.1906, 4.3372, 232.62)
    vrc['perfluoroethane'] =        VRMieComponent(1.8529, 138.01, 19.192, 5.7506, 3.9336, 211.46)
    vrc['perfluoropropane'] =       VRMieComponent(1.9401, 188.02, 22.627, 5.7506, 4.2983, 263.26)
    vrc['n-perfluorobutane'] =      VRMieComponent(2.1983, 238.03, 24.761, 5.7506, 4.4495, 290.49)
    vrc['n-perfluoropentane'] =     VRMieComponent(2.3783, 288.03, 29.750, 5.7506, 4.6132, 328.56)
    vrc['n-perfluorohexane'] =      VRMieComponent(2.5202, 338.04, 30.741, 5.7506, 4.7885, 349.30)
    vrc['fluorine'] =               VRMieComponent(1.3211,  38.00, 11.606,      6, 2.9554, 96.268)
    vrc['carbon dioxide'] =         VRMieComponent(1.5000,  44.01, 27.557, 5.1646, 3.1916, 231.88)
    vrc['benzene'] =                VRMieComponent(1.9163,  78.11, 14.798,      6, 4.0549, 372.59)
    vrc['toluene'] =                VRMieComponent(1.9977,  92.14, 16.334,      6, 4.2777, 409.73)
    vrc['ethane'].set_cross_e_kij(vrc['n-decane'], -0.0222)
    vrc['carbon dioxide'].set_cross_e_kij(vrc['n-decane'], 0.0500)

    # J. Phys. Chem. B 2013, 117, 2717−2733
    vrc['CO2'] =                VRMieComponent(1,  44.01, 23.000,      6, 3.7410, 353.55)
    vrc['CF4'] =                VRMieComponent(1,  88.00, 32.530,      6, 4.3500, 265.57)
    vrc['SF6'] =                VRMieComponent(1, 146.06, 19.020,   8.80, 4.8610, 440.49)
    vrc['HFO-1234yf'] =         VRMieComponent(2, 114.00, 16.540,      6, 3.9000, 257.39)
    # VRMieComponent.print_table()
    print(vrc['ethane'] + vrc['n-decane'])
    s = SAFTVRSystem().quick_set((vrc['methane'], 800), (vrc['n-decane'], 100))
    s.add_comp(vrc['n-decane'], 100)

    # s = SAFTVRSystem().quick_set((vrc['methane'], 800))
    s2 = SAFTVRSystem().quick_set((vrc['propane'], 1000),(vrc['ethane'], 1000),(vrc['methane'], 1000))#,(vrc['n-butane'], 1000),(vrc['n-pentane'], 1000),(vrc['n-hexane'], 1000),(vrc['n-decane'], 1000))
    s2.volume = 3000
    s.volume = Var(3000)
    # s.helmholtz()
    # print('Testing VR Mie System ======')
    sys.setrecursionlimit(3000)
    print(sys.getrecursionlimit())
    # print('{:18s}'.format('xi_1'),s._SAFTVRSystem__xi_m(1))
    # print('{:18s}'.format('xi_2'),s._SAFTVRSystem__xi_m(2))
    # print('{:18s}'.format('xi_3'),s._SAFTVRSystem__xi_m(3))
    # print('{:18s}'.format('a_hs'),s.a_hs())
    # print('{:18s}'.format('xi_x'),s._SAFTVRSystem__xi_x())
    # print('{:18s}'.format('a_1'),s.a_1())
    # print('{:18s}'.format('xi_x'),s._SAFTVRSystem__xi_sx())
    # print('{:18s}'.format('a_2'),s.a_2())
    # print('{:18s}'.format('a_3'),s.a_3())
    # print('{:18s}'.format('a_mono'),s.a_mono())
    # print('{:18s}'.format('gdhs'),s._SAFTVRSystem__gdhs(vrc['methane'].index, vrc['methane'].index))
    # print('{:18s}'.format('der_xix'),s._SAFTVRSystem__der_xi_x())
    # print('{:18s}'.format('g1'),s._SAFTVRSystem__g1(vrc['methane'].index, vrc['methane'].index))
    # print('{:18s}'.format('g2'),s._SAFTVRSystem__g2(vrc['methane'].index, vrc['methane'].index))
    # print('{:18s}'.format('gmieij'),s._SAFTVRSystem__gmieij(vrc['methane'].index,vrc['methane'].index))
    # print('{:18s}'.format('a_chain'),s.a_chain())
    # print('{:18s}'.format('helmholtz'),s.helmholtz())
    # print('{:18s}'.format('d a_ideal dV'),derivative(s.helmholtz_ideal(), s.volume))
    # print('{:18s}'.format('d a_res dV'),derivative(s.helmholtz_residual(), s.volume))

    # print('Testing System get_properties ======')
    # print('{:18s}'.format('Pressure (Pa):'), s.get_pressure(temperature=319, molar_volume=0.00014))
    # print('{:18s}'.format('Cv (J/(mol K)):'), s.get_cv(temperature=319, molar_volume=0.00014))

    CH3 =  GMieGroup(15.04982,    6., 4.077257, 256.7662, molar_weight=15.03502, shape_factor=0.5725512, id_seg=1, name="CH3")
    CH2 =  GMieGroup(19.87107,    6., 4.880081, 473.3893, molar_weight=14.02708, shape_factor=0.2293202, id_seg=1, name="CH2")
    COO =  GMieGroup(  31.189,    6.,   3.9939,   868.92, molar_weight=44.01,    shape_factor=0.65264,   id_seg=1, name="COO")
    CO2 =  GMieGroup(  26.408, 5.055,     3.05,  207.891, molar_weight=44.01,    shape_factor=0.847,     id_seg=2, name="CO2")
    CH =   GMieGroup(      8.,    6.,    5.295,   95.621, molar_weight=13.01864, shape_factor=0.0721,    id_seg=1, name="CH")
    C2H6 = GMieGroup(   10.16,    6.,    3.488,  165.513, molar_weight=30.07004, shape_factor=0.855,     id_seg=2, name="C2H6")
    CH4 =  GMieGroup(  12.504,    6.,    3.737,  152.575, molar_weight=16.04296, shape_factor=1,         id_seg=1, name="CH4")

    CH3.set_cp_ideal((19.5,-8.08e-3,1.53e-4,-9.67e-8))
    CH2.set_cp_ideal((-0.909,0.095,-5.44e-5,1.19e-8))
    C2H6.set_cp_ideal((12.4755767,1.44286617e-1,-2.3523953e-5,-8.99334179e-9))

    GMieGroup.combining_e_val(CH3, CH2, 350.77)
    GMieGroup.combining_e_val(CH3, COO, 402.75)
    GMieGroup.combining_e_val(COO, CH2, 498.86)
    GMieGroup.combining_e_val(CH3, CO2, 205.698)
    GMieGroup.combining_e_val(CH3, CH, 387.48)
    GMieGroup.combining_e_val(CH2, CO2, 276.453)
    GMieGroup.combining_e_val(CH2, CH, 506.21)
    GMieGroup.combining_e_val(C2H6, CO2, 175.751)

    GMieGroup.combining_e_val(CH4, CH3, 193.97079)
    GMieGroup.combining_lr_val(CH4, CH3, 12.62762)
    GMieGroup.combining_e_val(CH4, CH2, 243.12915)
    GMieGroup.combining_lr_val(CH4, CH2, 12.64155)
    GMieGroup.combining_e_val(CH4, CH, 297.2062)
    GMieGroup.combining_lr_val(CH4, CH, 10.996)
    GMieGroup.combining_e_val(CH4, CO2, 144.722)
    GMieGroup.combining_lr_val(CH4, CO2, 11.95)

    c = GMieComponent().quick_set((CH3,2), (CH2,2))
    print(c.mw, 3*15.03502 + 2*14.02708)
    print(c.get_gtypeii())
    c2 = GMieComponent().quick_set((CH3,2), (CH2,1))
    c3 = GMieComponent().quick_set((CH3,2), (CH2,3))
    c4 = GMieComponent().quick_set((CH3,2), (CH2,4))
    c1 = GMieComponent().quick_set((C2H6,1))

    s = SAFTgMieSystem().quick_set((c1, 1000), (c4, 500))
    s.volume = 3000
    s.helmholtz()

    print('Testing Gamma Mie System ======')
    print(s.groups)
    print('='*30)
    print('{:18s}'.format('seg_ratio'),s.segratio)
    print('{:18s}'.format('segden'),s.segden)
    print('{:18s}'.format('a_ideal'),s.a_ideal())
    print('{:18s}'.format('xi_0'),s._SAFTgMieSystem__xi_m(0))
    print('{:18s}'.format('xi_1'),s._SAFTgMieSystem__xi_m(1))
    print('{:18s}'.format('xi_2'),s._SAFTgMieSystem__xi_m(2))
    print('{:18s}'.format('xi_3'),s._SAFTgMieSystem__xi_m(3))
    print('{:18s}'.format('a_hs'),s.a_hs())
    print('{:18s}'.format('xi_x'),s._SAFTgMieSystem__xi_x())
    print('{:18s}'.format('a_1kl: '), s._SAFTgMieSystem__a_1kl(C2H6.index, C2H6.index))
    print('{:18s}'.format('a_1'),s.a_1())
    print('{:18s}'.format('xi_sx'),s._SAFTgMieSystem__xi_sx())
    print('{:18s}'.format('a_2'),s.a_2())
    print('{:18s}'.format('a_3'),s.a_3())
    print('{:18s}'.format('a_mono'),s.a_mono())
    print('{:18s}'.format('gdhs'),s._SAFTgMieSystem__gdhs(c1.index))
    print('{:18s}'.format('gdhs'),s._SAFTgMieSystem__gdhs(c4.index))
    print('{:18s}'.format('der_xix'),s._SAFTgMieSystem__der_xi_x())
    # print(s.ccomb)
    print('{:18s}'.format('g1'),s._SAFTgMieSystem__g1(c4.index))
    print('{:18s}'.format('g2'),s._SAFTgMieSystem__g2(c4.index))
    print('{:18s}'.format('gmieij'),s._SAFTgMieSystem__gmieij(c1.index))
    print('{:18s}'.format('gmieij'),s._SAFTgMieSystem__gmieij(c4.index))
    print('{:18s}'.format('a_chain'),s.a_chain())
    print('{:18s}'.format('helmholtz'),s.helmholtz())
    # print('{:18s}'.format('d a_ideal dV'),derivative(s.helmholtz_ideal(), s.volume))
    # print('{:18s}'.format('d a_res dV'),derivative(s.helmholtz_residual(), s.volume))
    print('='*20)
    print('Testing get properties')
    print('='*20)
    s = SAFTVRSystem().quick_set((vrc['SF6'], 1000))
    # co2 = GMieComponent().quick_set((CO2, 1))
    # s = SAFTgMieSystem().quick_set((co2, 1000))
    (pc, tc, rhoc) = s.critical_point(initial_t=325.9, v_nd=np.logspace(-4,-1,70), get_volume=False, get_density=True, print_results=False, print_progress=True)
    print('{:25s}{:8.3f}'.format('critical P (bar)', pc*cst.patobar))
    print('{:25s}{:8.3f}'.format('critical T (K)', tc))
    print('{:25s}{:8.3f}'.format('critical rho (mol/m3)', rhoc*1e-3*vrc['SF6'].mw))
    vget = s.single_phase_v(15e6, 550, print_results=False, v_crit=1/rhoc, supercritical=True)
    print('{:25s}{:8.3f}'.format('sp_v (1e-3 m3/mol)', vget * 1e3))
    print('{:25s}{:8.3f}'.format('rho (kg/m3)',1/vget*1e-3*vrc['SF6'].mw))
    s.set_molar_volume(vget)
    s.temperature = 550
    print('{:25s}{:8.3f}'.format('get_pressure (MPa)',s.get_pressure()*1e-6))
    print('{:25s}{:8.3f}'.format('get_entropy (J/mol K)',s.get_entropy()))
    print('{:25s}{:8.3f}'.format('get_u (J/mol)',s.get_u()))
    print('{:25s}{:8.3f}'.format('get_enthalpy (J/mol)',s.get_enthalpy()))
    print('{:25s}{:8.3f}'.format('get_gibbs (J/mol)',s.get_gibbs()))
    print('{:25s}{:8.3f}'.format('get_cv (J/mol K)',s.get_cv()))
    print('{:25s}{:8.3f}'.format('get_kappa (1/MPa)',s.get_kappa() * 1e6))
    print('{:25s}{:8.3f}'.format('get_cp (J/mol K)',s.get_cp()))
    print('{:25s}{:8.3f}'.format('get_cp (kJ/kg K)',s.get_cp()/vrc['SF6'].mw))
    print('{:25s}{:8.3f}'.format('get_w (m/s)',s.get_w()))  
    print('{:25s}{:8.3f}'.format('get_jt (K/MPa)',s.get_jt() *1e6))
    print('{:25s}'.format('dv2'),s.dV2(), s.n_molecules * cst.k * s.temperature / s.volume**2 + s.dV2(a='res'))
    print('{:25s}'.format('dt2'),s.dT2())
    print('{:25s}'.format('dtdv'),s.dTdV())
    xa = s.dTdV(a='res')
    print('{:25s}'.format('dtdv'), -s.n_molecules * cst.k / s.volume + xa[1])

    print('='*20)
    print('Testing get properties with SAFTgMie')
    print('='*20)

    nbutane = GMieComponent().quick_set((CH3,2), (CH2,1))
    npentane = GMieComponent().quick_set((CH3,2), (CH2,3))
    noctane = GMieComponent().quick_set((CH3,2), (CH2,6))
    ndecane = GMieComponent().quick_set((CH3,2), (CH2,8))

    C4s = SAFTgMieSystem().quick_set((nbutane, 100))
    C5s = SAFTgMieSystem().quick_set((npentane, 100))
    C8s = SAFTgMieSystem().quick_set((noctane, 100))
    C10s = SAFTgMieSystem().quick_set((ndecane, 100))

    print(nbutane.cp_int(300), nbutane.cp_t_int(300))
    print(npentane.cp_int(300), npentane.cp_t_int(300))

    # C5 speed of sound
    # (pc, tc, rhoc) = C5s.critical_point(initial_t=480., v_nd=np.logspace(-4,-1,50), get_volume=False, get_density=True, print_results=False, print_progress=False)
    # print(pc, tc, 1/rhoc)
    # trange = np.linspace(300,700,10)
    # vget = np.zeros(10)
    # w = np.zeros(10)
    # for i in range(len(trange)):
    #     t = trange[i]
    #     vget[i] = C5s.single_phase_v(5e6, t, print_results=False, v_crit=1/rhoc, supercritical=True)
    #     w[i] = C5s.get_w(temperature=t, molar_volume=vget[i])

    # fig, ax = plt.subplots()
    # ax.plot(trange, w, 'bo')
    # plt.show()

    # C4 isothermal compressibility
    # (pc, tc, rhoc) = C4s.critical_point(initial_t=380., v_nd=np.logspace(-4,-1,50), get_volume=False, get_density=True, print_results=False, print_progress=False)
    # print(pc, tc, 1/rhoc)
    # trange = np.linspace(300,700,15)
    # vget = np.zeros(15)
    # k = np.zeros(15)
    # for i in range(len(trange)):
    #     t = trange[i]
    #     vget[i] = C4s.single_phase_v(10e6, t, print_results=False, v_crit=1/rhoc, v_init=0.3/rhoc, supercritical=True)
    #     print(vget[i])
    #     k[i] = C4s.get_kappa(temperature=t, molar_volume=vget[i])
    # fig, ax = plt.subplots()
    # ax.plot(trange, k*1e9, 'bo')
    # print(k * 1e9)
    # plt.show()

    # C10 Cp
    (pc, tc, rhoc) = C10s.critical_point(initial_t=380., v_nd=np.logspace(-3.5,-0.5,50), get_volume=False, get_density=True, print_results=False, print_progress=False)
    print(pc, tc, 1/rhoc)
    trange = np.linspace(500,700,15)
    vget = np.zeros(15)
    cp = np.zeros(15)
    for i in range(len(trange)):
        t = trange[i]
        vget[i] = C10s.single_phase_v(3e6, t, print_results=False, v_crit=1/rhoc, v_init=0.3/rhoc, supercritical=True)
        cp[i] = C10s.get_cp(temperature=t, molar_volume=vget[i])
    fig, ax = plt.subplots()
    ax.plot(trange, cp, 'bo')
    print(cp)
    plt.show()

    '''
    comps = {}
    puresys = {}

    for i in range(2,11):
        key = 'C' + str(i)
        comps[key] = GMieComponent(i * 12.0107 + (i*2+2) * 1.00784)

        if i == 2:
            # comps[key].quick_set((CH3, 2))
            comps[key].quick_set((C2H6, 1))
        else:
            comps[key].quick_set((CH3, 2), (CH2, i-2))
        puresys[key] = SAFTgMieSystem().quick_set((comps[key], 1000))

    print('='*20)
    print('Testing n-alkanes')
    print('='*20)

    pc_data = np.array([])
    tc_data = np.array([])
    rhoc_data = np.array([])
    for i in range(2,5):
        key = 'C' + str(i)
        t_init = (i+1)**0.7 * 256.77 * 0.40772
        v_init =  (i * 12.0107 + (i*2+2) * 1.00784) * 0.001 / 200.
        v_range = np.logspace(math.log10(v_init/2), math.log10(v_init*2), 30)
        (pc, tc, rhoc) = puresys[key].critical_point(initial_t=t_init, v_nd=v_range, get_volume=False, get_density=True, print_results=False, print_progress=False)
        pc_data = np.append(pc_data, pc*cst.patobar)
        tc_data = np.append(tc_data, tc)
        rhoc_data = np.append(rhoc_data, rhoc)
        print('Critical point: pressure = {:6.3f} bar, temperature = {:6.3f} K, density = {:6.3f} mol/m3'.format(pc*cst.patobar, tc, rhoc))

    df = pd.DataFrame(np.column_stack([pc_data, tc_data, rhoc_data]))
    outputfile = 'saftgmie-alkane-crit-int.csv'
    # df.to_csv(outputfile, index=False, header=['p_c (bar)', 't_c (K)', 'rho_c (mol/m3)'])
    # print(f'Data generation complete. Output file: {outputfile}', ' '*5)
    print(df)

    carbon = np.array([])
    t_data = np.array([])
    p_data = np.array([])
    rhol = np.array([])
    rhog = np.array([])
    vl_data = np.array([])
    vv_data = np.array([])
    for i in range(2,5):
        key = 'C' + str(i)
        print('VLE data for {:3s} alkane'.format(key), '='*10)
        temp_range = np.linspace(tc_data[i-2]* 0.45, tc_data[i-2] * 0.995, 30) # for 30 points
        for j in range(len(temp_range)):
            t = temp_range[j]
            try: 
                if j == 0:
                    ig = (0.25/rhoc_data[i-2],100/rhoc_data[i-2])
                    # ig = (0.4 * (i * 12.0107 + (i*2+2) * 1.00784) * 0.001 / 200., 50 * (i * 12.0107 + (i*2+2) * 1.00784) * 0.001 / 200.)
                else:
                    ig = vle
                pv, vle = puresys[key].vapour_pressure(t, initial_guess=ig, get_volume=True, print_results=False)
                if abs(vle[0] - vle[1]) < 1e-6:
                    print(f'VLE points solver failed to converge at meaningful results at T={t:5.2f}, points too similar ({vle[0]:7.3e}, {vle[1]:7.3e})')
                    vle = ig
                else:
                    carbon = np.append(carbon, i)
                    t_data = np.append(t_data, t)
                    p_data = np.append(p_data, pv * cst.patobar)
                    vl_data = np.append(vl_data, min(vle))
                    vv_data = np.append(vv_data, max(vle))
                    rhol = np.append(rhol, 1/min(vle))
                    rhog = np.append(rhog, 1/max(vle))
                    print(f'Getting VLE at P = {pv*cst.patobar:5.2f} bar, T = {t:5.2f} K, v_l = {min(vle):7.3e}, v_v = {max(vle):7.3e}')
            except:
                print('VLE solver failed at T={t:5.2f} due to out of range operations. Current point aborted.')
        carbon = np.append(carbon, i)
        t_data = np.append(t_data, tc_data[i-2])
        p_data = np.append(p_data, pc_data[i-2])
        vl_data = np.append(vl_data, 1/rhoc_data[i-2])
        vv_data = np.append(vv_data, 1/rhoc_data[i-2])
        rhol = np.append(rhol, rhoc_data[i-2])
        rhog = np.append(rhog, rhoc_data[i-2])

    df = pd.DataFrame(np.column_stack([carbon, t_data, p_data, vl_data, vv_data, rhol, rhog]))
    outputfile = 'saftgm-gc-alkanes-vle-int.csv'
    print(df)
    '''

    # vnd = np.logspace(-4,-1, 70)
    # for lr in [8, 10, 12, 15, 20, 36]:
    #     test = VRMieComponent(1, 10, lr, 6, 3, 200)
    #     t = SAFTVRSystem().quick_set((test,1000))
    #     t_init = (lr**-0.5)/0.2 * 200
    #     v_init = 2* 3e-10**3 * cst.Na
    #     v_range = np.logspace(math.log10(v_init/2), math.log10(v_init)+1, 30)
    #     Pc, Tc, vc = t.critical_point(initial_t=t_init, v_nd=v_range, print_progress=False, get_volume=True, print_results=False)
    #     print('{:18s}'.format('T_crit*:'), Tc/200)
    #     print('{:18s}'.format('rho_crit* (mol/m3):'), 1/vc * cst.Na * 1e-30 * 3**3)


if __name__ == '__main__':
    main()