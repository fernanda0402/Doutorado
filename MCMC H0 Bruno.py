#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 13:56:20 2023

@author: usuario
"""

# Bibliotecas:
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize
from multiprocessing import Pool
import emcee
import time
import corner

# Dados:
t_data = [0.3372681281618887, 0.36363636363636365, 0.39525691699604737,
          0.41152263374485604, 0.4231908590774439, 0.4347826086956522,
          0.4909180166912126, 0.5263157894736842, 0.5319148936170213,
          0.5333333333333333, 0.5614823133071308, 0.5952380952380952,
          0.6277463904582549, 0.6756756756756757, 0.6764526821348847,
          0.6802721088435374, 0.689797889218459, 0.7019021548396153,
          0.7140816909454443, 0.7142857142857143, 0.7245326764237068,
          0.7396449704142013, 0.78125, 0.7874015748031495, 0.8333333333333334,
          0.8340283569641367, 0.8481764206955046, 0.8547008547008548,
          0.8928571428571428, 0.9174311926605504, 0.9345794392523364]

H_data = [186.5, 202.0, 140.0, 177.0, 160.0, 168.0, 154.0, 117.0, 90.0, 125.0,
          105.0, 92.0, 104.0, 97.0, 80.9, 89.0, 92.8, 87.1, 77.0, 95.0, 83.0,
          83.0, 88.8, 77.0, 72.9, 75.0, 75.0, 83.0, 68.6, 69.0, 69.0]

sig_data = [50.4, 40.0, 14.0, 18.0, 33.6, 17.0, 20.0, 23.0, 40.0, 17.0,
              12.0, 8.0, 13.0, 62.0, 9.0, 49.6, 12.9, 11.2, 10.2, 17.0, 13.5,
              14.0, 36.6, 14.0, 29.6, 5.0, 4.0, 8.0, 26.2, 12.0, 19.6]

df = pd.DataFrame({"t_data":t_data, "H_data":H_data, "sig_data":sig_data})

a = df["t_data"]
H = df["H_data"]
sigma = df["sig_data"]

Om_0 = 0.3

# Modelo LCDM:
def H(H0):
    Om = Om_0 * a**(-3)
    OL = 1 - Om_0
    H = H0 * np.sqrt(Om + OL)
    return H

def chi(H0):
    H_model = H(H0)
    chisq_vec = np.power((H_model - H_data)/sig_data, 2)
    return chisq_vec.sum()

def chisq_H(pars):
    H0 = pars
    return chi(H0)

print(chi(70))

# Priors:
H0_ini = 70

result = minimize(chisq_H, [H0_ini], bounds=((60, 80),))
H0min = result.x
print(result.x)

# Priors:
def lnprior(pars):
    H0 = pars
    if  60 < H0 < 80:
        return 0.0
    return -np.inf

def lnlike_H(pars):
    H0 = pars
    return -0.5*chisq_H(H0)

def lnprob_H(pars):
    lp = lnprior(pars)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike_H(pars)

with Pool() as pool:
    ndim, nwalkers, nsteps = 1, 15, 3000
    pos = [H0min + 1e-5*np.random.randn(ndim) for i in range(nwalkers)]
    sampler= emcee.EnsembleSampler(nwalkers, ndim, lnprob_H, pool=pool)
    sampler.run_mcmc(pos, nsteps,progress=True)
    end = time.time()
    test_chain= sampler.flatchain

samples1 = sampler.flatchain
samples1[np.argmax(sampler.flatlnprobability)]
np.savetxt("ResultsHLCDM2.txt", samples1, fmt="%s")

labels = ['H0']
fig = corner.corner(samples1, show_titles=True, labels=labels, plot_datapoints=True, quantiles=[0.16, 0.5, 0.84])
