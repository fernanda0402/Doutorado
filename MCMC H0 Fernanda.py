#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 14:20:58 2023

@author: usuario
"""

# Bibliotecas:
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from multiprocessing import Pool
import emcee
import time
import corner


# baixando os dados
data_Hz = np.genfromtxt('/home/usuario/Documentos/Dados/CC_Hz_data.csv', delimiter=', ')

z = data_Hz[:, 0]
H_data = data_Hz[:, 1]

sigma_H = data_Hz[:, 2]

df = pd.DataFrame({"z":z, "H_data":H_data, "sigma_H":sigma_H})
#print(df)

z = df["z"]
H_data = df["H_data"]
sigma_H = df["sigma_H"]

Om_0 = 0.3

# Modelo LCDM:
def H(H0):
    Om = Om_0 * (1+z)**(3)
    OL = 1 - Om_0
    H = H0 * np.sqrt(Om + OL)
    return H

def chi(H0):
    H_model = H(H0)
    chisq_vec = np.power((H_model - H_data)/sigma_H, 2)
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


# chi2

like_model =  sampler.get_log_prob(flat=False)
chi2_model = -2*like_model

print(chi2_model.min())

# chi2 reduzido

x2_rd = chi2_model.min()/(50-1)

print(x2_rd)








