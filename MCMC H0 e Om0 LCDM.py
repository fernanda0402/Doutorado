#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 13:13:57 2023

@author: usuario
"""

# CÓDIGO MCMC EXEMPLO BRUNO


# Bibliotecas:
import pandas as pd
import numpy as np
import scipy as sp
import math
from scipy.integrate import solve_ivp
from scipy import integrate
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from multiprocessing import Pool
import emcee
import time
import corner
#from chainconsumer import ChainConsumer




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
          0.8928571428571428, 0.9174311926605504, 0.9345794392523364]  # é o fator de escala

H_data = [186.5, 202.0, 140.0, 177.0, 160.0, 168.0, 154.0, 117.0, 90.0, 125.0,
          105.0, 92.0, 104.0, 97.0, 80.9, 89.0, 92.8, 87.1, 77.0, 95.0, 83.0, 
          83.0, 88.8, 77.0, 72.9, 75.0, 75.0, 83.0, 68.6, 69.0, 69.0]  # cronomêtros cósmicos

s_data = [50.4, 40.0, 14.0, 18.0, 33.6, 17.0, 20.0, 23.0, 40.0, 17.0, 
              12.0, 8.0, 13.0, 62.0, 9.0, 49.6, 12.9, 11.2, 10.2, 17.0, 13.5,
              14.0, 36.6, 14.0, 29.6, 5.0, 4.0, 8.0, 26.2, 12.0, 19.6]  # erro de H

df = pd.DataFrame({"t_data":t_data, "H_data":H_data, "s_data":s_data})
a = df["t_data"]
sigma = df["s_data"]

O_m0 = 0.2

# Modelo:
def HGR(a, H0):
    H_GR = H0*np.sqrt(O_m0*a**(-3) + (1 - O_m0))
    return H_GR


# likelihood:
y = df["H_data"]

def lnlike(theta, a, y, yerr):
    H0 = theta #theta (vetor) é o parâmetro que varia no fit. Cada componente dele vai variar no fit. É ele quem determina quem está variando 
    model =  HGR(a, H0)
    sigma2 = yerr ** 2 
    return -0.5 * np.sum((y - model) ** 2 / sigma2 ) #a exponencial da likelihood é o chi2

def lnprior(theta): # probabilidade a priori, só tem a ver com os dados, não com o modelo
    H0 = theta
    if 50 < H0 < 80:  # o intervalo de investigação
        return 0.0
    return -np.inf

def lnprob(theta, a, y, yerr):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, a, y, yerr)


yerr = sigma
data = (a, y, yerr)
nwalkers =  20   #The number of walkers in the ensemble.
niter = 2000 #número de interações
initial = np.array([68]) #chute inicial para os parâmetros livres
ndim = len(initial) #Number of dimensions in the parameter space.
p0 = [np.array(initial) + 1e-7 * np.random.randn(ndim) for i in range(nwalkers)]  # de onde ele vai sair + o tamanho do passo



# RUN MCMC

def main(p0,nwalkers,niter,ndim,lnprob,data):
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=data)

    print("Running burn-in...") #é a primeira fase do mcmc. É a fase que é jogada fora, porque é uma fase muito influenciada pelo chute inicial.
    p0, _, _ = sampler.run_mcmc(p0, 500, progress=True) # 100 é o número de interações. no mínimo 5000 para retirar. 
    sampler.reset()

    print("Running production...") #depois que tira o burn-in, a cadeia esquece do passado. 
    pos, prob, state = sampler.run_mcmc(p0, niter, progress=True)

    return sampler, pos, prob, state

sampler, pos, prob, state = main(p0,nwalkers,niter,ndim,lnprob,data)

#minha cadeia só vai ter resultado significativo (estatístico) se ela convergir

#fazer um plot do parâmetro em relação ao número de interações para saber qual o burn-in



burnin = 500
like_model =  sampler.get_log_prob(discard=burnin, flat=False)  # são as cadeias (a combinação dos valores de H0 e Om0 )
chi2_model = -2*like_model

np.savetxt('Hz.txt', like_model, fmt="%s")  # salvando as cadeias
print(chi2_model.min())


samples1 = sampler.flatchain  # faz a probabilidade
samples1[np.argmax(sampler.flatlnprobability)]
np.savetxt("Results(16pts).txt", samples1, fmt="%s")

labels = ['H0','O_m0']
fig = corner.corner(samples1,show_titles=True,labels=labels,plot_datapoints=True)  #gera a figura