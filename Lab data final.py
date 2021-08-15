# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 16:14:37 2021

@author: kate
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import pandas as pd

'''Functions'''
'''define the lorentzian fit for florescence rate/double frequency graph'''
def lorentzian( x, a, gam ):
    return a * gam**2 / ( gam**2 + ( x-397 )**2)

'''define inverse lorentzian fit to get frequency from the florescence'''
def inverselorentzian( y ):
    return -np.sqrt(((10450*36**2)/(y)-36**2))

'''define Gaussian for histogram fit'''
def gaussian(x,a,sigma):
    return a*np.exp(-(x+42.5)**2/(2*sigma**2))

def gaussian_2(x,a,sigma):
    return a*np.exp(-(x+33.5)**2/(2*sigma**2))

'''Data'''
'''data for florescence rate/double frequency graph'''
freq=np.array([160,162.5,165,167.5,170,172.5,175,177.5,180,182.5,185,187.5,190,192.5,195,197.5,200])

count=np.array([1839.83411,2104.664099,2389.732307,2570.733398,3001.301675,3145.661059,3653.057258,3653.057258,4765.194131,5615.672607,6588.57941,7614.769041,8961.689455,9530.184013,9702.824047,10290.55336,10253.09832])

SD=np.array([166.6834534,247.4573462,194.824433,232.4481066,283.8664437,573.633713,475.7925696,475.7925696,396.9784945,475.3375006,399.429309,749.6072297,500.1975053,821.0917898,748.6400623,605.5659375,577.6918205])

baseline=463.5012393

AOM=count-baseline

doublefreq=2*freq
freq_0=doublefreq-397


'''data for frequency/time graphs and histograms'''
wm_file = pd.read_csv(r"C:\Users\katie\OneDrive\Documents\Year 3\Lab data\wm file.csv")
sc_file = pd.read_csv(r'C:\Users\katie\OneDrive\Documents\Year 3\Lab data\sc file.csv')
time_wm=wm_file.iloc[:,[0]]
time_sc=sc_file.iloc[:,[0]]
fl_wm=wm_file.iloc[:,[1]]
fl_sc=sc_file.iloc[:,[1]]
freq_wm=inverselorentzian(fl_wm)
freq_sc=inverselorentzian(fl_sc)
freq_wm_lim = freq_wm[ (freq_wm >= -50) & (freq_wm <= -15) ]
freq_sc_lim = freq_sc[ (freq_sc >= -60) & (freq_sc <= -30) ]

'''Plots'''
'''florescence rate/double frequency graph'''
popt, pcurve=scipy.optimize.curve_fit(lorentzian, doublefreq[0:18], AOM[0:18], p0=([10450,36]))

perr= np.sqrt(np.diag(pcurve))

plt.plot(doublefreq, lorentzian(doublefreq, *popt), 'g--')

plt.scatter(doublefreq,AOM,marker='x')

plt.xlabel('2Xfrequency/Hz') 

plt.ylabel('flourescence rate/s') 

plt.title('Ion florescence vs 2X laser fequency')

plt.errorbar(doublefreq,AOM,yerr=SD, linestyle="None")
 
plt.show()
                      
'''corrected zero frequency florescence/double frequency graph'''
plt.scatter(AOM, (freq_0),marker='x')

plt.ylabel('frequency') 

plt.xlabel('flourescence count rate') 

plt.title('Laser frequency vs ion florescence count rate graph')
 
plt.show()


'''Wavemeter frequency/time graph'''
plt.ylabel('frequency') 

plt.xlabel('time/s') 

plt.title('Wavemeter frequency data vs time graph')

freq_wm_lim = freq_wm[ (freq_wm >= -50) & (freq_wm <= -15) ]

plt.scatter(time_wm, freq_wm_lim,marker='x')

plt.show()
print('Wavemeter data:')
print ('mean=',np.mean(freq_wm_lim))

print('minimum frequency=', np.min(freq_wm_lim))
print('maximum frequency=', np.max(freq_wm_lim))
print('frequency range=',np.max(freq_wm_lim)-np.min(freq_wm_lim) )

'''Scanning cavity frequency/time graph'''
plt.ylabel('frequency') 

plt.xlabel('time/s')

plt.title('Scanning cavity frequency data vs time graph')


plt.scatter(time_sc, freq_sc_lim ,marker='x')

plt.show()
print('Scanning cavity data:')
print('mean=',np.mean(freq_sc_lim))

print('minimum frequency=', np.min(freq_sc_lim))
print('maximum frequency=', np.max(freq_sc_lim))
print('frequency range=',np.max(freq_sc_lim)-np.min(freq_sc_lim) )

'''histogram plot'''
hist,bins,m=plt.hist(freq_sc_lim,bins=30)

hist=np.array(hist)

bincenters = np.array(np.mean(np.vstack([bins[0:-1],bins[1:]]), axis=0))

popt, pcurve=scipy.optimize.curve_fit(gaussian, bincenters,hist)

perr= np.sqrt(np.diag(pcurve))

plt.plot(bincenters, gaussian(bincenters, *popt), 'k--')  

plt.ylabel('Number of times laser reaches a frequency') 

plt.xlabel('Laser frequency/Hz')

plt.title('Popularity of laser fequency vs Scanning cavity laser frequency data')
      

plt.show()    



hist_2,bins_2,m=plt.hist(freq_wm_lim,bins=30)

hist_2=np.array(hist_2)

bincenters_2 = np.array(np.mean(np.vstack([bins_2[0:-1],bins_2[1:]]), axis=0))

popt, pcurve=scipy.optimize.curve_fit(gaussian_2, bincenters_2,hist_2)

perr= np.sqrt(np.diag(pcurve))

plt.plot(bincenters_2, gaussian_2(bincenters_2, *popt), 'k--')  

plt.ylabel('Number of times laser reaches a frequency') 

plt.xlabel('Laser frequency/Hz')

plt.title('Popularity of laser fequency vs Wavemeter laser frequency data')
      
      

plt.show()           