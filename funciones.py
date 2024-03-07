import random
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from astropy.wcs import WCS
from astropy.table import Table
import pandas as pd
import seaborn as sns
import scipy as sp
## recives N (number) gives array of N random index in range of MET, with repetition ##
def random_repeat(N , length):
    """Return N random lists of size length, with repetition."""
    pool = np.arange(length)
    result = np.random.choice(pool , N , replace = True)
    return result

## recives N (number) gives array of N random index in range of MET, with repetition ##
def random_exclude(N , length):
    """Return N random lists of size length, with no repetition."""
    pool = np.arange(length)
    result = np.random.choice(pool , N , replace = False)
    return result
## recives N (number), gives tables with indexs , parameters , spectra of the indexs ##
def random_extract(N ,mastar_spectra, params,repetition=True , reset_index = True):#gives a list with [indexs , parameters , spectra]
    """
    Return indexes, parameters and spectra randomized with size N.
    N -- integer of elements chosen.
    mastar_spectra -- NumPy ndarray where spectra will be picked from.
    params -- NumPy ndarray where parameters will be picked from.
    repetition -- bool determines if the elements will be picket with or without repetition (default True).
    reset_index -- bool determines if the dataframes will have new indexes or the original ones (default True).
    """
    length = len(params)
    if repetition == True:
        lista = random_repeat(N,length)
    else:
        lista = random_exclude(N,length)
    if reset_index == False:
        return [lista , params.loc[lista] , mastar_spectra.loc[lista]]
    else:
        return [lista , params.loc[lista].reset_index(drop=True) , mastar_spectra.loc[lista].reset_index(drop=True)]

    ## returns the weights and the res of the optimize process ##
def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.

    They weights are in effect first normalized so that they 
    sum to 1 (and so they must not all be 0).

    values, weights -- NumPy ndarrays with the same shape.
    """
    weights = weights/np.sum(weights)
    average = np.average(values, weights=weights)
    # Fast and numerically precise:
    variance = np.average((values-average)**2, weights=weights)
    return (average, np.sqrt(variance))

def get_weights(randomspectra , obs_spectrum):
    """
    Return weights and residual using the NonNegative Least Squares algorithm.
    randomspectra -- Pandas DataFrame.
    obs_spectrum -- NumPy ndarrays.
    """
    weights , res = sp.optimize.nnls(randomspectra.T, obs_spectrum)
    return weights , res

## returns the presesed spectrum, a sum with corresponding weights ##
def fm(randomspectra , weights , unique = False):
    """
    return the weighted average of the spectra.
    randomspectra -- Pandas DataFrame.
    weights -- NumPy ndarray
    """
    return np.sum(randomspectra.values * weights[:, None] , axis = 0)

def iterative_fits(mastar_spectra ,params, obs_spectrum, m ,N , repetition = False):
    """
    return m dataframes of N randomized indexes, parameters, spectra, weights and residual.
    mastar_spectra, params -- Pandas DataFrames to extract data.
    obs_spectrum -- NumPy ndarray.
    m, N -- integers of number of repetitions and size of each repetition respectively.
    repetition -- bool of the extraction with or without repetition.
    """
    length = len(mastar_spectra)
    index = []
    parameters = []
    spectra = []
    weights = []
    res = []
    for i in np.arange(m):
        
        index1 , parameters1 , spectra1 = random_extract(N ,mastar_spectra, params ,reset_index = True , repetition = repetition)
        weights1 , res1 = get_weights(spectra1 , obs_spectrum)
        index.append(index1)
        parameters.append(parameters1)
        spectra.append(spectra1)
        weights.append(weights1)
        res.append(res1)
    return index , parameters , spectra , weights , res

def min_res_fits(index , parameters , spectra , weights , res):
    """
    Return the dataframes with the minimum residual.
    index, parameters, spectra, weights, res -- lists.
    """
    index_min = np.argmin(res)
    min_index = index[index_min]
    min_parameters = parameters[index_min]
    min_spectra = spectra[index_min]
    min_weights = weights[index_min]
    min_res = res[index_min]
    
    indices = np.argsort(min_weights)
    sorted_min_weights = [min_weights[i] for i in indices]
    sorted_min_parameters = min_parameters.loc[indices]
    sorted_min_spectra = min_spectra.loc[indices]
    sorted_min_index = [min_index[i] for i in indices]
    return sorted_min_index , sorted_min_parameters , sorted_min_spectra , sorted_min_weights , min_res

def clear_zeros(data , weights):
    """Return list without the data which weight was null."""
    indices = np.nonzero(weights)[0]
    clear_data = data.loc[indices].reset_index(drop = True)
    return clear_data

def get_nonzero_spectra(spectra , parameters , weights):
    """Return the data of the dataframes where the weight was not null."""
    table , index = np.nonzero(weights)
    result_spectra=spectra[table[0]].loc[[index[0]]]
    result_parameters=parameters[table[0]].loc[[index[0]]]
    result_weights=[weights[table[0]][index[0]]]
    for i in np.arange(len(table)-1):
        result_spectra = result_spectra.append([spectra[table[i+1]].loc[[index[i+1]]]] , ignore_index=True)
        result_parameters = result_parameters.append([parameters[table[i+1]].loc[[index[i+1]]]] , ignore_index=True)
        result_weights.append(weights[table[i+1]][index[i+1]])
    how_many = len(table)
    return how_many , result_spectra , result_parameters , result_weights

def plot_kde(param1 , param2 , params , obs_param =[] , plot_obs = False):
    """
    Plot the kernel density estimation of param1 and param2.
    param1, param2 -- strings with the parameters.
    params -- Pandas Dataframe.
    obs_params -- tuple of the observed parameters.
    plot_obs -- bool choose to plot the observed parameters.
    """
    param1 = str(param1)
    param2 = str(param2)
    X1 , Y1 = np.mgrid[np.min(params[param1]):np.max(params[param1]):100j , np.min(params[param2]):np.max(params[param2]):100j]
    positions1 = np.vstack([X1.ravel() , Y1.ravel()])
    values1 = np.vstack([params[param1] , params[param2]])
    kernel1 = sp.stats.gaussian_kde(values1)
    K1 = np.reshape(kernel1(positions1).T , X1.shape)
    if plot_obs == True:
        plt.pcolormesh(X1 , Y1 , K1 , cmap = 'gray_r'), plt.scatter(obs_param[param1] , obs_param[param2] , color = 'r')
    else:
        plt.pcolormesh(X1 , Y1 , K1 , cmap = 'gray_r')
    return
def plot_spectra(m ,wavelength, spectra , weights  ,obs_spectra=[],plot_err = False):
    """plot m adjusted spectra or the error with the observed one."""
    plt.plot(wavelength , obs_spectra , c = 'k' , lw = 2 , label = 'Observed spectrum')
    if m == 1:
        if plot_err == True:
           
            plt.plot(wavelength , (fm(spectra , weights) - obs_spectra) -0.9*i)
        else:
            plt.plot(wavelength , fm(spectra , weights))
    
    if plot_err == True:
        for i in range(m):
            plt.plot(wavelength , (fm(spectra[i] , weights[i]) - obs_spectra) -0.9*i)
    else:
        for i in range(m):
            plt.plot(wavelength , fm(spectra[i] , weights[i]))
    
    return

def plot_mean_parameters(m , param1 , param2 , parameters , weights):
    """scatter plot the mean value of the parameters."""
    param1_mean = np.sum(parameters[param1] * weights)/np.sum(weights)
    param2_mean = np.sum(parameters[param2] * weights)/np.sum(weights)
    plt.scatter(param1_mean , param2_mean , c = 'k' , s=100 , label = 'Parameter average')
    
def map_parameters(m , param1, param2 , parameters , obs_param , weights , alpha = 1):
    """scatter plot each parameter of the DataFrame."""
    if m == 1:
        sc = plt.scatter(parameters[param1] , parameters[param2] , c=weights/np.sum(weights) ,cmap = 'YlGnBu',marker = '.')
        plt.colorbar(sc, label = 'pesos')
    else:
        for i in range(m):
            sc = plt.scatter(parameters[i][param1] , parameters[i][param2] , c=weights[i]/np.sum(weights[i]) ,cmap = 'YlGnBu', marker = '.' , vmin = 0, vmax = 1)
        plt.colorbar(sc, label = 'pesos')
    if len(obs_param[param1]) == 1:
        plt.scatter(obs_param[param1] , obs_param[param2] , c = 'darkturquoise' , marker = '*' , s=200 , label = 'Modelo de PES')
    else:
        plt.scatter(obs_param[param1] , obs_param[param2] , c = 'darkturquoise' , marker = '.' , s=50 , label = 'Modelo de PES')

        
#def plot_spectra_n_param()