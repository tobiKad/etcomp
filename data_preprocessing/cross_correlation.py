# %%
# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import statistics as stat
from scipy import stats
import glob
import os
import pandas as pd
from datetime import datetime
from scipy import signal
from numpy.random import default_rng
from scipy.signal import correlate


import seaborn as sns
import scipy.stats as stats
sns.set_context('talk',font_scale=.8)

# Resampling Labvanced Data
def resampleData(df):
    # 1.Resample data to 500Hz to have equal sample size
    # First to use resmaple function I have to change the unix timestamp (int) to datatime format
    df['time_lb'] = pd.to_datetime(df["time_lb"], unit='ms')
    # Then change index to converted column and use the function using mean value to resample from 30hz up to 500hz
    df = df.set_index('time_lb')
    # 500hz is 2ms
    print("The length of data frame is = "  + str(len(df)))
    print("any null values here?" + str(df.isnull().values.any()))
    # df_test = df.dropna()
    # print("The length of data frame is = "  + str(len(df_test)))
    df = df.resample('2ms').interpolate(limit_direction="both")
    # df = df.resample('2ms').interpolate()
    df.index = df.index.astype('int64') // 10** 6

    
    return df
def resampleDataEyelink(df):
    # 1.Resample data to 500Hz to have equal sample size
    # First to use resmaple function I have to change the unix timestamp (int) to datatime format
    df['time_el'] = pd.to_datetime(df["time_el"], unit='ms')
    # Then change index to converted column and use the function using mean value to resample from 30hz up to 500hz
    df = df.set_index('time_el')
    # 500hz is 2ms
    df = df.resample('2ms').interpolate()
    df.index = df.index.astype('int64') // 10** 6
    
    return df
# Take inteporlated data because we need data with equal index size and same sampling rate
# As a another argument we pass the array with Lags
def createLagSygCorrelation(df_interpolated):
    # Use numpy build in function for the cross correlation
    correlation = signal.correlate(df_interpolated.Y_el,df_interpolated.Y_lb,mode='same')
    # create a variable for the lag of the selected participant
    delay = np.argmax(correlation)-int(len(correlation)/2)
    # return delay(lag) for each participant
    return delay
def crosscorr(datax, datay, lag=0, wrap=False):
    """ Lag-N cross correlation. 
    Shifted data filled with NaNs 

    Parameters
    ----------
    lag : int, default 0
    datax, datay : pandas.Series objects of equal length
    Returns
    ----------
    crosscorr : float
    """
    if wrap:
        shiftedy = datay.shift(lag)
        shiftedy.iloc[:lag] = datay.iloc[-lag:].values
        return datax.corr(shiftedy)
    else: 
        return datax.corr(datay.shift(lag))


def slidingWindowCrossCorr (df):
    d1 = df['Y_lb']
    d2 = df['Y_el']
    ms = 1
    gaze_samples = 30
    rs = [crosscorr(d1,d2, lag) for lag in range(-int(ms*gaze_samples),int(ms*gaze_samples+1))]
    offset = np.floor(len(rs)/2)-np.argmax(rs)
    f,ax=plt.subplots(figsize=(14,3))
    ax.plot(rs)
    ax.axvline(np.ceil(len(rs)/2),color='k',linestyle='--',label='Center')
    ax.axvline(np.argmax(rs),color='r',linestyle='--',label='Peak synchrony')
    ax.set(title=f'Offset = {offset} gaze samples \Y_lb leads <> Y_el leads', xlabel='Offset',ylabel='Pearson r')
    offset = offset
    plt.legend()
    
import numpy as np
from numpy.fft import fft, ifft, fft2, ifft2, fftshift

def cross_correlation_using_fft(df):
    x = df.Y_lb
    y = df.Y_el
    f1 = fft(x)
    f2 = fft(np.flipud(y))
    cc = np.real(ifft(f1 * f2))
    return fftshift(cc)

# shift < 0 means that y starts 'shift' time steps before x # shift > 0 means that y starts 'shift' time steps after x
def compute_shift(df):
    x = df.Y_lb
    y = df.Y_el

    assert len(x) == len(y)
    c = cross_correlation_using_fft(df)
    assert len(c) == len(x)
    zero_index = int(len(x) / 2) - 1
    shift = zero_index - np.argmax(c)
    return shift



# %%



