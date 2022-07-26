
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
import scipy.stats as stats
sns.set_context('talk',font_scale=.8)

def formating_labvanced (df_lb):
    # Renaming the columns
    df_lb = df_lb.rename(columns={"value":"X_lb","Unnamed: 11":"Y_lb","Unnamed: 12":"time_lb","Unnamed: 13":'c'})
    df_lb['time_lb'] = df_lb['time_lb'].fillna(-9999)
    df_lb = df_lb[(df_lb[['time_lb']] != -9999).all(axis=1)]
    # Converting time column to 13 digit number
    df_lb.time_lb = df_lb.time_lb.apply(lambda x: int('%.0f' % x))


    return df_lb
def formating_target (df_lb):
    # Renaming the columns
    df_lb = df_lb.rename(columns={"value":"A","Unnamed: 11":"V","Unnamed: 12":"Y_target","Unnamed: 13":'X_target'})
    df_lb['timestamp'] = df_lb['timestamp'].fillna(-9999)
    df_lb = df_lb[(df_lb[['timestamp']] != -9999).all(axis=1)]
    # Converting time column to 13 digit number
    df_lb.timestamp = df_lb.timestamp.apply(lambda x: int('%.0f' % x))
    
    return df_lb

def formating_eyelink (df_el):
    df_el = df_el.rename(columns={'X':'X_el','Y':'Y_el','Time':'t'})
    df_el['time_el'] = df_el['t']

    df_el = df_el.set_index('t')
    ## Removing all missing data rows
    # df_el = df_el[(df_el[['X_el','Y_el']] != 0).all(axis=1)]
    return df_el

# Takes two parameters df_lb of each EyeTracker
def interpolation (df_el, df_lb):
# Interpolate data to have equal index size
    df_lb_temp = pd.concat([df_el, df_lb .index.to_frame()]).sort_index().interpolate()

    df_lb_temp = df_lb_temp[~df_lb_temp.index.duplicated(keep='first')]
    df_lb_interpolated = df_lb .merge(df_lb_temp, left_index=True, right_index=True, how='left')

    df_lb_interpolated = df_lb_interpolated.drop(columns=['time_lb'])
    df_lb_interpolated = df_lb_interpolated.reset_index()

    df_lb_interpolated = df_lb_interpolated.set_index('time_lb')
    return df_lb_interpolated

def syncNoInter(df_el, df_lb):
# Interpolate data to have equal index size
    df_lb_temp = pd.concat([df_el, df_lb .index.to_frame()]).sort_index()

    df_lb_temp = df_lb_temp[~df_lb_temp.index.duplicated(keep='first')]
    df_lb_interpolated = df_lb .merge(df_lb_temp, left_index=True, right_index=True, how='left')

    df_lb_interpolated = df_lb_interpolated.drop(columns=['time_lb'])
    df_lb_interpolated = df_lb_interpolated.reset_index()

    df_lb_interpolated .Y_el = df_lb_interpolated.Y_el.astype(int)
    df_lb_interpolated .X_el = df_lb_interpolated.X_el.astype(int)

    df_lb_interpolated = df_lb_interpolated.set_index('time_lb')
    return df_lb_interpolated



