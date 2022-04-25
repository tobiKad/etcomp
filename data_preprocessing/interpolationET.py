# %%
# class InterpolET():
#     def __init__(sdf_elf):

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
import scipy.stats as stats
sns.set_context('talk',font_scale=.8)

def formating_labvanced (df_lb):
    # Renaming the columns
    # df_lb = df_lb[df_lb['Task_Name'] == "large_grid"]
    df_lb = df_lb.rename(columns={"value":"X_lb","Unnamed: 11":"Y_lb","Unnamed: 12":"time_lb","Unnamed: 13":'c'})
    df_lb['time_lb'] = df_lb['time_lb'].fillna(0)
    df_lb = df_lb[(df_lb[['time_lb']] != 0).all(axis=1)]
    # Converting time column to 13 digit number
    df_lb.time_lb = df_lb.time_lb.apply(lambda x: int('%.0f' % x))

    # # Fill all nan values with 0
    df_lb[['X_lb','Y_lb']] = df_lb[['X_lb','Y_lb']].fillna(0)

    # # Convert the coordinates points from floats to intergers
    df_lb[['X_lb','Y_lb']] = df_lb[['X_lb','Y_lb']].apply(np.int64)
    df_lb = df_lb[(df_lb[['X_lb']] != 0).all(axis=1)]

    # # Drop NaNs
    df_lb.dropna(inplace=True)

    # df_lb = df_lb.set_index('time_lb', drop=True)
    return df_lb

def formating_eyelink (df_el):
    df_el.Time = df_el.Time.apply(lambda x: int('%.0f' % (x * 1)))
    # df_el['Time'] = df_el['Time'].apply(np.int64) * 1000
    # EYE LINK DATA rename the eydf_elink columns
    df_el = df_el.rename(columns={'X':'X_el','Y':'Y_el','Time':'t'})
    df_el['time_el'] = df_el['t']
    #If so ddf_elete this row
    df_el = df_el[(df_el[['X_el','Y_el','time_el']] != 0).all(axis=1)]
    df_el.dropna(inplace=True)
    #Check once agian
    (df_el == 0).sum(axis=0)
    df_el = df_el.set_index('t')

    return df_el

# Takes two parameters df_lb of each EyeTracker
def interpolation (df_el, df_lb):
# Interpolate data to have equal index size
    df_lb_temp = pd.concat([df_el, df_lb .index.to_frame()]).sort_index().interpolate()

    df_lb_temp = df_lb_temp[~df_lb_temp.index.duplicated(keep='first')]
    df_lb_interpolated = df_lb .merge(df_lb_temp, left_index=True, right_index=True, how='left')

    df_lb_interpolated = df_lb_interpolated.drop(columns=['time_lb'])
    df_lb_interpolated = df_lb_interpolated.reset_index()

    df_lb_interpolated .Y_el = df_lb_interpolated.Y_el.astype(int)
    df_lb_interpolated .X_el = df_lb_interpolated.X_el.astype(int)

    df_lb_interpolated = df_lb_interpolated.set_index('time_lb')
    return df_lb_interpolated

# %%



