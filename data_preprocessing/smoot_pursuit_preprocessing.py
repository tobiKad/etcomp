
# Import many dataFrame for the Algorithm Comparison:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib.lines import Line2D
from pyrsistent import b
from scipy import stats
import glob
import os
# pd.set_option('display.max_rows', 500)
from scipy.stats import gaussian_kde
from scipy.stats import sem
import seaborn as sns
from scipy.stats.mstats import winsorize

### Import our libraries
# some_file.py
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '../data_preprocessing/')
import fixation_plots as plots
from utility import utilitiesCalc
import interpolationET as inter

counter = 18
lb_data = []
el_data = []
trial_data = []
targets_data = []
subject = 0
targets_data = []
el_blinks = pd.DataFrame()
el_blnk = []
s_gaze = 0
all_target = pd.DataFrame()
temp_all_data = []
for i in range(1,counter):
    subject = subject + 1
    s_gaze = s_gaze + 1
    el_gaze = pd.read_csv("../data/el_data/p" + str(i) + '.csv', engine='python')
    el_blinks = pd.read_csv("../data/el_data/el_events/blinks/p" + str(i) + '_events.csv', engine='python')
    lb_gaze = pd.read_csv("../data/lb_data/timeseries_data/p" + str(i) + '_XYTC.csv', engine='python')
    all_targets = pd.read_csv("../data/lb_data/target_loc_data/p" + str(i) + '_XYVA.csv', engine='python')
    print('loads files from =' + str(i))
    lb_gaze = utilitiesCalc.addParticipantNumberCol(subject, lb_gaze)
    lb_gaze = inter.formating_labvanced (lb_gaze)
    lb_gaze = lb_gaze[lb_gaze['Task_Name'] == "smooth_pursuit"]

    el_gaze = utilitiesCalc.addParticipantNumberCol(subject, el_gaze)

    el_blinks = utilitiesCalc.addParticipantNumberCol(s_gaze, el_blinks)
    
    all_targets = utilitiesCalc.addParticipantNumberCol(subject, all_targets)
    all_targets = inter.formating_target(all_targets)
    all_targets = all_targets[all_targets['Task_Name'] == "smooth_pursuit"]
    
    

    def blinkExtractor(eyetracker,timecolumn):
        blinks = []
        df_blinks = pd.DataFrame()
        for index, row in el_blinks.iterrows():
            blink_offset = row['End']
            blink_onset = row['Start']
            within_blinks = eyetracker.loc[(eyetracker[timecolumn] <= blink_offset) & (eyetracker[timecolumn] >= blink_onset)]
            blinks.append(within_blinks)
        df_blinks = pd.concat(blinks, axis=0, ignore_index=False)
        eyetracker.drop(df_blinks.index, inplace=True)
        return eyetracker
    el = blinkExtractor(el_gaze, 'Time')
    lb = blinkExtractor(lb_gaze,'time_lb')
    el = el_gaze.rename(columns={'X':'X_el','Y':'Y_el','Time':'t'})
    lb = lb_gaze.set_index('time_lb')
    el.to_csv('../data/el_data/smooth_pursuit/p'+str(i)+'s_p.csv', index = False)
    lb.to_csv('../data/lb_data/smooth_pursuit/p'+str(i)+'s_p.csv', index = False)
    
    el = el.set_index('t')
    
    def interpolation ():
    # Interpolate data
        df_temp = pd.concat([el, lb .index.to_frame()]).sort_index().interpolate()

        df_temp = df_temp[~df_temp.index.duplicated(keep='first')]
        df_interpolated = lb .merge(df_temp, left_index=True, right_index=True, how='left')

        df_interpolated = df_interpolated.drop(columns=['time_lb'])
        df_interpolated = df_interpolated.reset_index()

        return df_interpolated
    df_interpolated = interpolation()
    # df_interpolated['time_lb'] = pd.to_datetime(df_interpolated["time_lb"], unit='ms')
    # df_interpolated = df_interpolated.set_index('time_lb')
    # df_interpolated = df_interpolated.resample('33ms').interpolate(limit_direction="both")
    # df_interpolated.reset_index(inplace=True)
    df_interpolated.to_csv('../data/el_data/smooth_pursuit/interpolated/p'+str(i)+'_s_p.csv', index = False)
    
    all_targets.set_index('timestamp')
    df_interpolated.set_index('time_lb')
    df_lb_temp = pd.concat([all_targets, df_interpolated .index.to_frame()]).sort_index().interpolate()
    df_lb_temp = df_lb_temp[~df_lb_temp.index.duplicated(keep='first')]
    df_targets_interpolated = df_interpolated .merge(df_lb_temp, left_index=True, right_index=True, how='left')
    df_targets_interpolated = df_targets_interpolated.reset_index()
    
    df_targets_interpolated.dropna(subset=['X_target'], how='all', inplace=True)
    df_targets_interpolated.dropna(subset=['Y_target'], how='all', inplace=True)
    df_targets_interpolated.dropna(subset=['V'], how='all', inplace=True)
    
    df_targets_interpolated = df_targets_interpolated[['Trial_Nr_x','time_lb','X_lb','Y_lb','c','X_el','Y_el','X_target','Y_target','V','Participant_Nr_x']]
    df_targets_interpolated = df_targets_interpolated.rename(columns={'Trial_Nr_x':'Trial_Nr','c':'C_lb','V':'V_Target','Participant_Nr_x':'Participant_Nr'})
    
    
    temp_all_data.append(df_targets_interpolated)
df = pd.concat(temp_all_data, axis=0, ignore_index=True)
df.to_csv('../data/smooth_pursuit_interpolated/smooth_pursuit_inter.csv', index = False)




