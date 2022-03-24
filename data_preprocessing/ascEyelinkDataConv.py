#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import Libraries
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import glob
import re


# In[2]:


# Import PyGaze library to encode EDF eyelink raw data format
import sys
sys.path.append('./PyGazeAnalyser-master/pygazeanalyser')
from edfreader import read_edf


# In[3]

# In[4]:


counter = 1
for files in sorted(glob.glob("./ascData/*.asc"),key=os.path.getmtime):    
    # Extract Raw Data
    data = files
    data_raw = read_edf(data, 'START', missing=0.0, debug=True)
    type(data_raw), len(data_raw), type(data_raw[0]), data_raw[0].keys()
    
    # Open Asci data and create a list 'lines' with each row line from the ASC data
 
    counter = counter +1
    asci_data = open(data, 'r')
    lines = []
    for line in asci_data:
        lines.append(line)
    # Iterate for each row
    for idx, line in enumerate(lines):
        if 'poczatek' in line:
            time_line = lines[idx].split()
            first_gaze = lines[idx-1].split()
            break

    # eyelink_time
    eyelink_time_start = int(time_line[1])
    # display_split = time_line[-1]
    first_gaze_after_rec = int(first_gaze[0])
    # Using re.findall()
    # Splitting text and number in string 
    display_split_unix = [re.findall(r'[\d\.\d]+', time_line[-1])[0] ]
    display_split_unix
    tracker_start = [re.findall(r'[\d\.\d]+', time_line[1])[0] ]
    tracker_start = int(float(tracker_start[0]))
    tracker_start
    # # #Converting to miliseconds
    display_time_ml_start = int(float(display_split_unix[0]) * 1000)
    display_time_ml_start = display_time_ml_start
    
    for idx, line in enumerate(lines):
        if 'koniec' in line:
            time_line = lines[idx].split()
            break

    eyelink_time_end = int(time_line[1])

    display_split = time_line[-1]
    display_split

    # # Using re.findall()
    # # Splitting text and number in string 
    display_split = [re.findall(r'[\d\.\d]+', time_line[-1])[0] ]
    display_split

    tracker_end = [re.findall(r'[\d\.\d]+', time_line[1])[0] ]
    tracker_end = int(float(tracker_end[0])) - 1
    tracker_end


    # #Converting to miliseconds
    display_time_ml_end = int(float(display_split[0]) * 1000)
    display_time_ml_end
    
    # Create columns for the data
    df_all = pd.DataFrame(columns = ['X', 'Y', 'Tracker_Time','Display_Time','Time'])
    df_all

    x = []
    y = []
    time = []

    for i in range(len(data_raw)):
        x = x + list(data_raw[i]['x'])
        y = y + list(data_raw[i]['y'])
        time = time + list(data_raw[i]['trackertime'])

    df_all.X = x
    df_all.Y = y
    df_all.Tracker_Time = time
    df_all.Display_Time = np.nan
    
    tracker_start = eyelink_time_start
    tracker_start = eyelink_time_end
    # tracker_start = df_all.head(1).Tracker_Time.values[0]
    # tracker_end = df_all.tail(1).Tracker_Time.values[0]
    
    df_all.loc[df_all['Tracker_Time'] == tracker_start, 'Display_Time'] = display_time_ml_start
    df_all.loc[df_all['Tracker_Time'] == tracker_end, 'Display_Time'] = display_time_ml_end
    diff_between_end = tracker_end - display_time_ml_end
    diff_between_start = tracker_start - display_time_ml_start
    df_all = df_all[(df_all.Display_Time == display_time_ml_start).idxmax():]
    a = (display_time_ml_end - display_time_ml_start)/(tracker_end-tracker_start)
    b = - tracker_start * a + display_time_ml_start
    
    df_all['Time'] = df_all['Tracker_Time'] * a + b
    df_all = df_all.drop(columns=['Display_Time'])
    df_all['Time'] = df_all['Tracker_Time'] - diff_between_end
    
    # df_all.Time = (df_all.Tracker_Time + df_all['Diff'])
    df_all.Time = df_all.Time.apply(lambda x: '%.3f' % x)


    print("saving data ")

    df_all.to_csv('./data/el_data/p' + str(counter) + '.csv', index = False)
    
    print('creating fixation events')
    # Create data frame for events
    df = pd.DataFrame(columns = ['x', 'y', 'Start', 'End', 'Participant Number'])
    # Parse event to have the beginning and end time
    for i in range(len(data_raw)):
        trial = i+1
        for j in range(len(data_raw[i]['events']['Efix'])):
            row = {'Trial':int(i), 'x':0, 'y':0, 'Start':0, 'End':0}

            X = data_raw[i]['events']['Efix'][j][3]
            Y = data_raw[i]['events']['Efix'][j][4]
            start = data_raw[i]['events']['Efix'][j][0]
            end = data_raw[i]['events']['Efix'][j][1]

            row['x'] = X
            row['y'] = Y
            row['Start'] = start
            row['End'] = end
            row['Participant Number'] = str(counter)

            df = df.append(row, ignore_index=True)
    df.Trial = df.Trial.astype(int)
    df['Start'] = abs((df.Start - diff_between_start) )
    df['End'] = abs((df.End - diff_between_end) )
    df.Start = df.Start.apply(lambda x: '%.0f' % x)
    df.End = df.End.apply(lambda x: '%.0f' % x)
    df.to_csv('./data/el_data/el_events/p' + str(counter) + '_events.csv', index = False)

