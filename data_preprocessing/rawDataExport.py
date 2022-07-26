class GazeFixationsExport():
    def __init__ (self):# %%
        #Import Libraries
        import os
        import sys
        import time
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle
        import glob
        import re

        # Load all ASC = Eyelink Raw data using build in library edfreader
        sys.path.append('./data_preprocessing/data_conversion/PyGazeAnalyser-master/pygazeanalyser')
        from edfreader import read_edf
        print('loaded EDF reader')

        ## Internal libraries
        from data_preprocessing import interpolationET
        from data_preprocessing import cross_correlation
        from data_preprocessing import fixation_plots

        # Loading data files from the directory
        for files in sorted(glob.glob("./ascData/p*.asc"),key=os.path.getmtime):

            print('loading subject file - ' + str(files))

            counter = 0
        for files in sorted(glob.glob("./ascData/p*.asc"),key=os.path.getmtime):
            counter = counter +1
            # Extract Raw Data
            data = files

            data_raw = read_edf(data, 'START', missing=0.0, debug=False)
            type(data_raw), len(data_raw), type(data_raw[0]), data_raw[0].keys()

            #     Open Asci data and create a list 'lines' with each row line from the ASC data
            asci_data = open(data, 'r')
            lines = []
            for line in asci_data:
                lines.append(line)
            # Iterate for each row
            for idx, line in enumerate(lines):
                if 'poczatek' in line:
                    time_line = lines[idx].split()      
                    break

            # Retrieve machine EYelink time in ms
            eyelink_time_start = int(time_line[1])
            print("Eyelink machine start time in ms " + str(eyelink_time_start))

            # Splitting text and number in string 
            display_split_unix = [re.findall(r'[\d\.\d]+', time_line[-1])[0] ]
            tracker_start = [re.findall(r'[\d\.\d]+', time_line[1])[0] ]
            tracker_start = int(float(tracker_start[0]))
            # # #Converting to miliseconds
            display_time_ml_start = int(float(display_split_unix[0]) * 1000)
            print("Eyelink machine start time in UNIX ms " + str(display_time_ml_start))

            # Same for the END koniec means end
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

            # #Converting to miliseconds
            display_time_ml_end = int(float(display_split[0]) * 1000)
            print("End time for Eyelink in UNIX timestamp ms " + str(display_time_ml_end))

            # Create columns for the data
            df_all = pd.DataFrame(columns = ['X', 'Y', 'Tracker_Time','Display_Time','Time'])
            # Create empty dataframe for Eyelink data

            x = []
            y = []
            time = []
            for i in range(len(data_raw)):
                x = x + list(data_raw[i]['x'])
                y = y + list(data_raw[i]['y'])
                time = time + list(data_raw[i]['trackertime'])

            #Checking if there are nan values

            df_all.X = x
            df_all.Y = y
            df_all.Tracker_Time = time
            df_all.Display_Time = np.nan

            # Compute the time for Eyelink substracting the last and first trigger time in ms from Unix timestamp
            diff_between_end = tracker_end - display_time_ml_end
            diff_between_start = tracker_start - display_time_ml_start

            # Then take this difference and subtract from all trackers time, then you would adjust the timestamp to 
            # the LAST reliable trigger
            df_all['Time'] = df_all['Tracker_Time'] - diff_between_end
            tracker_end = df_all.Time.tail(1).values[0]
            
            
            # Drop not used columns
            df_all = df_all.drop(columns=['Display_Time'])

            ## LOAD, FORMAT AND RESAMPLE up to 300hz LABVANCED DATA
            lb = pd.read_csv('./data/lb_data/timeseries_data/p'+ str(counter) + '_XYTC.csv')
            # Format to change the column name and remove between trials empty columns
            lb = interpolationET.formating_labvanced(lb)
            lb = lb.sort_values(by=['time_lb'])

            lb_resampled = cross_correlation.resampleData(lb)

            lb_resampled = lb_resampled[lb_resampled['timestamp'].notna()]

            # FORMAT AND RESAMPLE up to 300hz EYELINK
            el = interpolationET.formating_eyelink(df_all)
            el_resampled = cross_correlation.resampleDataEyelink(el)

            # Interpolate data to have equal size of the index and preparing it for the crosslag correlation:
            df_interpolated = interpolationET.interpolation (el_resampled, lb_resampled)
            # # Reset index to take off the timestamp column
            df_interpolated = df_interpolated.reset_index()

            # Calculating delay between two eyetrackers
            delay = cross_correlation.createLagSygCorrelation(df_interpolated)
            # Convert lag to ms
            ms_delay = delay*2
            print("Delay between Labvanced 300hz and Eyelink 300hz resampled = " + str(ms_delay))
            # Take the first Labvanced Timestamp this is the trigger which we did fix
            display_time_ml_start_first_task = df_interpolated.time_lb.head(1).values[0]
            # Fix the tracker start based on the Labvanced trigger + miliseconds_delay(lag)
            new_tracker_start = abs(display_time_ml_start_first_task + ms_delay)

            #  now for the linear model that interpolates all times, you use:
            # linear interpolation
            a = (display_time_ml_end - display_time_ml_start_first_task)/(tracker_end-new_tracker_start)
            b = - new_tracker_start * a + display_time_ml_start_first_task

            # Fix the time column
            df_all['Time'] = abs(df_all['Time'] * a + b)
            df_all.to_csv('./data/el_data/p' + str(counter) + '.csv', index = False)
            # Here we RE-NAME the column Time which we just correct with time_el, which is valid for other functions.
            el_crossCorrelated = interpolationET.formating_eyelink(df_all)

            ## Now we load data once again because we only wanted to resample data for the cross correlation,
            # Now we want to interpolate with data with the Labvanced 30Hz frame rate.
            lb_30hz = pd.read_csv('./data/lb_data/timeseries_data/p' + str(counter) + '_XYTC.csv')

            lb_30hz = interpolationET.formating_labvanced(lb_30hz)
            lb_30hz = lb_30hz.set_index('time_lb')
            # Sorting data to have large grid as a first
            lb_30hz = lb_30hz.sort_index(ascending=True)

            # Interpolating Eyelink after lag sync and with Labvanced data to create one dataframe.
            df_interpolated = interpolationET.interpolation (el_crossCorrelated, lb_30hz)

            # df_interpolated = df_interpolated.reset_index()

            df_interpolated.sort_index(ascending=True)
        #     df_interpolated.reset_index(inplace=True)

            delay = cross_correlation.createLagSygCorrelation(df_interpolated)
            ms_delay = delay*2
            print("Lag after cross correlation and inteporlation is = " + str(ms_delay))
            from math import atan2, degrees
            h = 45 # Monitor height in cm
            d = 60 # Distance between monitor and participant in cm
            r = 900 # Vertical resolution of the monitor
            # Calculate the number of degrees that correspond to a single pixel. This will
            # generally be a very small value, something like 0.03.
            deg_per_px = degrees(atan2(.5*h, d)) / (.5*r)
            print('%s degrees correspond to a single pixel' % deg_per_px)

            # Create column with the X and Y in degrees
            df_interpolated[['X_lb_vd','Y_lb_vd','X_el_vd','Y_el_vd']] = df_interpolated[['X_lb','Y_lb','X_el','Y_el']] * deg_per_px

            df_interpolated.to_csv('./data/all_data_interpolated/p' + str(counter) + '_interpolated.csv', index = False)

            ## Fixations Extraction Code

            print('creating fixation events')
            # Create data frame for events
            df = pd.DataFrame(columns = ['x', 'y', 'Start', 'End'])
            # Parse event to have the beginning and end time
            for i in range(len(data_raw)):
                trial = i+1
                for j in range(len(data_raw[i]['events']['Efix'])):
                    row = { 'x':0, 'y':0, 'Start':0, 'End':0}

                    x = data_raw[i]['events']['Efix'][j][3]
                    y = data_raw[i]['events']['Efix'][j][4]
                    start = data_raw[i]['events']['Efix'][j][0]
                    end = data_raw[i]['events']['Efix'][j][1]

                    row['x'] = x
                    row['y'] = y
                    row['Start'] = start
                    row['End'] = end

                    df = df.append(row, ignore_index=True)
            # Convert Start and End time using end trigger to unix timestamp
            df['Start'] = abs((df.Start - diff_between_end))
            df['End'] = abs((df.End - diff_between_end) )
            # Interpolate the fixations from starting and ending trigger
            df['Start'] = df['Start'] * a + b
            df['End'] = df['End'] * a + b
            df.Start = df.Start.apply(lambda x: '%.0f' % x)
            df.End = df.End.apply(lambda x: '%.0f' % x)
            df.to_csv('./data/el_data/el_events/p' + str(counter) + '_events.csv', index = False)
            print('fixation data saved, for participant =' + str(counter))

            df_blinks = pd.DataFrame(columns = ['Start', 'End'])
            # Parse event to have the beginning and end time
            for i in range(len(data_raw)):
                trial = i+1
                for j in range(len(data_raw[i]['events']['Eblk'])):
                    row = { 'Start':0, 'End':0}

                    start = data_raw[i]['events']['Eblk'][j][0]
                    end = data_raw[i]['events']['Eblk'][j][1]

                    row['Start'] = start
                    row['End'] = end

                    df_blinks = df_blinks.append(row, ignore_index=True)
            # Convert Start and End time using end trigger to unix timestamp
            df_blinks['Start'] = abs((df_blinks.Start - diff_between_end))
            df_blinks['End'] = abs((df_blinks.End - diff_between_end) )
            # Interpolate the fixations from starting and ending trigger
            df_blinks['Start'] = df_blinks['Start'] * a + b
            df_blinks['End'] = df_blinks['End'] * a + b
            df_blinks.Start = df_blinks.Start.apply(lambda x: '%.0f' % x)
            df_blinks.End = df_blinks.End.apply(lambda x: '%.0f' % x)
            df_blinks.to_csv('./data/el_data/el_events/blinks/p' + str(counter) + '_events.csv', index = False)
            print('blinks data saved, for participant =' + str(counter))




