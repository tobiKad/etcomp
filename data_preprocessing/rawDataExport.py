class GazeFixationsExport():
    def __init__ (self):# %%
        #Import Libraries
        import os
        import time
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle
        import glob
        import re
        # Loading data files from the directory
        # print('libraries loaded')

        import sys

        sys.path.append('./data_preprocessing/data_conversion/PyGazeAnalyser-master/pygazeanalyser')
        from edfreader import read_edf
        print('loaded EDF reader')

        ## Internal libraries
        from data_preprocessing import interpolationET
        from data_preprocessing import cross_correlation

        # Loading data files from the directory
        for files in sorted(glob.glob("./ascData/*.asc"),key=os.path.getmtime):

            print('loading subject file - ' + str(files))

        counter = 0
        modification_calc=True
        for files in sorted(glob.glob("./ascData/*.asc"),key=os.path.getmtime):
            counter = counter +1
            # Extract Raw Data
            data = files
            
        #     break
            data_raw = read_edf(data, 'START', missing=0.0, debug=False)
            type(data_raw), len(data_raw), type(data_raw[0]), data_raw[0].keys()

        #     Open Asci data and create a list 'lines' with each row line from the ASC data
        #     The counter is for starting with diffrent number is temporarly solution
        #     Finding the start (poczatek means start)
            
            print('loading participant nr = '+ str(counter))
            asci_data = open(data, 'r')
            lines = []
            for line in asci_data:
                lines.append(line)
            # Iterate for each row
            for idx, line in enumerate(lines):
                if 'poczatek' in line:
                    time_line = lines[idx].split()
                    break

            # eyelink_time
            eyelink_time_start = int(time_line[1])

        #     Splitting text and number in string 
            display_split_unix = [re.findall(r'[\d\.\d]+', time_line[-1])[0] ]
            tracker_start = [re.findall(r'[\d\.\d]+', time_line[1])[0] ]
            tracker_start = int(float(tracker_start[0]))
            # # #Converting to miliseconds
            display_time_ml_start = int(float(display_split_unix[0]) * 1000)
            display_time_ml_start = display_time_ml_start
            
            # Same for the end koniec means end
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


        #     # Compute difference between two Eyetrackers
            diff_between_end = tracker_end - display_time_ml_end
            diff_between_start = tracker_start - display_time_ml_start
        
    
            # Use the end trigger to get Tracker time Unix
            df_all['Time'] = df_all['Tracker_Time'] - diff_between_end
            tracker_end = df_all.Time.tail(1).values[0]

            # Find a and b

            # Drop not used columns
            df_all = df_all.drop(columns=['Display_Time'])
            # Convert data to the int instead of float
        #     df_all.Time = df_all.Time.apply(lambda x: '%.f' % x)
            
            ## LOAD LABVANCED DATA
            lb = pd.read_csv('./data/lb_data/timeseries_data/p' + str(counter) + '_XYTC.csv')
            
            # Format the Labvanced take only - Large Grid Task
            lb = interpolationET.formating_labvanced(lb)
            #     Resample Data
            lb_resampled = cross_correlation.resampleData(lb)

            
        #     lb = lb.set_index('time_lb')
            el = interpolationET.formating_eyelink(df_all)


            # Interpolate data to have equal size of the index:
            df_interpolated = interpolationET.interpolation (el, lb_resampled)
            df_interpolated = df_interpolated.reset_index()
            
            delay = cross_correlation.createLagSygCorrelation(df_interpolated)
            # Convert lag to ms
            ms_delay = delay*2
        
            # Take the first Labvanced Timestamp
            display_time_ml_start_first_task = df_interpolated.time_lb.head(1).values[0]
            # Fix the tracker start based on the Labvanced trigger + miliseconds_delay(lag)
            new_tracker_start = abs(display_time_ml_start_first_task + ms_delay)
            
            #  now for the linear model that interpolates all times, you use:
            display_time_ml_start = display_time_ml_start_first_task
            # tracker_start = display_time_ml_start_first_task +/- lag

            # linear interpolation
            a = (display_time_ml_end - display_time_ml_start)/(tracker_end-new_tracker_start)
            b = - new_tracker_start * a + display_time_ml_start
            

            df_all['Time'] = abs(df_all['Time'] * a + b)

        # #     # Same again
            el = interpolationET.formating_eyelink(df_all)

            lb = pd.read_csv('./data/lb_data/timeseries_data/p' + str(counter) + '_XYTC.csv')

            # Format the Labvanced take only - Large Grid Task
            lb = interpolationET.formating_labvanced(lb)
            lb = lb.set_index('time_lb')
            df_interpolated = interpolationET.interpolation (el, lb)
            df_interpolated = df_interpolated.reset_index()
        #     break
        # df_interpolated

            delay = cross_correlation.createLagSygCorrelation(df_interpolated)
            # fixation_plots.timeSeriesSyncPlot(df_interpolated.loc[0:100].time_lb, df_interpolated.loc[0:100].Y_lb, df_interpolated.loc[0:100].Y_el)
            # fixation_plots.timeSeriesSyncPlot(df_interpolated.loc[100:400].time_lb, df_interpolated.loc[100:400].Y_lb, df_interpolated.loc[100:400].Y_el)
            # fixation_plots.timeSeriesSyncPlot(df_interpolated.loc[-1:300].time_lb, df_interpolated.loc[-1:300].Y_lb, df_interpolated.loc[-1:300].Y_el)
            # fixation_plots.timeSeriesSyncPlot(df_interpolated.time_lb, df_interpolated.Y_lb, df_interpolated.Y_el)

            print('Delay after correction ' + str(delay) + 'for participant = ' + str(counter))
            print("saving data ")

            df_all.to_csv('./data/el_data/p' + str(counter) + '.csv', index = False)

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
            # break




