class GazeFixationsExport():
    def __init__ (self):
        diffArr = []
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
        sys.path.append('./PyGazeAnalyser-master/pygazeanalyser')
        from edfreader import read_edf
        print('loaded EDF reader')
        # Loading data files from the directory
        for files in sorted(glob.glob("./ascData/*.asc"),key=os.path.getmtime):
            
            print('loading subject file - ' + str(files))

        counter = 2
        modification_calc=True
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
        #             crossCoor_gaze = lines[idx-22].split()
                    break

            # eyelink_time
            eyelink_time_start = int(time_line[1])
            # display_split = time_line[-1]
        #     crossCoor_gaze = int(crossCoor_gaze[0])

        #     Using re.findall()
        #     Splitting text and number in string 
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
            
            
            



            # Setup the first point START    
            tracker_start = eyelink_time_start
            # Setup the last point END    
            tracker_end = eyelink_time_end
            # Setup this values in the dataframe in the beginning and at the end
            df_all.loc[df_all['Tracker_Time'] == tracker_start, 'Display_Time'] = display_time_ml_start
            df_all.loc[df_all['Tracker_Time'] == tracker_end, 'Display_Time'] = display_time_ml_end
            # Do the calculation of the linear interpolation
            
            # Compute difference between two Eyetrackers
            diff_between_end = tracker_end - display_time_ml_end
            diff_between_start = tracker_start - display_time_ml_start
            start_end_diff = diff_between_start - diff_between_end
            
            # Create modfied START trigger timestamp by substracting the delay value:
            tracker_start_modif = (tracker_start - start_end_diff)
            diffArr.append(start_end_diff)

      
            # Perform Linear interpolation with a new value for starting trigger
            # Code without correction
            ############
            a = (display_time_ml_end - display_time_ml_start)/(tracker_end-tracker_start)
            b = - tracker_start * a + display_time_ml_start
            ############
            # Code with correction
            # a = (display_time_ml_end - display_time_ml_start)/(tracker_end-tracker_start_modif)
            # b = - tracker_start_modif * a + display_time_ml_start


            


            # linear interpolation
            df_all['Time'] = df_all['Tracker_Time'] * a + b

            # Drop not used columns
            df_all = df_all.drop(columns=['Display_Time'])
            # Convert data to the int instead of float
            df_all.Time = df_all.Time.apply(lambda x: '%.f' % x)


            print("saving data ")

            df_all.to_csv('./data/el_data/p' + str(counter) + '.csv', index = False)

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
            # Interpolate the fixations from starting and ending trigger
            df['Start'] = df['Start'] * a + b
            df['End'] = df['End'] * a + b
            df.Start = df.Start.apply(lambda x: '%.0f' % x)
            df.End = df.End.apply(lambda x: '%.0f' % x)
            df.to_csv('./data/el_data/el_events/p' + str(counter) + '_events.csv', index = False)
            break
        print(diffArr)
