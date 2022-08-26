class HeadMovements():
    def __init__ (self):# %%
       # Import many dataFrame for the Algorithm Comparison:
        import pandas as pd
        import numpy as np
        ### Import our libraries
        # some_file.py
        import sys
        # insert at 1, 0 is the script path (or '' in REPL)
        sys.path.insert(1, '../data_preprocessing/')
        from data_preprocessing.utility import utilitiesCalc
        import data_preprocessing.interpolationET as inter
        counter = 21
        subject = 0
        el_blinks = pd.DataFrame()
        s_gaze = 0
        temp_all_data = []
        perc_loss_el_arr = []
        perc_loss_lb_arr = []
        std_data_loss_el = []
        std_data_loss_lb = [] 
        Task_Name = []
        for i in range(1,counter):
            subject = subject + 1
            s_gaze = s_gaze + 1
            el_gaze = pd.read_csv("./data/el_data/head_movments/p" + str(i) + '.csv', engine='python')
            el_blinks = pd.read_csv("./data/el_data/el_events/blinks_head/p" + str(i) + '_events.csv', engine='python')
            lb_gaze = pd.read_csv("./data/lb_data/head_gaze/p" + str(i) + '_XYTC.csv', engine='python')
        #     lb_head = pd.read_csv("../data/lb_data/head_positions/p" + str(i) + 'p1_Head_position.csv', engine='python')
            print('loads files from =' + str(i))
            #UNCOMENT that if you need to have data from specyfic task
        #     Task_Name = "Roll Movements"
        #     lb_gaze = lb_gaze[lb_gaze['Task_Name'] == Task_Name]
            lb_gaze = utilitiesCalc.addParticipantNumberCol(subject, lb_gaze)
            lb_gaze = utilitiesCalc.formating_labvanced (lb_gaze)

            starting_time = lb_gaze.time_lb.values[0]
            finishing_time = lb_gaze.time_lb.values[-1]
            
            lb_under_threshold = lb_gaze[lb_gaze['c'] <= 0.19579889650805132]


            lb_gaze_data_all = len(lb_gaze)
            perc_loss_lb = len(lb_under_threshold)/lb_gaze_data_all * 100
            print('Labvanced: for subject = ' +str(i) + 'data loss in % =' + str(perc_loss_lb))
            perc_loss_lb_arr.append(perc_loss_lb)
            mean_data_loss_lb = np.mean(perc_loss_lb_arr)
            std_data_loss_lb = np.std(perc_loss_lb_arr)
            # Now we can drop the "under threshold data" from gaze data
            lb_gaze.drop(lb_under_threshold.index, inplace=True)
            
            el_gaze = utilitiesCalc.addParticipantNumberCol(subject, el_gaze)
            el_gaze = el_gaze.loc[(el_gaze.Time <= finishing_time) & (el_gaze.Time >= starting_time)]
            
        #     el_blinks = el_blinks.loc[(el_blinks.End <= finishing_time) & (el_blinks.Start >= starting_time)]
        #     el_blinks = utilitiesCalc.addParticipantNumberCol(s_gaze, el_blinks)

            blinks = []
            df_blinks = pd.DataFrame()
            for index, row in el_blinks.iterrows():
                blink_offset = row['End']
                blink_onset = row['Start']
                within_blinks = el_gaze.loc[(el_gaze.Time <= blink_offset) & (el_gaze.Time >= blink_onset)]
        #         within_blinks = el_gaze.loc[(el_gaze.X == 0)] # Sanity check
                blinks.append(within_blinks)
            df_blinks = pd.concat(blinks, axis=0, ignore_index=False)
            el_gaze_data_all = len(el_gaze)
            if(len(df_blinks) !=0):
                perc_loss_el = len(df_blinks)/el_gaze_data_all * 100
                # print('Eyelink: for subject = ' +str(i) + 'data loss in % =' + str(perc_loss_el))
                perc_loss_el_arr.append(perc_loss_el)
                mean_data_loss_el = np.mean(perc_loss_el_arr)
                std_data_loss_el = np.std(perc_loss_el_arr)
            else:
                print('Eyelink: for subject = ' +str(i) + 'data loss in % = 0')
            el_gaze.drop(df_blinks.index, inplace=True)
            el = el_gaze.rename(columns={'X':'X_el','Y':'Y_el','Time':'t'})
            lb = lb_gaze.set_index('time_lb')
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
            df_interpolated.reset_index(inplace=True)
            
            temp_all_data.append(df_interpolated)
            df = pd.concat(temp_all_data, axis=0, ignore_index=True)
            
        # df.to_csv('../data/head_interpolated/head_inter.csv', index = False)
        if(Task_Name == "Roll Movements"):
            print('Labvanced - Roll Movements Mean data lost over all participants in % M=' + str(mean_data_loss_lb) + 'and SD=' +str(std_data_loss_lb))
            print('Eyelink - Roll Movements Mean data lost over all participants in % M=' + str(mean_data_loss_el) + 'and SD=' +str(std_data_loss_el))
        elif(Task_Name == "Yaw Movements"):
            print('Labvanced - Yaw Movements Mean data lost over all participants in % M=' + str(mean_data_loss_lb) + 'and SD=' +str(std_data_loss_lb))
            print('Eyelink - Yaw Movements Mean data lost over all participants in % M=' + str(mean_data_loss_el)  + 'and SD=' +str(std_data_loss_el))
        else:
            print('Labvanced - Overall Head Movments tasks Mean data lost over all participants in % M=' + str(mean_data_loss_lb)  + 'and SD=' +str(std_data_loss_lb))
            print('Eyelink - Overall Head Movments tasks Mean data lost over all participants in %  M=' + str(mean_data_loss_el)  + 'and SD=' +str(std_data_loss_el))