class SpPre():
    def __init__ (self):
        # Import many dataFrame for the Algorithm Comparison:
        import pandas as pd
        import numpy as np
        from data_preprocessing.utility import utilitiesCalc
         ## To switch off the error about the older versions.
        import warnings
        warnings.filterwarnings('ignore')
        counter = 22

        subject = 0
        el_blinks = pd.DataFrame()
        s_gaze = 0
        temp_all_data = []
        perc_loss_el_arr = []
        perc_loss_lb_arr = []
        for i in range(1,counter):
            subject = subject + 1
            s_gaze = s_gaze + 1
            el_gaze = pd.read_csv("./data/el_data/p" + str(i) + '.csv', engine='python')
            el_blinks = pd.read_csv("./data/el_data/el_events/blinks/p" + str(i) + '_events.csv', engine='python')
            lb_gaze = pd.read_csv("./data/lb_data/timeseries_data/p" + str(i) + '_XYTC.csv', engine='python')
            all_targets = pd.read_csv("./data/lb_data/target_loc_data/p" + str(i) + '_XYVA.csv', engine='python')
            print('loads files from =' + str(i))
            
            lb_gaze = utilitiesCalc.addParticipantNumberCol(subject, lb_gaze)
            lb_gaze = utilitiesCalc.formating_labvanced (lb_gaze)
            lb_gaze = lb_gaze[lb_gaze['Task_Name'] == "smooth_pursuit"]
            starting_time = lb_gaze.time_lb.values[0]
            finishing_time = lb_gaze.time_lb.values[-1]
            
            lb_under_threshold = lb_gaze[lb_gaze['c'] <= 0.19579889650805132]
            
            
            
            lb_gaze_data_all = len(lb_gaze)
            perc_loss_lb = len(lb_under_threshold)/lb_gaze_data_all * 100
            print('Labvanced: for subject = ' +str(i) + 'data loss in % =' + str(perc_loss_lb))
            perc_loss_lb_arr.append(perc_loss_lb)
            mean_data_loss_lb = np.mean(perc_loss_lb_arr)
            
            # Now we can drop the "under threshold data" from gaze data
            lb_gaze.drop(lb_under_threshold.index, inplace=True)
            
            el_gaze = utilitiesCalc.addParticipantNumberCol(subject, el_gaze)
            el_gaze = el_gaze.loc[(el_gaze.Time <= finishing_time) & (el_gaze.Time >= starting_time)]
            
            el_blinks = el_blinks.loc[(el_blinks.End <= finishing_time) & (el_blinks.Start >= starting_time)]
            el_blinks = utilitiesCalc.addParticipantNumberCol(s_gaze, el_blinks)
            
            all_targets = utilitiesCalc.addParticipantNumberCol(subject, all_targets)
            all_targets = utilitiesCalc.formating_target(all_targets)
            all_targets = all_targets[all_targets['Task_Name'] == "smooth_pursuit"]
            
            blinks = []
            df_blinks_el = pd.DataFrame()
            for index, row in el_blinks.iterrows():
                blink_offset = row['End']
                blink_onset = row['Start']
                within_blinks = el_gaze.loc[(el_gaze.Time <= blink_offset) & (el_gaze.Time >= blink_onset)]
                blinks.append(within_blinks)
            df_blinks_el = pd.concat(blinks, axis=0, ignore_index=False)
            el_gaze_data_all = len(el_gaze)
            perc_loss_el = len(df_blinks_el)/el_gaze_data_all * 100
            # print('Eyelink: for subject = ' +str(i) + 'data loss in % =' + str(perc_loss_el))
            perc_loss_el_arr.append(perc_loss_el)
            mean_data_loss_el = np.mean(perc_loss_el_arr)

            el_gaze.drop(df_blinks_el.index, inplace=True)

            
            el = el_gaze.rename(columns={'X':'X_el','Y':'Y_el','Time':'t'})
            lb = lb_gaze.set_index('time_lb')
            el.to_csv('./data/el_data/smooth_pursuit/p'+str(i)+'s_p.csv', index = False)
            lb.to_csv('./data/lb_data/smooth_pursuit/p'+str(i)+'s_p.csv', index = False)
            
            
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
            
            df_interpolated.to_csv('./data/el_data/smooth_pursuit/interpolated/p'+str(i)+'_s_p.csv', index = False)
            
            all_targets.set_index('timestamp')
            df_interpolated.set_index('time_lb')
            # Resampling
        #     df_interpolated['time_lb'] = pd.to_datetime(df_interpolated["time_lb"], unit='ms')
        #     df_interpolated = df_interpolated.set_index('time_lb')
        #     df_interpolated = df_interpolated.resample('33ms').interpolate(limit_direction="both")
        #     df_interpolated.index = df_interpolated.index.astype('int64') // 10** 6
            df_interpolated.reset_index(inplace=True)
            
            
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

        # print('Labvanced Mean data lost over all participants %M=' + str(mean_data_loss_lb))
        # print('Eyelink Mean data lost over all participants %M=' + str(mean_data_loss_el))
        df.to_csv('./data/smooth_pursuit_interpolated/smooth_pursuit_inter.csv', index = False)