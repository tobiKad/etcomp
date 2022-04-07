class lastFixLB():
    def __init__ (self):
        # Import many dataFrame for the Algorithm Comparison:
        import pandas as pd
        import numpy as np
       
        from data_preprocessing import fixation_plots as plots
        from data_preprocessing.utility import utilitiesCalc

        counter = 17
        tracker_type = "lb"

        for i in range(1,counter):
            df_lb = pd.read_csv('./data/lb_data/fixations_data/p' + str(i) + '_fixations.csv')
            df_lb_trial = pd.read_csv('./data/lb_data/lb_trial_pp/p' + str(i) + '_trial_pp.csv')
            # df_lb_trial = pd.read_csv('../data/lb_data/trial_data/p' + str(i) + '_trials.csv')
            
            
            # format whole data
            df_lb = utilitiesCalc.formating_timeseries(df_lb)
            df_lb_trial = utilitiesCalc.formating_trials(df_lb_trial)
            # Calculate the offset-fixations
            df_offsetFix = utilitiesCalc.calcOffsetFixation(tracker_type,df_lb_trial, df_lb)
            # Caluclate The Euclidan Distance
            df_offsetFix = utilitiesCalc.calcEuclideanDistance(df_offsetFix, df_lb_trial)
            # Add to the dataframe column with participant number:
            df_offsetFix = utilitiesCalc.addParticipantNumberCol(i, df_offsetFix)

            #Plot distance to target and Dispersion
            plots.distanceToTargetPlot(tracker_type, i, df_lb_trial, df_offsetFix)
            plots.dispersionGridScatter(tracker_type, i, df_lb_trial, df_offsetFix)
            plots.precisionScatterScatter(tracker_type, i, df_lb_trial, df_offsetFix)

            df_offsetFix.to_csv('./data/lb_data/last_fixation_data/' + str(i) + '_lb_fix.csv', index = False)
            df_lb_trial.to_csv('./data/lb_data/lb_trial_pp/p' + str(i) + '_trial_pp.csv', index = False)

        # Changing the Eyetracker type to Eyelink
        tracker_type = "el"

        for i in range(1,counter):
            df_el = pd.read_csv('./data/el_data/el_events/p' + str(i) + '_events.csv')
            df_lb_trial = pd.read_csv('./data/lb_data/lb_trial_pp/p' + str(i) + '_trial_pp.csv')
        
            # Calculate the offset-fixations
            df_el = utilitiesCalc.formating_el_data(df_el)
            df_offsetFix = utilitiesCalc.calcOffsetFixation(tracker_type, df_lb_trial, df_el)
            # Caluclate The Euclidan Distance
            df_offsetFix = utilitiesCalc.calcEuclideanDistance(df_offsetFix, df_lb_trial)
            # Add to the dataframe column with participant number:
            df_offsetFix = utilitiesCalc.addParticipantNumberCol(i, df_offsetFix)
            
        #     #Plot distance to target and Dispersion
            plots.distanceToTargetPlot(tracker_type, i, df_lb_trial, df_offsetFix)
            plots.precisionScatterScatter(tracker_type, i, df_lb_trial, df_offsetFix)
            
            df_offsetFix
            df_offsetFix.to_csv('./data/el_data/last_fixation_data/' + str(i) + '_lb_fix.csv', index = False)



