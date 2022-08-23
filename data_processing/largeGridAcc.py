class LargeGridAcc():
    def __init__ (self):# %%
        # Import many dataFrame for the Algorithm Comparison:
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.pyplot import cm
        from matplotlib.lines import Line2D
        from scipy import stats
        import glob
        import os
        # pd.set_option('display.max_rows', 500)
        from scipy.stats import gaussian_kde
        from scipy.stats import sem
        import seaborn as sns
        from scipy.stats.mstats import winsorize
        from scipy.stats import sem
        import pingouin as pg
        ### Import our libraries~
        # some_file.py
        import sys
        # insert at 1, 0 is the script path (or '' in REPL)
        sys.path.insert(1, '../data_preprocessing/')
        from data_preprocessing import fixation_plots as plots
        from data_preprocessing.utility import utilitiesCalc
        import warnings
        warnings.filterwarnings('ignore')

        ### graph settings #####
        ##### Set style options here #####
        sns.set_style("whitegrid")  # "white","dark","darkgrid","ticks"
        boxprops = dict(linestyle='-', linewidth=1.5, color='#00145A')
        flierprops = dict(marker='o', markersize=7,
                        linestyle='none')
        whiskerprops = dict(color='#00145A')
        capprops = dict(color='#00145A')
        medianprops = dict(linewidth=4, linestyle='-', color='green')
        meanprops=dict(marker="s" ,markerfacecolor="yellow", markersize=10, markeredgecolor="blue")
        # meanprops = dict("marker":"s","markerfacecolor":"white", "markeredgecolor":"blue")
        gt = 'means'
        ############## Loading the fixation for Eyelink, Labvanced and Trial Data ###############
        fixations = []
        el_data = []
        trial_data = []
        for files in sorted(glob.glob("./data/lb_data/last_fixation_data/*.csv"),key=os.path.getmtime):
            df_temp = pd.read_csv(files, index_col=False)
            fixations.append(df_temp)
            # Uncomment for debugging purposes
            # print(files)
        lb_fix = pd.concat(fixations, axis=0, ignore_index=True)
        # lb_fix = lb_fix[lb_fix['Task_Name'] == 'large_grid']
        for files in sorted(glob.glob("./data/el_data/last_fixation_data/*.csv"),key=os.path.getmtime):
            df_temp = pd.read_csv(files, index_col=False)
            el_data.append(df_temp)
            
        el_fix = pd.concat(el_data, axis=0, ignore_index=True)
        el_fix = el_fix[el_fix['Task_Name'] == 'large_grid']
        for files in sorted(glob.glob("./data/lb_data/lb_trial_pp/*.csv"),key=os.path.getmtime):
            df_temp = pd.read_csv(files, index_col=False)
            trial_data.append(df_temp)
        all_trials = pd.concat(trial_data, axis=0, ignore_index=True)
        all_trials = all_trials[all_trials['Task_Name'] == 'large_grid']
        ############## Removing Participants ################

        # Not enough targets detected Eyelink
        lb_fix = lb_fix[lb_fix['Participant_Nr'] != 16]
        el_fix = el_fix[el_fix['Participant_Nr'] != 16]

        # Bad calibration Labvanced and lag
        lb_fix = lb_fix[lb_fix['Participant_Nr'] != 17]
        el_fix = el_fix[el_fix['Participant_Nr'] != 17]

        # Delay between Labvanced 500hz and Eyelink 500hz resampled = -432504
        # Bad calibration Labvanced and lag
        lb_fix = lb_fix[lb_fix['Participant_Nr'] != 9]
        el_fix = el_fix[el_fix['Participant_Nr'] != 9]

        # Delay between Labvanced 500hz and Eyelink 500hz resampled = -68034
        # Bad calibration Labvanced and lag and # Not enough targets detected Eyetlink
        lb_fix = lb_fix[lb_fix['Participant_Nr'] != 11]
        el_fix = el_fix[el_fix['Participant_Nr'] != 11]

        el_fix.reset_index(drop=True, inplace=True)
        lb_fix.reset_index(drop=True, inplace=True)

        #### Spatial Distribution Setup ####
        ## Inner Grouping
        innerTrials = all_trials[(all_trials['targetX'].between(400, 1000)) & (all_trials['targetY'].between(300, 650))]
        innerFixELTrials = el_fix[(el_fix['targetX'].between(400, 1000)) & (el_fix['targetY'].between(300, 650))]
        innerFixLBTrials = lb_fix[(lb_fix['targetX'].between(400, 1000)) & (lb_fix['targetY'].between(300, 650))]

        ## Outer Grouping
        outerTrials = all_trials[(~all_trials['targetX'].between(400, 1000)) | (~all_trials['targetY'].between(300, 650))]
        outerFixELTrials = el_fix[(~el_fix['targetX'].between(400, 1000)) | (~el_fix['targetY'].between(300, 650))]
        outerFixLBTrials = lb_fix[(~lb_fix['targetX'].between(400, 1000)) | (~lb_fix['targetY'].between(300, 650))]

        all_trials[(~all_trials['targetX'].between(200, 1200)) | (~all_trials['targetY'].between(200, 750))]

        # Function to prepare data into the Anova, in order to the desired columns for analysis.
        def anovaDFpreparing(df, trackerType):
            df['Eytracker_type'] = trackerType
            
            
            if (trackerType == 'Labvanced'):
                
                anova_prep = df[['Eytracker_type','distance','Participant_Nr','Trial_Nr_x','targetX', 'targetY']]
                anova_prep.rename(columns={"Trial_Nr_x":'Trial_Nr'}, inplace=True)
                anova_prep.loc[ anova_prep['Trial_Nr'] < 28 , 'trial_division'] = 'first_half'
                anova_prep.loc[ anova_prep['Trial_Nr'] >= 28 , 'trial_division'] = 'second_half'
                
                anova_prep.loc[ (anova_prep['targetX'].between(400, 1000)) & (anova_prep['targetY'].between(300, 650)), 'target_lock'] = 'inner'
                anova_prep.loc[ (~anova_prep['targetX'].between(400, 1000)) | (~anova_prep['targetY'].between(300, 650)) , 'target_lock'] = 'outer'
        
            else:
                anova_prep = df[['Eytracker_type','distance','Participant_Nr','Trial_Nr','targetX', 'targetY']]
                anova_prep.loc[ anova_prep['Trial_Nr'] < 28 , 'trial_division'] = 'first_half'
                anova_prep.loc[ anova_prep['Trial_Nr'] >= 28 , 'trial_division'] = 'second_half'
                
                anova_prep.loc[ (anova_prep['targetX'].between(400, 1000)) & (anova_prep['targetY'].between(300, 650)), 'target_lock'] = 'inner'
                anova_prep.loc[ (~anova_prep['targetX'].between(400, 1000)) | (~anova_prep['targetY'].between(300, 650)) , 'target_lock'] = 'outer'

            return anova_prep
        lb_anova_prep = anovaDFpreparing(lb_fix,'Labvanced')
        el_anova_prep = anovaDFpreparing(el_fix,'Eyelink')

        all_dfs = [el_anova_prep,lb_anova_prep ]
        # The Anova section should be at the end of this file.
        df_anova = pd.concat(all_dfs).reset_index(drop=True)
        df_anova[["trial_division", "target_lock",'Trial_Nr','Participant_Nr','Eytracker_type','targetX','targetY']] = df_anova[["trial_division", "target_lock",'Trial_Nr','Participant_Nr','Eytracker_type','targetX','targetY']].astype('category')
        df_anova_target_labvanced = df_anova[df_anova['Eytracker_type'] == 'Labvanced']
        df_anova_target_eyelink = df_anova[df_anova['Eytracker_type'] == 'Eyelink']
        # Here only the gaze data was converted to pixels in previous steps, because events like fixation where create separebly the convertion is in the analysis.
        def pixtoDegrreTimeSeries(df):    
            from math import atan2, degrees
            h = 21 # Monitor height in cm
            d = 60 # Distance between monitor and participant in cm
            r = 900 # Vertical resolution of the monitor
            hor = 1440
            deg_per_px = degrees(atan2(.5*h, d)) / (.5*r)
            # print('%s degrees correspond to a single pixel' % deg_per_px)

            # Create column with the X and Y in degrees
            df[['x','y','distance']] = df[['x','y','distance']] * deg_per_px
        pixtoDegrreTimeSeries(lb_fix)
        pixtoDegrreTimeSeries(el_fix)
        # Same for trialData
        from math import atan2, degrees
        h = 21 # Monitor height in cm
        d = 60 # Distance between monitor and participant in cm
        r = 900 # Vertical resolution of the monitor
        deg_per_px = degrees(atan2(.5*h, d)) / (.5*r)
        # print('%s degrees correspond to a single pixel' % deg_per_px)

        # Create column with the X and Y in degrees
        all_trials[['targetX','targetY']] = all_trials[['targetX','targetY']] * deg_per_px

        ### Analysis By Subjects ###
        # Group by Participants and create variables
        lb_grouped_participant = lb_fix.groupby(['Participant_Nr'])
        el_grouped_participant = el_fix.groupby(['Participant_Nr'])
        # Means Value
        lb_means_patricipant = lb_grouped_participant['distance'].mean().reset_index(drop=True)
        el_means_patricipant = el_grouped_participant['distance'].mean().reset_index(drop=True)

        ## Create DataFrame for grouped by participants
        # This function create matrix N(participants)x2 where columns are Eyelink and Labvanced and rows are pariticpants.
        # Be careful with providing arguments to the function because you mind swap the means and Eyetracker labels!!!
        df_p_m = utilitiesCalc.twoMeansDFCreating(lb_means_patricipant, el_means_patricipant)
        
        # Use this variable if you wanna add title
        title = 'Accuracy: subjects comparison'
        subset = 'Participants'
        plots.compScatter(df_p_m, subset, title)
        # Shapiro + T-Test
        # plots.pairSampleTTest(df_p_m, subset)

        ### INNER AND OUTER ANALYSIS ###

        ### Converting to pixels once again ###
        innerFixLBTrials[['y','targetY','x','targetX','distance']] = innerFixLBTrials[['y','targetY','x','targetX','distance']] * deg_per_px
        innerFixELTrials[['y','targetY','x','targetX','distance']] = innerFixELTrials[['y','targetY','x','targetX','distance']] * deg_per_px

        outerFixLBTrials[['y','targetY','x','targetX','distance']] = outerFixLBTrials[['y','targetY','x','targetX','distance']] * deg_per_px
        outerFixELTrials[['y','targetY','x','targetX','distance']] = outerFixELTrials[['y','targetY','x','targetX','distance']] * deg_per_px
        # ## INNER
        # # Inner target group
        lb_inner_target_groupped = innerFixLBTrials.groupby(['targetX', 'targetY'])
        el_inner_target_groupped = innerFixELTrials.groupby(['targetX', 'targetY'])

        # Inner target group means
        lb_inner_target_means = lb_inner_target_groupped['distance'].mean().reset_index(drop=True)
        el_inner_target_means = el_inner_target_groupped['distance'].mean().reset_index(drop=True)

        ## OUTER
        # outer target group
        lb_outer_target_groupped = outerFixLBTrials.groupby(['targetX', 'targetY'])
        el_outer_target_groupped = outerFixELTrials.groupby(['targetX', 'targetY'])

        # outer target group means
        lb_outer_target_means = lb_outer_target_groupped['distance'].mean().reset_index(drop=True)
        el_outer_target_means = el_outer_target_groupped['distance'].mean().reset_index(drop=True)

        # Again just for plotting and T-Test comparison creating the matrix Nx2 for outer results
        df_outer_m = utilitiesCalc.twoMeansDFCreating(lb_outer_target_means, el_outer_target_means)
        df_inner_m = utilitiesCalc.twoMeansDFCreating(lb_inner_target_means, el_inner_target_means)
        #Plotting Spatial Comparison
        plots.spatialDivionScatter(df_inner_m, df_outer_m)

        ## Trial Analysis ##

        # Trial Grupping
        lb_grouped_trial = lb_fix.groupby(['Trial_Nr_x'])
        el_grouped_trail = el_fix.groupby(['Trial_Nr'])

        # Means Value
        lb_means_trial = lb_grouped_trial['distance'].mean().reset_index(drop=True)
        el_means_trial = el_grouped_trail['distance'].mean().reset_index(drop=True)

        ## Create DataFrame with fixation central tendency.reset_index
        df_t_m = utilitiesCalc.twoMeansDFCreating(lb_means_trial, el_means_trial)
        # plots.pairSampleTTest(df_t_m, 'Trials') 

        ## Dividing the trials into the half-half
        # First Half
        firstHalfTrials = all_trials[(all_trials['Trial_Nr'] < 28)]
        firstHalflb = lb_fix[(lb_fix['Trial_Nr_x'] < 28)]
        firstHalfEl = el_fix[(el_fix['Trial_Nr'] < 28)]
        lb_trials_groupped = firstHalflb.groupby(['Trial_Nr_x'])
        el_trials_groupped = firstHalfEl.groupby(['Trial_Nr'])
        trial_trials_grouped = firstHalfTrials.groupby(['Trial_Nr'])
        # # First Half trial group means
        f_lb_trials_means = lb_trials_groupped['distance'].mean().reset_index(drop=True)
        f_el_trials_means = el_trials_groupped['distance'].mean().reset_index(drop=True)

        ## Using function to create Mean Matrix for Trial, same way as it was for previous divisions.
        df_fhalf_m = utilitiesCalc.twoMeansDFCreating(f_lb_trials_means, f_el_trials_means)

        # Second Half
        secondHalfTrials = all_trials[(all_trials['Trial_Nr'] >= 28)]
        secondHalflb = lb_fix[(lb_fix['Trial_Nr_x'] >= 28)]
        secondHalfEl = el_fix[(el_fix['Trial_Nr'] >= 28)]

        lb_trials_groupped = secondHalflb.groupby(['Trial_Nr_x'])
        el_trials_groupped = secondHalfEl.groupby(['Trial_Nr'])
        trial_trials_grouped = secondHalfTrials.groupby(['Trial_Nr'])
        # # First Half trial group means
        s_lb_trials_means = lb_trials_groupped['distance'].mean().reset_index(drop=True)
        s_el_trials_means = el_trials_groupped['distance'].mean().reset_index(drop=True)

        df_shalf_m = utilitiesCalc.twoMeansDFCreating(s_lb_trials_means, s_el_trials_means)
        df_shalf_m
        ## Plot the results
        plots.temporalDivionScatter(df_fhalf_m, df_shalf_m)

        df_second_trial = utilitiesCalc.twoMeansDFCreating(s_lb_trials_means, s_el_trials_means)
        df_first_trial = utilitiesCalc.twoMeansDFCreating(f_lb_trials_means, f_el_trials_means)

        # YUNCOMMENT if you need t-test and normality check
        # plots.pairSampleTTest(df_first_trial, 'First half of the experiment') 
        # plots.pairSampleTTest(df_second_trial, 'Second half of the experiment') 

        ### ANOVA COMPARISON ###
        df_anova = df_anova.rename(columns={'trial_division':'temporal division','target_lock':'spatial division','Eytracker_type':'eyetracker type'})
        print(pg.anova(data=df_anova, dv='distance', between=['temporal division','spatial division','eyetracker type'], detailed=True).round(3))