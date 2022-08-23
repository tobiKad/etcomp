class LargeGridPrec():
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
        import statistics as stat

        ### Import our libraries
        # some_file.py
        import sys
        # insert at 1, 0 is the script path (or '' in REPL)
        sys.path.insert(1, '../data_preprocessing/')
        from data_preprocessing import fixation_plots as plots
        from data_preprocessing.utility import utilitiesCalc
        import warnings
        warnings.filterwarnings('ignore')

        ### graph setting
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

        fixations = []
        el_data = []
        trial_data = []
        for files in sorted(glob.glob("./data/lb_data/last_fixation_data/*.csv"),key=os.path.getmtime):
            df_temp = pd.read_csv(files, index_col=False)
            fixations.append(df_temp)
            
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

        # Ignoring the Participants:
        el_fix['Eye Tracker Type'] = 'Eyelink'
        lb_fix['Eye Tracker Type'] = 'Labvanced'
        # Not enough targets detected Eyelink
        lb_fix = lb_fix[lb_fix['Participant_Nr'] != 16]
        el_fix = el_fix[el_fix['Participant_Nr'] != 16]

        # Bad calibration Labvanced and lag
        lb_fix = lb_fix[lb_fix['Participant_Nr'] != 17]
        el_fix = el_fix[el_fix['Participant_Nr'] != 17]

        # Delay between Labvanced 300hz and Eyelink 300hz resampled = -432504
        # Bad calibration Labvanced and lag
        lb_fix = lb_fix[lb_fix['Participant_Nr'] != 9]
        el_fix = el_fix[el_fix['Participant_Nr'] != 9]

        # Delay between Labvanced 300hz and Eyelink 300hz resampled = -68034
        # Bad calibration Labvanced and lag and # Not enough targets detected Eyetlink
        lb_fix = lb_fix[lb_fix['Participant_Nr'] != 11]
        el_fix = el_fix[el_fix['Participant_Nr'] != 11]

        el_fix.reset_index(drop=True, inplace=True)
        lb_fix.reset_index(drop=True, inplace=True)

        el_fix = el_fix[(el_fix[['x', 'y']] > 0).all(1)]
        lb_fix = lb_fix[(lb_fix[['x', 'y']] > 0).all(1)]

        from math import atan2, degrees
        h = 21 # Monitor height in cm
        d = 60 # Distance between monitor and participant in cm
        ver = 900 # Vertical resolution of the monitor
        hor = 1440
        # Calculate the number of degrees that correspond to a single pixel. This will
        # generally be a very small value, something like 0.03.
        deg_per_px = degrees(atan2(.5*h, d)) / (.5*ver)

        # print('%s degrees correspond to a single pixel horizontal' % deg_per_px)
        df_el = el_fix[(el_fix['targetX'].between(900, 1000)) & (el_fix['targetY'].between(600, 650))]
        df_lb = lb_fix[(lb_fix['targetX'].between(900, 1000)) & (lb_fix['targetY'].between(600, 650))]
        df_t = all_trials[(all_trials['targetX'].between(900, 1000)) & (all_trials['targetY'].between(600, 650))]

        ## INNER OUTER GROUPING ##
        ## Inner Grouping
        innerTrials = all_trials[(all_trials['targetX'].between(400, 1000)) & (all_trials['targetY'].between(300, 650))]
        innerFixELTrials = el_fix[(el_fix['targetX'].between(400, 1000)) & (el_fix['targetY'].between(300, 650))]
        innerFixLBTrials = lb_fix[(lb_fix['targetX'].between(400, 1000)) & (lb_fix['targetY'].between(300, 650))]

        ## Outer Bigger Grouping

        outerTrials = all_trials[(~all_trials['targetX'].between(400, 1000)) | (~all_trials['targetY'].between(300, 650))]
        outerFixELTrials = el_fix[(~el_fix['targetX'].between(400, 1000)) | (~el_fix['targetY'].between(300, 650))]
        outerFixLBTrials = lb_fix[(~lb_fix['targetX'].between(400, 1000)) | (~lb_fix['targetY'].between(300, 650))]

        all_trials[(~all_trials['targetX'].between(200, 1200)) | (~all_trials['targetY'].between(200, 750))]

        #  Formating the groups to degrees
        innerFixLBTrials[['y','targetY','x','targetX','distance']] = innerFixLBTrials[['y','targetY','x','targetX','distance']] * deg_per_px
        innerFixELTrials[['y','targetY','x','targetX','distance']] = innerFixELTrials[['y','targetY','x','targetX','distance']] * deg_per_px


        outerFixLBTrials[['y','targetY','x','targetX','distance']] = outerFixLBTrials[['y','targetY','x','targetX','distance']] * deg_per_px
        outerFixELTrials[['y','targetY','x','targetX','distance']] = outerFixELTrials[['y','targetY','x','targetX','distance']] * deg_per_px

        ## INNER
        # Inner target group
        lb_inner_target_groupped = innerFixLBTrials.groupby(['targetX', 'targetY'])
        el_inner_target_groupped = innerFixELTrials.groupby(['targetX', 'targetY'])

        # Inner target group means
        lb_inner_target_means = lb_inner_target_groupped['distance'].mean().reset_index(drop=True)
        el_inner_target_means = el_inner_target_groupped['distance'].mean().reset_index(drop=True)

        ## Converting to degrees from pixels to all fixation (without grouping data)
        utilitiesCalc.pixtoDegrre(lb_fix)
        utilitiesCalc.pixtoDegrre(el_fix)
        # # Create column with the X and Y in degrees
        all_trials[['targetY','targetX']] = all_trials[['targetY','targetX']] * deg_per_px

        # Converting Trials
        utilitiesCalc.pixtoDegrre(df_el)
        utilitiesCalc.pixtoDegrre(df_lb)
        # Create column with the X and Y in degrees
        df_t[['targetY','targetX']] = df_t[['targetY','targetX']] * deg_per_px
        
        ## Preparing data for Plotting and calculation
        ## EYELINK STD.STD and STD.MEAN Caulcuation
        inner_el_std_std_x = innerFixELTrials.groupby(['targetX', 'targetY'])[['x']].std(ddof=1).std(ddof=1).values[0]
        el_inner_FixELTrial_means_x = innerFixELTrials.groupby(['targetX', 'targetY'])[['x']].std(ddof=1).mean().values[0]

        outer_el_std_std_x = outerFixELTrials.groupby(['targetX', 'targetY'])[['x']].std(ddof=1).std(ddof=1).values[0]
        outer_FixELTrial_means_x = outerFixELTrials.groupby(['targetX', 'targetY'])[['x']].std(ddof=1).mean().values[0]

        inner_el_std_std_y = innerFixELTrials.groupby(['targetX', 'targetY'])[['y']].std(ddof=1).std(ddof=1).values[0]
        el_inner_FixELTrial_means_y = innerFixELTrials.groupby(['targetX', 'targetY'])[['y']].std(ddof=1).mean().values[0]

        outer_el_std_std_y = outerFixELTrials.groupby(['targetX', 'targetY'])[['y']].std(ddof=1).std(ddof=1).values[0]
        outer_FixELTrial_means_y = outerFixELTrials.groupby(['targetX', 'targetY'])[['y']].std(ddof=1).mean().values[0]

        
        ## Labvanced STD.STD and STD.MEAN Caulcuation
        arr_el_means = [el_inner_FixELTrial_means_x,outer_FixELTrial_means_x, el_inner_FixELTrial_means_y ,outer_FixELTrial_means_y]
        arr_el_errors = [inner_el_std_std_x, outer_el_std_std_x, inner_el_std_std_y, outer_el_std_std_y]

        inner_lb_std_std_x = innerFixLBTrials.groupby(['targetX', 'targetY'])[['x']].std(ddof=1).std(ddof=1).values[0]
        inner_FixLBTrial_means_x = innerFixLBTrials.groupby(['targetX', 'targetY'])[['x']].std(ddof=1).mean().values[0]

        outer_lb_std_std_x = outerFixLBTrials.groupby(['targetX', 'targetY'])[['x']].std(ddof=1).std(ddof=1).values[0]
        outer_FixLBTrial_means_x = outerFixLBTrials.groupby(['targetX', 'targetY'])[['x']].std(ddof=1).mean().values[0]

        inner_lb_std_std_y = innerFixLBTrials.groupby(['targetX', 'targetY'])[['y']].std(ddof=1).std(ddof=1).values[0]
        inner_FixLBTrial_means_y = innerFixLBTrials.groupby(['targetX', 'targetY'])[['y']].std(ddof=1).mean().values[0]

        outer_lb_std_std_y = outerFixLBTrials.groupby(['targetX', 'targetY'])[['y']].std(ddof=1).std(ddof=1).values[0]
        outer_FixLBTrial_means_y = outerFixLBTrials.groupby(['targetX', 'targetY'])[['y']].std(ddof=1).mean().values[0]

        arr_lb_means = [inner_FixLBTrial_means_x,outer_FixLBTrial_means_x,inner_FixLBTrial_means_y,outer_FixLBTrial_means_y]
        arr_lb_errors = [inner_lb_std_std_x, outer_lb_std_std_x,inner_lb_std_std_y,outer_lb_std_std_y,]
        arr_lb_means = np.round(arr_lb_means, 2)
        arr_lb_errors = np.round(arr_lb_errors, 2)

        X = ['x inner','x outer',
            'y inner','y outer']
        fig, ax = plt.subplots(1, figsize=(9,3))  
        X_axis = np.arange(len(X))
        
        plt.bar(X_axis - 0.2, arr_lb_means, 0.4, yerr=arr_lb_errors, label = 'Labvanced', color='blue')
        plt.bar(X_axis + 0.2, arr_el_means, 0.4, yerr=arr_el_errors,label = 'Eyelink', color='red')

        plt.xticks(X_axis, X, fontsize=9)
        # plt.xlabel("Groups", fontsize=16)
        plt.ylabel("Visual degree [std]", fontsize=16)
        plt.title("Precision",fontsize=18)
        plt.legend()
        print('EYELINK')
        print('X Inner std of std for Eyelink = ' + str(format(round(inner_el_std_std_x, 2))))
        print('X Outer std of std for Eyelink = ' + str(format(round(outer_el_std_std_x, 2))))
        print('X Inner means of std for Eyelink = ' + str(format(round(el_inner_FixELTrial_means_x, 2))))
        print('X Outer means of std for Eyelink = ' + str(format(round(outer_FixELTrial_means_x, 2))))
        print('Y Inner std of std for Eyelink = ' + str(format(round(inner_el_std_std_y, 2))))
        print('Y Outer std of std for Eyelink = ' + str(format(round(outer_el_std_std_y, 2))))
        print('Y Inner means of std for Eyelink = ' + str(format(round(el_inner_FixELTrial_means_y, 2))))
        print('Y Outer means of std for Eyelink = ' + str(format(round(outer_FixELTrial_means_y, 2))))
     
        print('LABVANCED')
        # More Output to print
        print('X Inner std of std for Labvanced = ' + str(format(round(inner_lb_std_std_x, 2))))
        print('X Outer std of std for Labvanced = ' + str(format(round(outer_lb_std_std_x, 2))))
        print('X Inner means of std for Labvanced = ' + str(format(round(inner_FixLBTrial_means_x, 2))))
        print('X Outer means of std for Labvanced = ' + str(format(round(outer_FixLBTrial_means_x, 2))))
        print('Y Inner std of std for Labvanced = ' + str(format(round(inner_lb_std_std_y, 2))))
        print('Y Outer std of std for Labvanced = ' + str(format(round(outer_lb_std_std_y, 2))))
        print('Y Inner means of std for Labvanced = ' + str(format(round(inner_FixLBTrial_means_y, 2))))
        print('Y Outer means of std for Labvanced = ' + str(format(round(outer_FixLBTrial_means_y, 2))))
        print('X Inner std of std for Labvanced = ' + str(format(round(inner_el_std_std_x, 2))))
        print('X Outer std of std for Labvanced = ' + str(format(round(outer_el_std_std_x, 2))))
        plt.savefig('./analysis_graphs/largeGrid_prec_spatial.jpg')
        plt.show()

        ###GRAN MEANS###
        el_std_std_x = el_fix.groupby(['targetX', 'targetY'])[['x']].std(ddof=1).std(ddof=1).values[0]
        el_mean_std_x = el_fix.groupby(['targetX', 'targetY'])[['x']].std(ddof=1).mean().values[0]

        el_std_std_y = el_fix.groupby(['targetX', 'targetY'])[['y']].std(ddof=1).std(ddof=1).values[0]
        el_mean_std_y = el_fix.groupby(['targetX', 'targetY'])[['y']].std(ddof=1).mean().values[0]

        el_std_std_x_text = 'Precision for Eyelink standard deviation (of standard dev) for x (last fixation coordinates) = ' + str(format(round(el_std_std_x, 2))+' (visual degree)')
        el_std_mean_x_text = 'Precision for Eyelink mean (of standard dev) for x (last fixation coordinates) = ' + str(format(round(el_mean_std_x, 2))+' (visual degree)')

        el_std_std_y_text = 'Precision for Eyelink standard deviation (of standard dev) for y (last fixation coordinates) = ' + str(format(round(el_std_std_y, 2))+' (visual degree)')
        el_std_mean_y_text = 'Precision for Eyelink mean (of standard dev) for y (last fixation coordinates) = ' + str(format(round(el_mean_std_y, 2))+' (visual degree)')

        lb_std_std_x = lb_fix.groupby(['targetX', 'targetY'])[['x']].std(ddof=1).std(ddof=1).values[0]
        lb_mean_std_x = lb_fix.groupby(['targetX', 'targetY'])[['x']].std(ddof=1).mean().values[0]

        lb_std_std_y = lb_fix.groupby(['targetX', 'targetY'])[['y']].std(ddof=1).std(ddof=1).values[0]
        lb_mean_std_y = lb_fix.groupby(['targetX', 'targetY'])[['y']].std(ddof=1).mean().values[0]

        results_lb = [lb_mean_std_x, lb_mean_std_y]
        results_el = [el_mean_std_x, el_mean_std_y]

        error_lb = [lb_std_std_x, lb_std_std_y]
        error_el = [el_std_std_x, el_std_std_y]

        X = ['x','y']
        fig, ax = plt.subplots(1, figsize=(4, 3)) 
        X_axis = np.arange(len(X))

        # yerr=error  
        plt.bar(X_axis - 0.2, results_lb, 0.4, yerr=error_lb,label = 'Labvanced', color='blue')
        plt.bar(X_axis + 0.2, results_el, 0.4, yerr=error_el,label = 'Eyelink', color='red')

        plt.xticks(X_axis, X,fontsize=12)
        # plt.xlabel("Groups", fontsize=14)
        plt.ylabel("visual degree [std]", fontsize=14)
        plt.title("Precision",fontsize=16)
        plt.savefig('./analysis_graphs/largeGrid_prec_grand.jpg')
        plt.legend()

        ## ANOVA
        lb_fix['Eytracker_type'] = 'labvanced'
        el_fix['Eytracker_type'] = 'Eyelink'

        inner_lb_std_x = innerFixLBTrials.groupby(['targetX', 'targetY'])[['x']].std(ddof=1)
        inner_lb_std_x['Eye Tracker Type'] = 'Labvanced'
        inner_lb_std_x['Spatial Division'] = 'Inner'

        outer_lb_std_x = outerFixLBTrials.groupby(['targetX', 'targetY'])[['x']].std(ddof=1)
        outer_lb_std_x['Eye Tracker Type'] = 'Labvanced'
        outer_lb_std_x['Spatial Division'] = 'Outer'


        inner_lb_std_y = innerFixLBTrials.groupby(['targetX', 'targetY'])[['y']].std(ddof=1)
        inner_lb_std_y['Eye Tracker Type'] = 'Labvanced'
        inner_lb_std_y['Spatial Division'] = 'Inner'

        outer_lb_std_y = outerFixLBTrials.groupby(['targetX', 'targetY'])[['y']].std(ddof=1)
        outer_lb_std_y['Eye Tracker Type'] = 'Labvanced'
        outer_lb_std_y['Spatial Division'] = 'Outer'

        inner_el_std_x = innerFixELTrials.groupby(['targetX', 'targetY'])[['x']].std(ddof=1)
        inner_el_std_x['Eye Tracker Type'] = 'Eyelink'
        inner_el_std_x['Spatial Division'] = 'Inner'

        outer_el_std_x = outerFixELTrials.groupby(['targetX', 'targetY'])[['x']].std(ddof=1)
        outer_el_std_x['Eye Tracker Type'] = 'Eyelink'
        outer_el_std_x['Spatial Division'] = 'Outer'

        inner_el_std_y = innerFixELTrials.groupby(['targetX', 'targetY'])[['y']].std(ddof=1)
        inner_el_std_y['Eye Tracker Type'] = 'Eyelink'
        inner_el_std_y['Spatial Division'] = 'Inner'

        outer_el_std_y = outerFixELTrials.groupby(['targetX', 'targetY'])[['y']].std(ddof=1)
        outer_el_std_y['Eye Tracker Type'] = 'Eyelink'
        outer_el_std_y['Spatial Division'] = 'Outer'

        lb_x_std_spatial = [inner_lb_std_x,outer_lb_std_x ]
        df_x_lb = pd.concat(lb_x_std_spatial).reset_index(drop=True)

        el_x_std_spatial = [inner_el_std_x,outer_el_std_x ]
        df_x_el = pd.concat(el_x_std_spatial).reset_index(drop=True)

        lb_y_std_spatial = [inner_lb_std_y,outer_lb_std_y ]
        df_y_lb = pd.concat(lb_y_std_spatial).reset_index(drop=True)

        el_y_std_spatial = [inner_el_std_y,outer_el_std_y ]
        df_y_el = pd.concat(el_y_std_spatial).reset_index(drop=True)

        df_x_lb
        df_x_el
        df_anova_x = [df_x_lb, df_x_el]
        df_anova_x = pd.concat(df_anova_x).reset_index(drop=True)
        print(df_anova_x.isnull().sum())

        df_y_lb
        df_y_el
        df_anova_y = [df_y_lb, df_y_el]
        df_anova_y = pd.concat(df_anova_y).reset_index(drop=True)
        df_anova_y.isnull().sum()

        import pingouin as pg
        print('Anova for X coordinates')
        print(pg.anova(data=df_anova_x, dv='x', between=['Eye Tracker Type','Spatial Division'], detailed=True).round(3))
        print('Anova for Y coordinates')
        print(pg.anova(data=df_anova_y, dv='y', between=['Eye Tracker Type','Spatial Division'], detailed=True).round(3))