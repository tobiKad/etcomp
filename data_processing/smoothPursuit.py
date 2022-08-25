class SmoothPursuit():
    def __init__ (self):# %%
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt

        from scipy import stats
  
        # pd.set_option('display.max_rows', 500)~

        import seaborn as sns

        import scipy
        from scipy import signal, misc
        import pingouin as pg
        ### Import our libraries
        # some_file.py
        import sys
        # insert at 1, 0 is the script path (or '' in REPL)
        sys.path.insert(1, '../data_preprocessing/')

        ## To switch off the error about the older versions.
        import warnings
        warnings.filterwarnings('ignore')
        sns.set_style("whitegrid")  # "white","dark","darkgrid","ticks"
        medianprops = dict(linewidth=4, linestyle='-', color='green')
        meanprops=dict(marker="s" ,markerfacecolor="yellow", markersize=10, markeredgecolor="blue")
        # meanprops = dict("marker":"s","markerfacecolor":"white", "markeredgecolor":"blue")
        ## Loading and deleting subjects
        # df = pd.read_csv('../data/smooth_pursuit_interpolated/smooth_pursuit.csv')
        df= pd.read_csv('./data/smooth_pursuit_interpolated/smooth_pursuit_inter.csv')

        # Not enough targets detected Eyelink
        df = df[df['Participant_Nr'] != 16]

        # Bad calibration Labvanced and lag
        df = df[df['Participant_Nr'] != 17]

        # Delay between Labvanced 300hz and Eyelink 300hz resampled = -432504
        # Bad calibration Labvanced and lag
        df = df[df['Participant_Nr'] != 9]
        # Delay between Labvanced 300hz and Eyelink 300hz resampled = -68034
        # Bad calibration Labvanced and lag and # Not enough targets detected Eyetlink
        df = df[df['Participant_Nr'] != 11]
        df = df[df['Participant_Nr'] != 18]
        df.reset_index(drop=True, inplace=True)
        # df['t_from_0'] = df['time_lb'].diff().cumsum().fillna(0)
        df.head()

        # Establishing the Confidence level threshold

        ## Velocity calc function
        def calcVelocityForGaze(coor,velocity_col_name):
            # coor - the coordinate to calculate
            # gv_col_name the calculated column name
            df[velocity_col_name] = abs(df[coor].diff()).fillna(0) / df["time_lb"].diff()
            df[velocity_col_name] = df[velocity_col_name].fillna(0)
        calcVelocityForGaze('X_lb', 'velocity_X_lb')
        calcVelocityForGaze('Y_lb', 'velocity_Y_lb')
        calcVelocityForGaze('X_el', 'velocity_X_el')
        calcVelocityForGaze('Y_el', 'velocity_Y_el')
        ## Convert to Visual Degrees functions
        def changeToVisualDegrees(df):
            from math import atan2, degrees
            h = 21 # Monitor height in cm
            l = 42 # Montir length in cm
            d = 60 # Distance between monitor and participant in cm
            ver = 900 # Vertical resolution of the monitor
            hor = 1440
            # Calculate the number of degrees that correspond to a single pixel. This will
            # generally be a very small value, something like 0.03.
            deg_per_px = degrees(atan2(.5*h, d)) / (.5*ver)
            df[["X_lb","Y_lb","X_el","Y_el",'X_target','Y_target']] = df[["X_lb","Y_lb","X_el","Y_el",'X_target','Y_target']]*deg_per_px
            print('%s degrees correspond to a single pixel horizontal' % deg_per_px)
        changeToVisualDegrees(df)

        # Calculate the coorelation between all X and Y from Both Eyetrackers

        rx, px = stats.pearsonr(df.X_lb, df.X_el)

        print("The pearson correlation between Labvanced and Eyelink coefficient for X coordinates is (r=", str(format(round(rx,2))) + ')')
        print("The p-value between Labvanced and Eyelink for X coordinates is (p=" + str(format(round(px,2))) + ')')

        ry, py = stats.pearsonr(df.Y_lb, df.Y_el)

        print("The pearson correlation coefficient between Labvanced and Eyelink for Y coordinates is (r=" + str(format(round(ry,2)))+ ')')
        print("The p-value between Labvanced and Eyelink for Y coordinates is (p=" + str(format(round(py,2))) + ')')

        ### For visual inspection define function to take one participant and then pass to the function to visiaulize gaze data

        df_test = df.loc[(df['Participant_Nr']== 20)]
        df_trial = pd.DataFrame()
        df_p = df_test
        # df_trial = df_p.loc[(df['Trial_Nr']== 5)]
        df_p['t_from_0'] = df_p['time_lb'].diff().cumsum().fillna(0)
        fig, ax = plt.subplots(1, figsize=(12, 6))
        fig.patch.set_facecolor('white')
        plt.scatter(df_p.t_from_0/100, df_p.X_lb, color = 'blue', s=5, label='Labvanced')
        plt.scatter(df_p.t_from_0/100, df_p.X_el, color = 'red', s=5, label='Eyelink')
        # plt.plot(df_p.t_from_0, (df_p.c	 * 100), color = 'purple', linewidth = 2, label='Confidence Level')
        # plt.plot(df_p.time_lb, df_p.gap_between_trials, color = 'yellow', linewidth = 1, label='Trials-Break')
        # plt.scatter(df_p.t_from_0/100, df_p.X_target, color = 'green', s=3, label='Target Location')
        # plt.scatter(df_p.time_lb, df_p.V_Target, color = 'yellow', label='Velocity')
        ax.set_title('Gaze Data (X-coordinates)', fontname="Times New Roman",fontweight="bold", fontsize=16)
        ax.set_ylabel('X-coordinates (visual degrees)',fontsize=20)
        ax.set_xlabel('Time (seconds)',fontsize=20)
        txt = 'Correlation between EyeLink and Labvanced for x (r='+str(format(round(rx, 2)))+' p='+ str(format(round(px, 2)))+' visual degrees) \nand for y (r='+str(format(round(ry, 2)))+' p='+ str(format(round(py, 2)))+' visual degrees)'
        # plt.figtext(0.5, 0.95, txt, wrap=False, horizontalalignment='center', fontsize=22)
        print(txt)
        ax.legend(fontsize="x-large", loc='lower right', labelspacing=0.2, handletextpad=0.01, markerscale=1, shadow=True)
        ax.grid(linewidth = 0.4)
        # ax.legend()
        ax.grid()
        ax.set_ylim(-10,25)
        plt.gcf().autofmt_xdate()
        plt.show()

        fig, ax = plt.subplots(1, figsize=(12, 6))
        fig.patch.set_facecolor('white')
        plt.scatter(df_p.t_from_0/100, df_p.Y_lb, color = 'blue', s=5, label='Labvanced')
        plt.scatter(df_p.t_from_0/100, df_p.Y_el, color = 'red', s=5, label='Eyelink')
        # plt.scatter(df_trial.t_from_0/100, df_trial.X_target, color = 'green', s=3, label='Target Location')
        # plt.plot(df_p.t_from_0, (df_p.c	 * 100), color = 'purple', linewidth = 2, label='Confidence Level')
        # plt.plot(df_p.time_lb, df_p.gap_between_trials, color = 'yellow', linewidth = 1, label='Trials-Break')
        # plt.plot(df_p.time_lb, df_p.X_target, color = 'green', linewidth = 2, label='Target Location')
        #     plt.plot(df_p.time_lb, df_p.V_Target, color = 'yellow', linewidth = 2, label='Velocity')
        ax.set_title('Gaze Data (Y-coordinates)', fontname="Times New Roman",fontweight="bold", fontsize=16)
        ax.set_ylabel('Y-coordinates (visual degrees)',fontsize=20)
        ax.set_xlabel('Time (seconds)',fontsize=20)

        ax.set_ylim(-10,25)
        ax.legend()


        ax.legend(fontsize="x-large", loc='lower right', labelspacing=0.2, handletextpad=0.01, markerscale=1, shadow=True)
        ax.grid(linewidth = 0.4)
        plt.gcf().autofmt_xdate()
        ax.grid()
        plt.show()


        ## Smoothing for velocity, try withand without
        df['velocity_Y_el'] = scipy.signal.medfilt(df['velocity_Y_el'])
        df['velocity_X_el'] = scipy.signal.medfilt(df['velocity_X_el'])

        df['velocity_Y_lb'] = scipy.signal.medfilt(df['velocity_Y_lb'])
        df['velocity_X_lb'] = scipy.signal.medfilt(df['velocity_X_lb'])

        y_means_velo = df.groupby('Participant_Nr')["velocity_Y_el","velocity_Y_lb"].mean()
        x_means_velo = df.groupby('Participant_Nr')["velocity_X_el",'velocity_X_lb'].mean()
        ###
        def velocity_plot (x_means_velo,y_means_velo):    
            plt.figure(figsize =(12, 6) )

            ax1 = plt.subplot(1, 2, 1)
            ax1.set(ylim=(0, 1.30))
            #     plt.figure(figsize =(7, 12) )
            vals, names, xs = [],[],[]
            for i, col in enumerate(x_means_velo.columns):
                vals.append(x_means_velo[col].values)
                names.append(col)
                xs.append(np.random.normal(i + 1, 0.04, x_means_velo[col].values.shape[0]))  # adds jitter to the data points - can be adjusted
            plt.boxplot(vals, labels=names, medianprops=medianprops, meanprops = meanprops, showmeans=True, showbox=False)
            palette = ['r', 'b']
            for x, val, c in zip(xs, vals, palette):
                plt.scatter(x, val, alpha=0.4, color=c)
            plt.xlabel("X Coordinates Velocity", fontweight='normal', fontsize=22)
            plt.ylabel("Visual Degrees", fontweight='normal', fontsize=22) 

            #Outer

            ax3 = plt.subplot(1,2,2, sharey = ax1)
            ax3.set(ylim=(0, 1.30))
            vals, names, xs = [],[],[]
            for i, col in enumerate(y_means_velo.columns):
                vals.append(y_means_velo[col].values)
                names.append(col)
                xs.append(np.random.normal(i + 1, 0.04, y_means_velo[col].values.shape[0]))  # adds jitter to the data points - can be adjusted
            plt.boxplot(vals, labels=names, medianprops=medianprops, meanprops = meanprops, showmeans=True, showbox=False ,)
            palette = ['r', 'b']
            for x, val, c in zip(xs, vals, palette):
                plt.scatter(x, val, alpha=0.4, color=c)
            plt.xlabel("Y Coordinates Velocity", fontweight='normal', fontsize=22)

            plt.title('Velocity comparison after smoothing')
            sns.despine(bottom=True) # removes right and top axis lines
            #     plt.axhline(y=65, color='#ff3300', linestyle='--', linewidth=1, label='Threshold Value')
            plt.legend(bbox_to_anchor=(0.31, 1.06), loc=2, borderaxespad=0., framealpha=1, facecolor ='white', frameon=True)


            plt.show()
        velocity_plot (x_means_velo,y_means_velo)