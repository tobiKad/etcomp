class FreeView():
    def __init__ (self):# %%
        # Import many dataFrame for the Algorithm Comparison:
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        from scipy import stats
        import seaborn as sns
        ### Import our libraries
        # some_file.py
        import sys
        # insert at 1, 0 is the script path (or '' in REPL)
        sys.path.insert(1, '../data_preprocessing/')
        from math import atan2, degrees
        import warnings
        warnings.filterwarnings('ignore')
        ### graph setting
        ##### Set style options here #####
        sns.set_style("whitegrid")  # "white","dark","darkgrid","ticks"
        # df = pd.read_csv('../data/smooth_pursuit_interpolated/smooth_pursuit.csv')
        df= pd.read_csv('./data/free_view_interpolated/free_view_inter.csv')

        # Not enough targets detected Eyelink
        df = df[df['Participant_Nr_x'] != 16]

        # Bad calibration Labvanced and lag
        df = df[df['Participant_Nr_x'] != 17]

        # Delay between Labvanced 300hz and Eyelink 300hz resampled = -432504
        # Bad calibration Labvanced and lag
        df = df[df['Participant_Nr_x'] != 9]
        # Delay between Labvanced 300hz and Eyelink 300hz resampled = -68034
        # Bad calibration Labvanced and lag and # Not enough targets detected Eyetlink
        df = df[df['Participant_Nr_x'] != 11]
        df = df[df['Participant_Nr_x'] != 18]
        df.reset_index(drop=True, inplace=True)
        # df['t_from_0'] = df['time_lb'].diff().cumsum().fillna(0)

        def changeToVisualDegrees(df):
        
            h = 21 # Monitor height in cm
            l = 42 # Montir length in cm
            d = 60 # Distance between monitor and participant in cm
            ver = 900 # Vertical resolution of the monitor
            hor = 1440
            # Calculate the number of degrees that correspond to a single pixel. This will
            # generally be a very small value, something like 0.03.
            deg_per_px = degrees(atan2(.5*h, d)) / (.5*ver)
            df[["X_lb","Y_lb","X_el","Y_el"]] = df[["X_lb","Y_lb","X_el","Y_el"]]*deg_per_px
            # print('%s degrees correspond to a single pixel horizontal' % deg_per_px)
        changeToVisualDegrees(df)

        # Check which participant to remove
        df.dropna(inplace=True)

        contour_plot_lb = pd.DataFrame()
        contour_plot_el = pd.DataFrame()
        contour_plot_lb[['x', 'y']] = df[['X_lb', 'Y_lb']]
        contour_plot_lb['Tracker Type'] = 'Labvanced'
        contour_plot_el[['x', 'y']] = df[['X_el', 'Y_el']]
        contour_plot_el['Tracker Type'] = 'Eyelink'

        all_dfs = [contour_plot_lb,contour_plot_el ]
        df_counter_plot = pd.concat(all_dfs)

        sns.color_palette("rocket", as_cmap=True)
        sns.color_palette("hls", 8)
        g = sns.displot(df_counter_plot, x='x', y='y', kind='kde', cmap="mako", thresh=0, fill=True, col="Tracker Type", fontsize=22)
        g.set_axis_labels("X-coordinations (visual degree) ","Y-coordinations (visual degree)", fontsize=18)
        g.set(xlim=(0, 30), ylim=(0, 17.5))

        g.fig.subplots_adjust(top=0.9) # adjust the Figure in rp
        g.fig.suptitle('Gaze Data - Gaussian Density Plot', fontsize=16)
        sns.color_palette("hls", 8)
        plt.show()

        corrMatrix = df[['X_lb', 'Y_lb','X_el', 'Y_el']].corr('pearson')
        fig, ax = plt.subplots(1, figsize=(8,6))
        sns.heatmap(corrMatrix, annot=True)

        plt.title('Pearson Correlation between x and y coordinates ')
        plt.show()

        rx, px = stats.pearsonr(df.X_lb, df.X_el)

        print("The pearson correlation between Labvanced and Eyelink coefficient for X coordinates is (r=", str(format(round(rx,2))) + ')')
        print("The p-value between Labvanced and Eyelink for X coordinates is (p=" + str(format(round(px,2))) + ')')

        ry, py = stats.pearsonr(df.Y_lb, df.Y_el)

        print("The pearson correlation coefficient between Labvanced and Eyelink for Y coordinates is (r=" + str(format(round(ry,2)))+ ')')
        print("The p-value between Labvanced and Eyelink for Y coordinates is (p=" + str(format(round(py,2))) + ')')

        ### For the scan paths (scatter raw gaze comparison)

        df_one = df.loc[(df['Participant_Nr_x']== 19)]
        df_one.Trial_Nr.unique()
        for i in range(len(df_one.Trial_Nr.unique() + 1)):
            df_scatter = df_one.loc[(df_one['Trial_Nr']== i)]
            img = plt.imread("Muster02.jpg")
            g.set(xlim=(0, 30), ylim=(0, 17.5))
            ext = [0.0, 30.0, 0.00, 15]
            plt.scatter(df_scatter.X_el, df_scatter.Y_el, marker='o', c ='red', s=25, alpha=0.75,zorder=1)
            plt.scatter(df_scatter.X_lb, df_scatter.Y_lb, marker='o', c ='blue', s=25, alpha=0.75,zorder=1)
            plt.title('Scanpaths from one participant',fontsize=16)
            plt.xlabel('X-coordinations (visual degree)\n',fontsize=14)
            plt.ylabel('Y-coordinations (visual degree)',fontsize=14)
            plt.imshow(img, zorder=0, alpha=0.25, extent=ext)
            aspect=img.shape[0]/float(img.shape[1])*((ext[1]-ext[0])/(ext[3]-ext[2]))
            plt.gca().set_aspect(aspect)
            plt.show()