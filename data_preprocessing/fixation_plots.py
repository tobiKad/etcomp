import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib.lines import Line2D

def distanceToTargetPlot(eyetracker_type, subject_nr, df_trial, df_fixations):

        if (eyetracker_type=="lb" or eyetracker_type=="labvanced" or eyetracker_type=="Labvanced"):
                tracker_name = "LB"
        else:
                tracker_name = "EL"
        fig, ax = plt.subplots(1, figsize=(16,10))
        plt.gca().invert_yaxis()
        frows = 0
        lrows = df_fixations.index.size
        df_trial = df_trial.reset_index()

        t = df_fixations.loc[frows:lrows]
        t1 = df_trial.loc[frows:lrows]

        point1 = [t1.targetX, t1.targetY]
        point2 = [t.x, t.y]
        x_values = [point1[0], point2[0]]

        y_values = [point1[1], point2[1]]
        if(tracker_name=="EL"):
                plt.plot(x_values, y_values, color='red')
        else:
                plt.plot(x_values, y_values, color='blue')
        plt.scatter(t1.targetX, t1.targetY, marker='+', c='Black', s=100)
        plt.scatter(t.x, t.y, marker='^', s=100)
        plt.title('Graph 1) Distance to Target Scatter Plot: Patricipant ' + str(subject_nr) + ' with the fixation centroid grupped around the 56 target in the Large Gird Paradigm',fontsize=20)
        plt.xlabel('x coordinates in pixel units',fontsize=12)
        plt.ylabel('y coordinates in pixel units',fontsize=12)
        plt.savefig('./data_preprocessing/plots_images/'+str(tracker_name) + '_Fixations_Graphs/'+ str(subject_nr) + ''+str(tracker_name) + '_onsetfix_DistanceLine.jpg')
        plt.grid(True)
        plt.close('all')
def dispersionGridScatter(eyetracker_type, subject_nr, df_trial,  df_fixations, ):

        if (eyetracker_type=="lb" or eyetracker_type=="labvanced" or eyetracker_type=="Labvanced"):
                tracker_name = "LB"
        else:
                tracker_name = "EL"
        fig, ax = plt.subplots(1, figsize=(16,10))
        plt.scatter(df_trial.targetX, df_trial.targetY, marker='o', c=df_fixations.disp,s=df_fixations.disp*5, cmap="RdYlGn_r", edgecolor='black', linewidth=1, alpha=0.75, )
        cbar = plt.colorbar()
        cbar.set_label('Dispersion size in pixels')
        plt.gca().invert_yaxis()
        plt.title('Graph 2) Dispersion Distribution Scatter Plot: Participant number - '+str(subject_nr)+' last fixations centroid grupped around the 56 targets in the Large Gird Paradigm')
        plt.xlabel('x coordinates in pixel units')
        plt.ylabel('y coordinates in pixel units')
        plt.grid(True)
        plt.savefig('./data_preprocessing/plots_images/'+str(tracker_name) + '_Fixations_Graphs/' + str(subject_nr) + ''+str(tracker_name) + '_onsetfix_DistanceLine.jpg')
        plt.close('all')
def precisionScatterScatter(eyetracker_type, subject_nr, df_trial,  df_fixations, ):
    if (eyetracker_type=="lb" or eyetracker_type=="labvanced" or eyetracker_type=="Labvanced"):
        tracker_name = "LB"
    else:
        tracker_name = "EL"
    fig, ax = plt.subplots(1, figsize=(16,10))
    plt.scatter(df_trial.targetX, df_trial.targetY, marker='o', c=df_fixations.distance,s=df_fixations.distance*5, cmap="RdYlGn_r", edgecolor='black', linewidth=1, alpha=0.75, )
    cbar = plt.colorbar()
    cbar.set_label('Offset related fixation \ndistance to target size in pixels')
    plt.gca().invert_yaxis()
    plt.title( 'Graph 3) Distance Distribution Scatter Plot: results of participant ' + str(subject_nr) +' with the fixation centroid grupped around the 56 target in the Large Gird Paradigm.')
    plt.xlabel('x coordinates in pixel units')
    plt.ylabel('y coordinates in pixel units')
    plt.grid(True)
    plt.savefig('./data_preprocessing/plots_images/'+str(tracker_name) + '_Fixations_Graphs/'+ str(subject_nr) + ''+str(tracker_name) + '_onsetfix_DistanceTarget.jpg')
    plt.close('all')

def timeSeriesSyncPlot (time_lb, coor_lb, coor_el):
    x1 = time_lb
    y1 = coor_lb

    x2 = time_lb
    y2 = coor_el

    # plot
    plt.legend()
    plt.xlabel('Time in the miliseconds', fontsize=34)
    plt.ylabel('Coordinates in pixels', fontsize=34)
    plt.plot(x1,y1, c='b', label='Labvanced', linewidth=4, alpha=0.75)
    plt.plot(x2,y2, c='r', label='EyeLink', linewidth=4, alpha=0.75)
    plt.gcf().set_size_inches((20, 11))
    plt.gca().invert_yaxis()
    plt.ticklabel_format(useOffset=False)
    plt.ticklabel_format(useOffset=False, style='plain')
    plt.grid()
    plt.close('all')
    print('Blue Line represent timeseries data from Labvanced, the red line represents Eyelink')
def plotCentroidsToTargets(lb, et, trial_data):
    el_grouped = et.groupby(['targetX', 'targetY'])
    lb_grouped = lb.groupby(['targetX', 'targetY'])
    trials_grouped = trial_data.groupby(['targetX', 'targetY'])

    lb_grouped_fixations_by_trials = lb_grouped
    el_grouped_fixations_by_trials = el_grouped
    grouped_trials = trials_grouped
    # plt.plot(x_values, y_values, color='blue')

    point2 = [lb_grouped_fixations_by_trials.x.median(
    ), lb_grouped_fixations_by_trials.y.median()]
    point2_5 = [el_grouped_fixations_by_trials.x.median(
    ), el_grouped_fixations_by_trials.y.median()]
    point1 = [grouped_trials.targetX.first(), grouped_trials.targetY.first()]

    x_values = [point1[0], point2[0]]
    y_values = [point1[1], point2[1]]

    x_values_2 = [point1[0], point2_5[0]]
    y_values_2 = [point1[1], point2_5[1]]

    fig, ax = plt.subplots(1, figsize=(16, 10))
    plt.plot(x_values, y_values, color='blue')
    plt.plot(x_values_2, y_values_2, color='red')
    plt.gca().invert_yaxis()
    plt.scatter(grouped_trials.targetX.first(), grouped_trials.targetY.first(),
                marker='+', c='Black', s=300)
    plt.scatter(lb_grouped_fixations_by_trials.x.median(), lb_grouped_fixations_by_trials.y.median(
    ), s=100, marker='o', c='blue',  cmap="RdYlGn_r", edgecolor='black', linewidth=1, alpha=0.75,)

    plt.scatter(el_grouped_fixations_by_trials.x.median(), el_grouped_fixations_by_trials.y.median(
    ), s=100, marker='o', c='red',  cmap="RdYlGn_r", edgecolor='black', linewidth=1, alpha=0.75,)

    # cbar = plt.colorbar()
    # cbar.set_label('Median value of distance to target in pixels', fontsize=22)
    plt.title('Graph 1) Eyelink and Labvanced offset-fixations: Location of the X and Y coordinated grouped around targets: \nFor all participants with the fixation centroid as circle (red for Eyelink and blue for Labvanced) grupped around the 56 targets in the Large Gird Paradigm', fontsize=12)
    plt.xlabel('x coordinates in pixel units', fontsize=22)
    plt.ylabel('y coordinates in pixel units', fontsize=22)

    plt.grid(True)
    plt.savefig('../data_processing/analysis_graphs/centroids_targets.jpg')
    plt.show()
