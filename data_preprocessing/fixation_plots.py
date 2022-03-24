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
        plt.savefig('./'+str(tracker_name) + '_Fixations_Graphs/'+ str(subject_nr) + ''+str(tracker_name) + '_onsetfix_DistanceLine.jpg')
        plt.grid(True)
        plt.show()
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
        plt.savefig('./'+str(tracker_name) + '_Fixations_Graphs/' + str(subject_nr) + ''+str(tracker_name) + '_onsetfix_DistanceLine.jpg')
        plt.show()
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
    plt.savefig('./'+str(tracker_name) + '_Fixations_Graphs/'+ str(subject_nr) + ''+str(tracker_name) + '_onsetfix_DistanceTarget.jpg')
    plt.show()