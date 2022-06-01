import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib.lines import Line2D
import seaborn as sns
from scipy import stats
from scipy.stats import gaussian_kde
from scipy.stats import sem
import seaborn as sns
from scipy.stats.mstats import winsorize
from scipy.stats import sem

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

gt = 'means'

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
    plt.plot(x1,y1, c='b', label='Labvanced', linewidth=2, alpha=0.75)
    plt.plot(x2,y2, c='r', label='EyeLink', linewidth=2, alpha=0.75)
    plt.gcf().set_size_inches((20, 11))
    plt.gca().invert_yaxis()
    plt.ticklabel_format(useOffset=False)
    plt.ticklabel_format(useOffset=False, style='plain')
    plt.grid()
    plt.show()
    # plt.close('all')
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

def twoGroupsComparisonScatter(df, gt, title):
    # Setup colors
    colors = ['red', 'blue']
    #setup jitter
    jitter = 0.05
    df_x_jitter = pd.DataFrame(np.random.normal(loc=0, scale=jitter, size=df.values.shape), columns=df.columns)
    df_x_jitter += np.arange(len(df.columns))

    fig, ax = plt.subplots(1, figsize=(6,10))
    fig.patch.set_facecolor('white')
    # plot two groups
    for col in df:
        ax.plot(df_x_jitter[col], df[col], 'o', alpha=.60, zorder=1, ms=8, mew=1)
    ax.set_xticks(range(len(df.columns)))
    ax.set_xticklabels(df.columns)
    ax.set_xlim(-0.5,len(df.columns)-0.5)
    ax.set_title(title)
    ax.set_ylabel(str(gt) + 'of euclidian distance from fixation centroid to target in pixels')
    
    for idx in df.index:
        ax.plot(df_x_jitter.loc[idx,['Labvanced','Eyelink']], df.loc[idx,['Labvanced','Eyelink']], color = 'grey', linewidth = 0.5, linestyle = '--', zorder=-1, ) 
    plt.show()
    ax.grid()

def whiskerPlotEyetrackers(df, gt,title):

    
    fig = plt.figure(figsize =(8, 12) )


    vals, names, xs = [],[],[]
    for i, col in enumerate(df.columns):
        vals.append(df[col].values)
        names.append(col)
        xs.append(np.random.normal(i + 1, 0.04, df[col].values.shape[0]))  # adds jitter to the data points - can be adjusted
    plt.boxplot(vals, labels=names, notch=False, boxprops=boxprops, whiskerprops=whiskerprops,capprops=capprops, flierprops=flierprops, medianprops=medianprops, meanprops = meanprops, showmeans=True, showbox=True)
    palette = ['r', 'b']
    for x, val, c in zip(xs, vals, palette):
        plt.scatter(x, val, alpha=0.4, color=c)

    plt.xlabel("Type of Eyetracker", fontweight='normal', fontsize=14)
    plt.ylabel("Distance to target in pixels", fontweight='normal', fontsize=14)
#     if(method='GT'):
    plt.title(title)
    sns.despine(bottom=True) # removes right and top axis lines
    # plt.axhline(y=65, color='#ff3300', linestyle='--', linewidth=1, label='Threshold Value')
    plt.legend(bbox_to_anchor=(0.31, 1.06), loc=2, borderaxespad=0., framealpha=1, facecolor ='white', frameon=True)


    plt.show()

def whiskerCountPlotEyetrackers(df, gt,title):

    
    fig = plt.figure(figsize =(8, 12) )


    vals, names, xs = [],[],[]
    for i, col in enumerate(df.columns):
        vals.append(df[col].values)
        names.append(col)
        xs.append(np.random.normal(i + 1, 0.04, df[col].values.shape[0]))  # adds jitter to the data points - can be adjusted
    plt.boxplot(vals, labels=names, notch=False, boxprops=boxprops, whiskerprops=whiskerprops,capprops=capprops, flierprops=flierprops, medianprops=medianprops, meanprops = meanprops, showmeans=True, showbox=True)
    palette = ['r', 'b']
    for x, val, c in zip(xs, vals, palette):
        plt.scatter(x, val, alpha=0.4, color=c)

    plt.xlabel("Type of Eyetracker", fontweight='normal', fontsize=14)
    plt.ylabel("Total number of detected fixations during Free View Task", fontweight='normal', fontsize=14)
#     if(method='GT'):
    plt.title(title)
    sns.despine(bottom=True) # removes right and top axis lines
    # plt.axhline(y=65, color='#ff3300', linestyle='--', linewidth=1, label='Threshold Value')
    plt.legend(bbox_to_anchor=(0.31, 1.06), loc=2, borderaxespad=0., framealpha=1, facecolor ='white', frameon=True)


    plt.show()

def connectedDottScatter(df, subset, title):
    x_w_m = df['Labvanced']
    y_w_m = df['Eyelink']

    colors = 'br'
    N = len(x_w_m)
    fig, ax = plt.subplots(1, figsize=(12,8))
    for i in range(N):
        plt.plot([x_w_m[i], y_w_m[i]], [i, i], colors[int(x_w_m[i]<y_w_m[i])])
    plt.plot(x_w_m, np.arange(N), 'ks', markerfacecolor='b', label='Labvanced')
    plt.plot(y_w_m, np.arange(N), 'ro', markerfacecolor='r', label='Eyelink')
    plt.ylabel('Participant number')
    plt.xlabel('Mean Distance(pixels) to Target Values')
    plt.title(str(subset) +' sub-set This Plots shows on Y axis participants mean\n which are linked, the color of linked line is represents higher values between two participants. On the X axis we present value of central tendency.')
    plt.legend()
    plt.show()

def wilcoxonTtest(df, subset):
    
    lb_shapiro_norm_W = stats.shapiro(df['Eyelink'])
    el_shapiro_norm_W = stats.shapiro(df['Labvanced'])
    #         pair_tTest = stats.ttest_rel(df['Labvanced'], df['Eyelink'])
    t, p = stats.wilcoxon(df['Labvanced'], df['Eyelink'])
    x_w_m = df['Labvanced']
    y_w_m = df['Eyelink']
    
    lsem = sem(df['Labvanced'])
    
    esem = sem(df['Eyelink'])
    
    plt.style.use('seaborn-deep')
    # txt = ('Grpah2) Histogram which represents comparison of two Eyetracker winsorized 20% Means from all over the data \nLABVANCED: The Shapiro-Wilk test for normality first value is the W test value, and the second value it the p-value. ' + str(paired_Ttest_lb) + '\nEYELINK: The Shapiro-Wilk test for normality first value is the W test value, and the second value it the p-value. ' + str(paired_Ttest_lb) + '\n Results of pair-wise T-test' + str(stats.ttest_rel(df_p_m['Labvanced'], df_p_m['Eyelink'])) )
    
    bins = 10
    fig, ax = plt.subplots(1, figsize=(8,6))
    plt.hist([x_w_m, y_w_m], bins, label=['Labvanced', 'Eyelink'])
    plt.xlabel('Means of Distance to Target in pixels', fontsize=16)
    plt.ylabel('Number of means in bins', fontsize=16)
    plt.legend(loc='upper right')
    subset
    plt.title(str(subset) + ' sub-set Histogram which represents comparison of two Eyetracker Means from all over participants \nLABVANCED: The Shapiro-Wilk W='+str(format(lb_shapiro_norm_W[0], ".3f"))+' P='+str(format(lb_shapiro_norm_W[1], ".3f"))+ '\nEYELINK: The Shapiro-Wilk W='+str(format(el_shapiro_norm_W[0], ".3f"))+' P='+str(format(el_shapiro_norm_W[1], ".3f"))+'\nBecause both Means are not valid descriptor of central tendecy we use Wilcoxon Z=' + str(format(t,'.2f')) + ' P=' + str(format(p,'.2f') + 'p withouth formating ='+str(p)) + '\n Labvanced SEM = ' + str(lsem) + 'and for Eyelink SEM = ' + str(esem))
    # fig.text(.5, 0.9, txt, ha='center')
    plt.show()

def pairSampleTTest(df, subset):  
    lb_shapiro_norm_W = stats.shapiro(df['Labvanced'])
    el_shapiro_norm_W = stats.shapiro(df['Eyelink'])
    #         pair_tTest = stats.ttest_rel(df['Labvanced'], df['Eyelink'])
    t, p = stats.ttest_rel(df['Labvanced'], df['Eyelink'])

    plt.style.use('seaborn-deep')
    # txt = ('Grpah2) Histogram which represents comparison of two Eyetracker winsorized 20% Means from all over the data \nLABVANCED: The Shapiro-Wilk test for normality first value is the W test value, and the second value it the p-value. ' + str(paired_Ttest_lb) + '\nEYELINK: The Shapiro-Wilk test for normality first value is the W test value, and the second value it the p-value. ' + str(paired_Ttest_lb) + '\n Results of pair-wise T-test' + str(stats.ttest_rel(df_p_m['Labvanced'], df_p_m['Eyelink'])) )
    lsem = sem(df['Labvanced'])
    
    esem = sem(df['Eyelink'])
    
    bins = 10
    fig, ax = plt.subplots(1, figsize=(8,6))
    plt.hist([df['Labvanced'], df['Eyelink'] ], bins, label=['Labvanced', 'Eyelink'])
    plt.xlabel('Means of Distance to Target in pixels', fontsize=16)
    plt.ylabel('Number of particinats', fontsize=16)
    plt.legend(loc='upper right')
    subset
    plt.title(str(subset) + ' sub-set Histogram which represents comparison of two Eyetracker Means from all over participants \nLABVANCED: The Shapiro-Wilk W='+str(format(lb_shapiro_norm_W[0], ".3f"))+' P='+str(format(lb_shapiro_norm_W[1], ".3f"))+ '\nEYELINK: The Shapiro-Wilk W='+str(format(el_shapiro_norm_W[0], ".3f"))+' P='+str(format(el_shapiro_norm_W[1], ".3f")) + str(format(t,'.2f')) + ' p=' + str(format(p,'.2f') + '. \np-value withouth formating to verify if there is no formating mistake ='+str(p)) + '\n Labvanced SEM = ' + str(lsem) + 'and for Eyelink SEM = ' + str(esem))
    # fig.text(.5, 0.9, txt, ha='center')
    plt.show()
def centroidDistanceScatter(df_lb, df_el, df_trials, subset):
    
    ## Grouping by target fixations data
    lb_grouped = df_lb.groupby(['targetX', 'targetY']) 
    el_grouped = df_el.groupby(['targetX', 'targetY'])
    trial_grouped_target = df_trials.groupby(['targetX', 'targetY'])


    # Groupped by means
    lb_target_means = lb_grouped['distance'].mean().reset_index(drop=True)
    el_target_means = el_grouped['distance'].mean().reset_index(drop=True)
    
    lb_shapiro_norm_W = stats.shapiro(lb_target_means)
    el_shapiro_norm_W = stats.shapiro(el_target_means)
    
    lsem = sem(lb_target_means)
    esem = sem(el_target_means)
    
    # Points for labvanced
    point2lb = [lb_grouped.x.mean(), lb_grouped.y.mean()]
    point1lb = [trial_grouped_target.targetX.first(), trial_grouped_target.targetY.first()]

    x_values_lb = [point1lb[0], point2lb[0]]
    y_values_lb = [point1lb[1], point2lb[1]]

    # Points for Eyelink
    point2el = [el_grouped.x.mean(), el_grouped.y.mean()]
    point1el = [trial_grouped_target.targetX.first(), trial_grouped_target.targetY.first()]

    x_values_el = [point1el[0], point2el[0]]
    y_values_el = [point1el[1], point2el[1]]

    t, p = stats.ttest_rel(lb_target_means, el_target_means)

    fig, ax = plt.subplots(1, figsize=(16,10))
    #line for labvanced
    plt.plot(x_values_lb, y_values_lb, color='blue')

    #line for Eyelink
    plt.plot(x_values_el, y_values_el, color='red')

    ax.set_xlim(0,1440)
    ax.set_ylim(0,900)
    plt.gca().invert_yaxis()
    # Target
    plt.scatter(all_trials.targetX, all_trials.targetY, marker='+', c='Black', s=300)
    # Labvanced
    plt.scatter(lb_grouped.x.mean(), lb_grouped.y.mean(), marker='o', c ='blue', s=100, edgecolor='black', linewidth=1, alpha=0.75,)
    # Eyelink
    plt.scatter(el_grouped.x.mean(), el_grouped.y.mean(), marker='o', c ='red', s=100, edgecolor='black', linewidth=1, alpha=0.75,)

    plt.title(str(subset) + ' sub-sets of euclidian distances to target scatter plot: \nLABVANCED: The Shapiro-Wilk W='+str(format(lb_shapiro_norm_W[0], ".3f"))+' P='+str(format(lb_shapiro_norm_W[1], ".3f"))+ '\nEYELINK: The Shapiro-Wilk W='+str(format(el_shapiro_norm_W[0], ".3f"))+' P='+str(format(el_shapiro_norm_W[1], ".3f"))+'\nBecause both Means are valid descriptor of central tendecy (normaly distributed) we use Paired Sample T=test, t=' + str(format(t,'.2f')) + ' p=' + str(format(p,'.2f') + '. \np-value withouth formating to verify if there is no formating mistake ='+str(p)) + '\n Labvanced SEM = ' + str(lsem) + ' and for Eyelink SEM = ' + str(esem))
    plt.xlabel('x coordinates in pixel units',fontsize=16)
    plt.ylabel('y coordinates in pixel units',fontsize=16)
    
    plt.show()