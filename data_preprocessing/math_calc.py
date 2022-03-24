import numpy as np
import pandas as pd


def calcEuclideanDistance(df_fixations, df_trial):
    df_trial.drop(columns=['Unnamed: 0'])
    df_trial = df_trial.reset_index()
    x1 = df_fixations.x
    # x1
    x2 = df_trial.targetX
    # x2
    y1 = df_fixations.y
    # y1
    y2 = df_trial.targetY
    # y2
    d2 = np.square( x2 - x1 )  + np.square( y2 - y1 ) 

    distances = np.sqrt( d2 )
    df_fixations['distance'] = distances
    df_fixations = df_fixations.dropna(subset=['distance'])
    return df_fixations

def calcOffsetFixation(eyetracker_type,df_trial, df_gaze):
    df_trial['offsetFixation_idx'] = -9999
    for index, row in df_trial.iterrows():
        target_offset = row['target_offset']
        
    
        beforePreviousFix = df_gaze.loc[df_gaze.ts < target_offset]
        
    # print(beforePreviousFix)
        
        offsetfixation = beforePreviousFix.iloc[-1]
        if(offsetfixation.tf > target_offset):
            
            # print("Last Fixation \n" + str(offsetfixation))
            # print("Next Fixation \n" + str(nextfixation))
            # print("Target offset" + str(target_offset))
            # nextfixation = df_gaze.loc[beforePreviousFix.index[-1] + 1]
            df_trial.loc[index, "offsetFixation_idx"] = beforePreviousFix.index[-1]

    df_trial.drop(df_trial.loc[df_trial['offsetFixation_idx']==-9999].index, inplace=True)           
    df_trial = df_trial.set_index("offsetFixation_idx")
    df_offsetfixation = pd.merge(df_gaze, df_trial, left_index=True, right_index=True)
    df_offsetfixation["x"] = df_offsetfixation['x'].apply(lambda x: int('%.0f' % x))
    df_offsetfixation["y"] = df_offsetfixation['y'].apply(lambda x: int('%.0f' % x))
    if(eyetracker_type=="lb" or eyetracker_type=="labvanced" or eyetracker_type=="Labvanced"):
        df_offsetfixation["disp"] = pd.to_numeric(df_offsetfixation["disp"])    
    df_offsetfixation = df_offsetfixation.reset_index()

    return df_offsetfixation
def addParticipantNumberCol(participant_number, df):
    df['Participant_Nr'] = participant_number
    return df

def formating_timeseries(df_gaze):
    df_gaze = df_gaze.rename({'value': 'x', 'Unnamed: 11': 'y','Unnamed: 12': 'dur','Unnamed: 13': 'ts','Unnamed: 14': 'tf','Unnamed: 15': 'disp'}, axis='columns')
    df_gaze = df_gaze[df_gaze['Task_Name'] == 'large_grid']
    df_gaze = df_gaze.sort_values(by ='timestamp' )
    df_gaze["ts"] = pd.to_numeric(df_gaze["ts"],downcast='integer')
    df_gaze["tf"] = pd.to_numeric(df_gaze["tf"],downcast='integer')
    df_gaze["x"] = pd.to_numeric(df_gaze["x"],downcast='integer')
    df_gaze["y"] = pd.to_numeric(df_gaze["y"],downcast='integer') 
    df_gaze = df_gaze.dropna()

    df_gaze["ts"] = df_gaze['ts'].apply(lambda x: int('%.0f' % x))
    df_gaze["tf"] = df_gaze['tf'].apply(lambda x: int('%.0f' % x))

    return df_gaze
def formating_trials(df_trial):
    df_trial = df_trial[df_trial['Task_Name'] == 'large_grid']
    df_trial.button_pressed = df_trial.button_pressed.apply(lambda x: int('%.0f' % x))
    df_trial["targetX"] = pd.to_numeric(df_trial["targetX"],downcast='integer')
    df_trial["targetY"] = pd.to_numeric(df_trial["targetY"],downcast='integer') 
    df_trial['target_offset'] = df_trial['StartFrame'] + df_trial['target_visibility_afterTime']
    df_trial['target_offset'] = df_trial['target_offset'].apply(lambda x: int('%.0f' % x))
    
    return df_trial

def formating_el_data(df):
    df= df.rename({'Start':'ts', 'End':'tf'}, axis='columns')
    return df
    