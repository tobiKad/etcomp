from data_preprocessing import rawDataExport
from data_preprocessing import utility, calibration

# Convert all Raw Data:
print('Starting converting raw data')
rawDataExport.GazeFixationsExport()

print('Computing the calibration from Eyetrackers')
calibration.Calibration()

print('Running the gaze data synchronization')
## Here function

print('Running the Larger Grid Fixations analysis')
## Here FUNCTION

##Etc
