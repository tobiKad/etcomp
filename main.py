from data_preprocessing import rawDataExport, lastFixLB
from data_preprocessing import utility, calibration
from data_processing import fixationAnalysisLG

# # Convert all Raw Data:
# print('Starting converting raw data interpolation and sync')
# rawDataExport.GazeFixationsExport()

# print('Computing the calibration from Eyetrackers')
# calibration.Calibration()

print('Creating last fixation for boht Eyetrackers' )
lastFixLB.lastFixLB()

print('Running the Larger Grid Fixations analysis')
## Here FUNCTION

##Etc
