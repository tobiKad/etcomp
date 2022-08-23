from data_preprocessing import rawDataExport, lastFixLB, headRawDataExport
from data_preprocessing import utility, calibration
from data_processing import largeGridPrec, largeGridAcc, smoothPursuit
# from data_processing import fixationAnalysisLG

# # # Convert all Raw Data:
# print('Starting converting raw data interpolation and sync')
# rawDataExport.GazeFixationsExport()

# # # Head data exporting and interpolation
# print('Working on the Head Data')
# headRawDataExport.HeadExporter()

# print('Creating last fixation for both Eyetrackers and data formating')
# lastFixLB.lastFixLB()

print('Running the Larger Grid Fixations analysis')
# largeGridAcc.LargeGridAcc()
largeGridPrec.LargeGridPrec()
# largeGridPrecAnova.LargeGridPrecANOVA()

print('Running the Smooth Pursuit analysis')
# smoothPursuit.SmoothPursuit()

##Etc
