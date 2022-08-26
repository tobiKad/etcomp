from data_preprocessing import rawDataExport, lastFixLB, headRawDataExport, spPreProcessing, fvPreProcessing
from data_processing import largeGridPrec, largeGridAcc, smoothPursuit, freeView, headMovements

print('Starting converting raw data interpolation and sync')
rawDataExport.GazeFixationsExport()

print('Working on the Head Data')
headRawDataExport.HeadExporter()

print('Creating last fixation for both Eyetrackers and data formating')
lastFixLB.lastFixLB()

print('Running the Larger Grid Fixations analysis')
largeGridAcc.LargeGridAcc()
largeGridPrec.LargeGridPrec()

print('Running the Smooth Pursuit analysis')
spPreProcessing.SpPre()
smoothPursuit.SmoothPursuit()

print('Running Free View Analysis')
fvPreProcessing.FvPre()
freeView.FreeView()

print('Running  Head movments analysis')
headMovements.HeadMovements()
