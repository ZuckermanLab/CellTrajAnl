import numpy as np
import matplotlib.pyplot as plt
import ast, time
from IPython.display import clear_output

conditions = ['A1', 'A2', 'A3', 'A4', 'A5', 
              'B1', 'B2', 'B3', 'B4', 'B5', 
              'C1', 'C2', 'C3']
ligands = ['OSM1','EGF1','EGFTGFB1','TGFB1','PBS1',
           'OSM2','EGF2','EGFTGFB2','TGFB2','PBS2',
           'OSMEGFTGFB','OSMEGF','OSMTGFB']
nfovs = 4
fovs = [i for i in range(1, nfovs+1)]
fovs = np.array(fovs).astype(int)
# Filename with path
filePath = '/home/groups/ZuckermanLab/jalim/LI204601_INCUCYTE/crossCorrCCRnuc/'
sysName = 'LI204601_P'

# Cross correlations at all frame numbers (time points)
def get_cross_corr_all_frames(filename):
    cross_corr_data = []
    with open(filename, 'r') as file_in:
         for line in file_in:
             # Remove any leading/trailing whitespace
             line = line.strip()
             # If the line is not empty, convert it to a list and add to arrays
             if line:
                data = ast.literal_eval(line)
                cross_corr_data.append(data)
    corr_all_trajs = []
    for fi in range(0, len(cross_corr_data), 2):
        cross_corr = np.array(cross_corr_data[fi])
        corr_all_trajs.append(cross_corr)
    flat_cross_corr = np.concatenate(corr_all_trajs)
    return flat_cross_corr

# Cross correlations corresponding to a selected frame number (time point)
def get_cross_corr_single_time_point(filename, frame_id):
    cross_corr_single_frame = []
    with open(filename, 'r') as file_in:
        # Create an iterator over the file lines 
        file_iter = iter(file_in)
        try:
            while True:
                # Read two lines at a time: Cross correlations & the corresponding frame numbers
                line1 = next(file_iter).strip()
                line2 = next(file_iter).strip()
                # Parse line2 as a list of integers
                frame_numbers = ast.literal_eval(line2)
                if frame_id in frame_numbers:  # if the desired frame id is present 
                    index = frame_numbers.index(frame_id)  # Find the index of frame_id in frame_numbers
                    if line1:
                        data = ast.literal_eval(line1)
                        cross_corr_single_frame.append(data[index])
        except StopIteration:
            # End of file reached
            pass
    corr_all_trajs_single_frame = np.array(cross_corr_single_frame)
    return corr_all_trajs_single_frame

#Calculate average cross-correlations at different time points (frames) for all ligand conditions
ncond = len(conditions)

time_hours = [i for i in range(1, 49)]
time_hours = np.array(time_hours).astype(int)
frames = time_hours * 4 # Since the Data is collected every 15 minutes till 48 hours

for ic, cond in enumerate(conditions):
    corr_out_file = sysName+"_crossCorrRepNuc_"+ligands[ic]+".txt"
    with open(corr_out_file, 'a') as corr_out: 
         for i, frame_id in enumerate(frames):
             cross_corr_frame = [] # Store values averaged over all field of views
             for j, fov in enumerate(fovs):
                 filename = filePath+sysName+'_'+cond+'_'+str(fov)+'.dat'
                 cross_corr_model = get_cross_corr_single_time_point(filename, frame_id)
                 # Average cross-correlation for a selected frame and field of view
                 cross_corr_fov = np.nanmean(cross_corr_model)
                 cross_corr_frame.append(cross_corr_fov)
             corr_out.write(frame_id, np.nanmean(cross_corr_frame))
