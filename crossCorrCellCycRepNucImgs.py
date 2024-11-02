from matplotlib.patches import Patch
import sys, time, math
import matplotlib.pyplot as plt
import numpy as np
sys.path.append('/home/groups/ZuckermanLab/jalim/instalLocal/celltraj/celltraj')
import sctmCP_Dec15_2023 as cellTraj
import cellCycRepJCtrajsCP as nucMsksCCR # To import reporter & nuclear images along with cell masks
#import nucSCTM_CP as nucMsksCCR # To import reporter & nuclear images along with nuclear masks
from csaps import csaps
import string, pickle 
from datetime import date
from skimage import io, measure
from IPython.display import clear_output
from matplotlib.animation import FuncAnimation, FFMpegWriter
from skimage.measure import label, regionprops
from skimage.morphology import closing, square

# Trajectory Length for morphodynamical trajectory analysis
trajl = 1 # to be adjusted, leave it for now 
today = date.today()
date2day = today.strftime("%b%d-%Y")
sysName = 'LI204601_P'
figid = sysName+'_tlen'+str(trajl)+'_'+date2day
conditions = ['A1','A2','A3','A4','A5','B1','B2','B3','B4','B5','C1','C2','C3'] # LIGANDS (CONDITIONS)
tmSet = ['OSM1','EGF1','EGF+TGFB1','TGFB1','PBS1','OSM2','EGF2','EGF+TGFB2','TGFB2','PBS2',
         'OSM+EGF+TGFB','OSM+EGF','OSM+TGFB']
#conditions = ['A2'] # LIGANDS (CONDITIONS)
#tmSet = ['EGF1']
nConditions = len(tmSet) # Total number of Ligand Conditions
inds_tmSet = [i for i in range(nConditions)]
inds_tmSet = np.array(inds_tmSet).astype(int)
nfovs = 4
fovs = [i for i in range(1, nfovs + 1)]
fovs = np.array(fovs).astype(int)
dateSet = ['']
#pathSet = ['/home/groups/ZuckermanLab/jalim/LI204601_INCUCYTE/segsCellPose/bayesianTrackTest/znormRegPhaseImgs/']
pathSet = ['/home/groups/ZuckermanLab/jalim/LI204601_INCUCYTE/segsCellPose/bayesianTrackTest/']
imagingSet = [0 for i in range(nConditions)]
modelList = [None]*(nfovs*(nConditions))
modelList_conditions = np.zeros(nfovs*(nConditions)).astype(int)

i = 0
icond = 0
for cond in conditions:
    for fov in fovs:
        modelList_conditions[i] = icond
        modelList[i] = pathSet[imagingSet[icond]]+sysName+'_'+cond+'_'+str(fov)+dateSet[imagingSet[icond]]
        #print("Models: ",modelList[i])
        i = i + 1
    icond = icond + 1

nmodels = len(modelList)
modelSet = [None]*nmodels
indgood_models = np.array([]).astype(int)

for i in range(nmodels):
    try:
        objFile = modelList[i]+'.obj'
        objFileHandler = open(objFile,'rb')
        modelSet[i] = pickle.load(objFileHandler)
        print('loaded '+objFile+' with '+str(modelSet[i].cells_indSet.size)+' cells')
        objFileHandler.close()
        test = len(modelSet[i].linSet)
        indgood_models = np.append(indgood_models, i)
    except:
        print("ERROR in reading *.obj files")
        sys.exit(0)

# Total number of frames (image snapshots) in one condition per FOVs
nframes = 193 
cellnumber_stdSet = np.ones(nmodels)*np.inf
# range of frame indices where cell numbers are higher: ~70-98%
sframe = 70.*nframes/100.; sframe = math.ceil(sframe)
eframe = 98.5*nframes/100.; eframe = math.ceil(eframe)
cellnumber_frames = np.arange(sframe, eframe).astype(int)
cellnumber_std_cut = .50 # This was set to 0.10 by Jeremy 
frames = np.arange(nframes)
# Abscissas at which smoothing will be done using CSAPS package
abSmooth = np.linspace(frames[0], frames[-1], 10000)

for i in indgood_models:
    ncells = np.zeros(nframes)
    ncells_smooth = np.zeros_like(ncells)
    for iS in range(nframes):
        ncells[iS]=np.sum(modelSet[i].cells_frameSet == iS)
    # Cubic Spline Approximation (CSAPS) to smoothen the data
    splfov = csaps(frames, ncells/ncells[0], abSmooth, smooth = 0.98) # Scaled by ncells[0] to avoid large numbers
    ncells_smooth = splfov*ncells[0] # smoothened cell numbers reverse scaled back to original
    cellnumber_std = np.std(ncells[cellnumber_frames] - ncells_smooth[cellnumber_frames])/np.mean(ncells[cellnumber_frames])
    cellnumber_stdSet[i] = cellnumber_std # Standard Deviation in Cell Numbers

indhigh_std = np.where(cellnumber_stdSet > cellnumber_std_cut)[0]
indgood_models = np.setdiff1d(indgood_models, indhigh_std)

nf = len(tmSet)
inds_fovs_models = np.zeros(nmodels).astype(int)
inds_tmSet_models = np.zeros(nmodels).astype(int)
inds_imagingSet_models = np.zeros(nmodels).astype(int)
i = 0
icond = 0
for cond in conditions:
    for fov in fovs:
        inds_fovs_models[i] = fov
        inds_tmSet_models[i] = inds_tmSet[icond]
        inds_imagingSet_models[i] = imagingSet[icond]
        i = i + 1
    icond = icond + 1

for i in indgood_models:
    if inds_imagingSet_models[i] == 0:
        modelSet[i].Xf[np.isnan(modelSet[i].Xf)] = 0.0 #just replace with zeros for now? Not sure best...

nfeat_com = 3
Xf_com0 = np.zeros((0, nfeat_com))
for i in indgood_models:
    if inds_imagingSet_models[i] == 0:
        Xf_com0 = np.append(Xf_com0,modelSet[i].Xf_com, axis = 0)

av_dx = np.nanmean(Xf_com0[:, 0])
std_dx = np.nanstd(Xf_com0[:, 0])
for i in indgood_models:
    modelSet[i].Xf_com[:, 0] = (modelSet[i].Xf_com[:, 0] - av_dx)/std_dx

wctm = cellTraj.Trajectory() # import Trajectory object 

nfeat = modelSet[indgood_models[0]].Xf.shape[1]
Xf = np.zeros((0, nfeat))
indtreatment = np.array([])
indcellSet = np.array([])
for i in indgood_models:
    if inds_imagingSet_models[i] == 0:
        Xf = np.append(Xf, modelSet[i].Xf, axis = 0)
        indtreatment = np.append(indtreatment, i*np.ones(modelSet[i].Xf.shape[0]))
        indcellSet = np.append(indcellSet, modelSet[i].cells_indSet)

indtreatment = indtreatment.astype(int)
indcellSet = indcellSet.astype(int)
# Use the sklearn package (intended for ease of use over performance/scalability)
varCutOff = 10 
from sklearn.decomposition import PCA 
# n_components specifies the number of principal components to extract from the covariance matrix
pca = PCA(n_components = varCutOff) 
pca.fit(Xf) # builds the covariance matrix and "fits" the principal components
Xpca = pca.transform(Xf) # transforms the data into the pca representation
nPCs = Xpca.shape[1]

wctm.Xpca = Xpca
wctm.pca = pca
for i in indgood_models:
    if inds_imagingSet_models[i] == 0:
        indsf = np.where(indtreatment == i)[0]
        modelSet[i].Xpca = Xpca[indsf, :]

indgood_models = indgood_models[np.where(inds_imagingSet_models[indgood_models] == 0)[0]]

self = wctm
#wctm.trajl = trajl
all_trajSet = [None]*nmodels
# Get unique single-cell trajectories under different ligand conditions 
for i in indgood_models:
    modelSet[i].get_unique_trajectories()

# Import single-cell trajectory model from cell-cycle reporter' class
#sctm = nucMsksCCR.nucCellMasksCCRimgs()
sctm = nucMsksCCR.TrajectoryCCR()
sctm.visual = False
start_frame = 0
sysName = 'LI204601_P'
filepath = '/home/groups/ZuckermanLab/jalim/LI204601_INCUCYTE/crossCorrCCRnuc/'

def cross_correlation_image_regions(cropped_reporter_image, cropped_nuclear_image):

    # Flatten the images if they are 2D
    if cropped_reporter_image.ndim == 2:
        cropped_reporter_image = cropped_reporter_image.flatten()
    if cropped_nuclear_image.ndim == 2:
        cropped_nuclear_image = cropped_nuclear_image.flatten()
        
    # Compute correlation coefficient
    correlation_coefficient = np.corrcoef(cropped_reporter_image, cropped_nuclear_image)[0, 1]
    
    return correlation_coefficient

# Function to Z-normalize masked and non-masked images
def z_normalize(image):
    
    if np.ma.is_masked(image):
        # If the image is a masked array
        data = image.data
        mask = image.mask

        # Perform z-normalization on the data
        normalized_data = (data - np.nanmean(data)) / np.nanstd(data)

        # Create a new masked array with the normalized data and the original mask
        return np.ma.masked_array(normalized_data, mask = mask)
    else:
        # If the image is a regular array
        return (image - np.nanmean(image)) / np.nanstd(image)

# Function to convert array to a string format
def array_to_string(array):
     #print("Array within array_to_string(): ",array)
     if isinstance(array, (list, np.ndarray)):
         return ', '.join(map(str, array))
     elif isinstance(array, np.float64):
         return str(array)
     elif isinstance(array, np.int64):
         return str(array)
     else:
         raise TypeError("Input is not a list or np.ndarray")

def get_single_cell_borders(self, indcells=None, bordersize=10):
     if not hasattr(self,'fmskSet'):
            print("Foreground masks not found: provide or derive them")
            sys.exit(0)

     nx = self.imgSet.shape[1]
     ny = self.imgSet.shape[2];
     if indcells is None:
        indcells = np.arange(self.cells_indSet.size).astype(int)
     ncells = indcells.size
     cellborder_imgs = [None]*ncells
     cellborder_msks = [None]*ncells
     cellborder_fmsks = [None]*ncells
     ii = 0
     ic = indcells
     sys.stdout.write('extracting cellborders from frame '+str(self.cells_frameSet[ic])+' image '+str(self.cells_imgfileSet[ic])+'\n')
     img = self.imgSet[self.cells_indimgSet[ic]]
     msk = self.mskSet[self.cells_indimgSet[ic]]
     fmsk = self.fmskSet[self.cells_indimgSet[ic]]
     cellblocks = self.get_cell_blocks(msk)

     xmin = np.max(np.array([cellblocks[self.cells_indSet[ic], 0, 0] - bordersize, 0]))
     xmax = np.min(np.array([cellblocks[self.cells_indSet[ic], 0, 1] + bordersize, nx - 1]))
     ymin = np.max(np.array([cellblocks[self.cells_indSet[ic], 1, 0] - bordersize, 0]))
     ymax = np.min(np.array([cellblocks[self.cells_indSet[ic], 1, 1] + bordersize, ny - 1]))
 
     imgcell = img[xmin:xmax, :]
     imgcell = imgcell[:, ymin:ymax]
     mskcell = msk[xmin:xmax, :]
     mskcell = mskcell[:, ymin:ymax]
     fmskcell = fmsk[xmin:xmax, :]
     fmskcell = fmskcell[:, ymin:ymax]

     tightmskcell = msk[cellblocks[self.cells_indSet[ic], 0, 0]:cellblocks[self.cells_indSet[ic], 0, 1], :]
     tightmskcell = tightmskcell[:, cellblocks[self.cells_indSet[ic], 1, 0]:cellblocks[self.cells_indSet[ic], 1, 1]]
     (values, counts) = np.unique(tightmskcell[np.where(tightmskcell > 0)], return_counts = True)
     icell = values[np.argmax(counts)].astype(int)
     mskcell = mskcell == icell
 
     cellborder_imgs[ii] = imgcell.copy()
     cellborder_msks[ii] = mskcell.copy()
     cellborder_fmsks[ii] = fmskcell.copy()
     self.cellborder_imgs = cellborder_imgs
     self.cellborder_msks = cellborder_msks
     self.cellborder_fmsks = cellborder_fmsks
 
     self.xmin = xmin
     self.ymin = ymin
     self.xmax = xmax
     self.ymax = ymax

# Get borders of all cells in an image
def get_cellborders_image(self, indcells=None, bordersize=10):
          if not hasattr(self,'fmskSet'):
             print("Foreground masks not found: provide or derive them")
             sys.exit(0)
          nx = self.imgSet.shape[1]
          ny = self.imgSet.shape[2];
          if indcells is None:
              indcells = np.arange(self.cells_indSet.size).astype(int)
          ncells = indcells.size
          cellborder_imgs = [None]*ncells
          cellborder_msks = [None]*ncells
          cellborder_fmsks = [None]*ncells
          x_min = [None]*ncells; x_max = [None]*ncells
          y_min = [None]*ncells; y_max = [None]*ncells
          ip_frame = 100000
          ip_file = 100000
          ii = 0
          for ic in indcells:
              if not self.cells_imgfileSet[ic] == ip_file or not self.cells_frameSet[ic] == ip_frame:
                  sys.stdout.write('extracting cellborders from frame '+str(self.cells_frameSet[ic])+' image '+str(self.cells_imgfileSet[ic])+'\n')
                  img = self.imgSet[self.cells_indimgSet[ic]]
                  msk = self.mskSet[self.cells_indimgSet[ic]]
                  fmsk = self.fmskSet[self.cells_indimgSet[ic]]
                  cellblocks = self.get_cell_blocks(msk)
              xmin = np.max(np.array([cellblocks[self.cells_indSet[ic], 0, 0] - bordersize, 0]))
              xmax = np.min(np.array([cellblocks[self.cells_indSet[ic], 0, 1] + bordersize, nx - 1]))
              ymin = np.max(np.array([cellblocks[self.cells_indSet[ic], 1, 0] - bordersize, 0]))
              ymax = np.min(np.array([cellblocks[self.cells_indSet[ic], 1, 1] + bordersize, ny - 1]))
              imgcell = img[xmin:xmax, :]
              imgcell = imgcell[:, ymin:ymax]
              mskcell = msk[xmin:xmax, :]
              mskcell = mskcell[:, ymin:ymax]
              fmskcell = fmsk[xmin:xmax, :]
              fmskcell = fmskcell[:, ymin:ymax]
              tightmskcell = msk[cellblocks[self.cells_indSet[ic], 0, 0]:cellblocks[self.cells_indSet[ic], 0, 1], :]
              tightmskcell = tightmskcell[:, cellblocks[self.cells_indSet[ic], 1, 0]:cellblocks[self.cells_indSet[ic], 1, 1]]
              (values, counts) = np.unique(tightmskcell[np.where(tightmskcell > 0)], return_counts = True)
              icell = values[np.argmax(counts)].astype(int)
              mskcell = mskcell == icell
              cellborder_imgs[ii] = imgcell.copy()
              cellborder_msks[ii] = mskcell.copy()
              cellborder_fmsks[ii] = fmskcell.copy()
              x_min[ii] = xmin.copy()
              x_max[ii] = xmax.copy()
              y_min[ii] = ymin.copy()
              y_max[ii] = ymax.copy()
              ip_file = self.cells_imgfileSet[ic]
              ip_frame = self.cells_frameSet[ic]
              ii += 1
          self.cellborder_imgs = cellborder_imgs
          self.cellborder_msks = cellborder_msks
          self.cellborder_fmsks = cellborder_fmsks
          self.cellborder_inds = indcells.copy()
          self.x_min = x_min
          self.x_max = x_max
          self.y_min = y_min
          self.y_max = y_max

border_margin = 2

for im in indgood_models:
    model = filepath+sysName+'_'+conditions[inds_tmSet_models[im]]+'_'+str(inds_fovs_models[im])+'.h5'
    fileCC_dump = sysName+'_'+conditions[inds_tmSet_models[im]]+'_'+str(inds_fovs_models[im])+'.dat'
    print("Model name:", model)
    # Load cell cycle reporter and nuclear images and cytoplasm masks
    print('initializing...')
    sctm.initialize(model, sysName)
    # Get all the frames (193 at intervals of 15 minutes) for each model (condition)
    sctm.get_frames()
    end_frame = sctm.maxFrame
    # Get the cell-cycle reporter image, cytoplasmic & nuclear masks
    sctm.get_imageSet(start_frame, end_frame)
    sctm.get_nucImgSet(start_frame, end_frame)

    # Get frame indices where Microscopy imaging captures bad snaps & swap Cell-cycle and 
    # Nuclear channel images along with nuclear masks with the previous frame
    n_cells_prev = 0
    for frameid in range(nframes): 
        # If the number of cells is equal in two consecutive frames
        n_cells_curr = np.sum(modelSet[im].cells_frameSet == frameid)
        if n_cells_curr == n_cells_prev:
            sctm.imgSet[frameid] = sctm.imgSet[frameid-1]
            sctm.nucImgSet[frameid] = sctm.nucImgSet[frameid-1]
            #sctm.nmskSet[frameid] = sctm.nmskSet[frameid-1] 
        n_cells_prev = n_cells_curr
    
    # Set of all single-cell trajectories with absolute (irrespective of frames) cell indices
    num_single_cell_trajs = len(modelSet[im].trajectories)
    single_cell_traj_lengths = []
    for itraj in range(num_single_cell_trajs):
        single_cell_traj_inds = modelSet[im].trajectories[itraj] # Indices of single-cell trajectory across frames
        single_cell_traj_length = len(single_cell_traj_inds) # Length of a single-cell trajectory
        single_cell_traj_lengths.append(single_cell_traj_length)  
    indtrajs = np.argsort(single_cell_traj_lengths) # Length ordered array of unique single-cell trajectories 
    file_out = open(fileCC_dump, 'a') 
    ##################### Calculate cross-correlations along all single-cell trajectories ######################
    for tid in range(1, len(indtrajs) + 1):
        cell_traj = modelSet[im].trajectories[indtrajs[-tid]]
        len_single_cell_traj = cell_traj.size # Length of a single-cell trajectory or trajectory snippets
        # Cross-correlation between cell-cycle reporter & nuclear channel images along a single-cell trajectory
        frames_traj = []
        cross_corr = []
        for itt in range(len_single_cell_traj):
          
            indcells = np.array([cell_traj[itt]]) # Cell indices along a single-cell trajectory
            fid = modelSet[im].cells_frameSet[indcells[0]] # Frame index 
            #modelSet[im].get_single_cell_borders(indcells = indcells[0], bordersize = border_margin)
            get_single_cell_borders(modelSet[im], indcells = indcells[0], bordersize = border_margin)
            imgcell = modelSet[im].cellborder_imgs[0]
            mskcell = modelSet[im].cellborder_msks[0]
            fmskcell = modelSet[im].cellborder_fmsks[0]
            # Get cell-cell (CC) and cell-surroundings (CS) borders
            ccborder, csborder = modelSet[im].get_cc_cs_border(mskcell, fmskcell) 

            # Combine border information with cell interior to create a full mask for the entire cell
            cell_mask = np.logical_or(np.logical_or(ccborder, csborder), mskcell)
    
            reporter_image = sctm.imgSet[fid]
            nuclear_image = sctm.nucImgSet[fid]
            #nuclear_mask = sctm.nmskSet[fid]
 
            reporter_image = np.array(reporter_image)
            nuclear_image = np.array(nuclear_image)

            x_min, x_max = modelSet[im].xmin, modelSet[im].xmax
            y_min, y_max = modelSet[im].ymin, modelSet[im].ymax

            # Crop reporter images, nuclear images, and masks according to single-cell positions
            cropped_reporter_image = reporter_image[x_min:x_max, y_min:y_max]
            cropped_nuclear_image = nuclear_image[x_min:x_max, y_min:y_max]
            #cropped_nuclear_mask = nuclear_mask[x_min:x_max, y_min:y_max]

            # Mask out (ignore) the pixels that don't fit within the cell boundary
            cropped_reporter_image = np.ma.masked_where(cell_mask == 0, cropped_reporter_image)
            cropped_nuclear_image = np.ma.masked_where(cell_mask == 0, cropped_nuclear_image)
    
            # Z-normalize the area of interest within the images 
            cropped_reporter_image = z_normalize(cropped_reporter_image)
            cropped_nuclear_image = z_normalize(cropped_nuclear_image)
    
            # Cross-correlation between a masked cell region of cell-cycle reporter and nuclear channel images 
            cross_corr_frame = cross_correlation_image_regions(cropped_reporter_image, cropped_nuclear_image)
            # If cross-correlation is Zero (due to bad microscopy imaging), assign it from the previous frame
            if cross_corr_frame == 0.0: 
               cross_corr_frame = cross_corr[fid-1]
              
            cross_corr.append(cross_corr_frame)
            frames_traj.append(fid)

        #print("Cross correlations: ",cross_corr, "Frames along a single-cell trajectory: ",frames_traj)
        file_out.write('[' + array_to_string(cross_corr) + ']\n')
        file_out.write('[' + array_to_string(frames_traj) + ']\n')
    file_out.close()
#fig, axs = plt.subplots(1, 2, figsize = (7, 4))
#fig.tight_layout()  # Adjust subplots to fit into the figure area

############ Find the longest single-cell trajectory ############
def find_longest_single_cell_trajs():

    longest_trajectory = max(single_cell_traj_lengths) 
    if longest_trajectory > nframes: 
        print("Error: longest single-cell trajectory can't exceed total number of frames")
        sys.exit(0)
    #print("Longest single-cell trajectory length: ",longest_trajectory)
    ind_longest_trajectory = np.argmax(single_cell_traj_lengths) # Index of the longest single-cell trajectory
    #print("Index of longest single-cell trajectory: ",ind_longest_trajectory)
    cell_inds_longest_traj = modelSet[i].trajectories[ind_longest_trajectory] # Cell indices
    #print("Cell indices of the longest single-cell trajectory: ",cell_inds_longest_traj)
    return ind_longest_trajectory, cell_inds_longest_traj
