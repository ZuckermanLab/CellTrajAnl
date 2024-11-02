from matplotlib.patches import Patch
import sys, time, math
import matplotlib.pyplot as plt
import numpy as np
sys.path.append('/home/groups/ZuckermanLab/jalim/instalLocal/celltraj/celltraj')
import sctmCP_Dec15_2023 as cellTraj
import nucSCTM_CP as nucMsksCCR # To import reporter images, nuclear & cell masks
from csaps import csaps
import string, pickle 
from datetime import date
from skimage import io, measure
from IPython.display import clear_output
from matplotlib.animation import FuncAnimation, FFMpegWriter
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from sklearn.decomposition import PCA 

# Trajectory Length for morphodynamical trajectory analysis
trajl = 1 # to be adjusted, leave it for now 
today = date.today()
date2day = today.strftime("%b%d-%Y")
sysName = 'LI204601_P'
figid = sysName+'_tlen'+str(trajl)+'_'+date2day
try:
   conditions = [str(sys.argv[1])]
   tmSet = [str(sys.argv[2])]
   fovs = int(sys.argv[3])
except:
   print("Conditions couldn't be read!")
   sys.exit(0)

n_conditions = len(tmSet) # Total number of Ligand Conditions
inds_tmSet = [i for i in range(n_conditions)]
inds_tmSet = np.array(inds_tmSet).astype(int)
nfovs = 1
fovs = np.array(fovs).astype(int)
dateSet = ['']
objPathSet = ['/home/groups/ZuckermanLab/jalim/LI204601_INCUCYTE/segsCellPose/bayesianTrackTest/']
imagingSet = [0 for i in range(n_conditions)]
modelList = [None]*(nfovs*(n_conditions))
modelList_conditions = np.zeros(nfovs*(n_conditions)).astype(int)

i = 0
icond = 0
for cond in conditions:
    modelList_conditions[i] = icond
    modelList[i] = objPathSet[imagingSet[icond]]+sysName+'_'+cond+'_'+str(fovs)+dateSet[imagingSet[icond]]
    print("Models: ",modelList[i])
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
    inds_fovs_models[i] = fovs
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
all_trajSet = [None]*nmodels
# Get unique single-cell trajectories under different ligand conditions 
for i in indgood_models:
    modelSet[i].get_unique_trajectories()
    all_trajSet[i] = modelSet[i].trajectories.copy()

for i in indgood_models:
    modelSet[i].trajectories = all_trajSet[i].copy() # ALL Single-Cell trajectories

# Import single-cell trajectory model from cell-cycle reporter's class
sctm = nucMsksCCR.nucCellMasksCCRimgs()
sctm.visual = False; start_frame = 0
sysName = 'LI204601_G'
h5pathSet = '/home/groups/ZuckermanLab/jalim/LI204601_INCUCYTE/unNormReporterImgs/'

def unmask_data_from_images(image):
    if np.ma.is_masked(image): # If the image is a masked Numpy array
         data = image.data
    else: 
        data = image
    return image

def cross_correlation(cropped_reporter_image, cropped_nuclear_image):

     cropped_reporter_image = unmask_data_from_images(cropped_reporter_image)
     cropped_nuclear_image = unmask_data_from_images(cropped_nuclear_image)
     
     # Flatten the images if they are 2D
     if cropped_reporter_image.ndim == 2:
         cropped_reporter_image = cropped_reporter_image.flatten()
     if cropped_nuclear_image.ndim == 2:
         cropped_nuclear_image = cropped_nuclear_image.flatten()
 
     # Compute correlation coefficient
     correlation_coefficient = np.corrcoef(cropped_reporter_image, cropped_nuclear_image)[0, 1]
 
     return correlation_coefficient

def nuclei_within_cell_boundary(cropped_nuclear_mask, cropped_reporter_image,
                                cell_mask, hole_pixels = 3):
    # Apply closing to the nuclear mask 
    closed_nuclear_mask = closing(cropped_nuclear_mask, square(hole_pixels))

    # Label the regions in the closed nuclear mask
    labeled_nuclear_mask = label(closed_nuclear_mask)
    regions = regionprops(labeled_nuclear_mask, intensity_image = cropped_reporter_image)
    
    # Create a mask for only the nuclei within the cell boundary
    nuclei_within_cell_mask = np.zeros_like(cropped_nuclear_mask, dtype = bool)

    for region in regions:
        # Check if the labeled region centroid is within the cell mask
        if cell_mask[int(region.centroid[0]), int(region.centroid[1])]:
            # Combine masks using logical OR to keep the result as boolean
            nuclei_within_cell_mask = np.logical_or(nuclei_within_cell_mask, labeled_nuclear_mask == region.label)

    # Ensure the mask is a boolean type
    nuclei_within_cell_mask = nuclei_within_cell_mask.astype(bool)

    return nuclei_within_cell_mask

# Calculate Average Signal Intensity within Cytoplasm/Nuclear Regions
def mean_intensity(image_region):
    
    sum_value = np.nansum(image_region)
    count = np.sum(~np.isnan(image_region)).astype(int)
    if count == 0:
        mean_value = np.nan
    else:
        mean_value = sum_value / count
    
    return mean_value

def nucByCytoRatioSingleCell(cropped_reporter_image, nuclei_within_cell_mask, cell_mask):

    # Apply the nuclei mask to the reporter image
    nucleus_region = np.ma.masked_array(cropped_reporter_image, mask=np.logical_not(nuclei_within_cell_mask))
    cyto_region = np.ma.masked_array(cropped_reporter_image, mask=np.logical_and(nuclei_within_cell_mask, cell_mask))

    average_intensity_nuc = mean_intensity(nucleus_region)
    average_intensity_cyto = mean_intensity(cyto_region)

    if np.isfinite(average_intensity_nuc) and np.isfinite(average_intensity_cyto):
        nuc_by_cyto = average_intensity_nuc / average_intensity_cyto
    else:
        nuc_by_cyto = np.nan
        print("Cannot calculate Nuc/Cyto Ratio.")
    #print(f"Nucleus intensity: {average_intensity_nuc}, Cytoplasmic intensity: {average_intensity_cyto}, Ratio : {nuc_by_cyto}")
    
    return nuc_by_cyto, cyto_region, nucleus_region

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
     icount = 0
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
 
     cellborder_imgs[icount] = imgcell.copy()
     cellborder_msks[icount] = mskcell.copy()
     cellborder_fmsks[icount] = fmskcell.copy()
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
     icount = 0
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
         cellborder_imgs[icount] = imgcell.copy()
         cellborder_msks[icount] = mskcell.copy()
         cellborder_fmsks[icount] = fmskcell.copy()
         x_min[icount] = xmin.copy()
         x_max[icount] = xmax.copy()
         y_min[icount] = ymin.copy()
         y_max[icount] = ymax.copy()
         ip_file = self.cells_imgfileSet[ic]
         ip_frame = self.cells_frameSet[ic]
         icount += 1
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
    model = h5pathSet+sysName+'_'+conditions[inds_tmSet_models[im]]+'_'+str(fovs)+'.h5'
    fileCCinfo = sysName+'_'+conditions[inds_tmSet_models[im]]+'_'+str(fovs)+'.dat'
    print("Model name:", model)
    # Load cell cycle reporter and nuclear images and cytoplasm masks
    print('initializing...')
    sctm.initialize(model, sysName)
    # Get all the frames (193 at intervals of 15 minutes) for each model (condition)
    sctm.get_frames()
    end_frame = sctm.maxFrame
    # Get the cell-cycle & nuclear reporter images & nuclear masks
    sctm.get_imageSet(start_frame, end_frame)

    # Get frame indices where Microscopy imaging captures bad snaps & swap Cell-cycle and 
    # Nuclear channel images along with nuclear masks with the previous frame
    n_cells_prev = 0
    for frameid in range(nframes): 
        # If the number of cells is equal in two consecutive frames
        n_cells_curr = np.sum(modelSet[im].cells_frameSet == frameid)
        if n_cells_curr == n_cells_prev:
            sctm.imgSet[frameid] = sctm.imgSet[frameid-1] # Cell-cycle reporter images
            sctm.nmskSet[frameid] = sctm.nmskSet[frameid-1] # Nuclear reporter masks
        n_cells_prev = n_cells_curr
    
    # Set of all single-cell trajectories with absolute (irrespective of frames) cell indices
    num_single_cell_trajs = len(modelSet[im].trajectories)
    single_cell_traj_lengths = []
    file_out = open(fileCCinfo, 'a') 
    ##################### Calculate cross-correlations along all single-cell trajectories ######################
    for sctraj_indc in range(num_single_cell_trajs):
        cell_traj = modelSet[im].trajectories[sctraj_indc]
        len_single_cell_traj = cell_traj.size # Length of a single-cell trajectory or trajectory snippets
        # Cross-correlation between cell-cycle reporter & nuclear channel images along a single-cell trajectory
        frames_traj = []
        ratio_nbc = []
        cc_vals = []
        for itt in range(len_single_cell_traj):
          
            indcells = np.array([cell_traj[itt]]) # Cell indices along a single-cell trajectory
            frame_indc = modelSet[im].cells_frameSet[indcells[0]] # Frame index 
            get_single_cell_borders(modelSet[im], indcells = indcells[0], bordersize = border_margin)
            imgcell = modelSet[im].cellborder_imgs[0]
            mskcell = modelSet[im].cellborder_msks[0]
            fmskcell = modelSet[im].cellborder_fmsks[0]
            # Get cell-cell (CC) and cell-surroundings (CS) borders
            ccborder, csborder = modelSet[im].get_cc_cs_border(mskcell, fmskcell) 

            # Combine border information with cell interior to create a full mask for the entire cell
            cell_mask = np.logical_or(np.logical_or(ccborder, csborder), mskcell)

            reporter_image = sctm.imgSet[frame_indc]
            nuclear_image = sctm.nucImgSet[frame_indc]
            nuclear_mask = sctm.nmskSet[frame_indc]
 
            reporter_image = np.array(reporter_image)
            nuclear_image = np.array(nuclear_image)

            x_min, x_max = modelSet[im].xmin, modelSet[im].xmax
            y_min, y_max = modelSet[im].ymin, modelSet[im].ymax

            # Crop reporter images, nuclear images, and masks according to single-cell positions
            cropped_reporter_image = reporter_image[x_min:x_max, y_min:y_max]
            cropped_nuclear_image = nuclear_image[x_min:x_max, y_min:y_max]
            cropped_nuclear_mask = nuclear_mask[x_min:x_max, y_min:y_max]
            
            # Mask out (ignore) the pixels that don't fit within the cell boundary
            cropped_reporter_image = np.ma.masked_where(cell_mask == 0, cropped_reporter_image)
            cropped_nuclear_image = np.ma.masked_where(cell_mask == 0, cropped_nuclear_image)
            
            # Calculate cross-correlation between cell-cycle reporter and nuclear channel images 
            cc_frames = cross_correlation(cropped_reporter_image, cropped_nuclear_image)
            # Get Nuclei that are within the boundary of cell cytoplasm 
            nuclei_within_cell_mask = nuclei_within_cell_boundary(cropped_nuclear_mask, cropped_reporter_image,
                                                                  cell_mask, hole_pixels = 3)
            # Compute nuc/Cyto ratio and return those regions along with it 
            nuc_by_cyto = nucByCytoRatioSingleCell(cropped_reporter_image, nuclei_within_cell_mask, cell_mask)

            cc_vals.append(cc_frames)
            ratio_nbc.append(nuc_by_cyto)
            frames_traj.append(frame_indc)

        #print("Cross correlations: ",cross_corr, "Frames along a single-cell trajectory: ",frames_traj)
        file_out.write('[' + array_to_string(ratio_nbc) + ']\n')
        file_out.write('[' + array_to_string(cc_vals) + ']\n')
        file_out.write('[' + array_to_string(frames_traj) + ']\n')
    file_out.close()
