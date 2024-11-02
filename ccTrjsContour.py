####### Decompose CC values along single-cell trajectories onto Micro (Fine) States ########
# Note: CC trajectories (unsorted) are imported directly from files
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sys, os, time, math
sys.path.append('/home/groups/ZuckermanLab/jalim/instalLocal/celltraj/celltraj')
import jcTrajectory_CP as cellTraj
import h5py, pickle, subprocess
import umap, scipy
from csaps import csaps
import string, ast 
from joblib import dump, load
from datetime import date
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Trajectory Length for morphodynamical trajectory analysis
try:
   trajl = int(sys.argv[1])
except ValueError:
   print("Error in getting trajectory snippet length for morphodynamical analysis")
   sys.exit(0)
try:
   wells_flg = int(sys.argv[2]) # Flag to import data of a certain wells combinations
except ValueError:
   print("Wells flag is not given OR is not a valid integer")
   sys.exit(0)

if(wells_flg == 0):
  wellsInfo = 'Awells'
  conditions = ['A1','A2','A3','A4','A5','C1','C2','C3'] # LIGANDS (CONDITIONS) - A & Combination Wells
  tmSet = ['OSM1','EGF1','EGF+TGFB1','TGFB1','PBS1','OSM+EGF+TGFB','OSM+EGF','OSM+TGFB']
elif(wells_flg == 1):
  wellsInfo = 'Bwells'
  conditions = ['B1','B2','B3','B4','B5','C1','C2','C3'] # LIGANDS (CONDITIONS) - B & Combination Wells
  tmSet = ['OSM2','EGF2','EGF+TGFB2','TGFB2','PBS2','OSM+EGF+TGFB','OSM+EGF','OSM+TGFB']
else:
  wellsInfo = 'AllWells'
  conditions = ['A1','A2','A3','A4','A5','B1','B2','B3','B4','B5','C1','C2','C3'] # LIGANDS (CONDITIONS) - A, B, and Combination Wells
  tmSet = ['OSM1','EGF1','EGF+TGFB1','TGFB1','PBS1','OSM2','EGF2','EGF+TGFB2','TGFB2',
           'PBS2','OSM+EGF+TGFB','OSM+EGF','OSM+TGFB']

n_conditions = len(tmSet) # Total number of Ligand Conditions

today = date.today()
date2day = today.strftime("%b%d-%Y")
sysName = 'LI204601_P'
figid = sysName+'_tl'+str(trajl)+wellsInfo+'_'+date2day

# Indices for the ligands 
inds_tmSet = [i for i in range(n_conditions)]
inds_tmSet = np.array(inds_tmSet).astype(int)
nfovs = 4
fovs = [i for i in range(1, nfovs + 1)]
fovs = np.array(fovs).astype(int)
dateSet = ['']
pathSet = ['/home/groups/ZuckermanLab/jalim/LI204601_INCUCYTE/segsCellPose/bayesianTrackTest/']
imagingSet = [0 for i in range(n_conditions)]
modelList = [None]*(nfovs*(n_conditions))
modelList_conditions = np.zeros(nfovs*(n_conditions)).astype(int)

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
        indgood_models = np.append(indgood_models,i)
    except:
        print("ERROR in reading *.obj files")
        sys.exit(0)

# Total number of frames (image snapshots) in one condition per FOVs
n_frames = 193 
cellnumber_stdSet = np.ones(nmodels)*np.inf
# range of frame indices where cell numbers are higher: ~70-98%
sframe = 70.*n_frames/100.; sframe = math.ceil(sframe)
eframe = 98.5*n_frames/100.; eframe = math.ceil(eframe)
cellnumber_frames = np.arange(sframe, eframe).astype(int)
cellnumber_std_cut = .50 # This was set to 0.10 by Jeremy 
frames = np.arange(n_frames)
# Abscissas at which smoothing will be done using CSAPS package
abSmooth = np.linspace(frames[0], frames[-1], 10000)

for i in indgood_models:
    ncells = np.zeros(n_frames)
    ncells_smooth = np.zeros_like(ncells)
    for iS in range(n_frames):
        ncells[iS] = np.sum(modelSet[i].cells_frameSet == iS)
    # Cubic Spline Approximation (CSAPS) to smoothen the data
    splfov = csaps(frames, ncells/ncells[0], abSmooth, smooth = 0.98) # Scaled by ncells[0] to avoid large numbers
    ncells_smooth = splfov*ncells[0] # smoothened cell numbers reverse scaled back to original
    cellnumber_std = np.std(ncells[cellnumber_frames] - ncells_smooth[cellnumber_frames])/np.mean(ncells[cellnumber_frames])
    cellnumber_stdSet[i] = cellnumber_std # Standard Deviation in Cell Numbers
indhigh_std = np.where(cellnumber_stdSet > cellnumber_std_cut)[0]
indgood_models = np.setdiff1d(indgood_models, indhigh_std)

# get cell counts
n_conds = len(tmSet)
inds_tmSet_models = np.zeros(nmodels).astype(int)
inds_imagingSet_models = np.zeros(nmodels).astype(int)
i = 0
icond = 0
for cond in conditions:
    for fov in fovs:
        inds_tmSet_models[i] = inds_tmSet[icond] # Assign indices "tmSet" to all FOVs
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
        # Indices for each model for later access using them
        indtreatment = np.append(indtreatment, i*np.ones(modelSet[i].Xf.shape[0])) 
        indcellSet = np.append(indcellSet, modelSet[i].cells_indSet)

indtreatment = indtreatment.astype(int)
indcellSet = indcellSet.astype(int)

varCutOff = 10
from sklearn.decomposition import PCA #we will use the sklearn package (intended for ease of use over performance/scalability)
pca = PCA(n_components = varCutOff) #n_components specifies the number of principal components to extract from the covariance matrix
pca.fit(Xf) #builds the covariance matrix and "fits" the principal components
Xpca = pca.transform(Xf) #transforms the data into the pca representation
nPCs = Xpca.shape[1]

wctm.Xpca = Xpca
wctm.pca = pca
for i in indgood_models:
    if inds_imagingSet_models[i] == 0:
        indsf = np.where(indtreatment == i)[0]
        modelSet[i].Xpca = Xpca[indsf, :]

indgood_models = indgood_models[np.where(inds_imagingSet_models[indgood_models] == 0)[0]]

self = wctm
wctm.trajl = trajl
all_trajSet = [None]*nmodels
for i in indgood_models:
    modelSet[i].get_unique_trajectories()
    all_trajSet[i] = modelSet[i].trajectories.copy()

# Get snippets along with their full single-cell trajectory indices  
def get_snippets_with_traj_inds(self, seg_length):
    n_sctraj = len(self.trajectories) # Number of Single-Cell Trajectories 
    traj_segSet = np.zeros((0, seg_length)).astype(int)
    ind_map_snippet_fulltraj = np.array([])
    for ind_traj in range(n_sctraj):
        cell_traj = self.trajectories[ind_traj] # Select a single-cell trajectory 
        traj_len = cell_traj.size
        if traj_len >= seg_length:
            for ic in range(traj_len - seg_length + 1):
                traj_seg = cell_traj[ic:ic+seg_length]
                traj_segSet = np.append(traj_segSet, traj_seg[np.newaxis, :], axis = 0)
                # Save indices of all snippets corresponding to "FULL" single-cell trajectory 
                ind_map_snippet_fulltraj = np.append(ind_map_snippet_fulltraj, ind_traj)
                #print("Indices to map snippets to the full trajectory:",ind_map_snippet_fulltraj)
    return ind_map_snippet_fulltraj, traj_segSet

# Single-cell trajectories over the dimensionally reduced cell features
Xpcat = np.zeros((0, pca.n_components_*trajl + nfeat_com*trajl))
indtreatment_traj = np.array([])
indstack_traj = np.array([])
indframes_traj = np.array([])
indmodel_traj_snippets = np.array([])
cellinds0_traj = np.array([])
cellinds1_traj = np.array([])
cb_ratio_traj = np.array([])
for i in indgood_models:
    print('building trajectory data for model {}...'.format(i))
    modelSet[i].trajectories = all_trajSet[i].copy() # ALL Single-Cell trajectories 
    modelSet[i].trajl = trajl # Trajectory snippet length 
    # Get trajectory snippets of (all trajectories) a given length in a sliding window
    # modelSet[i].traj = modelSet[i].get_traj_segments(trajl)
    # Get trajectory snippets of (all trajectories) a given length in a sliding window and mapped with single-cell trajectory indices 
    modelSet[i].snippet_map_fulltraj_inds, modelSet[i].traj = get_snippets_with_traj_inds(modelSet[i], trajl)
    # Xpca (feature info) along the single-cell trajectory snippets, extracted directly from cell indices unique within a 'model' 
    data = modelSet[i].Xpca[modelSet[i].traj, :] # Dimensionally Reduced Featured Single-Cell Trajectories  
    datacom = modelSet[i].Xf_com[modelSet[i].traj, :] # Center of Mass (COM) 
    data = data.reshape(modelSet[i].traj.shape[0], modelSet[i].Xpca.shape[1]*trajl)
    datacom = datacom.reshape(modelSet[i].traj.shape[0], modelSet[i].Xf_com.shape[1]*trajl)
    data = np.append(data, datacom, axis = 1)
    indgood = np.where(np.sum(np.isnan(data), axis = 1) == 0)[0] # Consider models as "Good" that don't have NaN in "data" 
    data = data[indgood, :]
    modelSet[i].traj = modelSet[i].traj[indgood, :] # Cleaned trajectory snippets if any NaN 
    modelSet[i].snippet_map_fulltraj_inds = modelSet[i].snippet_map_fulltraj_inds[indgood]
    # Store all trajectory snippets of a given length (picked in a sliding window) 
    Xpcat = np.append(Xpcat, data, axis = 0) 
    indtreatment_traj = np.append(indtreatment_traj, i*np.ones(data.shape[0])) # Indices of Treatments (Models) Along Trajectory Snippets 
    indstacks = modelSet[i].cells_imgfileSet[modelSet[i].traj[:, 0]]
    indstack_traj = np.append(indstack_traj, indstacks)
    indframes = modelSet[i].cells_frameSet[modelSet[i].traj[:, 0]].astype(int) # Frame index at the start of the snippet
    indframes_traj = np.append(indframes_traj, indframes) # Starting Frame Indices of ALL snippets across ALL models (conditions)
    indtraj_snippets = modelSet[i].snippet_map_fulltraj_inds
    indmodel_traj_snippets = np.append(indmodel_traj_snippets, indtraj_snippets) # Save for all models: map of snippets to their single-cell trajectories
    cellinds0 = modelSet[i].traj[:, 0] # Cell indices at the start of snippets 
    cellinds0_traj = np.append(cellinds0_traj, cellinds0)
    cellinds1 = modelSet[i].traj[:, -1] # Cell indices at the end of snippets 
    cellinds1_traj = np.append(cellinds1_traj, cellinds1)
    cb_ratio_traj = np.append(cb_ratio_traj, modelSet[i].Xf[cellinds1, 77])

cellinds0_traj = cellinds0_traj.astype(int)
cellinds1_traj = cellinds1_traj.astype(int)

neigen = Xpcat.shape[1] # If embedded trajectories aren't UMAP'ed 
inds_conditions = [None]*n_conds
for imf in range(n_conds):
    indmodels = np.intersect1d(indgood_models, np.where(inds_tmSet_models == imf)[0])
    indstm = np.array([])
    for imodel in indmodels:
        indtm = np.where(indtreatment_traj == imodel)
        indstm = np.append(indstm, indtm)
    inds_conditions[imf] = indstm.astype(int).copy() # Condition (Model) specific trajectory snippet indices: Add up all FOVs

##### Apply UMAP on featured trajectory snippets for plotting only #####
neigen_umap = 2
reducer = umap.UMAP(n_neighbors=200, min_dist=0.1, n_components=neigen_umap, metric='euclidean')
trans = reducer.fit(Xpcat)
x_umap = trans.embedding_
indst = np.arange(x_umap.shape[0]).astype(int)
wctm.Xtraj = x_umap.copy()
wctm.indst = indst.copy()

##### Cluster single-cell trajectories of a given snippet length by using KMeans from deeptime 
from deeptime.clustering import KMeans
n_clusters = 200
model = KMeans(n_clusters = n_clusters,  # Place 200 cluster centers
               init_strategy = 'kmeans++',  # kmeans++ initialization strategy
               max_iter = 0,  # don't actually perform the optimization, just place centers
               fixed_seed = 13)
################################ Initial clustering ###############################
clustering = model.fit(Xpcat).fetch_model() # If embedded trajectories aren't UMAP'ed 

model.initial_centers = clustering.cluster_centers
model.max_iter = 5000
clusters = model.fit(Xpcat).fetch_model() # If embedded trajectories aren't UMAP'ed 
wctm.clusterst = clusters

# Consider (start) frames only in the range of 24+-6, i.e., 18-30 hours -> frame numbers: 72-120
# Note: images are collected at the interval of 15 minutes for 48 hours
fl = 72 
fu = 120 #fu = n_frames
# Indices of "Xpcat" that satisfy the condition across models 
indstw = np.where(np.logical_and(indframes_traj < fu, indframes_traj > fl))[0] 

# Get Cross correlations & respective frame numbers along all single-cell trajectories
def get_cross_corr_all_single_cell_trajs(filename):
      cross_corr_sctraj = []
      frame_num_sctraj = []
      with open(filename, 'r') as file_in:
          # Create an iterator over the file lines 
          file_iter = iter(file_in)
          try:
              while True:
                  # Read two lines at a time: Cross correlations & the corresponding frame numbers
                  line1 = next(file_iter).strip()
                  if line1:
                          data = ast.literal_eval(line1)
                          cross_corr_sctraj.append(data)
                          line2 = next(file_iter).strip()
                          # Parse line2 as a list of integers
                          frame_numbers = ast.literal_eval(line2)   
                          frame_num_sctraj.append(frame_numbers)
          except StopIteration:
              # End of file reached
              pass
      cross_corr_all_sctraj_file = cross_corr_sctraj
      frame_num_all_sctraj_file = frame_num_sctraj
      
      return cross_corr_all_sctraj_file, frame_num_all_sctraj_file

########## Location of CC files between the Reporter & Nuclear channel images ########### 
cc_path = "/home/groups/ZuckermanLab/jalim/LI204601_INCUCYTE/dcompCCtraj_states/"
indices_conditions = 0
frames_models = []
cc_vals_models = []

for imodel in indgood_models:
    if imodel % nfovs == 0:
      condition = conditions[indices_conditions]
      indices_conditions += 1
      fov_indc = 1
    else:
      fov_indc += 1
    cc_filename = f"{sysName}_{condition}_{fov_indc}.dat"
    cc_file = cc_path + cc_filename
    ### Import CC Values & Corresponding Frame Numbers from the Files for Each Model ###
    cross_correlations, frame_numbers = get_cross_corr_all_single_cell_trajs(cc_file)  
    
    frames_models.append(frame_numbers)
    cc_vals_models.append(cross_correlations)
      
cc_values_by_conditions = []

############# Map trajectory snippets on Microstates #############
for cond in range(n_conds):
    cc_values = []
    indstm = inds_conditions[cond] # Condition Specific Indices (Includes ALL Field of Views)
    indstwm = np.intersect1d(indstm, indstw) # Indices of trajectory snippets within frame range (72 - 120)
    good_model_indc = indtreatment_traj[indstwm].astype(int) # Get "Good" model indices of a given "Xpcat"
    fid_snippets = indframes_traj[indstwm].astype(int) # Map frame indices of snippets within a range
    indc_map_fulltraj_snippets = indmodel_traj_snippets[indstwm].astype(int) # Map Indices of Full Trajectory to Snippets 
    ################## Identify the "FULL" trajectory index and locate snippets within it ##################
    for iter_trj, ind_fulltraj in enumerate(indc_map_fulltraj_snippets):
        model_index = good_model_indc[iter_trj]
        possible_indices = np.where(frames_models[model_index][ind_fulltraj] == fid_snippets[iter_trj])  
        if possible_indices[0].size > 0:
          indc_cctraj_frames = possible_indices[0][0]
          cc_vals = cc_vals_models[model_index][ind_fulltraj][indc_cctraj_frames]
          cc_values.append(cc_vals)
        else:
          print(f"No match found for model: {model_index}, frame: {fid_micro_snippet[iter_trj]}, Full TRJ: {ind_fulltraj}")
          continue 
    cc_values_by_conditions.append(cc_values) 

# Save the CC values Mapping on Fine (Micro) States info in a single .npz file
cc_vals_filename = 'ccValsByConditions'+figid+'.npz'
np.savez(cc_vals_filename, cc_vals_by_conditions=cc_values_by_conditions)

def get_cdist2d(prob1):
    nx = prob1.shape[0]; ny = prob1.shape[1]
    prob1 = prob1/np.sum(prob1)
    prob1 = prob1.flatten()
    indprob1 = np.argsort(prob1)
    probc1 = np.zeros_like(prob1)
    probc1[indprob1] = np.cumsum(prob1[indprob1])
    probc1 = 1. - probc1
    probc1 = probc1.reshape((nx, ny))
    return probc1

def colorbar(mappable):
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(mappable, cax=cax)
    plt.sca(last_axes)
    return cbar

#Compute a 2D histogram with average values within each bin
def compute_histogram_with_values(x, y, values, bins):
    # x, y are coordinates; values are the associated values to average
    # bins are the binning specification.
    data, xedges, yedges = np.histogram2d(x, y, bins=bins)
    sums, _, _ = np.histogram2d(x, y, bins=bins, weights=values)
    counts, _, _ = np.histogram2d(x, y, bins=bins, weights=np.ones_like(values))
    
    average_values = np.divide(sums, counts, out=np.zeros_like(sums), where=counts!=0)
    return average_values, xedges, yedges

indtreatment_traj = indtreatment_traj.astype(int)
inds_imagingSet_traj = inds_imagingSet_models[indtreatment_traj]
indscc = np.where(cb_ratio_traj < np.inf)[0]
indstw_cc = np.intersect1d(indstw, indscc)
probSet = [None]*nmodels

nbins = 50
plt.clf()
plt.figure(figsize=(20, 12))
plt.subplot(5, 3, 1)

flat_cc_all_conditions = []
for condition in cc_values_by_conditions:
    for value in condition:
        flat_cc_all_conditions.extend(value)
flat_cc_all_conditions = np.array(flat_cc_all_conditions)  

#Compute a 2D histogram with average values within each bin
cc_grid, xedges, yedges = compute_histogram_with_values(x_umap[indstw_cc, 0], x_umap[indstw_cc, 1], flat_cc_all_conditions, nbins)
# Prepare meshgrid for contour plotting
xx, yy = np.meshgrid(0.5 * (xedges[:-1] + xedges[1:]), 0.5 * (yedges[:-1] + yedges[1:]))
cs = plt.contourf(xx, yy, cc_grid.T, levels=20, cmap=plt.cm.jet_r)  # you can choose a different colormap
cbar = colorbar(cs)
cbar.set_label('CC Values')
plt.title('Combined Cumulative Probability')
#plt.xlabel('UMAP 1')
#plt.ylabel('UMAP 2')
plt.axis('off')

for ic in range(n_conditions):
    tm = tmSet[ic]
    indstm = inds_conditions[ic]
    indstwm = np.intersect1d(indstm, indstw_cc)
    indstwm = np.intersect1d(indstwm, indscc)
    cc_values_condition = np.array(cc_values_by_conditions[ic])
    cc_grid, xedges, yedges = compute_histogram_with_values(x_umap[indstwm, 0], x_umap[indstwm, 1], cc_values_by_condition, nbins)
    xx, yy = np.meshgrid(0.5 * (xedges[:-1] + xedges[1:]), 0.5 * (yedges[:-1] + yedges[1:]))
    plt.subplot(5, 3, ic+2)
    cs = plt.contourf(xx, yy, cc_grid.T, levels=20, cmap=plt.cm.jet_r, extend='both')
    plt.title(tmSet[ic])
    cs.cmap.set_over('darkred')
    plt.axis('off')
    #plt.show()

plt.savefig('cc_vals_'+figid+'.png')
#plt.show()

plt.clf()
plt.figure(figsize = (20, 12))
plt.subplot(5, 3, 1)
prob1, xedges1, yedges1 = np.histogram2d(x_umap[indstw_cc, 0], x_umap[indstw_cc, 1], bins=nbins, density=True)
prob1c = get_cdist2d(prob1)
xx, yy = np.meshgrid(.5*xedges1[1:] + .5*xedges1[0:-1], .5*yedges1[1:] + .5*yedges1[0:-1])
raw_levels = np.percentile(prob1c, np.linspace(0, 100, 21)) # Levels according to the probabilities 
levels = np.unique(raw_levels)
# In case unique filtering makes levels non-monotonic or reduces their count significantly
if len(levels) < 2:
    # Fallback to linear levels if percentile-based levels are not suitable
    levels = np.linspace(np.min(prob1c), np.max(prob1c), 21)
cs = plt.contourf(xx, yy, prob1c.T, levels=levels, cmap=plt.cm.jet_r)
cbar = colorbar(cs)
cbar.set_label('Probability (%)')
cbar.ax.tick_params(labelsize=10)
plt.title('combined cumulative probability')
plt.axis('off')
for ic in range(n_conditions):
    tm = tmSet[ic]
    indstm = inds_conditions[ic]
    indstwm = np.intersect1d(indstm, indstw_cc)
    indstwm = np.intersect1d(indstwm, indscc)
    prob, xedges2, yedges2 = np.histogram2d(x_umap[indstwm, 0], x_umap[indstwm, 1], bins=[xedges1, yedges1], density=True)
    probc = get_cdist2d(prob)
    probSet[ic] = prob.copy()
    plt.subplot(5, 3, ic+2)
    cs = plt.contourf(xx, yy, probc.T, levels=levels, cmap=plt.cm.jet_r, extend='both')
    plt.title(tmSet[ic])
    cs.cmap.set_over('darkred')
    plt.axis('off')
    #plt.show()

plt.savefig('prob_'+figid+'.png')
#plt.show()