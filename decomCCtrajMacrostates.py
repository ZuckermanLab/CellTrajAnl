import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sys, os, time, math
sys.path.append('/home/groups/ZuckermanLab/jalim/instalLocal/celltraj/celltraj')
import jcTrajectory_CP as cellTraj
import cellCycRepJCtrajsCP as nucMsksCCR # To import reporter & nuclear images along with cell masks
import h5py, pickle, subprocess
import umap, scipy
from csaps import csaps
import string, ast 
from joblib import dump, load
from datetime import date
from skimage import io, measure
from skimage.measure import label, regionprops
from skimage.morphology import closing, square

######### Parameters of transitions between "macroscopic" states ##########
nstates_final = 7
border_margin = 2
max_states = 100
n_components_tMat = 15
trajl = 40 # Trajectory Length for morphodynamical trajectory analysis
wellsInfo = 'Awells'
conditions = ['A2'] # LIGANDS or CONDITIONS
tmSet = ['EGF1']

nConditions = len(tmSet) # Total number of Ligand Conditions
#os.environ['OMP_NUM_THREADS'] = '1'; os.environ['MKL_NUM_THREADS'] = '1'

today = date.today()
date2day = today.strftime("%b%d-%Y")
sysName = 'LI204601_P'
figid = sysName+'_tlen'+str(trajl)+'_'+date2day

# Indices for the ligands 
inds_tmSet = [i for i in range(nConditions)]
inds_tmSet = np.array(inds_tmSet).astype(int)
nfovs = 1
fovs = [i for i in range(1, nfovs + 1)]
fovs = np.array(fovs).astype(int)
dateSet = ['']
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
        indgood_models = np.append(indgood_models,i)
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
    data = modelSet[i].Xpca[modelSet[i].traj, :] 
    datacom = modelSet[i].Xf_com[modelSet[i].traj, :]
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
    indframes_traj = np.append(indframes_traj, indframes) # Starting Frame Indices of ALL snippets
    indtraj_snippets = modelSet[i].snippet_map_fulltraj_inds
    indmodel_traj_snippets = np.append(indmodel_traj_snippets, indtraj_snippets) # Save for all models: map of snippets to the sc trajectories
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

knn = 50
for i in indgood_models:
    modelSet[i].trajectories = all_trajSet[i].copy()

def get_trajectory_steps(self, inds=None, traj=None, Xtraj=None,
                         get_trajectories=True, nlag=1): # traj and Xtraj should be indexed same
    if inds is None:
        inds = np.arange(self.cells_indSet.size).astype(int)
    if get_trajectories:
        self.get_unique_trajectories(cell_inds=inds)
    if traj is None:
        traj = self.traj
    if Xtraj is None:
        x = self.Xtraj
    else:
        x = Xtraj
    trajp1 = self.get_traj_segments(self.trajl + nlag) # Get trajectory snippets @ snippet_length + nlag
    # Reversed index array inds_nlag is created to keep indices every nlag steps
    inds_nlag = np.flipud(np.arange(self.trajl + nlag - 1, -1, -nlag)).astype(int) # keep indices every nlag
    trajp1 = trajp1[:, inds_nlag]
    ntraj = trajp1.shape[0]
    
    neigen = Xpcat.shape[1]
    x0 = np.zeros((0, neigen))
    x1 = np.zeros((0, neigen))
    inds_trajp1 = np.zeros((0, 2)).astype(int)
    # Matching segments of trajectories and then appending the corresponding feature data from Xpcat to two arrays, 'x0' and 'x1'
    for itraj in range(ntraj):
        test0 = trajp1[itraj, 0:-1] # [0:self.trajl]
        test1 = trajp1[itraj, 1:] # [1:self.trajl+1]
        # traj[:, None] == test0[np.newaxis, :]: This expression compares each segment in 'traj' with 'test0'.
        # The use of [:, None] and [np.newaxis, :] reshapes the arrays for broadcasting, allowing element-wise
        # comparison between each segment in 'traj' and the segment 'test0'. 
        # .all(-1): This checks if all elements in a segment are equal, resulting in a boolean array where each
        # element represents whether a segment in 'traj' matches 'test0'.
        # .any(-1): This determines if there is any match in 'traj' for 'test0'. 'res0' is a boolean array indicating
        # which trajectories in 'traj' match 'test0'
        res0 = (traj[:, None] == test0[np.newaxis, :]).all(-1).any(-1)
        res1 = (traj[:, None] == test1[np.newaxis, :]).all(-1).any(-1)
        if np.sum(res0) == 1 and np.sum(res1) == 1: # If at least one matching array in 'traj' for both 'test0' & 'test1' 
            indt0 = np.where(res0)[0][0] # Collect indices where above matching happens
            indt1 = np.where(res1)[0][0] # Collect indices where above matching happens
            x0 = np.append(x0, np.array([Xpcat[indt0, :]]), axis = 0)
            x1 = np.append(x1, np.array([Xpcat[indt1, :]]), axis = 0)
            inds_trajp1 = np.append(inds_trajp1, np.array([[indt0, indt1]]), axis = 0)
        if itraj%100 == 0:
            sys.stdout.write('matching up trajectory '+str(itraj)+'\n')
    self.Xtraj0 = x0
    self.Xtraj1 = x1
    self.inds_trajp1 = inds_trajp1

dxs = np.zeros((nmodels, n_clusters, neigen))
x0set = np.zeros((0, neigen))
x1set = np.zeros((0, neigen))
inds_trajsteps_models = np.array([]).astype(int)
for i in indgood_models:
    print('getting flows from model: '+str(i))
    indstm = np.where(indtreatment_traj == i)[0]
    if indstm.size > 0:
        modelSet[i].Xtraj = Xpcat[indstm, 0:neigen]
        indstm_model = indstm - np.min(indstm) # index in the model
        if inds_imagingSet_models[i] == 1:
            modelSet[i].get_trajectory_steps(inds=None, get_trajectories=False, traj=modelSet[i].traj[indstm_model, :],
                                             Xtraj=modelSet[i].Xtraj[indstm_model, :])
        else:
            get_trajectory_steps(modelSet[i], inds=None, get_trajectories=False, traj=modelSet[i].traj[indstm_model, :], 
                                 Xtraj=modelSet[i].Xtraj[indstm_model, :])
        x0 = modelSet[i].Xtraj0
        x1 = modelSet[i].Xtraj1
        x0set = np.append(x0set, x0, axis=0)
        x1set = np.append(x1set, x1, axis=0)
        inds_trajsteps_models = np.append(inds_trajsteps_models, np.ones(x0.shape[0])*i)
        dx = x1 - x0
        for iclust in range(n_clusters):
            xc = np.array([clusters.cluster_centers[iclust, :]])
            dmatr = wctm.get_dmat(modelSet[i].Xtraj[modelSet[i].inds_trajp1[:, -1], :], xc) #get closest cells to cluster center
            indr = np.argsort(dmatr[:, 0])
            indr = indr[0:knn]
            cellindsr = modelSet[i].traj[[modelSet[i].inds_trajp1[indr, -1]], -1]
            dxs[i, iclust, :] = np.mean(dx[indr, :], axis=0)

def colorbar(mappable):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size = "5%", pad = 0.05)
    cbar = fig.colorbar(mappable, cax = cax)
    plt.sca(last_axes)
    return cbar
############################ Generate transition matrix ############################
centers_minima  = clusters.cluster_centers.copy()
nclusters = clusters.cluster_centers.shape[0]

# Assign "new data" to cluster centers
indc0 = clusters.transform(x0set).astype(int) # Indices of cluster centers where each snippets of 'x0set' belong 
indc1 = clusters.transform(x1set).astype(int) # Indices of cluster centers where each snippets of 'x1set' belong
wctm.get_transitionMatrixDeeptime(indc0, indc1, nclusters)
P = wctm.Mt.copy()

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
graph = csr_matrix(P > 0.)
n_components, labels = connected_components(csgraph=graph, directed=False, return_labels=True)
unique, counts = np.unique(labels, return_counts=True)
icc = unique[np.argmax(counts)]
indcc = np.where(labels == icc)[0]
centers_minima = centers_minima[indcc, :]

############### Using pyEmma for assignments only #################
import pyemma.coordinates as coor
clusters_minima = coor.clustering.AssignCenters(centers_minima, metric='euclidean')
######## Now clusters_minima will have attribute clusters_minima.clustercenters ########
nclusters = clusters_minima.clustercenters.shape[0]
indc0 = clusters_minima.assign(x0set)
indc1 = clusters_minima.assign(x1set)
wctm.get_transitionMatrixDeeptime(indc0, indc1, nclusters)
P = wctm.Mt.copy()

import pygpcca as gp
gpcca = gp.GPCCA(P, eta=None, z='LM', method='brandts')

neigen_umap = 0 # In this case
# Dump Transition Matrix for further analysis 
tmFileName = 'tMat_'+sysName+'_'+str(trajl)+'_'+date2day+'pc'+str(nPCs)+'u'+str(neigen_umap)+wellsInfo+'.joblib'
with open(tmFileName, 'wb') as fp:
     dump(P, fp, compress = 'zlib')

# Find Eigenvalues and Eigenvectors of the transition matrix "P"
H = .5*(P + np.transpose(P)) + .5j*(P - np.transpose(P))
w, v = np.linalg.eig(H)  
w = np.real(w)
indsort = np.argsort(w)
# Sorted eigenvalues & eigenvalues 
w = w[indsort] # Eigen Values
v = v[:, indsort] # Eigen Vectors
ncomp = n_components_tMat # Keep last "ncomp" eigenvectors
vr = np.multiply(w[-ncomp:], np.real(v[:, -ncomp:]))
vi = np.multiply(w[-ncomp:], np.imag(v[:, -ncomp:]))
vkin = np.append(vr, vi, axis = 1) # Eigenvectors (real & imaginary)

from sklearn.cluster import KMeans

################### Get kinetics of cell (macro or coarse) state transitions #####################
def get_kinetic_states_module(self, vkin, nstates_final, nstates_initial = None, pcut_final = .01,
                              max_states = 20, cluster_ninit = 10):
       nstates_good = 0
       nstates = nstates_initial
       vkinFit = vkin
       while nstates <= max_states:
            clusters_v = KMeans(n_clusters = nstates, init = 'k-means++',
                                n_init = cluster_ninit, max_iter = 5000, 
                                random_state = 0)
            clusters_v.fit(vkinFit) 
            stateSet = clusters_v.labels_
            state_probs = np.zeros(nstates)
            statesc, counts = np.unique(stateSet, return_counts = True)
            state_probs[statesc] = counts/np.sum(counts)
            print(np.sort(state_probs))
            nstates_good = np.sum(state_probs > pcut_final)
            print('{} states initial, {} states final'.format(nstates, nstates_good))
            print(nstates, "Current states", nstates_good, "Good states")
            nstates = nstates + 1
            if nstates_good >= nstates_final:
               break
       pcut = np.sort(state_probs)[-(nstates_final)] #nstates
       states_plow = np.where(state_probs < pcut)[0]
       # Assign (micro)states to predetermined state centers aka "macrostates" with probabilities less than 'pcut'
       for i in states_plow:
           indstate = np.where(stateSet == i)[0]
           for imin in indstate:
               dists = wctm.get_dmat(np.array([vkinFit[imin, :]]), vkinFit)[0] #closest in eigenspace
               dists[indstate] = np.inf
               ireplace = np.argmin(dists)
               stateSet[imin] = stateSet[ireplace]
       slabels, counts = np.unique(stateSet, return_counts = True)
       s = 0
       stateSet_clean = np.zeros_like(stateSet)
       for slabel in slabels:
           indstate = np.where(stateSet == slabel)[0]
           stateSet_clean[indstate] = s
           s = s + 1
       stateSet = stateSet_clean
       if np.max(stateSet) > nstates_final:
          print("returning ", np.max(stateSet)," states", nstates_final, "requested")
       return stateSet, nstates_good  

# Module to optimize pcut_final that shows the best clustering onto 7 states 
def get_kinetic_states(self, vkin, nstates_final, nstates_initial = None, pcut_final = .01,
                       max_states = 20, cluster_ninit = 10):
       if nstates_initial is None:
          nstates_initial = nstates_final
       nstates_good = 0
       while nstates_good < nstates_final or nstates_good > nstates_final:
                 stateSet, nstates_good = get_kinetic_states_module(wctm, vkin, nstates_final, 
                                                                     nstates_initial = nstates_initial, 
                                                                     pcut_final = pcut_final,
                                                                     max_states = max_states,
                                                                     cluster_ninit = cluster_ninit)
                 print("pcut_final = ",pcut_final)
                 pcut_final = pcut_final - 0.001 

       return stateSet

pcut_final = 0.055 # For EGF - Well 1
get_kstates = True
stateCenters = clusters_minima.clustercenters # Centers of microstates (fine-grain states) 
if get_kstates:
   stateSet = get_kinetic_states(wctm, vkin, nstates_final,
                                 nstates_initial = None, pcut_final = pcut_final, 
                                 max_states = max_states, cluster_ninit = 10)
   nstates = np.unique(stateSet).size
   objFile = 'stateSet_'+figid+'_nS'+str(nstates)+'.joblib'
   states_object = [clusters_minima, stateSet]
   with open(objFile, 'wb') as fpStates:
      dump(states_object, fpStates, compress = 'zlib')
else:
   objFile = 'stateSet_'+figid+'_nS'+str(nstates_initial)+'.joblib'
   with open(objFile, 'rb') as fpStates:
       states_object = load(fpStates)
   clusters_minima = states_object[0]
   stateSet = states_object[1]
   nstates = np.unique(stateSet).size

n_states = nstates
state_centers_minima = np.zeros((n_states, neigen))
for i in range(n_states):
    indstate = np.where(stateSet == i)[0]
    state_centers_minima[i, :] = np.median(stateCenters[indstate, :], axis=0)

state_labels = np.array(list(string.ascii_uppercase))[0:n_states]

# Consider frames only in the range of 24+-6, i.e., 18-30 hours -> frame numbers: 72-120
# Note: images are collected at the interval of 15 minutes for 48 hours
fl = 72 
fu = 120 #fu = nframes
indstw = np.where(np.logical_and(indframes_traj < fu, indframes_traj > fl))[0] # Indices 

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

# Import single-cell trajectory model from cell-cycle reporter' class
sctm = nucMsksCCR.TrajectoryCCR()
sctm.visual = False
start_frame = 0
# Location of the Reporter & Nuclear channel images along with cytoplasmic mask 
filepath = '/home/groups/ZuckermanLab/jalim/LI204601_INCUCYTE/crossCorrCCRnuc/'

for im in indgood_models:
    model = filepath+sysName+'_'+conditions[im]+'_1.h5'
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
    for fid in range(nframes): 
        # If the number of cells is equal in two consecutive frames
        n_cells_curr = np.sum(modelSet[im].cells_frameSet == fid)
        if n_cells_curr == n_cells_prev:
            sctm.imgSet[fid] = sctm.imgSet[fid-1]
            sctm.nucImgSet[fid] = sctm.nucImgSet[fid-1]
        n_cells_prev = n_cells_curr
    
    # Set of all single-cell trajectories with absolute (irrespective of frames) cell indices
    num_single_cell_trajs = len(modelSet[im].trajectories)
    ##################### Calculate cross-correlations along all single-cell trajectories ######################
    frames_model = []
    cc_vals_model = []
    for itraj in range(num_single_cell_trajs):
        cell_traj = modelSet[im].trajectories[itraj]
        sctraj_length = cell_traj.size # Length of a single-cell trajectory
        print(f'Single-cell trajectory: {itraj + 1}, Length: {sctraj_length}')
        # Cross-correlation between cell-cycle reporter & nuclear channel images along a single-cell trajectory
        frames_traj = []
        cross_corr = []
        for itt in range(sctraj_length):
          
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
 
            reporter_image = np.array(reporter_image)
            nuclear_image = np.array(nuclear_image)

            x_min, x_max = modelSet[im].xmin, modelSet[im].xmax
            y_min, y_max = modelSet[im].ymin, modelSet[im].ymax

            # Crop reporter images, nuclear images, and masks according to single-cell positions
            cropped_reporter_image = reporter_image[x_min:x_max, y_min:y_max]
            cropped_nuclear_image = nuclear_image[x_min:x_max, y_min:y_max]

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
        frames_model.append(frames_traj)
        cc_vals_model.append(cross_corr)

 Map trajectory snippets on Macrostates via Microstates
state_order = np.arange(n_states).astype(int)

for i in range(n_conds):
    indstm = inds_conditions[i] # Condition Specific Indices
    indstwm = np.intersect1d(indstm, indstw) # Indices of trajectory snippets within frame range (72 - 120)
    x0 = Xpcat[indstwm, :] # keep trajectory snippets that are within the given frame range
    ################ Mapping Microstates (Fine-grain) to Macrostates (Coarse-grain) #################
    indc_map_macro = stateSet[clusters_minima.assign(x0)] # Assign Microstates to Macrostates
    # Initialize Dictionary to store Cross-correlation Values of Snippets of Start Frames for each Macrostate 
    cc_values_by_macrostate = {}
    for si in state_order:
        state_label = chr(65 + si)  # Convert state index to the uppercase letters (A, B, C,...)
        cc_values_by_macrostate[state_label] = []
        #print(f'Macrostate: {state_label}')
        ind_map = np.where(indc_map_macro == si)[0] # Indices: Microstates -> Macrostate
        #print(f'x0 indices: {ind_map}')
        indc_map_Xpcat = indstwm[ind_map] # Mapping back to original Xpcat indices Corresponding to a Macrostate 
        #print(f'Xpcat indices: {indc_map_Xpcat}')
        fid_macro_snippet = indframes_traj[indc_map_Xpcat].astype(int) # Map frame indices of snippets onto Macrostates 
        indc_map_fulltraj_snippets = indmodel_traj_snippets[indc_map_Xpcat].astype(int) # Map Indices of Full Trajectory to Snippets 
        #print(f'Start Frame Indices of Snippets: {fid_macro_snippet}')
        ################## Identify the "FULL" trajectory index and locate snippets within it ##################
        for j, ind_fulltraj in enumerate(indc_map_fulltraj_snippets):
            possible_indices = np.where(frames_model[ind_fulltraj] == fid_macro_snippet[j])  
            indc_ccvals_traj = possible_indices[0][0]
            #print(f'Indices of Entire Traj: {ind_fulltraj}, Indices of CC_traj: {indc_ccvals_traj}')
            cc_vals = cc_vals_model[ind_fulltraj][indc_ccvals_traj]
            #print(f'CC values: {cc_vals}, Indices of Entire Traj: {ind_fulltraj}, Indices of CC trajectory: {indc_ccvals_traj}')
            cc_values_by_macrostate[state_label].append(cc_vals)

from scipy.stats import gaussian_kde
import matplotlib.ticker as ticker

fig, ax = plt.subplots(figsize=(10, 6))

# Plot each state's distribution
for state, values in cc_values_by_macrostate.items():
    kde = gaussian_kde(values)
    x_range = np.linspace(min(values), max(values), 1000)
    ax.plot(x_range, kde(x_range), label=f'{state}')

    # Highlight the mean value
    mean_x = np.mean(values)
    mean_y = kde(mean_x)
    ax.plot(mean_x, mean_y, 'o', color='k')

    # Add a vertical line for the mean
    ax.axvline(mean_x, color='k', linestyle='--', alpha=0.7)

legend = ax.legend(title='Macrostates')
plt.setp(legend.get_title(), fontsize='large')
ax.set_xlabel('Cross-correlation b/w Cell-cycle Reporter & Nuclear Channels of Start Frame of Snippets')
ax.set_ylabel('Frequency')

# Set major and minor ticks on the x-axis
ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
ax.xaxis.set_minor_locator(ticker.MultipleLocator())

# Format tick labels to a specified number of decimal places
ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))

# Set grid for minor ticks
#ax.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')
plt.show()

# Map trajectory snippets on the coarse states to generate a histogram 
state_order = np.arange(n_states).astype(int)
macrostates = np.array(state_labels)[state_order]
plt.clf()
plt.figure(figsize = (8, 5))
for i in range(n_conds):
    indstm = inds_conditions[i] # Condition Specific Indices
    indstwm = np.intersect1d(indstm, indstw) # Indices of trajectory snippets within frame range (72 - 120)
    x0 = Xpcat[indstwm, :] # keep trajectory snippets that are within the frame range
    # Know Indices & calculate (coarse) state probs by Mapping Fine-grain (Microstates) to Coarse-grain (Macrostates)
    indc_micro = clusters_minima.assign(x0) # Microstate cluster indices 
    indc_macro = stateSet[indc_micro] # Assignment of each microstate to the macrostates 
    states_centers, counts = np.unique(indc_macro, return_counts = True) # Counting how many times a macrostate is hit by a microstate 
    plt.bar(macrostates, counts)
plt.show()