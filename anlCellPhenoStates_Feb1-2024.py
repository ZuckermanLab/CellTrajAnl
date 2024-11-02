import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sys, os, time, math
sys.path.append('/home/groups/ZuckermanLab/jalim/instalLocal/celltraj/celltraj')
import jcTrajectory_CP as cellTraj
import h5py
import pickle
import subprocess
import umap
import scipy
from csaps import csaps
import string
from joblib import dump, load
from datetime import date

######### Parameters of transitions between "macroscopic" states ##########
nstates_final = 7
max_states = 100
pcut_final = 0.094
n_components_tMat = 15

# Trajectory Length for morphodynamical trajectory analysis
try:
   trajl = int(sys.argv[1])
except ValueError:
   print("Error in getting trajectory snippet length for morphodynamical analysis")
   sys.exit(0)
# Flag to import data of a certain wells combinations
try:
   wells_flg = int(sys.argv[2])
except ValueError:
   print("Wells flag is not given OR is not a valid integer")
   sys.exit(0)

if(wells_flg == 0):
  wellsInfo = 'Awells'
  conditions = ['A1','A2','A3','A4','A5','C1','C2','C3'] # LIGANDS (CONDITIONS)
  tmSet = ['OSM1','EGF1','EGF+TGFB1','TGFB1','PBS1','OSM+EGF+TGFB','OSM+EGF','OSM+TGFB']
elif(wells_flg == 1):
  wellsInfo = 'Bwells'
  conditions = ['B1','B2','B3','B4','B5','C1','C2','C3'] # LIGANDS (CONDITIONS)
  tmSet = ['OSM2','EGF2','EGF+TGFB2','TGFB2','PBS2','OSM+EGF+TGFB','OSM+EGF','OSM+TGFB']
else:
  wellsInfo = 'AllWells'
  conditions = ['A1','A2','A3','A4','A5','B1','B2','B3','B4','B5','C1','C2','C3'] # LIGANDS (CONDITIONS)
  tmSet = ['OSM1','EGF1','EGF+TGFB1','TGFB1','PBS1','OSM2','EGF2','EGF+TGFB2','TGFB2','PBS2','OSM+EGF+TGFB','OSM+EGF','OSM+TGFB']

nConditions = len(tmSet) # Total number of Ligand Conditions

#os.environ['OMP_NUM_THREADS'] = '1'; os.environ['MKL_NUM_THREADS'] = '1'

today = date.today()
date2day = today.strftime("%b%d-%Y")
sysName = 'LI204601_P'
figid = sysName+'_tlen'+str(trajl)+'_'+date2day

# Indices for the ligands 
inds_tmSet = [i for i in range(nConditions)]
inds_tmSet = np.array(inds_tmSet).astype(int)
nfovs = 4
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

#plt.clf()
#plt.figure(figsize = (8, 7))

with open('cellNumbers.dat', 'w', encoding = 'utf-8') as fp: # PRINT cell numbers in a file for each model
     for i in indgood_models:
        ncells = np.zeros(nframes)
        ncells_smooth = np.zeros_like(ncells)
        for iS in range(nframes):
           ncells[iS]=np.sum(modelSet[i].cells_frameSet==iS)
           fp.write(str(ncells[iS])+"\t")
           fp.write("\n")
        # Cubic Spline Approximation (CSAPS) to smoothen the data
        splfov = csaps(frames, ncells/ncells[0], abSmooth, smooth = 0.98) # Scaled by ncells[0] to avoid large numbers
        ncells_smooth = splfov*ncells[0] # smoothened cell numbers reverse scaled back to original
        cellnumber_std = np.std(ncells[cellnumber_frames] - ncells_smooth[cellnumber_frames])/np.mean(ncells[cellnumber_frames])
        cellnumber_stdSet[i] = cellnumber_std # Standard Deviation in Cell Numbers		
        #print("cellnumber_stdSet[",i,"] = ", cellnumber_std)
        #plt.plot(ncells/ncells[0], color = colModels[i], label = capModels[i])
        #plt.plot(ncells, color = colModels[i], label = capModels[i])
        #plt.plot(ncells/ncells[0]); plt.pause(.5)

indhigh_std = np.where(cellnumber_stdSet > cellnumber_std_cut)[0]
indgood_models = np.setdiff1d(indgood_models, indhigh_std)
#indgood_models = np.setdiff1d(indgood_models, np.array([51]).astype(int)) #missing comdx for EGF0_4

# get cell counts
nf = len(tmSet)
inds_tmSet_models = np.zeros(nmodels).astype(int)
inds_imagingSet_models = np.zeros(nmodels).astype(int)
i = 0
icond = 0
for cond in conditions:
    for fov in fovs:
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

# Single-cell trajectories over the dimensionally reduced features
Xpcat = np.zeros((0, pca.n_components_*trajl + nfeat_com*trajl))
indtreatment_traj = np.array([])
indstack_traj = np.array([])
indframes_traj = np.array([])
cellinds0_traj = np.array([])
cellinds1_traj = np.array([])
#cc_ratio_traj = np.array([])
cb_ratio_traj = np.array([])
for i in indgood_models:
    print('building trajectory data for model {}...'.format(i))
    modelSet[i].trajectories = all_trajSet[i].copy()
    modelSet[i].trajl = trajl
    modelSet[i].traj = modelSet[i].get_traj_segments(trajl)
    data = modelSet[i].Xpca[modelSet[i].traj, :] # Store Xpca along single-cell trajectory snippets
    datacom = modelSet[i].Xf_com[modelSet[i].traj, :]
    data = data.reshape(modelSet[i].traj.shape[0], modelSet[i].Xpca.shape[1]*trajl)
    datacom = datacom.reshape(modelSet[i].traj.shape[0], modelSet[i].Xf_com.shape[1]*trajl)
    data = np.append(data, datacom, axis = 1)
    indgood = np.where(np.sum(np.isnan(data), axis = 1) == 0)[0]
    data = data[indgood, :]
    modelSet[i].traj = modelSet[i].traj[indgood, :]
    Xpcat = np.append(Xpcat, data, axis = 0)
    indtreatment_traj = np.append(indtreatment_traj, i*np.ones(data.shape[0]))
    indstacks = modelSet[i].cells_imgfileSet[modelSet[i].traj[:, 0]]
    indstack_traj = np.append(indstack_traj, indstacks)
    indframes = modelSet[i].cells_frameSet[modelSet[i].traj[:, 0]]
    indframes_traj = np.append(indframes_traj, indframes)
    cellinds0 = modelSet[i].traj[:, 0]
    cellinds0_traj = np.append(cellinds0_traj, cellinds0)
    cellinds1 = modelSet[i].traj[:, -1]
    # cc_ratio = modelSet[i].cc_ratio[cellinds1]
    # cc_ratio_traj = np.append(cc_ratio_traj, cc_ratio)
    cellinds1_traj = np.append(cellinds1_traj, cellinds1)
    cb_ratio_traj = np.append(cb_ratio_traj, modelSet[i].Xf[cellinds1, 77])

cellinds0_traj = cellinds0_traj.astype(int)
cellinds1_traj = cellinds1_traj.astype(int)

get_embedding = True
neigen_umap = 2
if get_embedding:
    reducer = umap.UMAP(n_neighbors=200, min_dist=0.1, n_components=neigen_umap, metric='euclidean')
    trans = reducer.fit(Xpcat)
    x = trans.embedding_
    indst = np.arange(x.shape[0]).astype(int)
    wctm.Xtraj = x.copy()
    wctm.indst = indst.copy()
    #dump(x, sysName+'_trajl'+str(trajl)+'_d2embedding_'+date2day+'.joblib')
else:
    #x=load(sysName+'_trajl'+str(trajl)+'_d2embedding_'+date2day+'.joblib')
    pass

#neigen = x.shape[1]
neigen = Xpcat.shape[1] # If embedded trajectories aren't UMAP'ed 

inds_conditions = [None]*nf
for imf in range(nf):
    indmodels = np.intersect1d(indgood_models, np.where(inds_tmSet_models == imf)[0])
    indstm = np.array([])
    for imodel in indmodels:
        indtm = np.where(indtreatment_traj == imodel)
        indstm = np.append(indstm, indtm)
    inds_conditions[imf] = indstm.astype(int).copy()

##### Cluster single-cell trajectories of a given snippet length by using KMeans from deeptime 
from deeptime.clustering import KMeans
n_clusters = 200
model = KMeans(n_clusters = n_clusters,  # place 100 cluster centers
               init_strategy = 'kmeans++',  # kmeans++ initialization strategy
               max_iter = 0,  # don't actually perform the optimization, just place centers
               fixed_seed = 13)
################################ Initial clustering ###############################
clustering = model.fit(Xpcat).fetch_model() # If embedded trajectories aren't UMAP'ed 
#clustering = model.fit(x).fetch_model()

model.initial_centers = clustering.cluster_centers
model.max_iter = 5000
clusters = model.fit(Xpcat).fetch_model() # If embedded trajectories aren't UMAP'ed 
#clusters = model.fit(x).fetch_model()
wctm.clusterst = clusters

knn = 50
for i in indgood_models:
    modelSet[i].trajectories = all_trajSet[i].copy()

def get_trajectory_steps(self, inds=None, traj=None, Xtraj=None,
                         get_trajectories=True, nlag=1): #traj and Xtraj should be indexed same
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
    trajp1 = self.get_traj_segments(self.trajl + nlag)
    # Reversed index array inds_nlag is created to keep indices every nlag steps
    inds_nlag = np.flipud(np.arange(self.trajl + nlag - 1, -1, -nlag)).astype(int) #keep indices every nlag
    trajp1 = trajp1[:, inds_nlag]
    ntraj = trajp1.shape[0]
    #neigen = x.shape[1]
    neigen = Xpcat.shape[1]
    x0 = np.zeros((0, neigen))
    x1 = np.zeros((0, neigen))
    inds_trajp1 = np.zeros((0, 2)).astype(int)
    # Matching segments of trajectories and then appending the corresponding feature data from Xpcat to two arrays, 'x0' and 'x1'
    for itraj in range(ntraj):
        test0 = trajp1[itraj, 0:-1]
        test1 = trajp1[itraj, 1:]
        # traj[:, None] == test0[np.newaxis, :]: This expression compares each segment in 'traj' with 'test0'.
        # The use of [:, None] and [np.newaxis, :] reshapes the arrays for broadcasting, allowing element-wise
        # comparison between each segment in 'traj' and the segment 'test0'. 
        # .all(-1): This checks if all elements in a segment are equal, resulting in a boolean array where each
        # element represents whether a segment in 'traj' matches 'test0'.
        # .any(-1): This determines if there is any match in 'traj' for 'test0'. 'res0' is a boolean array indicating
        # which trajectories in 'traj' match 'test0'
        res0 = (traj[:, None] == test0[np.newaxis, :]).all(-1).any(-1)
        res1 = (traj[:, None] == test1[np.newaxis, :]).all(-1).any(-1)
        if np.sum(res0) == 1 and np.sum(res1) == 1:
            indt0 = np.where(res0)[0][0]
            indt1 = np.where(res1)[0][0]
            #x0 = np.append(x0, np.array([x[indt0, :]]), axis=0)
            #x1 = np.append(x1, np.array([x[indt1, :]]), axis=0)
            x0 = np.append(x0, np.array([Xpcat[indt0, :]]), axis=0)
            x1 = np.append(x1, np.array([Xpcat[indt1, :]]), axis=0)
            inds_trajp1 = np.append(inds_trajp1, np.array([[indt0, indt1]]), axis=0)
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
        #modelSet[i].Xtraj = x[indstm, 0:neigen]
        modelSet[i].Xtraj = Xpcat[indstm, 0:neigen]
        indstm_model = indstm - np.min(indstm) #index in model
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
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(mappable, cax=cax)
    plt.sca(last_axes)
    return cbar

indtreatment_traj = indtreatment_traj.astype(int)
inds_imagingSet_traj = inds_imagingSet_models[indtreatment_traj]

dxsav = np.mean(dxs, axis=0)
#x[np.isnan(x)]=0.0 #Seriously...

nbins = 20
#frames for a time window
#fl = 0
#fu = nframes 
fl = 72
fu = 120

plt.clf()
plt.figure(figsize = (9, 9))
indstw = np.where(np.logical_and(indframes_traj < fu, indframes_traj > fl))[0]
indscc = np.where(cb_ratio_traj < np.inf)[0]
indstw = np.intersect1d(indstw, indscc)
#indstw=np.intersect1d(indstw,np.where(inds_imagingSet_traj==1)[0])
probSet = [None]*nmodels

"""
plt.subplot(5, 3, 1)
prob1,xedges1,yedges1 = np.histogram2d(x[indstw, 0], x[indstw, 1], bins=nbins, density=True)
prob1c = get_cdist2d(prob1)
xx,yy = np.meshgrid(.5*xedges1[1:] + .5*xedges1[0:-1], .5*yedges1[1:] + .5*yedges1[0:-1])
levels = np.linspace(0, 1, 21)
cs = plt.contourf(xx, yy, prob1c.T, levels=levels, cmap=plt.cm.jet_r)
cbar = colorbar(cs)
#cbar.set_label('cumulative probability')
plt.title('combined cumulative distribution')
plt.axis('off')
for imf in range(nf):
    tm = tmSet[imf]
    indstm = inds_conditions[imf]
    indstwm = np.intersect1d(indstm, indstw)
    indstwm = np.intersect1d(indstwm, indscc)
    prob,xedges2,yedges2 = np.histogram2d(x[indstwm, 0], x[indstwm, 1], bins=[xedges1, yedges1], density=True)
    #prob = prob/np.sum(prob)
    probc = get_cdist2d(prob)
    probSet[imf] = prob.copy()
    plt.subplot(5, 3, imf+2)
    #levels = np.linspace(0,np.max(prob),100)
    cs = plt.contourf(xx, yy, probc.T, levels=levels, cmap=plt.cm.jet_r, extend='both')
    plt.title(tmSet[imf])
    cs.cmap.set_over('darkred')
    plt.axis('off')
    plt.pause(.1)

plt.savefig('prob_'+figid+'.png')

plt.clf()
plt.figure(figsize=(9,9))
plt.subplot(5,3,1)
plt.title('average')
plt.axis('off')
for ic in range(n_clusters):
    ax=plt.gca()
    ax.arrow(clusters.clustercenters[ic,0],clusters.clustercenters[ic,1],dxsav[ic,0],dxsav[ic,1],head_width=.2,linewidth=.5,color='white',alpha=1.0)

for i in range(nf):
    indmodels=np.where(inds_tmSet_models==i)[0]
    dxf=np.mean(dxs[indmodels,:,:],axis=0)
    plt.subplot(5,3,i+2)
    ax=plt.gca()
    for ic in range(n_clusters):
        ax.arrow(clusters.clustercenters[ic,0],clusters.clustercenters[ic,1],dxf[ic,0],dxf[ic,1],head_width=.2,linewidth=.5,color='white',alpha=1.0)
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.axis('off')
    plt.title(tmSet[i])
    plt.pause(.1)

plt.savefig('probflows_'+figid+'.png')

vsetList = [Xpcat[:, -3], Xpcat[:, -2], Xpcat[:, -1], cb_ratio_traj]
captionset = ['speed', 'alpha', 'cellcell_align', 'cellcell_contact']
#vset = cb_ratio_traj
#vset = cc_ratio_traj
#vset = Xpcat[:, -3]
nf = len(tmSet)
nbins = 20
for iv in range(len(vsetList)):
    vset = vsetList[iv]
    indg = np.where(np.logical_and(np.logical_not(np.isnan(vset)), np.logical_not(np.isinf(vset))))[0]
    plt.clf()
    plt.figure(figsize = (9, 9))
    plt.subplot(5, 3, 1)
    vdist1,xedges1,yedges1 = np.histogram2d(x[indg, 0], x[indg, 1], bins=nbins, weights=vset[indg])
    norm1,xedges1,yedges1 = np.histogram2d(x[indg, 0], x[indg, 1], bins=[xedges1, yedges1])
    vdist1 = np.divide(vdist1, norm1)
    indnan = np.where(np.isnan(vdist1))
    indgood = np.where(np.logical_and(np.logical_not(np.isnan(vdist1)), np.logical_not(np.isinf(vdist1))))
    xedges1c = .5*(xedges1[1:] + xedges1[0:-1])
    yedges1c = .5*(yedges1[1:] + yedges1[0:-1])
    xx,yy = np.meshgrid(xedges1c, yedges1c)
    #vdist1 = np.log(vdist1)
    levels = np.linspace(np.min(vdist1[indgood]), np.max(vdist1[indgood]), 20)
    #levels = np.linspace(0, np.max(vdist1[indgood]), 20)
    cs = plt.contourf(xx, yy, vdist1.T, cmap=plt.cm.jet, levels=levels)
    for ic in range(n_clusters):
        ax = plt.gca()
        ax.arrow(clusters.clustercenters[ic, 0], clusters.clustercenters[ic, 1], dxsav[ic,0], dxsav[ic,1],
                 head_width=.2, linewidth=.5, color='white', alpha=1.0)
    cs.cmap.set_over('darkred')
    cs.cmap.set_under('darkblue')
    cbar = colorbar(cs)
    cbar.set_label(captionset[iv])
    #cbar.set_label('cell-cell boundary fraction')
    #cbar.set_label('speed')
    #cbar.set_label('beta')
    plt.title('combined'+captionset[iv])
    plt.axis('off')
    plt.pause(3)
    #plt.savefig(captionset[iv]+'_flows_comb_'+figid+'.png')
    #plt.subplot(4,3,12)
    #cs=plt.contourf(xx,yy,vdist1.T,cmap=plt.cm.jet,levels=levels)
    #cs.cmap.set_over('darkred')
    #cs.cmap.set_under('darkblue')
    #plt.axis('off'); plt.title('combined')
    #plt.pause(.1)
    for i in range(nf-1):
        plt.subplot(5, 3, i+2)
        indstm = inds_conditions[i]
        indstm = np.intersect1d(indg, indstm)
        vdist1,xedges1,yedges1 = np.histogram2d(x[indstm, 0], x[indstm, 1], bins=nbins, weights=vset[indstm])
        norm1,xedges1,yedges1 = np.histogram2d(x[indstm, 0], x[indstm, 1], bins=[xedges1, yedges1])
        vdist1 = np.divide(vdist1, norm1)
        indnan = np.where(np.isnan(vdist1))
        indgood = np.where(np.logical_and(np.logical_not(np.isnan(vdist1)), np.logical_not(np.isinf(vdist1))))
        cs = plt.contourf(xx, yy, vdist1.T, cmap=plt.cm.jet, levels=levels)
        cs.cmap.set_over('darkred')
        cs.cmap.set_under('darkblue')
        #plt.xlabel('UMAP 1')
        #plt.ylabel('UMAP 2')
        plt.axis('off')
        plt.title(tmSet[i])
        plt.pause(.1)
    plt.savefig(captionset[iv]+'_'+figid+'.png')
"""
#frames for time window
#fl = 0
#fu = nframes 
fl = 72
fu = 120

nbins = 100
indstw = np.where(np.logical_and(indframes_traj < fu, indframes_traj > fl))[0]
"""
prob1,xedges1,yedges1 = np.histogram2d(x[indstw, 0], x[indstw, 1], bins=nbins, density=True)
prob1 = prob1/np.sum(prob1)
prob1 = scipy.ndimage.gaussian_filter(prob1, sigma=2)
xx,yy = np.meshgrid(.5*xedges1[1:] + .5*xedges1[0:-1], .5*yedges1[1:] + .5*yedges1[0:-1])
probSet = [None]*nf
for imf in range(nf):
    indstm = inds_conditions[imf]
    indstwm = np.intersect1d(indstm,indstw)
    prob,xedges2,yedges2 = np.histogram2d(x[indstwm, 0], x[indstwm, 1], bins=[xedges1, yedges1], density=True)
    prob = scipy.ndimage.gaussian_filter(prob, sigma=2)
    prob = prob/np.sum(prob)
    probSet[imf] = prob.copy()
"""
############################ Generate transition matrix ############################
centers_minima  = clusters.cluster_centers.copy()
nclusters = clusters.cluster_centers.shape[0]

# Assign "new data" to cluster centers
indc0 = clusters.transform(x0set).astype(int)
indc1 = clusters.transform(x1set).astype(int)
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
#### Now clusters_minima will have attribute clusters_minima.clustercenters
nclusters = clusters_minima.clustercenters.shape[0]
indc0 = clusters_minima.assign(x0set)
indc1 = clusters_minima.assign(x1set)
wctm.get_transitionMatrixDeeptime(indc0, indc1, nclusters)
P = wctm.Mt.copy()

import pygpcca as gp
gpcca = gp.GPCCA(P, eta=None, z='LM', method='brandts')

# Dump Transition Matrix for further analysis 
tmFileName = 'tMat_'+sysName+'_'+str(trajl)+'_'+date2day+'pc'+str(nPCs)+'u'+str(neigen_umap)+wellsInfo+'.joblib'
with open(tmFileName, 'wb') as fp:
     dump(P, fp, compress = 'zlib')

# Find Eigen Values and Eigen vectors of the transition matrix "P"
H = .5*(P + np.transpose(P)) + .5j*(P - np.transpose(P))
w, v = np.linalg.eig(H)  
w = np.real(w)
indsort = np.argsort(w)
w = w[indsort] # Eigen Values
v = v[:, indsort] # Eigen Vectors
ncomp = n_components_tMat # Keep last "ncomp" eigen vectors
vr = np.multiply(w[-ncomp:], np.real(v[:, -ncomp:]))
vi = np.multiply(w[-ncomp:], np.imag(v[:, -ncomp:]))
vkin = np.append(vr, vi, axis = 1)
#plt.plot(w)

################### Get kinetics of cell (macro) state transitions #####################
from sklearn.cluster import KMeans

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
            statesc,counts = np.unique(stateSet, return_counts = True)
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
       # Assign (micro)states to existing state centers aka macrostates with probabilities less than 'pcut'
       for i in states_plow:
           indstate = np.where(stateSet == i)[0]
           for imin in indstate:
               dists = wctm.get_dmat(np.array([vkinFit[imin, :]]), vkinFit)[0] #closest in eigen space
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

# Module to optimize pcut_final that shows best clustering onto 7 states 
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

get_kstates = True
stateCenters = clusters_minima.clustercenters
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

state_labels = np.array(list(string.ascii_uppercase))[0:nstates]
"""
nbins = probSet[0].shape[0]
plt.close('all')
plt.clf()
plt.figure(figsize = (6, 5))
prob1,xedges1,yedges1 = np.histogram2d(x[:, 0], x[:, 1], bins=nbins, density=True)
prob1 = scipy.ndimage.gaussian_filter(prob1, sigma=2)
xx1,yy1 = np.meshgrid(.5*xedges1[1:] + .5*xedges1[0:-1], .5*yedges1[1:] + .5*yedges1[0:-1])
pts = np.array([xx1.flatten(), yy1.flatten()]).T
indpts = clusters_minima.assign(pts)
states = stateSet[indpts]
states = states[prob1.flatten() > np.min(prob1[prob1 > 0])]
pts = pts[prob1.flatten() > np.min(prob1[prob1 > 0]), :]
plt.contourf(xx1, yy1, prob1.T, cmap=plt.cm.gray_r, levels=20, alpha=.3)
plt.scatter(pts[:,0], pts[:,1], s=10, c=states, cmap=plt.cm.jet, marker='.', alpha=0.5)
plt.scatter(clusters_minima.clustercenters[:, 0], clusters_minima.clustercenters[:, 1], s=100, c=stateSet, cmap=plt.cm.jet)

for istate in range(n_states):
    plt.text(state_centers_minima[istate, 0], state_centers_minima[istate, 1], str(state_labels[istate]))

plt.pause(.1)
plt.savefig('kineticstates_'+figid+'_nS_'+str(nstates)+'.png')
"""
state_probs = np.zeros((nf, n_states))
fl = 72
fu = 120
#fu = nframes
cell_states = clusters_minima
indstw = np.where(np.logical_and(indframes_traj < fu, indframes_traj > fl))[0]
for i in range(nf):
    indstm = inds_conditions[i]
    indstwm = np.intersect1d(indstm, indstw)
    #x0 = x[indstwm, :]
    x0 = Xpcat[indstwm, :]
    indc0 = stateSet[clusters_minima.assign(x0)]
    statesc,counts = np.unique(indc0, return_counts=True)
    state_probs[i, statesc] = counts/np.sum(counts)

state_order = np.arange(n_states).astype(int)
plt.clf()
plt.imshow(state_probs[:, state_order], cmap=plt.cm.gnuplot)
cbar = plt.colorbar()
cbar.set_label('state probabilities')
# We want to show all ticks...
ax = plt.gca()
ax.set_yticks(np.arange(len(tmSet)))
ax.set_xticks(np.arange(nstates))
ax.set_xticklabels(np.array(state_labels)[state_order])
ax.set_yticklabels(tmSet)
# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=10, ha="right",rotation_mode="anchor")
plt.pause(.1);

plt.savefig('stProbs_'+figid+'_nS'+str(nstates)+'pc'+str(nPCs)+'u'+str(neigen_umap)+wellsInfo+'.png')
np.savetxt('stProbs_'+figid+'_nS'+str(nstates)+'pc'+str(nPCs)+'u'+str(neigen_umap)+wellsInfo+'.dat', state_probs)

"""
states_x = stateSet[cell_states.assign(x)]
inds_states = [None]*n_states
for i in range(n_states):
    indstate = np.where(states_x == i)[0]
    inds_states[i] = indstate

vset = Xpcat[:, -3]
plt.clf()
ax = plt.gca()
#vset = np.log2(cc_ratio_traj)
for i in range(n_states):
    ii = state_order[i]
    vplot = ax.violinplot(vset[inds_states[ii]], positions=[i+1], showmeans=True, showextrema=False) #,quantiles=[.05,.95])
    vplot['cmeans'].set_color('black')
    for pc in vplot['bodies']:
        pc.set_facecolor('black')
        #pc.set_edgecolor('black')
        #pc.set_alpha(1)
    plt.pause(.1)

ax.set_xticks(range(1, n_states+1))
ax.set_xticklabels(np.array(state_labels)[state_order])
#plt.ylabel('log2(nuc/cyto cc-ratio)')
#plt.ylabel(r'cell-cell local alignment $\langle \hat{v}_1 \cdot \hat{v}_2 \rangle$')
#plt.ylabel('speed (z-score)')
plt.ylabel('speed (z-score)')
plt.xlabel('states')
plt.pause(.1)
plt.savefig('speed_'+figid+'.png')

from adjustText import adjust_text

istate=5
indcells_traj=inds_states[istate]
indmodels=indtreatment_traj[indcells_traj]
indcells_model=cellinds1_traj[indcells_traj]
for ic in [100,500,1000,1500,2000]: #range(indcells_traj.size):
    model_sctm=modelSet[indgood_models[indmodels[ic]]]
    celltraj=model_sctm.get_cell_trajectory(indcells_model[ic])
    model_sctm.visual=True
    model_sctm.show_cells(celltraj)
    plt.pause(1)
    #plt.savefig('cell'+str(ic)+'_state'+str(istate)+'_'+figid+'.png')
"""
