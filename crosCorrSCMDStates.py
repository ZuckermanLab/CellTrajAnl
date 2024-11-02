import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sys, os, time
sys.path.append('/home/groups/ZuckermanLab/jalim/instalLocal/celltraj/celltraj')
import trajCellPoseSr as cellTraj
import h5py
import pickle
import subprocess
import pandas
import re
import scipy
import string, itertools
from scipy import stats
from scipy.integrate import simps
from datetime import date

sctm = cellTraj.cellPoseTraj()

nstates_init = 7
today = date.today()
date2day = today.strftime("%b%d-%Y")
trajl = int(sys.argv[1])
nPCs = int(sys.argv[2])
nUMP = int(sys.argv[3])
wellInfo = sys.argv[4]
trajl_cc = int(sys.argv[5])

if trajl is None:
    print("Error: Provide trajectory length")
    sys.exit(0)

figid = 'LI204601_P_tlen'+str(trajl)+'_'+date2day+'_nS'+str(nstates_init) 
datapath = os.getcwd()

# read cross-correlations for several ligand conditions 
cond0 = ['OSM1','EGF1','EGFTGFB1','TGFB1','PBS1','OSMEGFTGFB','OSMEGF','OSMTGFB']
cond1 = ['OSM2','EGF2','EGFTGFB2','TGFB2','PBS2','OSMEGFTGFB','OSMEGF','OSMTGFB']
cond = []
cond = np.append(cond0[:-3], cond1)
ncond = len(cond)
ncond0 = len(cond0)
lenCCR = trajl_cc  # For the time being, may subject to change later
cross_corr_rep_nuc = np.zeros((ncond0, lenCCR)) # Cross-correlations b/w cell-cycle reporter & nuclear channel images
for icond in range(ncond0):
    cor_file = f'LI204601_P_CCRN_SCT_{trajl_cc}_{cond0[icond]}.dat'
    print("Loading file: ", cor_file)
    cor_file_path = datapath+'/../'+cor_file
    data = np.loadtxt(cor_file_path)
    cross_corr_cond = data[:, 1]
    cross_corr_rep_nuc[icond, :] = cross_corr_cond

cross_corr_rep_nuc = np.array(cross_corr_rep_nuc)

inds_dataset0 = np.zeros(ncond0).astype(int)
#inds_dataset1 = np.ones(ncond1).astype(int)
#inds_dataset = np.append(inds_dataset0, inds_dataset1)
inds_dataset = inds_dataset0
inds0 = np.where(inds_dataset == 0)[0]
#inds1 = np.where(inds_dataset == 1)[0]
nsamples = inds_dataset.size
#tmSet = cond0 + cond1
tmSet = cond0 
nLigConds = len(tmSet)

# get morphodynamical state probabilities from imaging analysis: To Change
scmdFilePath = "/home/groups/ZuckermanLab/jalim/LI204601_INCUCYTE/bulkRNAseq/scripts/"
stProbFile = scmdFilePath+'stProbs_LI204601_P_tlen'+str(trajl)+'_nS'+str(nstates_init)+'pc'+str(nPCs)+'u'+str(nUMP)+wellInfo+'wellsComb.dat' 

if not stProbFile:
  print("ERROR in reading state probability file")
  sys.exit(0)

state_probs = np.loadtxt(stProbFile)
tmSet_imaging = np.array(['OSM1','EGF1','EGFTGFB1','TGFB1','PBS1','OSMEGFTGFB','OSMEGF','OSMTGFB',
                          'OSM2','EGF2','EGFTGFB2','TGFB2','PBS2'])
tmfSet = tmSet #so much fun with names
inds_tmfSet_imaging = np.array([]).astype(int)
for imf in range(len(tmfSet)):
    tm = tmfSet[imf]
    inds_tmfSet_imaging = np.append(inds_tmfSet_imaging, np.where(tmSet_imaging == tm)[0])

inds_tmfSet_Imaging = inds_tmfSet_imaging
state_probs = state_probs[inds_tmfSet_imaging, :]
print("List of all conditions:", np.array(tmSet)[inds_tmfSet_imaging]) # Test + training sets 

def get_state_decomposition(cross_corr_rep_nuc, state_probs, ncombinations=500, inds_tm_training=None,
                            save_file=None, visual=False, verbose=True, nchunk=100):
    nStates = state_probs.shape[1] # number of morphodynamic states
    ntr = state_probs.shape[0] # training set conditions
    ntr_measured = cross_corr_rep_nuc.shape[0] # log-fold change values of RNA levels corresponding to training set
    nCCRNs = cross_corr_rep_nuc.shape[1] # Number of CC values along a single-cell trajectory
    if nStates > ntr:
        print(f'error, more states than conditions in state probabilities')
        return
    if nStates > ntr_measured:
        print(f'error, more states than measured bulk conditions')
        return
    cross_corr_rep_nuc_states = np.ones((nStates, nCCRNs))*np.nan
    if inds_tm_training is None:
        inds_tm_training = np.arange(ntr).astype(int)
    ntr_training = inds_tm_training.size
    comb_trainarray = np.array(list(itertools.combinations(inds_tm_training, nStates)))
    ncomb = comb_trainarray.shape[0]
    print(f'{ncomb} possible combinations of {ntr} training measurements decomposed into {nStates} states')
    if ncombinations > ncomb:
        ncombinations = ncomb
    print(f'using {ncombinations} of {ncomb} possible training set combinations randomly per feature')
    for icc in range(nCCRNs):
        # Generate a uniform random sequence from np.arange(ncomb) of size "ncombinations"
        indr = np.random.choice(ncomb, ncombinations, replace=False)
        v_states_comb = np.zeros((ncombinations, nStates))
        for icomb in range(ncombinations):
            indcomb = comb_trainarray[indr[icomb]] # Pick randomized index to remove bias 
            # Pick a ligand condition randomly and use its cross-correlation values
            v_treatments = cross_corr_rep_nuc[indcomb, icc] 
            # Least square linear optimization for each Gene --> solving state_probs*x = v_treatments (fold-change)  
            res = scipy.optimize.lsq_linear(state_probs[indcomb, :], v_treatments, bounds=(lb, ub), verbose=1)
            v_states_comb[icomb, :] = res.x.copy() # x (contribution of each state) is returned from scipy.optimize.lsq_linear 
        v_states = np.mean(v_states_comb, axis=0)
        cross_corr_rep_nuc_states[:, icc] = v_states.copy() # log-fold change of a selected gene across morphodynamic states
    if save_file is not None:
        np.save(save_file, cross_corr_rep_nuc_states)
    return cross_corr_rep_nuc_states

# Drop Cross-correlations & state probabilities for the ligand
# condition "PBS" to make it "7" conditions "7" Morphodynamic states 
ind_pbs = 4 
state_probs = np.delete(state_probs, ind_pbs, axis = 0)
cross_corr_rep_nuc = np.delete(cross_corr_rep_nuc, ind_pbs, axis = 0)

nStates = state_probs.shape[1] # number of morphodynamic states
lb = np.zeros(nStates)
ub = np.ones(nStates)*np.inf
      
get_counts = True
if get_counts:
   cross_corr_rep_nuc_states = get_state_decomposition(cross_corr_rep_nuc, state_probs, ncombinations = 500,
                                                       inds_tm_training= None, 
                                                       save_file='statDecom_'+figid+'.npy')
else:
   cross_corr_rep_nuc_states = np.load('statDecom_'+figid+'.npy')

"""
# Plot the state decomposed cross-correlations
plt.clf()

fig, axs = plt.subplots(figsize= (10, 5)) 

n_frames = cross_corr_rep_nuc.shape[1]
frames = np.arange(1, n_frames + 1)
colors = ['red', 'blue', 'green', 'pink', 'purple', 'magenta', 'cyan']
colors = np.array(colors)

for i in range(nStates):
    axs.plot(frames, cross_corr_rep_nuc_states[i], color=colors[i])

plt.xlabel('Frames')
plt.ylabel('Cross correlations across Morphodynamic states')
fig.tight_layout()
plt.savefig(figid+'_CCvalsSCMTstatesPC'+str(nPCs)+'u'+str(nUMP)+wellInfo+'wellsComb.png',
            dpi = 400, bbox_inches = 'tight')
"""
