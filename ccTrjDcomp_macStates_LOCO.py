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

trajl = None
nstates_init = 7
today = date.today()
date2day = today.strftime("%b%d-%Y")
trajl = int(sys.argv[1])
nPCs = int(sys.argv[2])
nUMP = int(sys.argv[3])
wellInfo = sys.argv[4]
corr_time_pt = int(sys.argv[5])

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
cross_corr_rep_nuc0 = [] # Cross-correlations b/w cell-cycle reporter & nuclear channel images
for icond in range(ncond0):
    cor_file = f'LI204601_P_avgCorrRepNuc_{cond0[icond]}.dat'
    print("Loading file: ", cor_file)
    cor_file_path = datapath+'/../'+cor_file
    data = np.loadtxt(cor_file_path)
    cross_corr_cond = data[:, 1]
    hours_cond = data[:, 0].astype(int) / 4 # Convert Frame number to Hours 
    if corr_time_pt in hours_cond: 
       indc_tpt = np.where(hours_cond == corr_time_pt)[0]
       cross_corr_rep_nuc0.append(cross_corr_cond[indc_tpt])

cross_corr_rep_nuc = np.array(cross_corr_rep_nuc0)

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

state_probs_ = np.loadtxt(stProbFile)
tmSet_imaging = np.array(['OSM1','EGF1','EGFTGFB1','TGFB1','PBS1','OSMEGFTGFB','OSMEGF','OSMTGFB',
                          'OSM2','EGF2','EGFTGFB2','TGFB2','PBS2'])
tmfSet = tmSet #so much fun with names
inds_tmfSet_imaging = np.array([]).astype(int)
for imf in range(len(tmfSet)):
    tm = tmfSet[imf]
    inds_tmfSet_imaging = np.append(inds_tmfSet_imaging, np.where(tmSet_imaging == tm)[0])

inds_tmfSet_Imaging = inds_tmfSet_imaging
state_probs_ = state_probs_[inds_tmfSet_imaging, :]
print("List of all conditions:", np.array(tmSet)[inds_tmfSet_imaging]) # Test + training sets 

def get_predictedFC(state_probs_test, statesCCRN):
    n_test = state_probs_test.shape[0]
    nStates = state_probs_test.shape[1]
    cross_corr_rep_nuc_predicted = np.ones((n_test))*np.nan
    
    for itr in range(n_test):
        statep = state_probs_test[itr, :]
        cross_corr_rep_nuc_predicted[itr] = (np.tile(statep, 1)*statesCCRN.T).sum(-1)
    
    return cross_corr_rep_nuc_predicted

def get_state_decomposition(cross_corr_rep_nuc, state_probs, ncombinations=500, inds_tm_training=None,
                            save_file=None, visual=False, verbose=True, nchunk=100):
    nStates = state_probs.shape[1] # number of morphodynamic states
    ntr = state_probs.shape[0] # training set conditions
    ntr_measured = cross_corr_rep_nuc.shape[0] # log-fold change values of RNA levels corresponding to training set
    if nStates > ntr:
        print(f'error, more states than conditions in state probabilities')
        return
    if nStates > ntr_measured:
        print(f'error, more states than measured bulk conditions')
        return
    cross_corr_rep_nuc_states = np.ones((nStates))*np.nan
    if inds_tm_training is None:
        inds_tm_training = np.arange(ntr).astype(int)
    ntr_training = inds_tm_training.size
    comb_trainarray = np.array(list(itertools.combinations(inds_tm_training, nStates)))
    ncomb = comb_trainarray.shape[0]
    print(f'{ncomb} possible combinations of {ntr} training measurements decomposed into {nStates} states')
    if ncombinations > ncomb:
        ncombinations = ncomb
    print(f'using {ncombinations} of {ncomb} possible training set combinations randomly per feature')
    # Generate a uniform random sequence from np.arange(ncomb) of size "ncombinations"
    indr = np.random.choice(ncomb, ncombinations, replace=False)
    v_states_comb = np.zeros((ncombinations, nStates))
    for icomb in range(ncombinations):
        indcomb = comb_trainarray[indr[icomb]] # Pick randomized index to remove bias 
        v_treatments = cross_corr_rep_nuc[indcomb] # Pick a ligand condition randomly and use its cross-correlation values
        v_treatments = v_treatments.flatten()
        # Least square linear optimization for each Gene --> solving state_probs*x = v_treatments (fold-change)  
        res = scipy.optimize.lsq_linear(state_probs[indcomb, :], v_treatments, bounds=(lb, ub), verbose=1)
        v_states_comb[icomb, :] = res.x.copy() # x (contribution of each state) is returned from scipy.optimize.lsq_linear 
    v_states = np.mean(v_states_comb, axis=0)
    cross_corr_rep_nuc_states[:] = v_states.copy() # log-fold change of a selected gene across morphodynamic states
    if save_file is not None:
        np.save(save_file, cross_corr_rep_nuc_states)
    return cross_corr_rep_nuc_states

# Function to calculate Z-score
def z_score(value, mean, std):
    return (value - mean) / std

plt.clf()
plt.figure(figsize = (9, 6))
ax = plt.gca()

visual = False
# Initialize lists to store the correlation results
#corr_results_pred, corr_results_rand = [], []

################# MODIFY INDICES AND DATA ACCORDING TO WHETHER A CONDITION IS EXCLUDED FROM TRAINING #################
loco = True

dumpFile = figid+'_LOCO_crosCorrPC'+str(nPCs)+'u'+str(nUMP)+wellInfo+'wellsComb.dat'
with open(dumpFile, 'a') as fp:
    for iTest in range(nLigConds):
    
        inds_tm_training = np.arange(nLigConds).astype(int) 
        inds_tm_test = np.array([iTest]).astype(int) # leaving one "LIGAND" condition out (LOCO), just test from combo data
        LOCO = tmSet_imaging[inds_tm_test]
        LOCO = ''.join(LOCO) # convert string list to string
        inds_tm_training = np.setdiff1d(inds_tm_training, inds_tm_test) # remove LOCO index from the training set
        ############# Update state probabilities and log-fold change values as per "inds_tm_training" #############
        state_probs_loco = state_probs_[inds_tm_training, :]
        cross_corr_rep_nuc_all = cross_corr_rep_nuc
        cross_corr_rep_nuc_loco = cross_corr_rep_nuc[inds_tm_training]
        #print("Cross correlations LOCO: ",cross_corr_rep_nuc_loco)
        inds_tmfSet_imaging = np.arange(len(inds_tm_training), dtype = int)
        inds_tm_training = inds_tmfSet_imaging # Update training indices after LOCO
        state_probs = state_probs_loco[inds_tmfSet_imaging, :] # state probabilities of the training set
        nStates = state_probs.shape[1] # Number of Macroscopic (morphodynamic) states 
        ntr = state_probs.shape[0] # Number of training conditions 
        state_probs = state_probs[inds_tmfSet_imaging, 0:nStates]
        ntr_training = inds_tm_training.size
        lb = np.zeros(nStates)
        ub = np.ones(nStates)*np.inf
        
        get_counts = True
        if get_counts:
            cross_corr_rep_nuc_states = get_state_decomposition(cross_corr_rep_nuc_loco, state_probs, ncombinations=500,
                                                                inds_tm_training=inds_tm_training, 
                                                                save_file='statDecom_'+figid+LOCO+'.npy')
        else:
            cross_corr_rep_nuc_states = np.load('statDecom_'+figid+LOCO+'.npy')
             
        # Predict cross-correlation values of the test set whereas the model was trained on remaining conditions (training set)
        state_probs_LOCO = state_probs_[inds_tm_test, 0:nStates] # State probabilities of the "Test Set"
        cross_corr_rep_nuc_predicted = get_predictedFC(state_probs_LOCO, cross_corr_rep_nuc_states)
         
        nConds_test = len(inds_tm_test) # Number of Ligand conditions in "Test Set"
        
        corrSet_pred = cross_corr_rep_nuc_predicted # cross-correlation values prediction of test set(s)
        corrSet_real = cross_corr_rep_nuc_all[iTest] # cross-correlation values of test condition(s)
        ######################### how unique are state probabilities #########################
        nrandom = 500
        corrSet_rand = np.zeros(nrandom) # Correlation of NULL model and real values
        for ir in range(nrandom):
            state_probs_r = np.zeros_like(state_probs_LOCO) # state probabilities random -> NULL model
            for itr in range(nConds_test):
                rp = np.random.rand(nStates) # Random probability of each training set  
                rp = rp/np.sum(rp)
                state_probs_r[itr, :] = rp.copy()
            cross_corr_rep_nuc_null = get_predictedFC(state_probs_r, cross_corr_rep_nuc_states) # cross-correlations as per NULL model state probabilities
            corrSet_rand[ir] = cross_corr_rep_nuc_null # cross-correlation values of test set(s) from the NULL model 
                
        # Calculate the mean and standard deviation of the null model distribution
        null_mean = np.mean(corrSet_rand)
        null_std = np.std(corrSet_rand)
        # Calculate Z-scores for model prediction and experimentally measured
        z_score_mod_pred = z_score(corrSet_pred, null_mean, null_std)
        z_score_exp_measured = z_score(corrSet_real, null_mean, null_std)
         
        """ 
        itr = 0 # Index = 0 in case of LOCO
        data_for_condition = corrSet_rand[:] # Null model predictions 
        reference_value = corrSet_pred # Predictions of this model
        null_model_exceed_pred = np.where(data_for_condition > reference_value, 1, 0) # binary_sequence
        #null_model_exceed_pred = data_for_condition[data_for_condition > reference_value] # Data values 
         
        if null_model_exceed_pred.size > 0 :
           sum_null = np.sum(null_model_exceed_pred) 
        else: 
           sum_null = 0.
          
        fp.write(f"Model Prediction: {reference_value}, Number of time Null model predicts above the model: {sum_null}\n")    
        """
         
        fp.write(f"Z-score of Model Prediction: {z_score_mod_pred}, and experimentally measured: {z_score_exp_measured}\n")    
         
        ################################# Plot model predictions for LOCO  ################################
        vplot = ax.violinplot(corrSet_rand[:], positions = [iTest + 1],
                              showmeans = True, showextrema = False, quantiles = [.025, .975])
        for partname in ('cmeans', 'cquantiles'):
            vp = vplot[partname]
            vp.set_edgecolor('black')
        if iTest == 0:
           plt.scatter(iTest + 1, corrSet_pred, s=100, c = 'red', marker = 'd', label = 'Model predicted')
           plt.scatter(iTest + 1, corrSet_real, s=100, c = 'blue', marker = 'd', label = 'Experimentally measured')
           plt.legend(loc='best')
        else:   
           plt.scatter(iTest + 1, corrSet_pred, s=100, c = 'red', marker = 'd')
           plt.scatter(iTest + 1, corrSet_real, s=100, c = 'blue', marker = 'd')
        for pc in vplot['bodies']:
            pc.set_facecolor('black')
            pc.set_edgecolor('black')
            # pc.set_alpha(1)
        plt.pause(.1)
    
ax.set_xticks(np.arange(1, len(inds_tmfSet_Imaging) + 1))
ax.set_xticklabels(np.array(tmfSet)[inds_tmfSet_Imaging])
plt.setp(ax.get_xticklabels(), rotation = 30, ha = "right", rotation_mode = "anchor")
plt.ylabel('Cross correlations')
plt.title('Model predicted, experimentally measured, and Null model cross-correlations')
#plt.ylim(-1., 1.)
plt.tight_layout()
plt.savefig(figid+'_LOCO_crosCorrPC'+str(nPCs)+'u'+str(nUMP)+wellInfo+'wellsComb.png', dpi = 300, bbox_inches = 'tight')
plt.close()
