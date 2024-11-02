import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('/home/groups/ZuckermanLab/jalim/instalLocal/celltraj/celltraj')
import trajCellPoseSr
import h5py
import pickle
import os
import subprocess
import time
import pandas
import re
import scipy
import string, itertools
from scipy import stats
from scipy.integrate import simps
from datetime import date

trajl = None
nstates_init = 7
today = date.today()
date2day = today.strftime("%b%d-%Y")
trajl = int(sys.argv[1])
nPCs = int(sys.argv[2])
nUMP = int(sys.argv[3])
wellInfo = sys.argv[4]

if trajl is None:
    print("Error: Provide trajectory length")
    sys.exit(0)

figid = 'LI204601_P_tlen'+str(trajl)+'_'+date2day+'_nS'+str(nstates_init) 
datapath = os.getcwd()
seqFile0 = 'MDDligandCombRNAseqLog2TPM_proteinCoding.csv'
seqData0 = pandas.read_csv(datapath+'/'+seqFile0)

# Create a filter for log2(TPM) > 0.5 
ind_minexpr = np.where(np.sum(seqData0.iloc[:, 3:] > 0.5, axis=1) >= 3)[0]
geneNames0 = seqData0['hgnc_symbol']
ind_nan = np.where(np.logical_not(pandas.isna(seqData0['hgnc_symbol'])))[0] # also genes with names
ensembl_gene_id0 = seqData0['ensembl_gene_id']
ind_expressed = np.intersect1d(ind_minexpr, ind_nan) # Indices of genes that are expressed (excluding NaN value members)
gene_names = geneNames0[ind_expressed] # Genes that are expressed
ensembl_gene_ids = ensembl_gene_id0[ind_expressed]

# read in DEseq2 files
#cond0 = ['PBS','EGF','OSM','TGFB','OSMEGF','EGFTGFB','OSMTGFB','OSMTGFBEGF']
cond0 = ['OSM','EGF','EGFTGFB','TGFB','PBS','OSMTGFBEGF','OSMEGF','OSMTGFB']
ncond0 = len(cond0)
deseq0 = [None]*ncond0
for icond in range(ncond0):
    seqfile = f'analysis_R/deseq2_DE_lfcshrink_ligands_{cond0[icond]}_vs_CTRL.csv'
    deseq0[icond] = pandas.read_csv(datapath+'/'+seqfile)
"""
cond1 = ['PBS','EGF','HGF','OSM','BMP2','IFNG','TGFB']
ncond1 = len(cond1)
deseq1 = [None]*ncond1
for icond in range(ncond1):
    seqfile = f'analysis_R/MDD_deseq2_DE_lfcshrink_ligand_{cond1[icond]}_vs_ctrl.csv'
    deseq1[icond] = pandas.read_csv(datapath+'/'+seqfile)

cond1 = ['PBS1','EGF1','HGF','OSM1','BMP2EGF','IFNGEGF','EGFTGFB1'] #fixing condition names for consistency
"""
# now match the genes in the two datasets together, we will do into protein coding nGenes0
nGenes0 = ind_expressed.size
inds_dataset0 = np.zeros(ncond0).astype(int)
#inds_dataset1 = np.ones(ncond1).astype(int)
#inds_dataset = np.append(inds_dataset0,inds_dataset1)
inds_dataset = inds_dataset0
inds0 = np.where(inds_dataset == 0)[0]
#inds1 = np.where(inds_dataset == 1)[0]
nsamples = inds_dataset.size
x_lfc = np.ones((nsamples, nGenes0))*np.nan # logarithmic fold change
x_padj = np.ones((nsamples, nGenes0))*np.nan
seq_genes0 = deseq0[0]['Unnamed: 0']
#seq_genes1 = deseq1[0]['Unnamed: 0']

for i in range(nGenes0):
    if i%100 == 0:
        print(f'matching gene {str(i)} of {str(nGenes0)}')
    gene_name = ensembl_gene_ids.iloc[i]
    indgene1 = np.where(seq_genes0 == gene_name)[0] 
    #indgene1 = np.where(seq_genes1 == gene_name)[0] 
    #if indgene1.size > 0:
    if indgene1.size > 0:
        for icond in range(ncond0):
            lfc = deseq0[icond].iloc[ind_expressed[i]]['log2FoldChange']
            padj = deseq0[icond].iloc[ind_expressed[i]]['padj']
            x_lfc[inds0[icond], i] = lfc
            x_padj[inds0[icond], i] = padj
"""
        for icond in range(ncond1):
            lfc = deseq1[icond].iloc[indgene1]['log2FoldChange']
            padj = deseq1[icond].iloc[indgene1]['padj']
            x_lfc[inds1[icond], i] = lfc
            x_padj[inds1[icond], i] = padj
"""
#tmSet = cond0 + cond1
tmSet = cond0 
nLigConds = len(tmSet)
sctm = trajCellPoseSr.cellPoseTraj()

inds_finite = np.where(np.isfinite(np.sum(x_lfc, axis=0)))[0]
x_lfc = x_lfc[:, inds_finite]
x_padj = x_padj[:, inds_finite]
gene_names = gene_names.iloc[inds_finite]
ensembl_gene_ids = ensembl_gene_ids.iloc[inds_finite]
Xpca, pca = sctm.get_pca_fromdata(x_lfc, var_cutoff = .95)
colorSet = ['gray', 'gold', 'red', 'blue', 'orange', 'green', 'purple', 'brown',
            'gray', 'gold', 'lightblue', 'red', 'lightgreen', 'darkred', 'green']

visual = False
if visual:
    plt.figure(figsize = (8, 6))
    from adjustText import adjust_text
    texts = [None]*Xpca.shape[0]
    for i in range(nLigConds):
        plt.scatter(Xpca[i, 0], Xpca[i, 1], s = 50, c = colorSet[i])
        text = plt.text(Xpca[i, 0], Xpca[i, 1], tmSet[i], color = colorSet[i])
        texts[i] = text
    plt.xlabel('PCA1'); plt.ylabel('PCA2')
    adjust_text(texts, arrowprops=dict(arrowstyle = '->', color='black'))
    plt.pause(.1)
    #plt.savefig('crossdata_pca_conditions_'+date2day+'.png')

# get morphodynamical state probabilities from imaging analysis: To Change
stProbFile = 'stProbs_LI204601_P_tlen'+str(trajl)+'_nS'+str(nstates_init)+'pc'+str(nPCs)+'u'+str(nUMP)+wellInfo+'wellsComb.dat' 

if not stProbFile:
  print("ERROR in reading state probability file")
  sys.exit(0)

state_probs_ = np.loadtxt(stProbFile)
tmSet_imaging = np.array(['OSM','EGF','EGFTGFB','TGFB','PBS','OSMTGFBEGF','OSMEGF','OSMTGFB',
                          'PBS1','EGF1','OSM1','TGFB1','OSMEGF1'])
tmfSet = tmSet #so much fun with names
inds_tmfSet_imaging = np.array([]).astype(int)
for imf in range(len(tmfSet)):
    tm = tmfSet[imf]
    inds_tmfSet_imaging = np.append(inds_tmfSet_imaging, np.where(tmSet_imaging == tm)[0])

inds_tmfSet_Imaging = inds_tmfSet_imaging
state_probs_ = state_probs_[inds_tmfSet_imaging, :]
print("List of all conditions:", np.array(tmSet)[inds_tmfSet_imaging]) # Test + training sets 

def get_predictedFC(state_probs_test, statesFC):
    n_test = state_probs_test.shape[0]
    nStates = state_probs_test.shape[1]
    nGenes = statesFC.shape[1]
    x_FC_predicted = np.ones((n_test, nGenes))*np.nan
    
    for itr in range(n_test):
        statep = state_probs_test[itr, :]
        x_FC_predicted[itr, :] = (np.tile(statep, (nGenes, 1))*statesFC.T).sum(-1)
    
    return x_FC_predicted

def get_state_decomposition(x_fc, state_probs, ncombinations=500, inds_tm_training=None,
                            save_file=None, visual=False, verbose=True, nchunk=100, gene_names=None):
    nStates = state_probs.shape[1] # number of morphodynamic states
    ntr = state_probs.shape[0] # training set conditions
    nGenes = x_fc.shape[1]
    ntr_measured = x_fc.shape[0] # log-fold change values of RNA levels corresponding to training set
    if nStates > ntr:
        print(f'error, more states than conditions in state probabilities')
        return
    if nStates > ntr_measured:
        print(f'error, more states than measured bulk conditions')
        return
    x_fc_states = np.ones((nStates, nGenes))*np.nan
    if inds_tm_training is None:
        inds_tm_training = np.arange(ntr).astype(int)
    ntr_training = inds_tm_training.size
    comb_trainarray = np.array(list(itertools.combinations(inds_tm_training, nStates)))
    ncomb = comb_trainarray.shape[0]
    print(f'{ncomb} possible combinations of {ntr} training measurements decomposed into {nStates} states')
    if ncombinations > ncomb:
        ncombinations = ncomb
    print(f'using {ncombinations} of {ncomb} possible training set combinations randomly per feature')
    for ig in range(nGenes): # LOOP OVER NUMBER OF GENES
        # Generate a uniform random sequence from np.arange(ncomb) of size "ncombinations"
        indr = np.random.choice(ncomb, ncombinations, replace=False)
        if ig%nchunk == 0 and verbose:
            print(f'decomposing gene {ig} of {nGenes}')
            if save_file is not None:
                np.save(save_file, x_fc_states)
        v_states_comb = np.zeros((ncombinations, nStates))
        for icomb in range(ncombinations):
            indcomb = comb_trainarray[indr[icomb]] # Pick randomized index to remove bias 
            v_treatments = x_fc[indcomb, ig] # Pick a ligand condition randomly and use its RNA levels
            # Least square linear optimization for each Gene --> solving state_probs*x = v_treatments (fold-change)  
            res = scipy.optimize.lsq_linear(state_probs[indcomb, :], v_treatments, bounds=(lb, ub), verbose=1)
            v_states_comb[icomb, :] = res.x.copy() # x (contribution of each state) is returned from scipy.optimize.lsq_linear 
        v_states = np.mean(v_states_comb, axis=0)
        x_fc_states[:, ig] = v_states.copy() # log-fold change of a selected gene across morphodynamic states
        if ig%nchunk == 0 and visual:
            plt.clf()
            plt.plot(v_states_comb.T, 'k.')
            plt.plot(v_states.T, 'b-', linewidth=2)
            if gene_names is None:
                plt.title(f'{ig} of {nGenes}')
            else:
                plt.title(str(gene_names.iloc[ig])+' gene '+str(ig)+' of '+str(nGenes))
            plt.pause(.1)
    if save_file is not None:
        np.save(save_file, x_fc_states)
    return x_fc_states

plt.clf()
plt.figure(figsize = (9, 6))
ax = plt.gca()

# Initialize lists to store the correlation results
#corr_results_pred, corr_results_rand = [], []

################# MODIFY INDICES AND DATA ACCORDING TO WHETHER A CONDITION IS EXCLUDED FROM TRAINING #################
loco = True

dumpFile = figid+'_LOCO_correlations-PC'+str(nPCs)+'u'+str(nUMP)+wellInfo+'wellsComb.dat'
with open(dumpFile, 'a') as fp:
    for iTest in range(nLigConds):
    
        inds_tm_training = np.arange(nLigConds).astype(int) 
        inds_tm_test = np.array([iTest]).astype(int) # leaving one "LIGAND" condition out (LOCO), just test from combo data
        LOCO = tmSet_imaging[inds_tm_test]
        LOCO = ''.join(LOCO) # convert string list to string
        inds_tm_training = np.setdiff1d(inds_tm_training, inds_tm_test) # remove LOCO index from the training set
        ############# Update state probabilities and log-fold change values as per "inds_tm_training" #############
        state_probs_loco = state_probs_[inds_tm_training, :]
        x_lfc_loco = x_lfc[inds_tm_training, :]
        #print(x_lfc_loco[:,10])
        inds_tmfSet_imaging = np.arange(len(inds_tm_training), dtype = int)
        inds_tm_training = inds_tmfSet_imaging # Update training indices after LOCO
        state_probs = state_probs_loco[inds_tmfSet_imaging, :] # state probabilities of the training set
        nStates = state_probs.shape[1] # Number of Macroscopic (morphodynamic) states 
        ntr = state_probs.shape[0] # Number of training conditions 
        state_probs = state_probs[inds_tmfSet_imaging, 0:nStates]
        seq_genes = gene_names.reset_index(drop = True)
        ntr_training = inds_tm_training.size
        lb = np.zeros(nStates)
        ub = np.ones(nStates)*np.inf
        
        # First pass, just with training set
        nGenes = x_lfc.shape[1]
        # Element-wise raise 2 to the power of x_lfc --> Eliminate Log @ base 2
        x_fc = 2**x_lfc_loco # Log-fold change values of training set
        x_fc_all = 2**x_lfc # Log-fold change values of test & training sets
        
        # state_names = np.array(list(string.ascii_uppercase))[0:nStates] Not sure now...
        get_counts = True
        if get_counts:
            x_fc_states = get_state_decomposition(x_fc, state_probs, ncombinations=500, inds_tm_training=inds_tm_training, 
                                                  save_file=None, visual=visual, gene_names=gene_names)
                                                  #save_file='statefc_production_'+figid+'_'+LOCO+'.npy', visual=visual, gene_names=gene_names)
        else:
            x_fc_states = np.load('statefc_production_'+figid+LOCO+'.npy')
             
        # Predict fold-change values of the test set whereas the model was trained on remaining conditions (training set)
        state_probs_LOCO = state_probs_[inds_tm_test, 0:nStates] # State probabilities of the "Test Set"
        x_fc_predicted = get_predictedFC(state_probs_LOCO, x_fc_states)
        #x_fc_predicted = get_predictedFC(state_probs, x_fc_states)
        x_lfc_predicted = np.log2(x_fc_predicted) # Convert from fold-change to log fold-change
           
        print(f'{tmSet[iTest]} corr: {np.corrcoef(x_lfc[iTest, :], x_lfc_predicted[0, :])[0, 1]:.2f}')
        
        nConds_test = len(inds_tm_test) # Number of Ligand conditions in "Test Set"
        
        ######################### how unique are state probabilities #########################
        nrandom = 500
        corrSet_pred = np.zeros(nConds_test) # Correlation between predicted and real values
        corrSet_predrand = np.zeros((nrandom, nConds_test)) # Correlation between NULL model and predicted values
        corrSet_rand = np.zeros((nrandom, nConds_test)) # Correlation of NULL model and real values
        for ir in range(nrandom):
            state_probs_r = np.zeros_like(state_probs_LOCO) # state probabilities random -> NULL model
            for itr in range(nConds_test):
                rp = np.random.rand(nStates) # Random probability of each training set  
                rp = rp/np.sum(rp)
                state_probs_r[itr, :] = rp.copy()
            x_fc_null = get_predictedFC(state_probs_r, x_fc_states) # Fold-change values as per NULL model state probabilities
            for itr in range(nConds_test):
                lfc_pred = np.log2(x_fc_predicted[itr, :]) # Log-fold change prediction of test set(s)
                lfc_real = np.log2(x_fc_all[iTest, :]) # Log-fold change of test condition(s)
                #.5*x_counts_all[indcombos[i,0],:]+.5*x_counts_all[indcombos[i,1],:]
                lfc_null = np.log2(x_fc_null[itr, :]) # Log-fold change of test set(s) from the NULL model 
                df = pandas.DataFrame(np.array([lfc_pred, lfc_null, lfc_real]).T)
                rhoSet = df.corr().to_numpy() # Convert pandas data type to Numpy array 
                corrSet_pred[itr] = rhoSet[0, 2]
                corrSet_rand[ir, itr] = rhoSet[1, 2]
                corrSet_predrand[ir, itr] = rhoSet[0, 1]
                print(tmfSet[iTest]+f' correlation: prediction {rhoSet[0, 2]:.2f}, null {rhoSet[1, 2]:.2f} prednull {rhoSet[0, 1]:.2f}, ir: {ir} of {nrandom}')
                
        itr = 0 # Index = 0 in case of LOCO
        data_for_condition = corrSet_rand[:, itr] 
        reference_value = corrSet_pred[itr] 
        null_model_exceed_pred = np.where(data_for_condition > reference_value, 1, 0) # binary_sequence
        #null_model_exceed_pred = data_for_condition[data_for_condition > reference_value] # Data values 
         
        if null_model_exceed_pred.size > 0 :
           sum_null = np.sum(null_model_exceed_pred) 
        else: 
           sum_null = 0.
          
        fp.write(f"Model Prediction: {reference_value}, Number of time Null model predicts above the model: {sum_null}\n")    
        
        ################################# Plot model predictions for LOCO  ################################
        vplot = ax.violinplot(corrSet_rand[:, itr], positions = [iTest + 1],
                              showmeans = True, showextrema = False, quantiles = [.025, .975])
        for partname in ('cmeans', 'cquantiles'):
            vp = vplot[partname]
            vp.set_edgecolor('black')
        plt.scatter(iTest + 1, corrSet_pred[itr], s=100, c = 'red', marker = 'd')
        for pc in vplot['bodies']:
            pc.set_facecolor('black')
            pc.set_edgecolor('black')
            # pc.set_alpha(1)
        plt.pause(.1)
    
ax.set_xticks(np.arange(1, len(inds_tmfSet_Imaging) + 1))
ax.set_xticklabels(np.array(tmfSet)[inds_tmfSet_Imaging])
plt.setp(ax.get_xticklabels(), rotation = 30, ha = "right", rotation_mode = "anchor")
plt.ylabel('correlation'); plt.ylim(-1., 1.)
plt.savefig(figid+'_LOCO_correlationsPC'+str(nPCs)+'u'+str(nUMP)+wellInfo+'wellsComb.png')
plt.close()

#np.savetxt(figid+'_LOCO_avgCorrPC'+str(nPCs)+'u'+str(nUMP)+wellInfo+'wellsComb.dat', corr_results_pred)

"""
# make heat map at ligand conditions and cell (macro) states
import seaborn as sns
import pandas as pd
# get list of most top Nv variable genes
Nv = 5000
nticks = 30
plt.clf()
plt.figure(figsize = (16, 10))
gvars = np.std(x_lfc, axis=0)
indvar = np.argsort(gvars)[-Nv:]
tick_genes = gene_names.iloc[indvar][-nticks:]
df_cond = pd.DataFrame(data = x_lfc[:, indvar].T, index = gene_names.iloc[indvar], columns = tmSet)
hmap = sns.clustermap(df_cond, cmap = "seismic", cbar_kws = {'label': 'log2 fold-change', 'orientation': 'horizontal'},
                      cbar_pos = (.05, .9, .3, .05), col_cluster = False, vmin = -10, vmax = 10)
reord_inds = hmap.dendrogram_row.reordered_ind.copy()
tick_locs = np.zeros(nticks)
for itick in range(nticks):
    tick_locs[itick] = np.where(seq_genes[indvar[reord_inds]] == tick_genes.iloc[itick])[0][0]+.5

#hmap.ax_heatmap.set_yticks(tick_locs)
#hmap.ax_heatmap.set_yticklabels(tick_genes)
plt.setp(hmap.ax_heatmap.get_xticklabels(), rotation=40)
plt.pause(.1)
plt.savefig(figid+'_'+LOCO+'_heatmap_cond.png')
plt.close()

plt.clf()
plt.figure(figsize = (16, 10))
indstates = np.arange(nStates).astype(int)
x_lfc_states = np.log2(x_fc_states + np.min(x_fc)) # add in a null of min measured fc
gvars = np.std(x_lfc_states, axis=0)
indvar = np.argsort(gvars)[-Nv:]
df_states = pd.DataFrame(data = x_lfc_states[:, indvar][indstates, :].T, 
                         index = gene_names.iloc[indvar], columns = state_names[indstates])
hmap = sns.clustermap(df_states, cmap = "seismic", cbar_kws = {'label': 'log2 fold-change', 'orientation': 'horizontal'},
                      cbar_pos = (.05, .9, .3, .05), col_cluster = False, vmin = -10, vmax = 10)
plt.pause(.1)
plt.savefig(figid+'_'+LOCO+'_heatmap_states.png')
plt.close()

# redo along estimated paths with new state orders
# get enrichments for transitions
import gseapy as gp

toSet = np.array([1, 2, 3, 4, 5, 6, 7, 1, 0]) 
fromSet = np.array([0, 1, 2, 3, 4, 5, 6, 7, 1]) 
#library_name = 'ChEA_2022'
#library_name = 'GO_Biological_Process_2021'
#library_name = 'MSigDB_Hallmark_2020'
for library_name in ['MSigDB_Hallmark_2020']: #'KEGG_2021_Human','WikiPathways_2019_Human','ChEA_2022']:
    for i in range(fromSet.size):
        lfc = x_lfc_states[toSet[i], :] - x_lfc_states[fromSet[i], :]
        indsgenes = np.argsort(lfc)
        indsgenes = np.flipud(indsgenes)
        gene_list = gene_names.iloc[indsgenes]
        ngenes = gene_list.size
        rnk = pandas.DataFrame(data=lfc[indsgenes], index=gene_list)
        names = gp.get_library_name(organism='Human')
        gene_sets = library_name
        prerank = gp.prerank(rnk = rnk, gene_sets = library_name,
                             processes = 1,
                             combination_num = 1000, # reduce number to speed up testing
                             outdir = f'gsea/states_ord_{library_name}_{figid}/prerank_{state_names[fromSet[i]]}{state_names[toSet[i]]}',
                             format = 'png', seed = 6)
        print(f'gsea/states_ord_{library_name}_{figid}/prerank_{state_names[fromSet[i]]}{state_names[toSet[i]]}')
        prerank.res2d.sort_index().head()

for library_name in ['MSigDB_Hallmark_2020']: #'KEGG_2021_Human','WikiPathways_2019_Human','ChEA_2022']:
    for i in range(nStates):
        lfc = x_lfc_states[i, :]
        indsgenes = np.argsort(lfc)
        indsgenes = np.flipud(indsgenes)
        gene_list = gene_names.iloc[indsgenes]
        ngenes = gene_list.size
        rnk = pandas.DataFrame(data = lfc[indsgenes], index = gene_list)
        names = gp.get_library_name(organism = 'Human')
        gene_sets = library_name
        prerank = gp.prerank(rnk = rnk, gene_sets = library_name,
                             processes = 1,
                             combination_num = 1000, # reduce number to speed up testing
                             outdir = f'gsea/states_ord_{library_name}_{figid}/prerank_{state_names[i]}',
                             format = 'png', seed = 6)
        print(f'gsea/states_ord_{library_name}_{figid}/prerank_{state_names[i]}')
        prerank.res2d.sort_index().head()
"""
