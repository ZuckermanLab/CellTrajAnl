import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sys, h5py, pickle, os
import subprocess, time
import seaborn as sns
import pandas as pd
import re, scipy, string, itertools
from scipy import stats
from datetime import date
from mpl_toolkits.axes_grid1 import make_axes_locatable
sys.path.append('/home/groups/ZuckermanLab/jalim/instalLocal/celltraj/celltraj')
import trajCellPoseSr

n_macrostates = 8
today = date.today()
date2day = today.strftime("%b%d-%Y")
trajl = int(sys.argv[1])
nPCs = int(sys.argv[2])
nUMP = int(sys.argv[3])
wellInfo = sys.argv[4]

if trajl is None:
    print("Error: Provide trajectory length")
    sys.exit(0)

figid = 'LI204601_P_tlen'+str(trajl)+'_'+date2day+'_nS'+str(n_macrostates) 
datapath = os.getcwd()
seqFile0 = 'MDDligandCombRNAseqLog2TPM_proteinCoding.csv'
seqData0 = pd.read_csv(datapath+'/'+seqFile0)

# Create a filter for log2(TPM) > 0.5 
ind_minexpr = np.where(np.sum(seqData0.iloc[:, 3:] > 0.5, axis=1) >= 3)[0]
geneNames0 = seqData0['hgnc_symbol']
ind_nan = np.where(np.logical_not(pd.isna(seqData0['hgnc_symbol'])))[0] # also genes with names
ensembl_gene_id0 = seqData0['ensembl_gene_id']
ind_expressed = np.intersect1d(ind_minexpr, ind_nan) # Indices of genes that are expressed (excluding NaN value members)
gene_names = geneNames0[ind_expressed] # Genes that are expressed
ensembl_gene_ids = ensembl_gene_id0[ind_expressed]

# read in DEseq2 files
cond0 = ['OSM','EGF','EGFTGFB','TGFB','PBS','OSMTGFBEGF','OSMEGF','OSMTGFB']
ncond0 = len(cond0)
deseq0 = [None]*ncond0
for icond in range(ncond0):
    seqfile = f'analysis_R/deseq2_DE_lfcshrink_ligands_{cond0[icond]}_vs_CTRL.csv'
    deseq0[icond] = pd.read_csv(datapath+'/'+seqfile)
# now match the genes in the two datasets together, we will do into protein coding nGenes0
nGenes0 = ind_expressed.size
inds_dataset0 = np.zeros(ncond0).astype(int)
inds_dataset = inds_dataset0
inds0 = np.where(inds_dataset == 0)[0]
nsamples = inds_dataset.size
x_lfc = np.ones((nsamples, nGenes0))*np.nan # logarithmic fold change
x_padj = np.ones((nsamples, nGenes0))*np.nan
seq_genes0 = deseq0[0]['Unnamed: 0']

for i in range(nGenes0):
    if i%100 == 0:
        print(f'matching gene {str(i)} of {str(nGenes0)}')
    gene_name = ensembl_gene_ids.iloc[i]
    indgene1 = np.where(seq_genes0 == gene_name)[0] 
    if indgene1.size > 0:
        for icond in range(ncond0):
            lfc = deseq0[icond].iloc[ind_expressed[i]]['log2FoldChange']
            padj = deseq0[icond].iloc[ind_expressed[i]]['padj']
            x_lfc[inds0[icond], i] = lfc
            x_padj[inds0[icond], i] = padj
tmSet = cond0 
n_conditions = len(tmSet)
sctm = trajCellPoseSr.cellPoseTraj()

inds_finite = np.where(np.isfinite(np.sum(x_lfc, axis=0)))[0]
x_lfc = x_lfc[:, inds_finite]
x_padj = x_padj[:, inds_finite]
gene_names = gene_names.iloc[inds_finite]
ensembl_gene_ids = ensembl_gene_ids.iloc[inds_finite]
Xpca, pca = sctm.get_pca_fromdata(x_lfc, var_cutoff = .95)
colorSet = ['gray', 'gold', 'red', 'blue', 'orange', 'green', 'purple', 'brown',
            'gray', 'gold', 'lightblue', 'red', 'lightgreen', 'darkred', 'green']

# get morphodynamical state probabilities from imaging analysis: To Change
stProbFile = f"stProbs_LI204601_P_tlen{trajl}_nS{n_macrostates}pc{nPCs}u{nUMP}{wellInfo}wellsComb.dat"

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
state_probs = state_probs_[inds_tmfSet_imaging, :]
print("List of all conditions:", np.array(tmSet)[inds_tmfSet_imaging]) # Test + training sets 

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
visual = None
inds_tm_training = np.arange(n_conditions).astype(int) 
nStates = state_probs.shape[1] # Number of Macroscopic (morphodynamic) states 
seq_genes = gene_names.reset_index(drop = True)
lb = np.zeros(nStates)
ub = np.ones(nStates)*np.inf
  
nGenes = x_lfc.shape[1]
# Element-wise raise 2 to the power of x_lfc --> Eliminate Log @ base 2
x_fc = 2**x_lfc # Log-fold change values for all conditions
      
state_names = np.array(list(string.ascii_uppercase))[0:nStates]
get_counts = True
if get_counts:
   x_fc_states = get_state_decomposition(x_fc, state_probs, ncombinations=500, inds_tm_training=inds_tm_training, 
                                         save_file=None, visual=visual, gene_names=gene_names)
else:
   x_fc_states = np.load('statefc_production_'+figid+'.npy')
             
n_HVG = 5000  # number of highly variable genes
nticks = 30  # number of tick labels on the right y-axis

plt.figure(figsize=(12, 12))

indstates = np.arange(nStates).astype(int)
x_lfc_states = np.log2(x_fc_states + np.min(x_fc)) #add in a null of min measured fc
gvars = np.std(x_lfc_states, axis=0)
indvar = np.argsort(gvars)[-n_HVG:]
tick_genes = gene_names.iloc[indvar][-nticks:]

df_states = pd.DataFrame(data = x_lfc_states[:, indvar][indstates, :].T,
                         index = gene_names.iloc[indvar], columns = state_names[indstates])
# Generate the clustermap
hmap = sns.clustermap(df_states, figsize=(12, 12), cmap="seismic",
                      col_cluster=True, row_cluster=True, vmin=-10, vmax=10)

#plt.text(0.005, 0.01, 'log2 fold-change', rotation = 90)
hmap.cax.set_position([0.04, 0.05, 0.02, 0.2])  
#hmap.cax.set_title('log2 fold-change')
#hmap.cax.title.set_position((0.5, 5))  

plt.setp(hmap.ax_heatmap.yaxis.get_majorticklabels(), fontsize=10, rotation=0)
plt.setp(hmap.ax_heatmap.xaxis.get_majorticklabels(), fontsize=10, rotation=0)

#plt.show()
plt.savefig(figid+'_'+'RNAlevStates.png', dpi = 500, bbox_inches = 'tight')
plt.close()