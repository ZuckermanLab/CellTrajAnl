import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import numpy as np
import pandas as pd
import sys, os, time, math
import h5py 
sys.path.append('/home/groups/ZuckermanLab/jalim/instalLocal/celltraj/celltraj')
import jcTrajectory_CP as cellTraj
import pickle, subprocess
import umap, scipy, json 
from csaps import csaps
import string, ast 
from joblib import dump, load
from datetime import date
from mpl_toolkits.axes_grid1 import make_axes_locatable

from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix,\
precision_score, f1_score, recall_score, ConfusionMatrixDisplay
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler

from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Reshape, Dense, Input, Dropout, LeakyReLU, BatchNormalization
from keras.optimizers import Adam, RMSprop, SGD, Adagrad, Adadelta, Nadam
from keras.regularizers import l2, l1
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical

import ray
from ray import tune

# Get the number of CPUs from the environment variable
n_cpus = int(os.environ['NCPU'])

trajl = 1
today = date.today()
date2day = today.strftime("%b%d-%Y")
sysName = 'LI204601_P'
figid = f"{sysName}_{date2day}"
fovs = ['A2_1', 'A2_2', 'A2_3', 'B2_1', 'B2_3', 'B2_4']
nfovs = len(fovs)
pathSet = '/home/groups/ZuckermanLab/jalim/LI204601_INCUCYTE/segsCellPose/bayesianTrackTest/'
modelList = [None]*(nfovs)
modelList_conditions = np.zeros(nfovs).astype(int)
for i in range(nfovs):
    modelList_conditions[i] = i
    modelList[i] = f"{pathSet}{sysName}_{fovs[i]}"
    #print("Model Info: ",modelList[i])

nmodels = len(modelList)
modelSet = [None]*nmodels
indgood_models = np.array([]).astype(int)
for i in range(nmodels):
    try:
        objFile = modelList[i]+'.obj'
        objFileHandler = open(objFile,'rb')
        modelSet[i] = pickle.load(objFileHandler)
        print(f"loaded {objFile} with {modelSet[i].cells_indSet.size} cells")
        objFileHandler.close()
        test = len(modelSet[i].linSet)
        indgood_models = np.append(indgood_models, i)
    except:
        print("ERROR in reading *.obj files")
        sys.exit(0)

n_frames = 193 # Total number of frames (image snapshots) in one condition per FOVs
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
for i in indgood_models:
    modelSet[i].Xf[np.isnan(modelSet[i].Xf)] = 0.0
n_COMfeatures = 3
Xf_com0 = np.zeros((0, n_COMfeatures))
for i in indgood_models:
    Xf_com0 = np.append(Xf_com0,modelSet[i].Xf_com, axis = 0)

av_dx = np.nanmean(Xf_com0[:, 0])
std_dx = np.nanstd(Xf_com0[:, 0])
for i in indgood_models:
    modelSet[i].Xf_com[:, 0] = (modelSet[i].Xf_com[:, 0] - av_dx)/std_dx
wctm = cellTraj.Trajectory() # import Trajectory object 
# Cell features: Zernike (49), Haralick (13), Shape (15), Boundary (15) --> total 92
n_features = modelSet[indgood_models[0]].Xf.shape[1]
Xf = np.zeros((0, n_features))
indtreatment = np.array([])
indcellSet = np.array([])

for i in indgood_models:
    Xf = np.append(Xf, modelSet[i].Xf, axis = 0)
    # Indices for each model for later access using them
    indtreatment = np.append(indtreatment, i*np.ones(modelSet[i].Xf.shape[0])) 
    indcellSet = np.append(indcellSet, modelSet[i].cells_indSet)

indtreatment = indtreatment.astype(int)
indcellSet = indcellSet.astype(int)
for i in indgood_models:
    indsf = np.where(indtreatment == i)[0]
    modelSet[i].Xf = Xf[indsf, :]
self = wctm
all_trajSet = [None]*nmodels
for i in indgood_models:
    print(f"Get single-cell trajectories of model: {i}")
    modelSet[i].get_unique_trajectories()
    all_trajSet[i] = modelSet[i].trajectories.copy()

latent_dim=256 # Latent (Bottleneck) Dimension of Variational Auto-encoder 
path_vaeFeat="/home/exacloud/gscratch/ZuckermanLab/jalim/o2vae/"
for i in indgood_models:
    data_vae = np.load(f'{path_vaeFeat}features_vae{latent_dim}_{sysName[:-1]}{fovs[i]}.npz')
    features_vae = data_vae['embeddings']
    #labels_vae = data_vae['labels']
    modelSet[i].Xf_vae = features_vae

# Get snippets along with their full single-cell trajectory indices  
def get_snippets_with_traj_inds(self, seg_length): 
    n_sctraj = len(self.trajectories) # Number of Single-Cell Trajectories 
    traj_segSet = np.zeros((0, seg_length)).astype(int)
    ind_map_snippet_fulltraj = np.array([])
    for ind_traj in range(n_sctraj):
        cell_traj = self.trajectories[ind_traj] # Select a single-cell trajectory
        traj_len = cell_traj.size
        #print("Length of a Single-Cell Trajectory:",traj_len)
        if traj_len >= seg_length:
            for ic in range(traj_len - seg_length):
                traj_seg = cell_traj[ic:ic+seg_length]
                traj_segSet = np.append(traj_segSet, traj_seg[np.newaxis, :], axis = 0)
                # Save indices of all snippets corresponding to "FULL" single-cell trajectory 
                ind_map_snippet_fulltraj = np.append(ind_map_snippet_fulltraj, ind_traj)
                #print("Indices to map snippets to the full trajectory:",ind_map_snippet_fulltraj)
    return ind_map_snippet_fulltraj, traj_segSet

trajectory_lengths = np.array([1, 4, 8, 20, 40])
trajl = trajectory_lengths[0]
Xf_traj = np.zeros((0, modelSet[0].Xf_vae.shape[1]*trajl))
indtreatment_traj = np.array([])
indstack_traj = np.array([])
indframes_traj = np.array([])
indmodel_traj_snippets = np.array([])
for i in indgood_models:
    print(f'Building trajectory data for model: {i}')
    modelSet[i].trajectories = all_trajSet[i].copy() # ALL Single-Cell trajectories 
    modelSet[i].trajl = trajl # Trajectory snippet length 
    # Get trajectory snippets of (all trajectories) a given length in a sliding window and mapped with single-cell trajectory indices 
    modelSet[i].snippet_map_fulltraj_inds, modelSet[i].traj = get_snippets_with_traj_inds(modelSet[i], trajl)
    data = modelSet[i].Xf_vae[modelSet[i].traj, :] 
    data = data.reshape(modelSet[i].traj.shape[0], modelSet[i].Xf_vae.shape[1]*trajl)
    indgood = np.where(np.sum(np.isnan(data), axis = 1) == 0)[0] # Consider models as "Good" that don't have NaN in "data" 
    data = data[indgood, :]
    modelSet[i].traj = modelSet[i].traj[indgood, :] # Cleaned trajectory snippets if any NaN 
    modelSet[i].snippet_map_fulltraj_inds = modelSet[i].snippet_map_fulltraj_inds[indgood]
    # Store all trajectory snippets of a given length (picked in a sliding window) 
    Xf_traj = np.append(Xf_traj, data, axis = 0) 
    indtreatment_traj = np.append(indtreatment_traj, i*np.ones(data.shape[0])) # Indices of Treatments (Models) Along Trajectory Snippets 
    indstacks = modelSet[i].cells_imgfileSet[modelSet[i].traj[:, 0]]
    indstack_traj = np.append(indstack_traj, indstacks)
    ind_frames = modelSet[i].cells_frameSet[modelSet[i].traj[:, 0]].astype(int) # Frame indices at the start of snippets
    ind_frames = ind_frames + trajl # Frame indices at the end of snippets
    indframes_traj = np.append(indframes_traj, ind_frames) # Starting Frame Indices of ALL snippets
    indtraj_snippets = modelSet[i].snippet_map_fulltraj_inds
    indmodel_traj_snippets = np.append(indmodel_traj_snippets, indtraj_snippets) # Save for all models: map of snippets to the sc trajectories

def get_cellCycInfoTrajs(file_info):
    with open(file_info, 'r') as fp:
        lines = fp.readlines()

    nuc2cytoRatio = []
    CC_vals = []
    frames = []

    for line1, line2, line3 in zip(lines[::3], lines[1::3], lines[2::3]):
        line1 = np.array(line1.strip()[1:-1].split(', '), dtype=float)
        line2 = np.array(line2.strip()[1:-1].split(', '), dtype=float)
        line3 = np.array(line3.strip()[1:-1].split(', '), dtype=int)

        mask = ~np.isnan(line1)
        line1 = line1[mask]
        line2 = line2[mask]
        line3 = line3[mask]

        nuc2cytoRatio.append(line1)
        CC_vals.append(line2)
        frames.append(line3)

    return nuc2cytoRatio, CC_vals, frames

file_path = '/home/groups/ZuckermanLab/jalim/LI204601_INCUCYTE/cellCycInfoRawReporterImgs/'
sname_reporter = "LI204601_G"
X = [None] * nfovs
y = [None] * nfovs

for i in range(nfovs):
    # Get cross-correlation values along all single-cell trajectories  
    cc_filename = f"{file_path}{sname_reporter}_{fovs[i]}.dat"
    n2c_ratio, cc_vals, frame_numbers = get_cellCycInfoTrajs(cc_filename)

    valid_indstm = []
    nuc2Cyto_ratio_last_frames = []
    model_indc = i
    indstm = np.where(indtreatment_traj == model_indc)[0]
    fid_snippets = indframes_traj[indstm].astype(int) # Map frame indices of snippets
    indc_map_fulltraj_snippets = indmodel_traj_snippets[indstm].astype(int) # Map Indices of Full Trajectory to Snippets 
    
    for j, ind_fulltraj in enumerate(indc_map_fulltraj_snippets):
        # Look for a snippet frame within its full trajectory
        possible_indices = np.where(frame_numbers[ind_fulltraj] == fid_snippets[j])[0]
        if len(possible_indices) > 0:
            nbc_ratio = n2c_ratio[ind_fulltraj][possible_indices[0]]
            nuc2Cyto_ratio_last_frames.append(nbc_ratio)
            valid_indstm.append(indstm[j]) 
         
    nuc2Cyto_ratio_last_frames = np.array(nuc2Cyto_ratio_last_frames)
    X[i] = Xf_traj[valid_indstm]  # Use the valid indices to index into Xf_traj
    y[i] = nuc2Cyto_ratio_last_frames
    print(f"Preparing data of {fovs[i]} field of view.")

def trim_cc_vals(ratio_values, num_bins, target_density):
    df = pd.DataFrame(ratio_values, columns=['CC'])
    df['original_index'] = df.index # Include original indices in the DataFrame
    
    counts, bin_edges = np.histogram(df['CC'], bins=num_bins) # Calculate histogram without plotting
    df['bin'] = pd.cut(df['CC'], bins=bin_edges, labels=False, include_lowest=True) # Create a bin label based on the bin_edges

    trimmed_data = []
    
    for i in range(num_bins):
        bin_filter = (df['bin'] == i) # Filter the DataFrame to get data only in this bin
        bin_data = df[bin_filter]

        # If the number of items in the bin is greater than target_density, sample down
        if bin_data.shape[0] > target_density:
           sampled_data = bin_data.sample(n=target_density, random_state=42)
        else:
           sampled_data = bin_data
        trimmed_data.append(sampled_data) # Append the sampled or full bin data to the list
    
    trimmed_df = pd.concat(trimmed_data) # Concatenate all trimmed data back into a DataFrame
    trimmed_df = trimmed_df.sort_values('original_index') # Sorting by original index to preserve the original data order
    
    return trimmed_df

scaler_nn = MinMaxScaler() # Scale features in the range [0, 1]

fovs_indc = [i for i in range(nfovs)]
fovs_indc = np.array(fovs_indc)
fovs_inds_test = np.array([3, 4, 5])
train_fovs = np.setdiff1d(fovs_indc, fovs_inds_test)

X_train = np.zeros((0, X[0].shape[1]))
y_train = np.zeros(0)
for it_fov in train_fovs:
    X_train = np.append(X_train, X[it_fov], axis = 0)
    y_train = np.append(y_train, y[it_fov], axis = 0) # Nuc/Cyto Ratio

target_density = 2000; n_bins = 50 
############################ Trim Training Data ############################ 
trimmed_cc_train = trim_cc_vals(y_train, n_bins, target_density)
original_indices_train = []
y_train_trim = []
    
for it in range(trimmed_cc_train['original_index'].shape[0]):
    original_indices_train.append(trimmed_cc_train['original_index'].iloc[it])
    y_train_trim.append(trimmed_cc_train['CC'].iloc[it])
    
original_indices_train = np.array(original_indices_train)
y_train_trim = np.array(y_train_trim)
X_train_trim = X_train[original_indices_train, :]

######### Scale All Features in the Same Range #########
X_train_nn = scaler_nn.fit_transform(X_train_trim)
test_fov = 5 # Select Test FOV
X_test = X[test_fov]
y_test = y[test_fov]
X_test_nn = scaler_nn.transform(X_test)

def class_conditions(data, boundaries):
    conditions = []
    if len(boundaries) == 1:  # Special case for two classes
        conditions.append(data < boundaries[0])
        conditions.append(data >= boundaries[0])
    else:
        for i in range(len(boundaries) + 1):
            if i == 0:
                condition = data < boundaries[i]
            elif i == len(boundaries):
                condition = data >= boundaries[i-1]
            else:
                condition = (data >= boundaries[i-1]) & (data < boundaries[i])
            conditions.append(condition)
    return conditions

def multiclass_classification(data_train, data_test, boundaries):
    class_labels = list(range(len(boundaries) + 1))
    if len(boundaries) == 1:  # Special case for two classes
        class_labels = [0, 1]
    num_classes = len(class_labels)
    
    y_train_class = np.select(class_conditions(data_train, boundaries), class_labels)
    y_test_class = np.select(class_conditions(data_test, boundaries), class_labels)
    
    y_train_class_onehot = to_categorical(y_train_class)
    y_test_class_onehot = to_categorical(y_test_class)
    
    return y_train_class, y_test_class, y_train_class_onehot, y_test_class_onehot

class_boundaries = [0.90, 2.70]  # For 3 classes
#clas_boundaries = [0.88] # For 2 classes
num_classes = int(len(class_boundaries) + 1)

y_train_class, y_test_class, y_train_class_onehot, y_test_class_onehot = multiclass_classification(y_train_trim,
                                                                                                   y_test, 
                                                                                                   class_boundaries)
num_classes = len(class_boundaries)
input_shape = (X_train_nn.shape[1], ) # Define input shape
# Oversample the minority class
over_sampler = RandomOverSampler(random_state=42, sampling_strategy='not majority')
X_train_resampled, y_train_resampled = over_sampler.fit_resample(X_train_nn, y_train_class)

def create_model(config, input_shape, num_classes):
    inputs = Input(shape=input_shape)
    x = Reshape((input_shape[0], 1, 1))(inputs)

    x = Conv2D(config['conv1_filters'], kernel_size=(3, 1), padding='same', kernel_regularizer=l2(config['conv1_l2_reg']))(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 1))(x)
    x = Dropout(config['dropout1'])(x)

    x = Conv2D(config['conv2_filters'], kernel_size=(3, 1), padding='same', kernel_regularizer=l1(config['conv2_l1_reg']))(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 1))(x)
    x = Dropout(config['dropout2'])(x)

    x = Flatten()(x)

    x = Dense(config['dense1_units'], kernel_regularizer=l2(config['dense1_l2_reg']))(x)
    x = LeakyReLU()(x)
    x = Dropout(config['dropout3'])(x)

    x = Dense(config['dense2_units'], kernel_regularizer=l1(config['dense2_l1_reg']))(x)
    x = LeakyReLU()(x)
    x = Dropout(config['dropout4'])(x)

    outputs = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model

def train_model(config, input_shape, num_classes, X_train_resampled, y_train_resampled):
    model = create_model(config, input_shape, num_classes)
    
    optimizer = config['optimizer']
    if optimizer == 'adam':
        optimizer = Adam(learning_rate=config['learning_rate'])
    elif optimizer == 'rmsprop':
        optimizer = RMSprop(learning_rate=config['learning_rate'])
    elif optimizer == 'sgd':
        optimizer = SGD(learning_rate=config['learning_rate'], momentum=config['momentum'])
    elif optimizer == 'adagrad':
        optimizer = Adagrad(learning_rate=config['learning_rate'])
    elif optimizer == 'adadelta':
        optimizer = Adadelta(learning_rate=config['learning_rate'])
    elif optimizer == 'nadam':
        optimizer = Nadam(learning_rate=config['learning_rate'])
    
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_loss', 
                                   restore_best_weights=True,
                                   patience=config['patience'],
                                   min_delta=config['min_delta'])

    history = model.fit(X_train_resampled,
                        to_categorical(y_train_resampled),
                        epochs=config['epochs'],
                        batch_size=config['batch_size'], 
                        validation_split=0.2,
                        callbacks=[early_stopping])

    return history.history['val_accuracy'][-1]

def tune_hyperparameters(input_shape, num_classes, X_train_resampled, y_train_resampled):
    config = {
        'conv1_filters': tune.grid_search([16, 32, 64]),
        'conv1_l2_reg': tune.uniform(0.001, 0.1),
        'dropout1': tune.uniform(0.1, 0.5),
        'conv2_filters': tune.grid_search([32, 64, 128]),
        'conv2_l1_reg': tune.uniform(0.001, 0.1),
        'dropout2': tune.uniform(0.1, 0.5),
        'dense1_units': tune.grid_search([64, 128, 256]),
        'dense1_l2_reg': tune.uniform(0.001, 0.1),
        'dropout3': tune.uniform(0.1, 0.5),
        'dense2_units': tune.grid_search([32, 64, 128]),
        'dense2_l1_reg': tune.uniform(0.001, 0.1),
        'dropout4': tune.uniform(0.1, 0.5),
        'optimizer': tune.choice(['adam', 'rmsprop', 'sgd', 'adagrad', 'adadelta', 'nadam']),
        'learning_rate': tune.uniform(0.0001, 0.01),
        'momentum': tune.uniform(0.5, 0.9),
        'patience': tune.grid_search([10, 15, 20]),
        'min_delta': tune.uniform(0.0001, 0.01),
        'epochs': tune.grid_search([30, 50, 70]),
        'batch_size': tune.grid_search([10, 20, 30])
    }
    
    analysis = tune.run(
        lambda config: train_model(config, input_shape, num_classes, X_train_resampled, y_train_resampled),
        config=config,
        num_samples=10,
        resources_per_trial={'cpu': n_cpus}
    )

    print('Best hyperparameters:', analysis.best_config)
    print('Best validation accuracy:', analysis.best_result)

    # Add some logging statements to track progress
    print('Hyperparameter tuning complete!')
    print('Total time elapsed:', time.time() - start_time)

tune_hyperparameters(input_shape, num_classes, X_train_resampled, y_train_resampled)
