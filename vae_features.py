import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from IPython.display import clear_output

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--project_name", help="Project Name")
parser.add_argument("--model_name", help="Model Name")
parser.add_argument("--sys_name", help="System Name")
parser.add_argument("--latent_dim", help="Latent Dimensions")
args = parser.parse_args()

model_name = args.model_name
sys_name = args.sys_name
project_name = args.project_name
latent_dim = args.latent_dim

figure_id = f'{latent_dim}_{project_name}_{sys_name}'
model_path=f"/home/groups/ZuckermanLab/jalim/egf_m2cO2vae/"

import sys
sys.path.append(model_path)
import importlib
import train_loops
import run
from utils import utils
import wandb
import logging
from pathlib import Path
from configs.config_LI204601 import config # Load Pretrained Model Configuration
from torchvision.utils import make_grid
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using Device: {device}')

importlib.reload(utils)

dset, loader, dset_test, loader_test = run.get_datasets_from_config(config) # get datasets specified by config

config.model.encoder.n_channels = dset[0][0].shape[0]  # image channels
model = run.build_model_from_config(config)

# Get pre-trained model weights
pretrained_model_path = os.path.join(model_path, f"wandb/offline-run-20240814_122540-{model_name}/files/model.pt")

if pretrained_model_path:
   print(f"Successfully loaded pre-trained model!")
else:
   print(f"Pre-trained couldn't be found")
   sys.exit(0)

model_checkpoint = torch.load(pretrained_model_path)
model.to(device).cpu().train() # if keys don't match, try many combinations of putting it on and off cuda
missing_keys, unexpected_keys = model.load_state_dict(model_checkpoint['model_state_dict'], strict=False)
# checking that the only missing keys from the state_dict are this one type
assert all(['_basisexpansion' in k for k in missing_keys]) 

#Get Latent Dimension Embedding of Training & Test Sets
from utils import eval_utils
importlib.reload(eval_utils)
model.eval().cpu() 
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device).cpu()

embeddings_train, labels_train = utils.get_model_embeddings_from_loader(model, 
                                                                        loader,
                                                                        return_labels=True) # Training Set

embeddings_test, labels_test = utils.get_model_embeddings_from_loader(model, 
                                                                      loader_test,
                                                                      return_labels=True) # Test Set

######### Save Latent Dimension Embeddings of the Training & Test Sets for Downstream Analysis ##########
# Clone Torch Tensors into Numpy arrays
embed_train = embeddings_train.clone().numpy()
embed_test = embeddings_test.clone().numpy()
labels_train_ = labels_train.clone().numpy()
labels_test_ = labels_test.clone().numpy()

embedding_data = np.concatenate((embed_train, embed_test))
labels_data = np.concatenate((labels_train_, labels_test_)).astype(int)

ind_sort = np.argsort(labels_data)
labels_data = labels_data[ind_sort]
embedding_data = embedding_data[ind_sort]

np.savez(f'features_vae{figure_id}.npz', embeddings=embedding_data, labels=labels_data)
