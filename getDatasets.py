import matplotlib.pyplot as plt
import numpy as np
import sys, os
sys.path.append('/home/groups/ZuckermanLab/jalim/instalLocal/celltraj/celltraj')
import vaeSCMDT as cellTraj
import h5py, pickle, subprocess, random
import torch
from torch.utils.data import DataLoader, Dataset, Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as F
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

model_name = "LI204601_P_A2_1"
filepath = "/home/groups/ZuckermanLab/jalim/LI204601_INCUCYTE/segsCellPose/scTrajAnl/"
imgFileInfo = f"{filepath}{model_name}.h5"

sctm = cellTraj.Trajectory()
print('initializing...')
sctm.initialize(imgFileInfo, model_name)
sctm.get_frames()
start_frame = 0
end_frame = sctm.maxFrame
sctm.visual = False
sctm.get_imageSet(start_frame, end_frame)
sctm.imgSet_t = np.zeros((end_frame - start_frame + 1, 3))
sctm.get_imageSet_trans() # Register cell masks
#sctm.get_imageSet_trans_turboreg # Register phase channel images
sctm.get_cell_data()
sctm.get_cell_images()
sctm.prepare_cell_images()

class PhaseImageScaler:
    def __init__(self):
        self.scaler = MinMaxScaler()

    def scale_image(self, image):
        return self.scaler.fit_transform(image)

class PhaseImageDataset(Dataset):
    def __init__(self, cell_images, cell_masks, cell_labels):
        self.cell_images = cell_images
        self.cell_masks = cell_masks
        self.cell_labels = cell_labels
        self.scaler = PhaseImageScaler()
        self.image_size = int(np.ceil(np.sqrt(cell_images.shape[1])))

    def __len__(self):
        return len(self.cell_images)

    def __getitem__(self, idx):
        image = self.cell_images[idx].reshape(self.image_size, self.image_size)
        scaled_image = self.scaler.scale_image(image)
        mask = self.cell_masks[idx].reshape(self.image_size, self.image_size)
        label = self.cell_labels[idx]
        return (
            torch.tensor(scaled_image, dtype=torch.float32).unsqueeze(0),
            torch.tensor(mask, dtype=torch.float32).unsqueeze(0),
            torch.tensor(label, dtype=torch.long)
        )

class PhaseImageDataLoader:
    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def get_data(self):
        X, M, y = [], [], []
        for batch in tqdm(self.loader):
            X.append(batch[0])
            M.append(batch[1])
            y.append(batch[2])
        X = torch.cat(X)
        M = torch.cat(M)
        y = torch.cat(y)
        y = y.squeeze()
        return X, M, y

cell_labels = [i for i in range(sctm.X.shape[0])]
cell_labels = np.array(cell_labels).astype(int)
cell_images = sctm.X
cell_masks = sctm.Xm

dataset = PhaseImageDataset(cell_images, cell_masks, cell_labels)
data_loader = PhaseImageDataLoader(dataset)

X, M, y = data_loader.get_data()

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
M_train, M_test, ym_train, ym_test = train_test_split(M, y, test_size=0.2, random_state=42)

# Saving Phase Channel Images as torch tensors
torch.save(X_train, f'{model_name}-X_train.sav')
torch.save(X_test, f'{model_name}-X_test.sav')
torch.save(y_train, f'{model_name}-y_train.sav')
torch.save(y_test, f'{model_name}-y_test.sav')
# Saving Masks as torch tensors
torch.save(M_train, f'{model_name}-X_train_msks.sav')
torch.save(M_test, f'{model_name}-X_test_msks.sav')
torch.save(ym_train, f'{model_name}-y_train_msks.sav')
torch.save(ym_test, f'{model_name}-y_test_msks.sav')