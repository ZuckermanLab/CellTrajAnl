import numpy as np
import torch
import argparse

path_datasets = f"/home/groups/ZuckermanLab/jalim/LI204601_INCUCYTE/o2vae_datasets/"
sys_name = "LI204601_P_"

parser = argparse.ArgumentParser()

parser.add_argument("--well_A", help="A wells info")
parser.add_argument("--well_B", help="B wells info")
args = parser.parse_args()
 
A_well = args.well_A
B_well = args.well_B

good_fovs = np.array([[1, 3, 4], [1, 3, 4]]) # "A" & "B" wells
wells = np.array([A_well, B_well])

def load_and_concatenate_data(wells, good_fovs, sys_name, path_datasets):
    X_train, X_test, y_train, y_test = None, None, None, None
    for i, well in enumerate(wells):
        for fov in range(good_fovs[i].size):
            model = sys_name + well
            fov = good_fovs[i][fov]
            X_train_fov = torch.load(f'{path_datasets}{model}_{fov}-X_train.sav')
            X_test_fov = torch.load(f'{path_datasets}{model}_{fov}-X_test.sav')
            y_train_fov = torch.load(f'{path_datasets}{model}_{fov}-y_train.sav')
            y_test_fov = torch.load(f'{path_datasets}{model}_{fov}-y_test.sav')
            if X_train is None:
                X_train = X_train_fov
                X_test = X_test_fov
                y_train = y_train_fov
                y_test = y_test_fov
            else:
                X_train = torch.cat((X_train, X_train_fov), dim=0)
                X_test = torch.cat((X_test, X_test_fov), dim=0)
                y_train = torch.cat((y_train, y_train_fov), dim=0)
                y_test = torch.cat((y_test, y_test_fov), dim=0)
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = load_and_concatenate_data(wells, good_fovs, sys_name, path_datasets)

new_data_path = f"{path_datasets}egf_goodFOVs_all/"

torch.save(X_train, f'{new_data_path}X_train.sav')
torch.save(X_test, f'{new_data_path}X_test.sav')
torch.save(y_train, f'{new_data_path}y_train.sav')
torch.save(y_test, f'{new_data_path}y_test.sav')