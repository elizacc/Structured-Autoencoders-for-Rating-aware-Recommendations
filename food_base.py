# %%
import numpy as np
import pandas as pd


from torch.utils.data import Dataset
import torch
import torch.nn as nn
import torch.optim as optim

from dataprep import full_preproccessing
from utils import *


def set_random_seed(seed):
#     torch.cuda.manual_seed(seed)
    np.random.seed(seed)

set_random_seed(42)


# %%
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# %%
device

# %% [markdown]
# # Data preprocessing

if __name__ == "__main__":
    # %%
    data = pd.read_csv('../../e.makhneva/data/FoodCom/Food_com.csv')
    data.rename(columns={'user_id': 'userid', 'recipe_id': 'movieid', "date": "timestamp"}, inplace=True)
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data['rating'] = data['rating'] + 1

    # %%
    training, testset_valid, holdout_valid, testset, holdout, data_description, data_index = full_preproccessing(data)

    # %%
    class MVDataset(Dataset):
        def __init__(self, data, augment=False):
            self.data = data
            self.augment = augment

        def __len__(self):
            return self.data.shape[0]

        def __getitem__(self, idx):
            if self.augment:
                num_noise = np.random.randint(0, int(0.1*self.data.shape[1]))
                idxs = torch.randint(0, self.data.shape[1], size=(num_noise,))
                noised_input = self.data[idx].detach().clone().to_dense()
                noised_input[idxs] = 0
                
                useridx = np.zeros_like(noised_input.cpu())
                itemidx = np.arange(self.data.shape[1])
                noised_input = torch.sparse_coo_tensor(np.array([itemidx,]), noised_input,
                                                    size=torch.Size((data_description["n_items"],)), dtype=torch.float32)
                return noised_input, self.data[idx]
            else:
                return self.data[idx], self.data[idx]

    # %%
    class baseAE(nn.Module):
        def __init__(self, n_items, hid):
            super(baseAE, self).__init__()
            self.V = nn.Linear(n_items, hid)
    #         torch.nn.init.xavier_uniform_(self.V.weight)
            self.VT = nn.Linear(hid, n_items)
    #         torch.nn.init.xavier_uniform_(self.VT.weight)
            self.relu = nn.ReLU()

        def forward(self, x):
            # encode
            x = self.V(x)
            x = self.relu(x)
            # decode
            output = self.VT(x)
    #         output = self.relu(output)
            return output

    # %%
    def base_model(h, data_description, device):
        ae = baseAE(data_description['n_items'], h).to(device)
        criterion = nn.BCEWithLogitsLoss().to(device)
        optimizer = optim.Adam(ae.parameters())
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        
        return ae, criterion, optimizer, scheduler

    # %%
    grid = 2**np.arange(4, 11)

    # %%
    sizes = 2**np.arange(4,10)
    for batch_size in sizes:
        tuning_pipeline_augment(training, testset_valid, holdout_valid, data_description, base_model, grid, device, MVDataset, batch_size=int(batch_size))


