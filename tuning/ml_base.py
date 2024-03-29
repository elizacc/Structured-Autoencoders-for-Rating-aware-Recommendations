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
print('device:', device)

# %% [markdown]
# # Data preprocessing

if __name__ == "__main__":
    # %%
    data = pd.read_csv('../../e.makhneva/data/ml-1m/ml-1m.csv')
    data.rename(columns = {'userId' : 'userid', 'movieId' : 'movieid'}, inplace = True)

    # %%
    training, testset_valid, holdout_valid, testset, holdout, data_description, data_index = full_preproccessing(data)

    # %%
    class MVDataset(Dataset):
        def __init__(self, data, data_description, augment=False):
            self.augment = augment
            useridx = data[data_description['users']].values
            itemidx = data[data_description['items']].values
            values = np.ones(len(itemidx), dtype=np.float32)
            self.n_items = data_description['n_items']

            self.matrix = torch.sparse_coo_tensor(np.array([useridx, itemidx]), torch.tensor(values),
                                                  size=torch.Size(
                                                      (data_description["n_users"], data_description["n_items"])),
                                                  dtype=torch.float32)

        def __len__(self):
            return self.matrix.shape[0]

        def __getitem__(self, idx):
            if self.augment:
                num_noise = np.random.randint(0, int(0.1*self.matrix.shape[1]))
                idxs = torch.randint(0, self.matrix.shape[1], size=(num_noise,))
                noised_input = self.matrix[idx].detach().clone().to_dense()
                noised_input[idxs] = 0
                
                # useridx = np.zeros_like(noised_input.cpu())
                itemidx = np.arange(self.matrix.shape[1])
                noised_input = torch.sparse_coo_tensor(np.array([itemidx,]), noised_input,
                                                    size=torch.Size((data_description["n_items"],)), dtype=torch.float32)
                return noised_input, self.matrix[idx]
            else:
                return self.matrix[idx], self.matrix[idx]

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
        tuning_pipeline_augment(training, testset_valid, holdout_valid, data_description, base_model, device, grid, MVDataset, batch_size=int(batch_size), tensor_model=False)


