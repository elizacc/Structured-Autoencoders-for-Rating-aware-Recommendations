import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.parametrizations import orthogonal
from torch.nn import functional as F
# from IPython.display import clear_output

from dataprep import transform_indices, full_preproccessing
from utils import *


def set_random_seed(seed):
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
# %%
# answer = binary matrix (no ratings)
# answer = binary matrix (no ratings)
class MVDataset(Dataset):
    def __init__(self, data, data_description, augment=False):
        useridx = data[data_description['users']].values
        itemidx = data[data_description['items']].values
        feedbackidx = data[data_description['feedback']].values
        values = np.ones(len(itemidx), dtype=np.float32)
        self.n_items = data_description['n_items']
        self.n_ratings = data_description['n_ratings']
        
        self.tensor = torch.sparse_coo_tensor(np.array([useridx, itemidx, feedbackidx-1]), torch.tensor(values),
                                            size=torch.Size((data_description["n_users"], data_description["n_items"], data_description['n_ratings'])))
        self.matrix = torch.sparse_coo_tensor(np.array([useridx, itemidx]), torch.tensor(values),
                                      size=torch.Size((data_description["n_users"], data_description["n_items"])), dtype=torch.float32)
        
        self.augment = augment

    def __len__(self):
        return self.tensor.shape[0]

    def __getitem__(self, idx):
        if self.augment:
            num_noise = np.random.randint(0, int(0.1 * self.tensor.shape[1]))
            idxs = torch.randint(0, self.tensor.shape[1], size=(num_noise,))
            noised_input = self.tensor[idx].detach().clone().to_dense()
            noised_input[idxs] = 0

            itemidx = np.arange(self.tensor.shape[1])
            ratingidx = np.arange(self.tensor.shape[2])
            itemidx, ratingidx = np.meshgrid(itemidx, ratingidx)
            noised_input = torch.sparse_coo_tensor(np.array([itemidx.flatten(), ratingidx.T.flatten(),]),
                                                   noised_input.flatten(),
                                                   size=torch.Size((self.n_items, self.n_ratings,)),
                                                   dtype=torch.float32)
            return noised_input, self.matrix[idx]
        else:
            return self.tensor[idx], self.matrix[idx]

# %% [markdown]


def main():
    set_random_seed(42)
    data = pd.read_csv('../../e.makhneva/data/Video_Games/Amazon_Video_Games.csv')
    data.rename(columns={'reviewerID': 'userid', 'asin': 'movieid', "overall": "rating", "unixReviewTime": "timestamp"},
                inplace=True)

    # %%
    training, testset_valid, holdout_valid, testset, holdout, data_description, data_index = full_preproccessing(data)
    # %%
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('device:', device)
    # %% [markdown]
    # print('Model: triangular matrix')

    # %%
    def triu_init(m):
        if isinstance(m, nn.Linear):
            with torch.no_grad():
                m.weight.copy_(torch.tril(m.weight))

    def get_zero_grad_hook(mask):
        def hook(grad):
            return grad * mask

        return hook

    # %%
    # class triangularAE(nn.Module):
    #     def __init__(self, n_items, n_ratings, hid1, hid2):
    #         super(triangularAE, self).__init__()
    #         self.V = nn.Linear(n_items, hid1, bias=False)
    #         torch.nn.init.xavier_uniform_(self.V.weight)
    #         self.W = nn.Linear(n_ratings, hid2, bias=False)
    #         torch.nn.init.xavier_uniform_(self.W.weight)
    #         self.L = nn.Linear(n_ratings, n_ratings, bias=False)
    #         torch.nn.init.xavier_uniform_(self.L.weight)
    #         triu_init(self.L)
    #         #         self.norm = nn.LayerNorm(n_ratings)
    #         self.vec = nn.Linear(n_ratings, 1)
    #         torch.nn.init.xavier_uniform_(self.vec.weight)
    #
    #         self.relu = nn.ReLU()
    #
    #     def forward(self, x):
    #         # encode
    #         x = self.L(x)
    #         x = self.relu(x)
    #         x = self.W(x)
    #         x = self.relu(x)
    #         xT = torch.transpose(x, -1, -2)
    #         yT = self.V(xT)
    #         y = torch.transpose(yT, -1, -2)
    #         y = self.relu(y)
    #         # decode
    #         output = F.linear(y, self.W.weight.T)
    #         output = self.relu(output)
    #         outputT = torch.transpose(output, -1, -2)
    #         outputT = torch.linalg.solve(self.L.weight, outputT)
    #         outputT = self.relu(outputT)
    #         outputT = F.linear(outputT, self.V.weight.T)
    #         output = torch.transpose(outputT, -1, -2)
    #
    #         #         output = self.relu(output)
    #         # vec
    #         output = self.vec(output).squeeze(-1)
    #         return output
    #
    # # %%
    # def triangular_model(h, data_description, device):
    #     h1, h2 = h
    #     ae = triangularAE(data_description['n_items'], data_description['n_ratings'], h1, h2).to(device)
    #     criterion = nn.BCEWithLogitsLoss().to(device)
    #     optimizer = optim.Adam(ae.parameters())
    #     scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    #
    #     mask = torch.tril(torch.ones_like(ae.L.weight))
    #     ae.L.weight.register_hook(get_zero_grad_hook(mask))
    #
    #     return ae, criterion, optimizer, scheduler

    # %% [markdown]
    # ### Tuning

    # %%
    # grid1 = 2 ** np.arange(4, 11)
    # grid2 = np.arange(3, 6)
    # grid = np.meshgrid(grid2, grid1)
    # grid = list(zip(grid[1].flatten(), grid[0].flatten()))
    #
    # # %%
    # sizes = 2 ** np.arange(9, 10)
    # for batch_size in sizes:
    #     print('Batch size:', batch_size)
    #     tuning_pipeline_augment(training, testset_valid, holdout_valid, data_description, triangular_model, device,
    #                             grid[-3:], MVDataset, batch_size=int(batch_size))

    # %% [markdown]
    # print('Model: triangular banded matrix')
    #
    # # %%
    # class bandedLinear(nn.Module):
    #     def __init__(self, num_features, bias: bool = True, device=None, dtype=None):
    #         factory_kwargs = {'device': device, 'dtype': dtype}
    #         self.device = device
    #         super().__init__()
    #         self.num_features = num_features
    #         self.weight = nn.Parameter(torch.empty((num_features, 1), **factory_kwargs))
    #         if bias:
    #             self.bias = nn.Parameter(torch.empty(num_features, **factory_kwargs))
    #         else:
    #             self.register_parameter('bias', None)
    #         self.reset_parameters()
    #
    #     def reset_parameters(self):
    #         nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
    #         if self.bias is not None:
    #             fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
    #             bound = 1 / np.sqrt(fan_in) if fan_in > 0 else 0
    #             nn.init.uniform_(self.bias, -bound, bound)
    #
    #     def forward(self, input):
    #         M = torch.zeros((self.num_features, self.num_features), device=self.weight.device)
    #         for i in range(self.num_features):
    #             d = torch.ones(self.num_features - i, device=self.weight.device) * self.weight[i]
    #             M = M + torch.diag(d, diagonal=-i)
    #         return F.linear(input, M, self.bias)
    #
    #     def extra_repr(self):
    #         return 'num_features={}, bias={}'.format(
    #             self.num_features, self.bias is not None
    #         )
    #
    # # %%
    # class triangularbandedAE(nn.Module):
    #     def __init__(self, n_items, n_ratings, hid1, hid2):
    #         super(triangularbandedAE, self).__init__()
    #         self.V = nn.Linear(n_items, hid1)
    #         torch.nn.init.xavier_uniform_(self.V.weight)
    #         self.W = nn.Linear(n_ratings, hid2)
    #         torch.nn.init.xavier_uniform_(self.W.weight)
    #         self.L = bandedLinear(n_ratings, bias=False)
    #         torch.nn.init.xavier_uniform_(self.L.weight)
    #         #         self.norm = nn.LayerNorm(n_ratings)
    #         self.vec = nn.Linear(n_ratings, 1)
    #         torch.nn.init.xavier_uniform_(self.vec.weight)
    #
    #         self.relu = nn.ReLU()
    #
    #     def forward(self, x):
    #         M = torch.zeros((self.L.weight.shape[0], self.L.weight.shape[0]), device=self.L.weight.device)
    #         for i in range(self.L.weight.shape[0]):
    #             d = torch.ones(self.L.weight.shape[0] - i, device=self.L.weight.device) * self.L.weight[i]
    #             M = M + torch.diag(d, diagonal=-i)
    #         M.require_grad = True
    #
    #         # encode
    #         x = torch.matmul(x, M.T)
    #         x = self.relu(x)
    #         x = self.W(x)
    #         x = self.relu(x)
    #         xT = torch.transpose(x, -1, -2)
    #         yT = self.V(xT)
    #         y = torch.transpose(yT, -1, -2)
    #         y = self.relu(y)
    #         # decode
    #         output = F.linear(y, self.W.weight.T)
    #         output = self.relu(output)
    #         outputT = torch.transpose(output, -1, -2)
    #         outputT = torch.linalg.solve(self.L.weight, outputT)
    #         outputT = self.relu(outputT)
    #         outputT = F.linear(outputT, self.V.weight.T)
    #         output = torch.transpose(outputT, -1, -2)
    #         # output = self.relu(output)
    #         # vec
    #         output = self.vec(output).squeeze(-1)
    #         return output
    #
    # # %%
    # def triangular_banded_model(h, data_description, device):
    #     h1, h2 = h
    #     ae = triangularbandedAE(data_description['n_items'], data_description['n_ratings'], h1, h2).to(device)
    #     criterion = nn.BCEWithLogitsLoss().to(device)
    #     optimizer = optim.Adam(ae.parameters())
    #     scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    #
    #     return ae, criterion, optimizer, scheduler
    #
    # # %% [markdown]
    # # ### Tuning
    #
    # # %%
    # grid1 = 2 ** np.arange(4, 11)
    # grid2 = np.arange(3, 6)
    # grid = np.meshgrid(grid2, grid1)
    # grid = list(zip(grid[1].flatten(), grid[0].flatten()))
    #
    # # %%
    # sizes = 2 ** np.arange(4, 10)
    # for batch_size in sizes:
    #     print('Batch size:', batch_size)
    #     tuning_pipeline_augment(training, testset_valid, holdout_valid, data_description, triangular_banded_model,
    #                             device, grid, MVDataset, batch_size=int(batch_size))

    # %% [markdown]
    # print('Model: square root matrix')
    #
    # # %%
    # class squarerootAE(nn.Module):
    #     def __init__(self, n_items, n_ratings, hid1, hid2):
    #         super(squarerootAE, self).__init__()
    #         self.V = nn.Linear(n_items, hid1, bias=False)
    #         torch.nn.init.xavier_uniform_(self.V.weight)
    #         self.W = nn.Linear(n_ratings, hid2, bias=False)
    #         torch.nn.init.xavier_uniform_(self.W.weight)
    #         self.L = nn.Linear(n_ratings, n_ratings, bias=False)
    #         torch.nn.init.xavier_uniform_(self.L.weight)
    #         #         self.norm = nn.LayerNorm(n_ratings)
    #         self.vec = nn.Linear(n_ratings, 1)
    #         torch.nn.init.xavier_uniform_(self.vec.weight)
    #
    #         self.relu = nn.ReLU()
    #     def forward(self, x):
    #         # encode
    #         x = self.L(x)
    #         x = self.relu(x)
    #         x = self.W(x)
    #         x = self.relu(x)
    #         xT = torch.transpose(x, -1, -2)
    #         yT = self.V(xT)
    #         y = torch.transpose(yT, -1, -2)
    #         y = self.relu(y)
    #         # decode
    #         output = F.linear(y, self.W.weight.T)
    #         output = self.relu(output)
    #         outputT = torch.transpose(output, -1, -2)
    #         outputT = torch.linalg.solve(self.L.weight, outputT)
    #         outputT = self.relu(outputT)
    #         outputT = F.linear(outputT, self.V.weight.T)
    #         output = torch.transpose(outputT, -1, -2)
    #
    #         #         output = self.relu(output)
    #         # vec
    #         output = self.vec(output).squeeze(-1)
    #         return output

        # def forward(self, x):
        #     # encode
        #     x = self.L(x)
        #     #         x = self.norm(x)
        #     x = self.relu(x)
        #     x = self.W(x)
        #     x = self.relu(x)
        #     xT = torch.transpose(x, -1, -2)
        #     yT = self.V(xT)
        #     y = torch.transpose(yT, -1, -2)
        #     y = self.relu(y)
        #     # decode
        #     output = torch.matmul(y, self.W.weight)
        #     output = self.relu(output)
        #     outputT = torch.transpose(output, -1, -2)
        #     outputT = torch.linalg.solve(self.L.weight, outputT)
        #     outputT = self.relu(outputT)
        #     outputT = torch.matmul(outputT, self.V.weight.T)
        #     output = torch.transpose(outputT, -1, -2)
        #     # vec
        #     output = self.vec(output).squeeze(-1)
        #     return output

    # %%
    # def square_root_model(h, data_description, device):
    #     h1, h2 = h
    #     ae = squarerootAE(data_description['n_items'], data_description['n_ratings'], h1, h2).to(device)
    #     criterion = nn.BCEWithLogitsLoss().to(device)
    #     optimizer = optim.Adam(ae.parameters())
    #     scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    #
    #     return ae, criterion, optimizer, scheduler
    #
    # # %% [markdown]
    # # ### Tuning
    #
    # # %%
    # grid1 = 2 ** np.arange(4, 11)
    # grid2 = np.arange(3, 6)
    # grid = np.meshgrid(grid2, grid1)
    # grid = list(zip(grid[1].flatten(), grid[0].flatten()))
    #
    # # %%
    # sizes = 2 ** np.arange(4, 10)
    # for batch_size in sizes:
    #     print('Batch size:', batch_size)
    #     tuning_pipeline_augment(training, testset_valid, holdout_valid, data_description, square_root_model, device,
    #                             grid, MVDataset, batch_size=int(batch_size))

    # %% [markdown]
    # print('Model: triangular layers are different')
    #
    # # %%
    # def triu_init(m):
    #     if isinstance(m, nn.Linear):
    #         with torch.no_grad():
    #             m.weight.copy_(torch.tril(m.weight))
    #
    # def get_zero_grad_hook(mask):
    #     def hook(grad):
    #         return grad * mask
    #
    #     return hook
    #
    # # %%
    # class twotriangularAE(nn.Module):
    #     def __init__(self, n_items, n_ratings, hid1, hid2):
    #         super(twotriangularAE, self).__init__()
    #         self.V = nn.Linear(n_items, hid1, bias=False)
    #         torch.nn.init.xavier_uniform_(self.V.weight)
    #         self.W = nn.Linear(n_ratings, hid2, bias=False)
    #         torch.nn.init.xavier_uniform_(self.W.weight)
    #         self.L = nn.Linear(n_ratings, n_ratings, bias=False)
    #         torch.nn.init.xavier_uniform_(self.L.weight)
    #         triu_init(self.L)
    #         self.LTinv = nn.Linear(n_ratings, n_ratings, bias=False)
    #         torch.nn.init.xavier_uniform_(self.LTinv.weight)
    #         triu_init(self.LTinv)
    #
    #         #         self.norm = nn.LayerNorm(n_ratings)
    #         self.vec = nn.Linear(n_ratings, 1)
    #         torch.nn.init.xavier_uniform_(self.vec.weight)
    #
    #         self.relu = nn.ReLU()
    #
    #     def forward(self, x):
    #         # encode
    #         x = self.L(x)
    #         x = self.relu(x)
    #         x = self.W(x)
    #         x = self.relu(x)
    #         xT = torch.transpose(x, -1, -2)
    #         yT = self.V(xT)
    #         y = torch.transpose(yT, -1, -2)
    #         y = self.relu(y)
    #         # decode
    #         output = F.linear(y, self.W.weight.T)
    #         output = self.relu(output)
    #         output = self.LTinv(output)
    #         output = self.relu(output)
    #         outputT = torch.transpose(output, -1, -2)
    #         outputT = F.linear(outputT, self.V.weight.T)
    #         output = torch.transpose(outputT, -1, -2)
    #
    #         #         output = self.relu(output)
    #         # vec
    #         output = self.vec(output).squeeze(-1)
    #         return output
    #
    # # %%
    # def twotriangular_model(h, data_description, device):
    #     h1, h2 = h
    #     ae = twotriangularAE(data_description['n_items'], data_description['n_ratings'], h1, h2).to(device)
    #     criterion = nn.BCEWithLogitsLoss().to(device)
    #     optimizer = optim.Adam(ae.parameters())
    #     scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    #
    #     mask = torch.tril(torch.ones_like(ae.L.weight))
    #     ae.L.weight.register_hook(get_zero_grad_hook(mask))
    #
    #     mask = torch.tril(torch.ones_like(ae.LTinv.weight))
    #     ae.LTinv.weight.register_hook(get_zero_grad_hook(mask))
    #
    #     return ae, criterion, optimizer, scheduler

    # %% [markdown]
    # ### Tuning

    # %%
    # grid1 = 2 ** np.arange(4, 11)
    # grid2 = np.arange(3, 6)
    # grid = np.meshgrid(grid2, grid1)
    # grid = list(zip(grid[1].flatten(), grid[0].flatten()))
    #
    # # %%
    # sizes = 2 ** np.arange(4, 10)
    # for batch_size in sizes:
    #     print('Batch size:', batch_size)
    #     tuning_pipeline_augment(training, testset_valid, holdout_valid, data_description, twotriangular_model, device,
    #                             grid, MVDataset, batch_size=int(batch_size))

    # # %% [markdown]
    # print('Model: encoder and decoder layers different')
    #
    # # %%
    # def triu_init(m):
    #     if isinstance(m, nn.Linear):
    #         with torch.no_grad():
    #             m.weight.copy_(torch.tril(m.weight))
    #
    # def get_zero_grad_hook(mask):
    #     def hook(grad):
    #         return grad * mask
    #
    #     return hook
    #
    # # %%
    # class vartriangularAE(nn.Module):
    #     def __init__(self, n_items, n_ratings, hid1, hid2):
    #         super(vartriangularAE, self).__init__()
    #         self.V = nn.Linear(n_items, hid1, bias=False)
    #         torch.nn.init.xavier_uniform_(self.V.weight)
    #         self.VT = nn.Linear(hid1, n_items, bias=False)
    #         torch.nn.init.xavier_uniform_(self.VT.weight)
    #         self.W = nn.Linear(n_ratings, hid2, bias=False)
    #         torch.nn.init.xavier_uniform_(self.W.weight)
    #         self.WT = nn.Linear(hid2, n_ratings, bias=False)
    #         torch.nn.init.xavier_uniform_(self.WT.weight)
    #         self.L = nn.Linear(n_ratings, n_ratings, bias=False)
    #         torch.nn.init.xavier_uniform_(self.L.weight)
    #         triu_init(self.L)
    #         self.LTinv = nn.Linear(n_ratings, n_ratings, bias=False)
    #         torch.nn.init.xavier_uniform_(self.LTinv.weight)
    #         triu_init(self.LTinv)
    #
    #         #         self.norm = nn.LayerNorm(n_ratings)
    #         self.vec = nn.Linear(n_ratings, 1)
    #         torch.nn.init.xavier_uniform_(self.vec.weight)
    #
    #         self.relu = nn.ReLU()
    #
    #     def forward(self, x):
    #         # encode
    #         x = self.L(x)
    #         x = self.relu(x)
    #         x = self.W(x)
    #         x = self.relu(x)
    #         xT = torch.transpose(x, -1, -2)
    #         yT = self.V(xT)
    #         y = torch.transpose(yT, -1, -2)
    #         y = self.relu(y)
    #         # decode
    #         output = self.WT(y)
    #         output = self.relu(output)
    #         output = self.LTinv(output)
    #         output = self.relu(output)
    #         outputT = torch.transpose(output, -1, -2)
    #         outputT = self.VT(outputT)
    #         output = torch.transpose(outputT, -1, -2)
    #
    #         #         output = self.relu(output)
    #         # vec
    #         output = self.vec(output).squeeze(-1)
    #         return output
    #
    # # %%
    # def vartriangular_model(h, data_description, device):
    #     h1, h2 = h
    #     ae = vartriangularAE(data_description['n_items'], data_description['n_ratings'], h1, h2).to(device)
    #     criterion = nn.BCEWithLogitsLoss().to(device)
    #     optimizer = optim.Adam(ae.parameters())
    #     scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    #
    #     mask = torch.tril(torch.ones_like(ae.L.weight))
    #     ae.L.weight.register_hook(get_zero_grad_hook(mask))
    #
    #     mask = torch.tril(torch.ones_like(ae.LTinv.weight))
    #     ae.LTinv.weight.register_hook(get_zero_grad_hook(mask))
    #
    #     return ae, criterion, optimizer, scheduler
    #
    # # %% [markdown]
    # # ### Tuning
    #
    # # %%
    # grid1 = 2 ** np.arange(4, 11)
    # grid2 = np.arange(3, 6)
    # grid = np.meshgrid(grid2, grid1)
    # grid = list(zip(grid[1].flatten(), grid[0].flatten()))
    #
    # # %%
    # sizes = 2 ** np.arange(8, 10)
    # for batch_size in sizes:
    #     print('Batch size:', batch_size)
    #     tuning_pipeline_augment(training, testset_valid, holdout_valid, data_description, vartriangular_model, device,
    #                             grid, MVDataset, batch_size=int(batch_size))

    # %% [markdown]
    print('Model: encoder and decoder different, individual rating layer')

    # %%
    def triu_init(m):
        if isinstance(m, nn.Linear):
            with torch.no_grad():
                m.weight.copy_(torch.tril(m.weight))

    def get_zero_grad_hook(mask):
        def hook(grad):
            return grad * mask

        return hook

    # %%
    class varindtriangularAE(nn.Module):
        def __init__(self, n_items, n_ratings, hid1, hid2):
            super(varindtriangularAE, self).__init__()
            self.V = nn.Linear(n_items, hid1, bias=False)
            torch.nn.init.xavier_uniform_(self.V.weight)
            self.VT = nn.Linear(hid1, n_items, bias=False)
            torch.nn.init.xavier_uniform_(self.VT.weight)
            self.W = nn.Linear(n_ratings, hid2, bias=False)
            torch.nn.init.xavier_uniform_(self.W.weight)
            self.WT = nn.Linear(hid2, n_ratings, bias=False)
            torch.nn.init.xavier_uniform_(self.WT.weight)
            self.L = nn.Linear(n_ratings, n_ratings, bias=False)
            torch.nn.init.xavier_uniform_(self.L.weight)
            triu_init(self.L)
            self.LTinv = nn.Linear(n_ratings, n_ratings, bias=False)
            torch.nn.init.xavier_uniform_(self.LTinv.weight)
            triu_init(self.LTinv)

            #         self.norm = nn.LayerNorm(n_ratings)
            self.vec = nn.Linear(n_items, 1)
            torch.nn.init.xavier_uniform_(self.vec.weight)

            self.relu = nn.ReLU()

        def forward(self, input):
            # encode
            x = self.L(input)
            x = self.relu(x)
            x = self.W(x)
            x = self.relu(x)
            xT = torch.transpose(x, -1, -2)
            yT = self.V(xT)
            y = torch.transpose(yT, -1, -2)
            y = self.relu(y)
            # decode
            output = self.WT(y)
            output = self.relu(output)
            output = self.LTinv(output)
            output = self.relu(output)
            outputT = torch.transpose(output, -1, -2)
            outputT = self.VT(outputT)
            output = torch.transpose(outputT, -1, -2)

            #         output = self.relu(output)
            # vec
            inputT = torch.transpose(input, -1, -2)
            rating_layer = self.vec(inputT)
            output = torch.matmul(output, rating_layer).squeeze(-1)
            return output

    # %%
    def varindtriangular_model(h, data_description, device):
        h1, h2 = h
        ae = varindtriangularAE(data_description['n_items'], data_description['n_ratings'], h1, h2).to(device)
        criterion = nn.BCEWithLogitsLoss().to(device)
        optimizer = optim.Adam(ae.parameters())
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

        mask = torch.tril(torch.ones_like(ae.L.weight))
        ae.L.weight.register_hook(get_zero_grad_hook(mask))

        mask = torch.tril(torch.ones_like(ae.LTinv.weight))
        ae.LTinv.weight.register_hook(get_zero_grad_hook(mask))

        return ae, criterion, optimizer, scheduler

    # %% [markdown]
    # ### Tuning

    # %%
    grid1 = 2 ** np.arange(4, 11)
    grid2 = np.arange(3, 6)
    grid = np.meshgrid(grid2, grid1)
    grid = list(zip(grid[1].flatten(), grid[0].flatten()))

    # %%
    sizes = 2 ** np.arange(4, 10)
    for batch_size in sizes:
        print('Batch size:', batch_size)
        tuning_pipeline_augment(training, testset_valid, holdout_valid, data_description, varindtriangular_model,
                                device, grid, MVDataset, batch_size=int(batch_size))

    # %% [markdown]
    # ## Model: output -- tensor

    # %%
    # class TensorDataset(Dataset):
    #     def __init__(self, data, data_description):
    #         useridx = data[data_description['users']].values
    #         itemidx = data[data_description['items']].values
    #         feedbackidx = data[data_description['feedback']].values
    #         values = np.ones(len(itemidx), dtype=np.float32)

    #         self.tensor = torch.sparse_coo_tensor(np.array([useridx, itemidx, feedbackidx-1]), torch.tensor(values),
    #                                             size=torch.Size((data_description["n_users"], data_description["n_items"], data_description['n_ratings'])))

    #     def __len__(self):
    #         return self.tensor.shape[0]

    #     def __getitem__(self, idx):
    #         return self.tensor[idx], self.tensor[idx]

    # # %%
    # train_dataset = MVDataset(training, data_description)
    # train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    # # %%
    # class AE(nn.Module):
    #     def __init__(self, n_items, n_ratings, hid1, hid2, hid3):
    #         super(AE, self).__init__()
    #         self.V = nn.Linear(n_items, hid1)
    #         self.W = nn.Linear(hid3, hid2)
    #         self.L = nn.Linear(n_ratings, hid3)
    #         triu_init(self.L)
    # #         self.vec = nn.Linear(n_ratings, 1)
    # #         self.tanh = nn.Tanh()

    #     def forward(self, x):
    #         # encode
    #         x = self.L(x)
    #         x = self.W(x)
    #         xT = torch.transpose(x, -1, -2)
    #         yT = self.V(xT)
    #         y = torch.transpose(yT, -1, -2)
    #         # decode
    #         output = torch.matmul(y, self.W.weight)
    #         output = torch.matmul(output, self.L.weight)
    #         output = torch.transpose(torch.matmul(torch.transpose(output, -1, -2), self.V.weight), -1, -2)
    #         # vec
    # #         output = self.tanh(output)
    # #         output = self.vec(output).squeeze(-1)
    #         return output

    # ae = AE(data_description['n_items'], data_description['n_ratings'], 100, 50, 20).to(device)
    # criterion = nn.BCEWithLogitsLoss().to(device)
    # optimizer = optim.Adam(ae.parameters())

    # # %%
    # mask = torch.triu(torch.ones_like(ae.L.weight))
    # # Register with hook
    # ae.L.weight.register_hook(get_zero_grad_hook(mask))

    # # %%
    # # Training the AE
    # n_epochs = 100
    # history = []

    # for epoch in range(1, n_epochs + 1):
    #     train_loss = 0
    #     for batch in train_dataloader:
    #         optimizer.zero_grad()

    #         user_tensor, true_user_tensor = batch

    #         input_tensor = user_tensor.to_dense().to(device)
    #         target = true_user_tensor.to_dense().to(device)

    #         output = ae(input_tensor)
    #         target.require_grad = False # we don't use it in training

    #         loss = criterion(output, target)
    #         loss.backward()
    #         optimizer.step()
    #         train_loss += loss.data.item()

    #     history.append(train_loss / len(train_dataloader))

    #     print('epoch: '+str(epoch)+' loss: '+str(train_loss / len(train_dataloader)))

    # # %%
    # plt.plot(history)

    # # %%
    # # Testing the AE
    # test_loss = 0

    # for user in testset.userid.unique():
    #     itemidx = testset.loc[testset.userid == user, data_description['items']].values
    #     feedbackidx = testset.loc[testset.userid == user, data_description['feedback']].values
    #     values = np.ones(len(itemidx), dtype=np.float32)

    #     user_tensor_test = torch.sparse_coo_tensor(np.array([itemidx, feedbackidx-1]), torch.tensor(values),
    #                               size=torch.Size((data_description["n_items"], data_description['n_ratings']))).to_dense().to(device).unsqueeze(0)
    #     target = user_tensor_test.clone()

    #     output = ae(user_tensor_test)
    #     target.require_grad = False

    #     loss = criterion(output, target)
    #     test_loss += loss.data.item()

    # print('test loss: '+str(test_loss / testset.userid.nunique()))

    # # %%
    # scores = torch.zeros((len(testset.userid.unique()), data_description['n_items']))
    # for i, user in enumerate(testset.userid.unique()):
    #     itemidx = testset.loc[testset.userid == user, data_description['items']].values
    #     feedbackidx = testset.loc[testset.userid == user, data_description['feedback']].values
    #     values = np.ones(len(itemidx), dtype=np.float32)

    #     user_matrix_test = torch.sparse_coo_tensor(np.array([itemidx, feedbackidx-1]), torch.tensor(values),
    #                               size=torch.Size((data_description["n_items"], data_description['n_ratings']))).to_dense().unsqueeze(0).to(device)

    #     output = ae(user_matrix_test)
    #     scores[i] = output[0][:, -1].T

    # scores = scores.detach().numpy()

    # # %%
    # # our
    # downvote_seen_items(scores, testset, data_description)
    # make_prediction(scores, holdout, data_description)


# ## Preprocess data
print('Functions defined')
if __name__ == '__main__':
    print('here')
    main()