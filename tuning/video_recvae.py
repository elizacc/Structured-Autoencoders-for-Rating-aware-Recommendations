import torch.nn as nn
import torch.optim as optim
# from torch.nn import functional as F
from model import VAE

from dataprep import full_preproccessing
from utils_vae import *


def set_random_seed(seed):
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)


# %%
# answer = binary matrix (no ratings)
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
            num_noise = np.random.randint(0, int(0.1 * self.matrix.shape[1]))
            idxs = torch.randint(0, self.matrix.shape[1], size=(num_noise,))
            noised_input = self.matrix[idx].detach().clone().to_dense()
            noised_input[idxs] = 0

            # useridx = np.zeros_like(noised_input.cpu())
            itemidx = np.arange(self.matrix.shape[1])
            noised_input = torch.sparse_coo_tensor(np.array([itemidx, ]), noised_input,
                                                   size=torch.Size((self.n_items,)), dtype=torch.float32)
            return noised_input, self.matrix[idx]
        else:
            return self.matrix[idx], self.matrix[idx]


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

    def recvae(h, data_description, device):
        h1, h2, gamma = h
        ae = VAE(h1, h2, data_description['n_items']).to(device)
        criterion = nn.BCEWithLogitsLoss().to(device)
        optimizer = optim.Adam(ae.parameters(), lr=5 * 10e-4)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

        return ae, criterion, optimizer, scheduler

    # %% [markdown]
    # ### Tuning

    # %%
    grid1 = 2 ** np.arange(4, 11)
    grid2 = 2 ** np.arange(4, 11)
    gamma = 0.001 * np.arange(1, 11)
    grid = np.meshgrid(grid2, grid1, gamma)
    grid = list(zip(grid[0].flatten(), grid[1].flatten(), grid[2].flatten()))

    # %%
    sizes = [500]  # 2 ** np.arange(4, 10)
    for batch_size in sizes:
        print('Batch size:', batch_size)
        tuning_pipeline_augment(training, testset_valid, holdout_valid, data_description, recvae,
                                device, grid, MVDataset, batch_size=int(batch_size), tensor_model=False)


# ## Preprocess data
if __name__ == '__main__':
    main()