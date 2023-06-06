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
        useridx = pd.factorize(data[data_description['users']])[0]
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
    data = pd.read_csv('../../e.makhneva/data/FoodCom/Food_com.csv')
    data.rename(columns={'user_id': 'userid', 'recipe_id': 'movieid', "date": "timestamp"}, inplace=True)
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data['rating'] = data['rating'] + 1

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


    print('Alpha: 2')
    h = (1024, 128, 0.01)
    training_testing_pipeline_augment(training, testset_valid, holdout_valid, testset, holdout, data_description,
                                      recvae, h, device, MVDataset, batch_size=500)

    print('Alpha: 3')
    h = (1024, 128, 0.007)
    training_testing_pipeline_augment(training, testset_valid, holdout_valid, testset, holdout, data_description,
                                      recvae, h, device, MVDataset, batch_size=500)

    print('Alpha: 4')
    h = (64, 32, 0.001)
    training_testing_pipeline_augment(training, testset_valid, holdout_valid, testset, holdout, data_description,
                                      recvae, h, device, MVDataset, batch_size=500)

    print('Alpha: 5')
    h = (128, 64, 0.008)
    training_testing_pipeline_augment(training, testset_valid, holdout_valid, testset, holdout, data_description,
                                      recvae, h, device, MVDataset, batch_size=500)

    print('Alpha: 6')
    h = (256, 64, 0.004)
    training_testing_pipeline_augment(training, testset_valid, holdout_valid, testset, holdout, data_description,
                                      recvae, h, device, MVDataset, batch_size=500)


# ## Preprocess data
if __name__ == '__main__':
    main()
