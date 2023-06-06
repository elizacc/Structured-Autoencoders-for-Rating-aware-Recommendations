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
    data = pd.read_csv('../../e.makhneva/data/Video_Games/Amazon_Video_Games.csv')
    data.rename(columns={'reviewerID': 'userid', 'asin': 'movieid', "overall": "rating", "unixReviewTime": "timestamp"},
                inplace=True)

    # %%
    training, testset_valid, holdout_valid, testset, holdout, data_description, data_index = full_preproccessing(data)

    max_id = training.userid.max()
    mapping = {user: user + max_id for user in testset_valid.userid.unique()}
    testset_valid['userid'] = testset_valid['userid'].map(mapping)
    holdout_valid['userid'] = holdout_valid['userid'].map(mapping)
    train_val = pd.concat((training, testset_valid, holdout_valid))

    train_val[data_description['users']] = pd.factorize(train_val[data_description['users']])[0]

    data_description = dict(
        users=data_index['users'].name,
        items=data_index['items'].name,
        feedback='rating',
        n_users=train_val.userid.nunique(),
        n_items=len(data_index['items']),
        n_ratings=train_val['rating'].nunique(),
        min_rating=train_val['rating'].min(),
        test_users=holdout[data_index['users'].name].drop_duplicates().values,
        n_test_users=holdout[data_index['users'].name].nunique()
    )
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

    print('Alpha: 2')
    h = (64, 128, 0.01)
    training_testing_pipeline_augment(training, testset_valid, holdout_valid, testset, holdout, data_description,
                                      recvae, h, device, MVDataset, batch_size=500)

    print('Alpha: 3')
    h = (128, 64, 0.006)
    training_testing_pipeline_augment(training, testset_valid, holdout_valid, testset, holdout, data_description,
                                      recvae, h, device, MVDataset, batch_size=500)

    print('Alpha: 4')
    h = (512, 64, 0.08)
    training_testing_pipeline_augment(training, testset_valid, holdout_valid, testset, holdout, data_description,
                                      recvae, h, device, MVDataset, batch_size=500)

    print('Alpha: 5')
    h = (64, 16, 0.009)
    training_testing_pipeline_augment(training, testset_valid, holdout_valid, testset, holdout, data_description,
                                      recvae, h, device, MVDataset, batch_size=500)


# ## Preprocess data
if __name__ == '__main__':
    main()
