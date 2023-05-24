import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F

from dataprep import full_preproccessing
from utils import *


def set_random_seed(seed):
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)


# %%
# answer = binary matrix (no ratings)
class MVDataset(Dataset):
    def __init__(self, data, data_description, augment=False):
        useridx = pd.factorize(data[data_description['users']])[0]
        itemidx = data[data_description['items']].values
        feedbackidx = data[data_description['feedback']].values
        values = np.ones(len(itemidx), dtype=np.float32)
        self.n_items = data_description['n_items']
        self.n_ratings = data_description['n_ratings']

        self.tensor = torch.sparse_coo_tensor(np.array([useridx, itemidx, feedbackidx - 1]), torch.tensor(values),
                                              size=torch.Size((data_description["n_users"], data_description["n_items"],
                                                               data_description['n_ratings'])))
        self.matrix = torch.sparse_coo_tensor(np.array([useridx, itemidx]), torch.tensor(values),
                                              size=torch.Size(
                                                  (data_description["n_users"], data_description["n_items"])),
                                              dtype=torch.float32)

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
            noised_input = torch.sparse_coo_tensor(np.array([itemidx.flatten(), ratingidx.T.flatten(), ]),
                                                   noised_input.flatten(),
                                                   size=torch.Size((self.n_items, self.n_ratings,)),
                                                   dtype=torch.float32)
            return noised_input, self.matrix[idx]
        else:
            return self.tensor[idx], self.matrix[idx]


# %% [markdown]


def main():
    set_random_seed(42)
    data = pd.read_csv('../../e.makhneva/data/ml-1m/ml-1m.csv')
    data.rename(columns = {'userId' : 'userid', 'movieId' : 'movieid'}, inplace = True)

    # %%
    training, testset_valid, holdout_valid, testset, holdout, data_description, data_index = full_preproccessing(data)
    # %%
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('device:', device)
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

    print('Alpha: 2')
    h = (128, 4)
    training_testing_pipeline_augment(training, testset_valid, holdout_valid, testset, holdout, data_description,
                                      varindtriangular_model, h, device, MVDataset, batch_size=16, tensor_model=True)

    print('Alpha: 3')
    h = (512, 5)
    training_testing_pipeline_augment(training, testset_valid, holdout_valid, testset, holdout, data_description,
                                      varindtriangular_model, h, device, MVDataset, batch_size=256, tensor_model=True)

    print('Alpha: 4')
    h = (512, 5)
    training_testing_pipeline_augment(training, testset_valid, holdout_valid, testset, holdout, data_description,
                                      varindtriangular_model, h, device, MVDataset, batch_size=16, tensor_model=True)

    print('Alpha: 5')
    h = (512, 5)
    training_testing_pipeline_augment(training, testset_valid, holdout_valid, testset, holdout, data_description,
                                      varindtriangular_model, h, device, MVDataset, batch_size=16, tensor_model=True)


# ## Preprocess data
if __name__ == '__main__':
    main()