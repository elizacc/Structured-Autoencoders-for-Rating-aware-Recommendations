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
    data = pd.read_csv('../../e.makhneva/data/Video_Games/Amazon_Video_Games.csv')
    data.rename(columns={'reviewerID': 'userid', 'asin': 'movieid', "overall": "rating", "unixReviewTime": "timestamp"},
                inplace=True)

    # %%
    training, testset_valid, holdout_valid, testset, holdout, data_description, data_index = full_preproccessing(data)
    # %%
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('device:', device)

    def training_testing_pipeline_augment(training, testset_valid, holdout_valid, data_description, model_init, h,
                                          device, MVDataset, tensor_model=False, batch_size=16, early_stop=50,
                                          n_epochs=1000):
        train_dataset = MVDataset(training, data_description, augment=True)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        user_tensor_val, target_val = prepare_tensor(testset_valid, data_description, tensor_model)
        val_num_batches = int(np.ceil(target_val.shape[0] / batch_size))

        print('Hidden sizes:', h)

        model, criterion, optimizer, scheduler = model_init(h, data_description, device)

        # Training the AE
        history = []
        val_history = []

        hrs2 = []
        mrrs2 = []
        cs2 = []
        ndcgs2 = []

        hrs3 = []
        mrrs3 = []
        cs3 = []
        ndcgs3 = []

        hrs4 = []
        mrrs4 = []
        cs4 = []
        ndcgs4 = []

        hrs5 = []
        mrrs5 = []
        cs5 = []
        ndcgs5 = []

        hrs6 = []
        mrrs6 = []
        cs6 = []
        ndcgs6 = []

        prev_matt2 = [0]
        prev_matt3 = [0]
        prev_matt4 = [0]
        prev_matt5 = [0]
        prev_matt6 = [0]

        for epoch in range(1, n_epochs + 1):
            train_loss = 0

            for batch in train_dataloader:
                optimizer.zero_grad()

                input_tensor, target = batch
                input_tensor, target = input_tensor.to_dense().to(device), target.to_dense().to(device)

                output = model(input_tensor)
                target.require_grad = False  # we don't use it in training

                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                train_loss += loss.data.item()

            scheduler.step()
            history.append(train_loss / len(train_dataloader))

            test_loss = 0
            scores = torch.zeros((testset_valid.userid.nunique(), data_description['n_items']))

            with torch.no_grad():
                model.eval()
                for batch in range(val_num_batches):
                    input_tensor = user_tensor_val[batch * batch_size: (batch + 1) * batch_size].to(device)
                    target = target_val[batch * batch_size: (batch + 1) * batch_size].to(device)

                    output = model(input_tensor)
                    target.require_grad = False

                    test_loss += criterion(output, target)
                    scores[batch * batch_size: (batch + 1) * batch_size] = output
                model.train()

            scores = scores.detach().cpu().numpy()
            val_loss = test_loss / val_num_batches
            val_history.append(val_loss.item())

            downvote_seen_items(scores, testset_valid, data_description)

            prev_matt2, hrs2, mrrs2, cs2, ndcgs2 = predict_and_check(model, scores, holdout_valid, data_description,
                                                                     hrs2,
                                                                     mrrs2, cs2, ndcgs2, 2, prev_matt2, epoch, h)
            prev_matt3, hrs3, mrrs3, cs3, ndcgs3 = predict_and_check(model, scores, holdout_valid, data_description,
                                                                     hrs3,
                                                                     mrrs3, cs3, ndcgs3, 3, prev_matt3, epoch, h)
            prev_matt4, hrs4, mrrs4, cs4, ndcgs4 = predict_and_check(model, scores, holdout_valid, data_description,
                                                                     hrs4,
                                                                     mrrs4, cs4, ndcgs4, 4, prev_matt4, epoch, h)
            prev_matt5, hrs5, mrrs5, cs5, ndcgs5 = predict_and_check(model, scores, holdout_valid, data_description,
                                                                     hrs5,
                                                                     mrrs5, cs5, ndcgs5, 5, prev_matt5, epoch, h)
            prev_matt6, hrs6, mrrs6, cs6, ndcgs6 = predict_and_check(model, scores, holdout_valid, data_description,
                                                                     hrs6,
                                                                     mrrs6, cs6, ndcgs6, 6, prev_matt6, epoch, h)

            # stop = epoch if epoch < early_stop else epoch-early_stop
            if len(prev_matt2) >= early_stop and len(prev_matt3) >= early_stop and len(
                    prev_matt4) >= early_stop and len(prev_matt5) >= early_stop and len(prev_matt6) >= early_stop:
                print(f'Current epoch {epoch}')
                break

        # Testing the AE
        check_test(model, criterion, user_tensor_val, target_val, testset_valid, holdout_valid, data_description,
                   val_num_batches, 2, h, device, batch_size=batch_size, dcg=True)
        check_test(model, criterion, user_tensor_val, target_val, testset_valid, holdout_valid, data_description,
                   val_num_batches, 3, h, device, batch_size=batch_size, dcg=True)
        check_test(model, criterion, user_tensor_val, target_val, testset_valid, holdout_valid, data_description,
                   val_num_batches, 4, h, device, batch_size=batch_size, dcg=True)
        check_test(model, criterion, user_tensor_val, target_val, testset_valid, holdout_valid, data_description,
                   val_num_batches, 5, h, device, batch_size=batch_size, dcg=True)
        check_test(model, criterion, user_tensor_val, target_val, testset_valid, holdout_valid, data_description,
                   val_num_batches, 6, h, device, batch_size=batch_size, dcg=True)

        print('Test loss:', val_history[-min(early_stop, epoch)])
        print('Train loss:', history[-min(early_stop, epoch)])

        print()
        print()

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


    class triangularAE(nn.Module):
        def __init__(self, n_items, n_ratings, hid1, hid2):
            super(triangularAE, self).__init__()
            self.V = nn.Linear(n_items, hid1, bias=False)
            torch.nn.init.xavier_uniform_(self.V.weight)
            self.W = nn.Linear(n_ratings, hid2, bias=False)
            torch.nn.init.xavier_uniform_(self.W.weight)
            self.L = nn.Linear(n_ratings, n_ratings, bias=False)
            torch.nn.init.xavier_uniform_(self.L.weight)
            triu_init(self.L)
            #         self.norm = nn.LayerNorm(n_ratings)
            self.vec = nn.Linear(n_ratings, 1)
            torch.nn.init.xavier_uniform_(self.vec.weight)

            self.relu = nn.ReLU()

        def forward(self, x):
            # encode
            x = self.L(x)
            x = self.relu(x)
            x = self.W(x)
            x = self.relu(x)
            xT = torch.transpose(x, -1, -2)
            yT = self.V(xT)
            y = torch.transpose(yT, -1, -2)
            y = self.relu(y)
            # decode
            output = F.linear(y, self.W.weight.T)
            output = self.relu(output)
            outputT = torch.transpose(output, -1, -2)
            outputT = torch.linalg.solve(self.L.weight, outputT)
            outputT = self.relu(outputT)
            outputT = F.linear(outputT, self.V.weight.T)
            output = torch.transpose(outputT, -1, -2)

            #         output = self.relu(output)
            # vec
            output = self.vec(output).squeeze(-1)
            return output

    # %%
    def triangular_model(h, data_description, device):
        h1, h2 = h
        ae = triangularAE(data_description['n_items'], data_description['n_ratings'], h1, h2).to(device)
        criterion = nn.BCEWithLogitsLoss().to(device)
        optimizer = optim.Adam(ae.parameters())
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

        mask = torch.tril(torch.ones_like(ae.L.weight))
        ae.L.weight.register_hook(get_zero_grad_hook(mask))

        return ae, criterion, optimizer, scheduler

    h = (256, 4)
    training_testing_pipeline_augment(training, testset_valid, holdout_valid, data_description,
                                      triangular_model, h, device, MVDataset, batch_size=512, tensor_model=True)




# ## Preprocess data
print('Functions defined')
if __name__ == '__main__':
    print('here')
    main()