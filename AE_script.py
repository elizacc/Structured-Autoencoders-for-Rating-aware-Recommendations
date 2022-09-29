import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from polara import get_movielens_data
from polara.preprocessing.dataframes import leave_one_out, reindex

from dataprep import transform_indices
from evaluation import topn_recommendations, downvote_seen_items



# Utils
def model_evaluate(recommended_items, holdout, holdout_description, alpha=3, topn=10, dcg=False):
    itemid = holdout_description['items']
    rateid = holdout_description['feedback']
    if alpha == None:
        alpha = np.median(holdout[rateid].unique())
    n_test_users = recommended_items.shape[0]
    holdout_items = holdout[itemid].values
    assert recommended_items.shape[0] == len(holdout_items), f"{recommended_items.shape} != {holdout_items.shape}"
    
    hits_mask = recommended_items[:, :topn] == holdout_items.reshape(-1, 1)
    pos_mask = (holdout[rateid] >= alpha).values
    neg_mask = (holdout[rateid] < alpha).values
    
    # HR calculation
    #hr = np.sum(hits_mask.any(axis=1)) / n_test_users
    hr_pos = np.sum(hits_mask[pos_mask].any(axis=1)) / n_test_users
    hr_neg = np.sum(hits_mask[neg_mask].any(axis=1)) / n_test_users
    hr = hr_pos + hr_neg
    
    # MRR calculation
    hit_rank = np.where(hits_mask)[1] + 1.0
    mrr = np.sum(1 / hit_rank) / n_test_users
    pos_hit_rank = np.where(hits_mask[pos_mask])[1] + 1.0
    mrr_pos = np.sum(1 / pos_hit_rank) / n_test_users
    neg_hit_rank = np.where(hits_mask[neg_mask])[1] + 1.0
    mrr_neg = np.sum(1 / neg_hit_rank) / n_test_users
    
    # Matthews correlation
    TP = np.sum(hits_mask[pos_mask]) # + 
    FP = np.sum(hits_mask[neg_mask]) # +
    cond = (hits_mask.sum(axis = 1) == 0)
    FN = np.sum(cond[pos_mask])
    TN = np.sum(cond[neg_mask])
    N = TP+FP+TN+FN
    S = (TP+FN)/N
    P = (TP+FP)/N
    C = (TP/N - S*P) / np.sqrt(P*S*(1-P)*(1-S))
    
    # DCG calculation
    if dcg:
        pos_hit_rank = np.where(hits_mask[pos_mask])[1] + 1.0
        neg_hit_rank = np.where(hits_mask[neg_mask])[1] + 1.0
        ndcg = np.mean(1 / np.log2(pos_hit_rank+1))
        ndcl = np.mean(1 / np.log2(neg_hit_rank+1))
    
    # coverage calculation
    n_items = holdout_description['n_items']
    cov = np.unique(recommended_items).size / n_items
    if dcg:
        return hr, hr_pos, hr_neg, mrr, mrr_pos, mrr_neg, cov, C, ndcg, ndcl
    else:
        return hr, hr_pos, hr_neg, mrr, mrr_pos, mrr_neg, cov, C

def make_prediction(tf_scores, holdout, data_description):
    for n in [5, 10, 20]:
        tf_recs = topn_recommendations(tf_scores, n)
        hr, hr_pos, hr_neg, mrr, mrr_pos, mrr_neg, cov, C = model_evaluate(tf_recs, holdout, data_description, topn=n)
        if n == 10:
            mrr10 = mrr
            hr10 = hr
            c10 = C
        print(f"HR@{n} = {hr:.4f}, MRR@{n} = {mrr:.4f}, Coverage@{n} = {cov:.4f}")
        print(f"HR_pos@{n} = {hr_pos:.4f}, HR_neg@{n} = {hr_neg:.4f}")
        print(f"MRR_pos@{n} = {mrr_pos:.4f}, MRR_neg@{n} = {mrr_neg:.4f}")
        print(f"Matthews@{n} = {C:.4f}")
        print("-------------------------------------")
    
    return mrr10, hr10, c10

def valid_mlrank(mlrank):
    '''
    Only allow ranks that are suitable for truncated SVD computations
    on unfolded compressed tensor (the result of ttm product in HOOI).
    '''
    #s, r1, r2, r3 = mlrank
    s, r1, r3 = mlrank
    r2 = r1
    #print(s, r1, r2, r3)
    return r1*r2 > r3 and r1*r3 > r2 and r2*r3 > r1


def full_preproccessing(data = None):
    if (data is None):
        data = get_movielens_data("ml-1m.zip", include_time=True)
    test_timepoint = data['timestamp'].quantile(
    q=0.8, interpolation='nearest'
    )
    
    labels, levels = pd.factorize(data.movieid)
    data.movieid = labels

    labels, levels = pd.factorize(data.userid)
    data.userid = labels
    
#     if (data["rating"].nunique() > 5):
#         data["rating"] = data["rating"] * 2
        
    data["rating"] = data["rating"].astype(int)

    test_data_ = data.query('timestamp >= @test_timepoint')
    train_data_ = data.query(
    'userid not in @test_data_.userid.unique() and timestamp < @test_timepoint'
    )
    
    training, data_index = transform_indices(train_data_.copy(), 'userid', 'movieid')
    test_data = reindex(test_data_, data_index['items'])

    testset_, holdout_ = leave_one_out(
    test_data, target='timestamp', sample_top=True, random_state=0
    )
    testset_valid_, holdout_valid_ = leave_one_out(
        testset_, target='timestamp', sample_top=True, random_state=0
    )

    test_users_val = np.intersect1d(testset_valid_.userid.unique(), holdout_valid_.userid.unique())
    testset_valid = testset_valid_.query('userid in @test_users_val').sort_values('userid')
    holdout_valid = holdout_valid_.query('userid in @test_users_val').sort_values('userid')

    test_users = np.intersect1d(testset_.userid.unique(), holdout_.userid.unique())
    testset = testset_.query('userid in @test_users').sort_values('userid')
    holdout = holdout_.query('userid in @test_users').sort_values('userid')
    
    assert holdout_valid.set_index('userid')['timestamp'].ge(
        testset_valid
        .groupby('userid')
        ['timestamp'].max()
    ).all()

    data_description = dict(
        users = data_index['users'].name,
        items = data_index['items'].name,
        feedback = 'rating',
        n_users = len(data_index['users']),
        n_items = len(data_index['items']),
        n_ratings = training['rating'].nunique(),
        min_rating = training['rating'].min(),
        test_users = holdout_valid[data_index['users'].name].drop_duplicates().values, # NEW
        n_test_users = holdout[data_index['users'].name].nunique() # NEW CHECK
    )

    return training, testset_valid, holdout_valid, testset, holdout, data_description, data_index


training, testset_valid, holdout_valid, testset, holdout, data_description, data_index = full_preproccessing()

useridx = training[data_description['users']].values
itemidx = training[data_description['items']].values
# feedbackidx = training[data_description['feedback']].values
values = np.ones(len(itemidx))

user_tensor_train = torch.sparse_coo_tensor(np.array([useridx, itemidx]), torch.tensor(values),
                                      size=torch.Size((data_description["n_users"], data_description["n_items"])), dtype=torch.float32)


class SimpleDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx]

class baseAE(nn.Module):
    def __init__(self, n_items, hid):
        super(baseAE, self).__init__()
        self.V = nn.Linear(n_items, hid)
        self.VT = nn.Linear(hid, n_items)
        self.relu = nn.ReLU()

    def forward(self, x):
        # encode
        x = self.V(x)
        # decode
        output = self.VT(x)
        output = self.relu(output)
        return output


train_dataset = SimpleDataset(user_tensor_train)
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)

bae = baseAE(data_description['n_items'], 200).to(device)
criterion = nn.BCEWithLogitsLoss().to(device)
optimizer = optim.Adam(bae.parameters())


# Training the simpleAE
n_epochs = 100
history = []

for epoch in range(1, n_epochs + 1):   
    train_loss = 0
    for user_matrix in train_dataloader:
        optimizer.zero_grad()
        
        input_tensor = user_matrix.to_dense().to(device)
        
        target = input_tensor.clone()
        output = bae(input_tensor)
        target.require_grad = False # we don't use it in training

        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.data.item()
        
    history.append(train_loss / len(train_dataloader))
        
    print('epoch: '+str(epoch)+' loss: '+str(train_loss / len(train_dataloader)))


scores = torch.zeros((len(testset.userid.unique()), data_description['n_items']))
for i, user in enumerate(testset.userid.unique()):
    itemidx = testset.loc[testset.userid == user, data_description['items']].values
    feedbackidx = testset.loc[testset.userid == user, data_description['feedback']].values
    values = np.ones(len(itemidx), dtype=np.float32)

    user_matrix_test = torch.sparse_coo_tensor(np.array([itemidx]), torch.tensor(values),
                              size=torch.Size((data_description["n_items"], ))).to_dense().unsqueeze(0).to(device)
    
    output = bae(user_matrix_test)
    scores[i] = output[0].T

        
scores = scores.detach().numpy()

# base
downvote_seen_items(scores, testset, data_description)
make_prediction(scores, holdout, data_description)