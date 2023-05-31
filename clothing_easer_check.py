# %%
import numpy as np
import pandas as pd
# from tqdm.notebook import tqdm

from scipy.sparse import csr_matrix

from dataprep import full_preproccessing
from utils import *
# from RecVAE.utils import *
# from RecVAE.model import VAE as RecVAE

# %%
def set_random_seed(seed):
    np.random.seed(seed)
set_random_seed(42)

# %%
data = pd.read_csv('../../e.makhneva/data/Clothing_Shoes_and_Jewelry/Amazon_Clothing_Shoes_and_Jewelry.csv')
data.rename(columns={'reviewerID': 'userid', 'asin': 'movieid', "overall": "rating", "unixReviewTime": "timestamp"},
            inplace=True)

# %%
training, testset_valid, holdout_valid, testset, holdout, data_description, data_index = full_preproccessing(data)

train_val = pd.concat((training, testset_valid, holdout_valid))

data_description = dict(
    users = data_index['users'].name,
    items = data_index['items'].name,
    feedback = 'rating',
    n_users = len(data_index['users']),
    n_items = len(data_index['items']),
    n_ratings = training['rating'].nunique(),
    min_rating = training['rating'].min(),
    test_users = holdout[data_index['users'].name].drop_duplicates().values,
    n_test_users = holdout[data_index['users'].name].nunique()
)

# %%
def matrix_from_observations(data, data_description):
    useridx = data[data_description['users']]
    itemidx = data[data_description['items']]
    values = data[data_description['feedback']]
    return csr_matrix((values, (useridx, itemidx)), shape=(useridx.values.max() + 1, data_description["n_items"]), dtype='f8')

def easer(data, data_description, lmbda=500):
    X = matrix_from_observations(data, data_description)
    G = X.T.dot(X)
    diag_indices = np.diag_indices(G.shape[0])
    G[diag_indices] += lmbda
    P = np.linalg.inv(G.A)
    B = P / (-np.diag(P))
    B[diag_indices] = 0
    
    return B

def easer_scoring(params, data, data_description):
    item_factors = params
    test_data = data.assign(
        userid = pd.factorize(data['userid'])[0]
    )
    test_matrix = matrix_from_observations(test_data, data_description)
    scores = test_matrix.dot(item_factors)
    return scores

easer_params = easer(train_val, data_description, lmbda=20)
easer_scores = easer_scoring(easer_params, testset, data_description)
downvote_seen_items(easer_scores, testset, data_description)
make_prediction(easer_scores, holdout, data_description, dcg=True, alpha=2)

easer_params = easer(train_val, data_description, lmbda=10)
easer_scores = easer_scoring(easer_params, testset, data_description)
downvote_seen_items(easer_scores, testset, data_description)
make_prediction(easer_scores, holdout, data_description, dcg=True, alpha=3)

easer_params = easer(train_val, data_description, lmbda=10)
easer_scores = easer_scoring(easer_params, testset, data_description)
downvote_seen_items(easer_scores, testset, data_description)
make_prediction(easer_scores, holdout, data_description, dcg=True, alpha=4)

easer_params = easer(train_val, data_description, lmbda=430)
easer_scores = easer_scoring(easer_params, testset, data_description)
downvote_seen_items(easer_scores, testset, data_description)
make_prediction(easer_scores, holdout, data_description, dcg=True, alpha=5)

