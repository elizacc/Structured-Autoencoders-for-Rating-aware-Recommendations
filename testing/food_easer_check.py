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
data = pd.read_csv('../../e.makhneva/data/FoodCom/Food_com.csv')
data.rename(columns={'user_id': 'userid', 'recipe_id': 'movieid', "date": "timestamp"}, inplace=True)
data['timestamp'] = pd.to_datetime(data['timestamp'])
data['rating'] = data['rating'] + 1

# %%
training, testset_valid, holdout_valid, testset, holdout, data_description, data_index = full_preproccessing(data)

max_id = training.userid.max()
mapping = {user: user + max_id for user in testset_valid.userid.unique()}
testset_valid['userid'] = testset_valid['userid'].map(mapping)
holdout_valid['userid'] = holdout_valid['userid'].map(mapping)
train_val = pd.concat((training, testset_valid, holdout_valid))

train_val[data_description['users']] = pd.factorize(train_val[data_description['users']])[0]

data_description = dict(
    users = data_index['users'].name,
    items = data_index['items'].name,
    feedback = 'rating',
    n_users = train_val.userid.nunique(),
    n_items = len(data_index['items']),
    n_ratings = train_val['rating'].nunique(),
    min_rating = train_val['rating'].min(),
    test_users = holdout[data_index['users'].name].drop_duplicates().values,
    n_test_users = holdout[data_index['users'].name].nunique()
)


# %%
def matrix_from_observations(data, data_description):
    useridx = data[data_description['users']]
    itemidx = data[data_description['items']]
    values = data[data_description['feedback']]
    return csr_matrix((values, (useridx, itemidx)), shape=(useridx.values.max() + 1, data_description["n_items"]),
                      dtype='f8')


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
        userid=pd.factorize(data['userid'])[0]
    )
    test_matrix = matrix_from_observations(test_data, data_description)
    scores = test_matrix.dot(item_factors)
    return scores


easer_params = easer(train_val, data_description, lmbda=180)
easer_scores = easer_scoring(easer_params, testset, data_description)
downvote_seen_items(easer_scores, testset, data_description)
make_prediction(easer_scores, holdout, data_description, dcg=True, alpha=2)

easer_params = easer(train_val, data_description, lmbda=180)
easer_scores = easer_scoring(easer_params, testset, data_description)
downvote_seen_items(easer_scores, testset, data_description)
make_prediction(easer_scores, holdout, data_description, dcg=True, alpha=3)

easer_params = easer(train_val, data_description, lmbda=250)
easer_scores = easer_scoring(easer_params, testset, data_description)
downvote_seen_items(easer_scores, testset, data_description)
make_prediction(easer_scores, holdout, data_description, dcg=True, alpha=4)

easer_params = easer(train_val, data_description, lmbda=10)
easer_scores = easer_scoring(easer_params, testset, data_description)
downvote_seen_items(easer_scores, testset, data_description)
make_prediction(easer_scores, holdout, data_description, dcg=True, alpha=5)

easer_params = easer(train_val, data_description, lmbda=450)
easer_scores = easer_scoring(easer_params, testset, data_description)
downvote_seen_items(easer_scores, testset, data_description)
make_prediction(easer_scores, holdout, data_description, dcg=True, alpha=6)

