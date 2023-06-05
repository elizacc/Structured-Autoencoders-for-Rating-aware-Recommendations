# %%
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# from tqdm.notebook import tqdm

from polara.lib.tensor import hooi
from polara.lib.sparse import tensor_outer_at
from scipy.sparse import csr_matrix
from scipy.linalg import solve_triangular
from sa_hooi import sa_hooi, form_attention_matrix, get_scaling_weights

from dataprep import full_preproccessing
from utils import *
# from RecVAE.utils import *
# from RecVAE.model import VAE as RecVAE


# %%
def set_random_seed(seed):
    np.random.seed(seed)
set_random_seed(42)

    # %%
data = pd.read_csv('../../e.makhneva/data/ml-1m/ml-1m.csv')
data.rename(columns={'userId': 'userid', 'movieId': 'movieid'}, inplace=True)

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
def tf_model_build(config, data, data_description, attention_matrix=np.array([])):
    userid = data_description["users"]
    itemid = data_description["items"]
    feedback = data_description["feedback"]

    idx = data[[userid, itemid, feedback]].values
    idx[:, -1] = idx[:, -1] - data_description['min_rating']  # works only for integer ratings!
    val = np.ones(idx.shape[0], dtype='f8')

    n_users = data_description["n_users"]
    n_items = data_description["n_items"]
    n_ratings = data_description["n_ratings"]
    shape = (n_users, n_items, n_ratings)
    core_shape = config['mlrank']
    num_iters = config["num_iters"]

    if (attention_matrix.shape[0] == 0):
        attention_matrix = form_attention_matrix(
            data_description['n_ratings'],
            **config['params'],
            format='csr'
        ).A

    item_popularity = (
        pd.Series(np.ones((n_items,)))
        .reindex(range(n_items))
        .fillna(1)
        .values
    )
    scaling_weights = get_scaling_weights(item_popularity, scaling=config["scaling"])

    u0, u1, u2 = sa_hooi(
        idx, val, shape, config["mlrank"],
        attention_matrix=attention_matrix,
        scaling_weights=scaling_weights,
        max_iters=config["num_iters"],
        parallel_ttm=False,
        randomized=config["randomized"],
        growth_tol=config["growth_tol"],
        seed=config["seed"],
        iter_callback=None,
    )

    return u0, u1, u2, attention_matrix


# %%
config = {
    "scaling": 1,
    "mlrank": (30, 30, 5),
    "n_ratings": data_description['n_ratings'],
    "num_iters": 4,
    "params": None,
    "randomized": True,
    "growth_tol": 1e-4,
    "seed": 42
}


# %%
def tf_scoring(params, data, data_description):
    user_factors, item_factors, feedback_factors, attention_matrix = params
    userid = data_description["users"]
    itemid = data_description["items"]
    feedback = data_description["feedback"]

    data = data.sort_values(userid)
    useridx = data[userid]
    itemidx = data[itemid].values
    ratings = data[feedback].values
    ratings = ratings - data_description['min_rating']

    n_users = useridx.nunique()
    n_items = data_description['n_items']
    n_ratings = data_description['n_ratings']

    # inv_attention = np.linalg.inv(attention_matrix.A) # change
    inv_attention = solve_triangular(attention_matrix, np.eye(5), lower=True)
    # np.testing.assert_almost_equal(inv_attention, inv_attention_)

    tensor_outer = tensor_outer_at('cpu')

    inv_aT_feedback = (inv_attention.T @ feedback_factors)[-1, :]

    scores = tensor_outer(
        1.0,
        item_factors,
        attention_matrix @ feedback_factors,
        itemidx,
        ratings
    )
    scores = np.add.reduceat(scores, np.r_[0, np.where(np.diff(useridx))[0] + 1])  # sort by users
    scores = np.tensordot(
        scores,
        inv_aT_feedback,
        axes=(2, 0)
    ).dot(item_factors.T)

    return scores

def center_and_rescale_score(x, func=None):

    if func is None:
        func = np.arctan

    return func(x - 3)

# %%
print('Trigonometry matrix')
eucl_matrix = np.zeros((5, 5))

for i in range(5):
    for j in range(5):

        k, l = center_and_rescale_score(i + 1), center_and_rescale_score(j + 1)

        diff = abs(k - l)

        eucl_matrix[i, j] = 1 / (diff + 1)

similarity = eucl_matrix

a = np.linalg.cholesky(similarity)

attention_matrix = csr_matrix(a)

config['mlrank'] = (1024, 1024, 4)
tf_params = tf_model_build(config, train_val, data_description, attention_matrix=attention_matrix.A)
seen_data = testset
tf_scores = tf_scoring(tf_params, seen_data, data_description)
downvote_seen_items(tf_scores, seen_data, data_description)

print('alpha: 2')
make_prediction(tf_scores, holdout, data_description, dcg=True, alpha=2)
print('alpha: 3')
make_prediction(tf_scores, holdout, data_description, dcg=True, alpha=3)

config['mlrank'] = (512, 512, 4)
tf_params = tf_model_build(config, train_val, data_description, attention_matrix=attention_matrix.A)
seen_data = testset
tf_scores = tf_scoring(tf_params, seen_data, data_description)
downvote_seen_items(tf_scores, seen_data, data_description)

print('alpha: 4')
make_prediction(tf_scores, holdout, data_description, dcg=True, alpha=4)

config['mlrank'] = (2048, 2048, 4)
tf_params = tf_model_build(config, train_val, data_description, attention_matrix=attention_matrix.A)
seen_data = testset
tf_scores = tf_scoring(tf_params, seen_data, data_description)
downvote_seen_items(tf_scores, seen_data, data_description)

print('alpha: 5')
make_prediction(tf_scores, holdout, data_description, dcg=True, alpha=5)