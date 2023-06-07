import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import norm, svds

# from polara.lib.tensor import hooi
# from polara.lib.sparse import tensor_outer_at

from dataprep import transform_indices, full_preproccessing
from utils import *
# from RecVAE.utils import *
# from RecVAE.model import VAE as RecVAE

def set_random_seed(seed):
#     torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    
# fix_torch_seed(42)
set_random_seed(42)

data = pd.read_csv('../../e.makhneva/data/Video_Games/Amazon_Video_Games.csv')
data.rename(columns={'reviewerID': 'userid', 'asin': 'movieid', "overall": "rating", "unixReviewTime": "timestamp"},
                inplace=True)

training, testset_valid, holdout_valid, testset, holdout, data_description, data_index = full_preproccessing(data)

def matrix_from_observations(data, data_description):
    useridx = data[data_description['users']]
    itemidx = data[data_description['items']]
    values = data[data_description['feedback']]
    return csr_matrix((values, (useridx, itemidx)), shape=(useridx.values.max() + 1, data_description["n_items"]), dtype='f8')

def build_svd_model(config, data, data_description):
    source_matrix = matrix_from_observations(data, data_description)
    #print(source_matrix.shape)
    D = norm(source_matrix, axis=0)
    A = source_matrix.dot(diags(D**(config['f']-1)))
    _, _, vt = svds(A, k=config['rank'], return_singular_vectors='vh')
#     singular_values = s[::-1]
    item_factors = np.ascontiguousarray(vt[::-1, :].T)
    return item_factors

def svd_model_scoring(params, data, data_description):
    item_factors = params
    test_data = data.assign(
        userid = pd.factorize(data['userid'])[0]
    )
    test_matrix = matrix_from_observations(test_data, data_description)
    #print(test_matrix.shape, item_factors.shape)
    scores = test_matrix.dot(item_factors) @ item_factors.T
    return scores


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

svd_params = build_svd_model({'rank':1536, 'f':0.4}, train_val, data_description)
svd_scores = svd_model_scoring(svd_params, testset, data_description)
downvote_seen_items(svd_scores, testset, data_description)

make_prediction(svd_scores, holdout, data_description, dcg=True, alpha=2)
make_prediction(svd_scores, holdout, data_description, dcg=True, alpha=3)

svd_params = build_svd_model({'rank':64, 'f':1.0}, train_val, data_description)
svd_scores = svd_model_scoring(svd_params, testset, data_description)
downvote_seen_items(svd_scores, testset, data_description)
make_prediction(svd_scores, holdout, data_description, dcg=True, alpha=4)

svd_params = build_svd_model({'rank':96, 'f':0.7}, train_val, data_description)
svd_scores = svd_model_scoring(svd_params, testset, data_description)
downvote_seen_items(svd_scores, testset, data_description)

make_prediction(svd_scores, holdout, data_description, dcg=True, alpha=5)