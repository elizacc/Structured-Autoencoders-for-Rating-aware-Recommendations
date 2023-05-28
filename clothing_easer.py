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

if __name__ == '__main__':
    # %%
    data = pd.read_csv('../../e.makhneva/data/Clothing_Shoes_and_Jewelry/Amazon_Clothing_Shoes_and_Jewelry.csv')
    data.rename(columns={'reviewerID': 'userid', 'asin': 'movieid', "overall": "rating", "unixReviewTime": "timestamp"},
                inplace=True)

    # %%
    training, testset_valid, holdout_valid, testset, holdout, data_description, data_index = full_preproccessing(data)

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

    lambda_grid = np.arange(10, 510, 10)

    # %%
    hr_tf = {}
    mrr_tf = {}
    c_tf = {}
    for lmbda in lambda_grid:
        easer_params = easer(training, data_description, lmbda=lmbda)
        easer_scores = easer_scoring(easer_params, testset_valid, data_description)
        downvote_seen_items(easer_scores, testset_valid, data_description)
        easer_recs = topn_recommendations(easer_scores, topn=10)
        for alpha in [2,3,4,5]:
            hr, hr_pos, hr_neg, mrr, mrr_pos, mrr_neg, cov, C = model_evaluate(easer_recs, holdout_valid, data_description, alpha=alpha)
            hr_tf[f'lambda={lmbda}, alpha={alpha}'] = hr
            mrr_tf[f'lambda={lmbda}, alpha={alpha}'] = mrr
            c_tf[f'lambda={lmbda}, alpha={alpha}'] = C

    # %%
    print(c_tf)


