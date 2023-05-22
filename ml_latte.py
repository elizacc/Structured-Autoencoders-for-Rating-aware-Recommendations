# %%
import numpy as np
import pandas as pd

# import polara
# from polara import get_movielens_data
# from polara.preprocessing.dataframes import leave_one_out, reindex
# from polara.evaluation.pipelines import random_grid
# from polara.lib.tensor import hooi
from polara.lib.sparse import tensor_outer_at

# from dataprep import transform_indices
from evaluation import topn_recommendations, model_evaluate, downvote_seen_items
from sa_hooi import sa_hooi, form_attention_matrix, get_scaling_weights

import scipy
from scipy.sparse import csr_matrix
from scipy.linalg import solve_triangular
# from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, rbf_kernel, laplacian_kernel

# %%
training, testset_valid, holdout_valid, testset, holdout, data_description = full_preproccessing(data)

# %%
def tf_model_build(config, data, data_description, attention_matrix=np.array([])):
    userid = data_description["users"]
    itemid = data_description["items"]
    feedback = data_description["feedback"]

    idx = data[[userid, itemid, feedback]].values
    idx[:, -1] = idx[:, -1] - data_description['min_rating'] # works only for integer ratings!
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
            format = 'csr'
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
        attention_matrix = attention_matrix,
        scaling_weights = scaling_weights,
        max_iters = config["num_iters"],
        parallel_ttm = False,
        randomized = config["randomized"],
        growth_tol = config["growth_tol"],
        seed = config["seed"],
        iter_callback = None,
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
    
    #inv_attention = np.linalg.inv(attention_matrix.A) # change
    inv_attention = solve_triangular(attention_matrix, np.eye(5), lower=True)
    #np.testing.assert_almost_equal(inv_attention, inv_attention_)
    
    tensor_outer = tensor_outer_at('cpu')

    inv_aT_feedback = (inv_attention.T @ feedback_factors)[-1, :]
        
    scores = tensor_outer(
        1.0,
        item_factors,
        attention_matrix @ feedback_factors,
        itemidx,
        ratings
    )
    scores = np.add.reduceat(scores, np.r_[0, np.where(np.diff(useridx))[0]+1]) # sort by users
    scores = np.tensordot(
        scores,
        inv_aT_feedback,
        axes=(2, 0)
    ).dot(item_factors.T)
    
    return scores

# %%
grid1 = 2**np.arange(4, 12)
grid2 = np.arange(2, 5)
grid = np.meshgrid(grid1, grid2)

# %%
print('Linear matrix')
config["params"] = {'decay_factor': 1, 'exponential_decay': False, 'reverse': False}

hr_tf = {}
mrr_tf = {}
c_tf = {}
for r12, r3 in zip(grid[0].flatten(), grid[1].flatten()):
    config['mlrank'] = (r12, r12, r3)
    tf_params = tf_model_build(config, training, data_description, attention_matrix=np.array([]))
    tf_scores = tf_scoring(tf_params, testset_valid, data_description)
    downvote_seen_items(tf_scores, testset_valid, data_description)
    tf_recs = topn_recommendations(tf_scores, topn=10)
    for alpha in [2,3,4,5]:
        hr, hr_pos, hr_neg, mrr, mrr_pos, mrr_neg, cov, C = model_evaluate(tf_recs, holdout_valid, data_description, alpha=alpha)
        hr_tf[f'r={r}, f={f:.2f}, alpha={alpha}'] = hr
        mrr_tf[f'r={r}, f={f:.2f}, alpha={alpha}'] = mrr
        c_tf[f'r={r}, f={f:.2f}, alpha={alpha}'] = C

# %%
print(c_tf)

# %%
print('Custom matrix')
config["params"] = {}
similarity_matrix = np.array([[1, 0.85, 0.5, 0.1, 0], [0.85, 1, 0.65, 0.2, 0.05], [0.5, 0.65, 1, 0.75, 0.2], [0.1, 0.2, 0.75, 1, 0.85], [0, 0.05, 0.2, 0.85, 1]])
attention_matrix = scipy.linalg.sqrtm(similarity_matrix).real

hr_tf = {}
mrr_tf = {}
c_tf = {}
for r12, r3 in zip(grid[0].flatten(), grid[1].flatten()):
    config['mlrank'] = (r12, r12, r3)
    tf_params = tf_model_build(config, training, data_description, attention_matrix=attention_matrix)
    tf_scores = tf_scoring(tf_params, testset_valid, data_description)
    downvote_seen_items(tf_scores, testset_valid, data_description)
    tf_recs = topn_recommendations(tf_scores, topn=10)
    for alpha in [2,3,4,5]:
        hr, hr_pos, hr_neg, mrr, mrr_pos, mrr_neg, cov, C = model_evaluate(tf_recs, holdout_valid, data_description, alpha=alpha)
        hr_tf[f'r={r}, f={f:.2f}, alpha={alpha}'] = hr
        mrr_tf[f'r={r}, f={f:.2f}, alpha={alpha}'] = mrr
        c_tf[f'r={r}, f={f:.2f}, alpha={alpha}'] = C

# %%
print('Exponential decay matrix')
exponential_attentions_list = [
    {'decay_factor': 1, 'exponential_decay': True, 'reverse': True},
    {'decay_factor': 1, 'exponential_decay': True, 'reverse': False}
]
for params in exponential_attentions_list:
    config["params"] = params
    print(params)
    
    hr_tf = {}
    mrr_tf = {}
    c_tf = {}
    for r12, r3 in zip(grid[0].flatten(), grid[1].flatten()):
        config['mlrank'] = (r12, r12, r3)
        tf_params = tf_model_build(config, training, data_description, attention_matrix=np.array([]))
        tf_scores = tf_scoring(tf_params, testset_valid, data_description)
        downvote_seen_items(tf_scores, testset_valid, data_description)
        tf_recs = topn_recommendations(tf_scores, topn=10)
        for alpha in [2,3,4,5]:
            hr, hr_pos, hr_neg, mrr, mrr_pos, mrr_neg, cov, C = model_evaluate(tf_recs, holdout_valid, data_description, alpha=alpha)
            hr_tf[f'r={r}, f={f:.2f}, alpha={alpha}'] = hr
            mrr_tf[f'r={r}, f={f:.2f}, alpha={alpha}'] = mrr
            c_tf[f'r={r}, f={f:.2f}, alpha={alpha}'] = C

# %%
print('Eucledian distance matrix')
eucl_matrix = np.zeros((5, 5))

for i in range(5):
    for j in range(5):
        eucl_matrix[i, j] = 1.0 / np.exp(abs(i - j)) if i != j else 1#5 + 1e-2
        
a = np.linalg.cholesky(eucl_matrix)

#for i in range(5):
#    a[i, i] = 1e-5

attention_matrix = csr_matrix(a)

hr_tf = {}
mrr_tf = {}
c_tf = {}
for r12, r3 in zip(grid[0].flatten(), grid[1].flatten()):
    config['mlrank'] = (r12, r12, r3)
    tf_params = tf_model_build(config, training, data_description, attention_matrix=attention_matrix)
    tf_scores = tf_scoring(tf_params, testset_valid, data_description)
    downvote_seen_items(tf_scores, testset_valid, data_description)
    tf_recs = topn_recommendations(tf_scores, topn=10)
    for alpha in [2,3,4,5]:
        hr, hr_pos, hr_neg, mrr, mrr_pos, mrr_neg, cov, C = model_evaluate(tf_recs, holdout_valid, data_description, alpha=alpha)
        hr_tf[f'r={r}, f={f:.2f}, alpha={alpha}'] = hr
        mrr_tf[f'r={r}, f={f:.2f}, alpha={alpha}'] = mrr
        c_tf[f'r={r}, f={f:.2f}, alpha={alpha}'] = C

# %%
print('Eucledian distance matrix variation')
eucl_matrix = np.zeros((5, 5))

for i in range(5):
    for j in range(5):
        eucl_matrix[i, j] = abs(i - j) / np.exp(abs(i - j)) if i != j else 1#5 + 1e-2
        
a = np.linalg.cholesky(eucl_matrix)

#for i in range(5):
#    a[i, i] = 1e-5

attention_matrix = csr_matrix(a)

hr_tf = {}
mrr_tf = {}
c_tf = {}
for r12, r3 in zip(grid[0].flatten(), grid[1].flatten()):
    config['mlrank'] = (r12, r12, r3)
    tf_params = tf_model_build(config, training, data_description, attention_matrix=attention_matrix)
    tf_scores = tf_scoring(tf_params, testset_valid, data_description)
    downvote_seen_items(tf_scores, testset_valid, data_description)
    tf_recs = topn_recommendations(tf_scores, topn=10)
    for alpha in [2,3,4,5]:
        hr, hr_pos, hr_neg, mrr, mrr_pos, mrr_neg, cov, C = model_evaluate(tf_recs, holdout_valid, data_description, alpha=alpha)
        hr_tf[f'r={r}, f={f:.2f}, alpha={alpha}'] = hr
        mrr_tf[f'r={r}, f={f:.2f}, alpha={alpha}'] = mrr
        c_tf[f'r={r}, f={f:.2f}, alpha={alpha}'] = C

# %% [markdown]
# ## Rating distribution attention

# %%
print('Rating distribution matrix')
rating_dist = []

total_cnt = training.shape[0]

for i in range(5):
    val = training.query(f'rating == {i + 1}').count()[0] / total_cnt
    
    rating_dist.append(val)

rat_dist_matrix = np.zeros((5, 5))

for i in range(5):
    for j in range(5):
        diff = abs(rating_dist[i] - rating_dist[j])
        rat_dist_matrix[i, j] = diff / np.exp(diff) if i != j else 1. + 1e-1
        
a = np.linalg.cholesky(rat_dist_matrix)

for i in range(5):
    a[i, i] = 1e-5

attention_matrix = csr_matrix(a)
#rat_dist_matrix

hr_tf = {}
mrr_tf = {}
c_tf = {}
for r12, r3 in zip(grid[0].flatten(), grid[1].flatten()):
    config['mlrank'] = (r12, r12, r3)
    tf_params = tf_model_build(config, training, data_description, attention_matrix=attention_matrix)
    tf_scores = tf_scoring(tf_params, testset_valid, data_description)
    downvote_seen_items(tf_scores, testset_valid, data_description)
    tf_recs = topn_recommendations(tf_scores, topn=10)
    for alpha in [2,3,4,5]:
        hr, hr_pos, hr_neg, mrr, mrr_pos, mrr_neg, cov, C = model_evaluate(tf_recs, holdout_valid, data_description, alpha=alpha)
        hr_tf[f'r={r}, f={f:.2f}, alpha={alpha}'] = hr
        mrr_tf[f'r={r}, f={f:.2f}, alpha={alpha}'] = mrr
        c_tf[f'r={r}, f={f:.2f}, alpha={alpha}'] = C

# %% [markdown]
# ## Trigonometry scale attention

# %%
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
        
print(similarity)
    
a = np.linalg.cholesky(similarity)        

attention_matrix = csr_matrix(a)

hr_tf = {}
mrr_tf = {}
c_tf = {}
for r12, r3 in zip(grid[0].flatten(), grid[1].flatten()):
    config['mlrank'] = (r12, r12, r3)
    tf_params = tf_model_build(config, training, data_description, attention_matrix=attention_matrix)
    tf_scores = tf_scoring(tf_params, testset_valid, data_description)
    downvote_seen_items(tf_scores, testset_valid, data_description)
    tf_recs = topn_recommendations(tf_scores, topn=10)
    for alpha in [2,3,4,5]:
        hr, hr_pos, hr_neg, mrr, mrr_pos, mrr_neg, cov, C = model_evaluate(tf_recs, holdout_valid, data_description, alpha=alpha)
        hr_tf[f'r={r}, f={f:.2f}, alpha={alpha}'] = hr
        mrr_tf[f'r={r}, f={f:.2f}, alpha={alpha}'] = mrr
        c_tf[f'r={r}, f={f:.2f}, alpha={alpha}'] = C


