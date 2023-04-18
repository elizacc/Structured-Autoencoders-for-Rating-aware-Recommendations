import numpy as np
import pandas as pd


def downvote_seen_items(scores, data, data_description):
    itemid = data_description['items']
    userid = data_description['users']
    # get indices of observed data
    sorted = data.sort_values(userid)
    item_idx = sorted[itemid].values
    user_idx = sorted[userid].values
    user_idx = np.r_[False, user_idx[1:] != user_idx[:-1]].cumsum()
    # downvote scores at the corresponding positions
    seen_idx_flat = np.ravel_multi_index((user_idx, item_idx), scores.shape)
    np.put(scores, seen_idx_flat, scores.min() - 1)


def topn_recommendations(scores, topn=10):
    recommendations = np.apply_along_axis(topidx, 1, scores, topn)
    return recommendations


def topidx(a, topn):
    parted = np.argpartition(a, -topn)[-topn:]
    return parted[np.argsort(-a[parted])]


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
        ndcg = np.sum(1 / np.log2(hit_rank+1)) / n_test_users
        ndcg_pos = np.sum(1 / np.log2(pos_hit_rank+1)) / n_test_users
        ndcg_neg = np.sum(1 / np.log2(neg_hit_rank+1)) / n_test_users
    
    # coverage calculation
    n_items = holdout_description['n_items']
    cov = np.unique(recommended_items).size / n_items
    if dcg:
        return hr, hr_pos, hr_neg, mrr, mrr_pos, mrr_neg, cov, C, ndcg, ndcg_pos, ndcg_neg
    else:
        return hr, hr_pos, hr_neg, mrr, mrr_pos, mrr_neg, cov, C
    

def make_prediction(tf_scores, holdout, data_description, disp=True, dcg=False, alpha=3):
    for n in [5, 10, 20]:
        tf_recs = topn_recommendations(tf_scores, n)
        if dcg:
            hr, hr_pos, hr_neg, mrr, mrr_pos, mrr_neg, cov, C, ndcg, ndcg_pos, ndcg_neg = model_evaluate(tf_recs, holdout, data_description, topn=n, dcg=dcg, alpha=alpha)
        else:
            hr, hr_pos, hr_neg, mrr, mrr_pos, mrr_neg, cov, C = model_evaluate(tf_recs, holdout, data_description, topn=n, alpha=alpha)
        if n == 10:
            mrr10 = mrr
            hr10 = hr
            c10 = C
            ndcg10 = ndcg
        if dcg:
            results = pd.DataFrame({f"HR@{n}": hr, f"MRR@{n}": mrr, f"Coverage@{n}": cov, f"NCDG@{n}": ndcg,
                                    f"HR_pos@{n}": hr_pos, f"HR_neg@{n}": hr_neg,
                                    f"MRR_pos@{n}": mrr_pos, f"MRR_neg@{n}": mrr_neg,
                                    f"NCDG_pos@{n}": ndcg_pos, f"NDCG_neg@{n}": ndcg_neg,
                                    f"Matthews@{n}": C}, index=[n])
        else:
            results = pd.DataFrame({f"HR@{n}": hr, f"MRR@{n}": mrr, f"Coverage@{n}": cov,
                                    f"HR_pos@{n}": hr_pos, f"HR_neg@{n}": hr_neg,
                                    f"MRR_pos@{n}": mrr_pos, f"MRR_neg@{n}": mrr_neg,
                                    f"Matthews@{n}": C}, index=[n])
        if disp:
            display(results)

    if dcg:
        return mrr10, hr10, c10, ndcg10
    else:
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