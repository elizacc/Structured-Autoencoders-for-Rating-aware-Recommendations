from polara import get_movielens_data
import pandas as pd
import numpy as np
from polara.preprocessing.dataframes import leave_one_out, reindex

def transform_indices(data, users, items):
    data_index = {}
    for entity, field in zip(['users', 'items'], [users, items]):
        idx, idx_map = to_numeric_id(data, field)
        data_index[entity] = idx_map
        data.loc[:, field] = idx
    return data, data_index

def to_numeric_id(data, field):
    idx_data = data[field].astype("category")
    idx = idx_data.cat.codes
    idx_map = idx_data.cat.categories.rename(field)
    return idx, idx_map

def full_preproccessing(data = None, userid='userid', itemid='movieid', rating='rating', timestamp='timestamp', ucore=5, icore=5):
    if (data is None):
        data = get_movielens_data("ml-1m.zip", include_time=True)

    print(f'There are {data[userid].nunique()} users')
    #p-core filtering
    i_count = data.groupby(itemid)[userid].count()
    data = data[data[itemid].isin(i_count[i_count >= icore].index)]
    
    u_count = data.groupby(userid)[itemid].count()
    data = data[data[userid].isin(u_count[u_count >= ucore].index)]

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