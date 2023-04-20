import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from tqdm.notebook import tqdm
from torch.utils.data import Dataset, DataLoader


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

def prepare_tensor(data, data_description, tensor_model=False):
    useridx = pd.factorize(data[data_description['users']])[0]
    itemidx = data[data_description['items']].values
    feedbackidx = data[data_description['feedback']].values
    values = np.ones(len(itemidx), dtype=np.float32)
    user_tensor_test = torch.sparse_coo_tensor(np.array([useridx, itemidx, feedbackidx-1]), torch.tensor(values),
                                      size=torch.Size((data[data_description['users']].nunique(), data_description["n_items"], data_description['n_ratings']))).to_dense()
    target_test = torch.sparse_coo_tensor(np.array([useridx, itemidx]), torch.tensor(values),
                                      size=torch.Size((data[data_description['users']].nunique(), data_description["n_items"], ))).to_dense()
    if tensor_model:
        return user_tensor_test, target_test
    else:
        return target_test, target_test
    
def predict_and_check(model, scores, holdout, data_description, hrs, mrrs, cs, ndcgs, alpha, prev_matt, epoch, h, disp=False, dcg=True):
    mrr10, hr10, c10, ndcg10 = make_prediction(scores, holdout, data_description, disp=disp, dcg=dcg, alpha=alpha)
    hrs.append(hr10)
    mrrs.append(mrr10)
    cs.append(c10)
    ndcgs.append(ndcg10)

    if np.max(prev_matt) < cs[-1] or epoch == 1:
        prev_matt = [cs[-1]]
        torch.save(model.state_dict(), f'best_ae_{h}_{alpha}.pt')
#     elif prev_matt[-1] < cs[-1]:
#         prev_matt = [cs[-1]]
    else:
        prev_matt.append(cs[-1])
        
    return prev_matt

def check_test(model, criterion, user_tensor_test, target_test, testset, holdout, data_description, test_num_batches, alpha, h, device, batch_size=16, dcg=True):
    test_loss = 0
    scores = torch.zeros((len(testset.userid.unique()), data_description['n_items']))
    
    model.load_state_dict(torch.load(f'best_ae_{h}_{alpha}.pt'))
    with torch.no_grad():
        for batch in range(test_num_batches):
            input_tensor = user_tensor_test[batch * batch_size: (batch+1) * batch_size].to(device)
            target = target_test[batch * batch_size: (batch+1) * batch_size].to(device)

            output = model(input_tensor)
            target.require_grad = False

            test_loss += criterion(output, target)
            scores[batch * batch_size: (batch+1) * batch_size] = output

    test_loss = test_loss / test_num_batches
    scores = scores.detach().cpu().numpy()
    downvote_seen_items(scores, testset, data_description)
    print(f'Results for alpha={alpha}')
    mrr10, hr10, c10, ndcg10 = make_prediction(scores, holdout, data_description, dcg=dcg, alpha=alpha)

def tuning_pipeline(training, testset_valid, holdout_valid, data_description, model_init, device, grid, batch_size=16, tensor_model=False, early_stop=50, n_epochs=1000):
    user_tensor_train, target_train = prepare_tensor(training, data_description, tensor_model)
    user_tensor_val, target_val = prepare_tensor(testset_valid, data_description, tensor_model)
    
    num_batches = int(np.ceil(user_tensor_train.shape[0] / batch_size))
    val_num_batches = int(np.ceil(target_val.shape[0] / batch_size))

    for h in tqdm(grid):
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

        prev_matt2 = [0]
        prev_matt3 = [0]
        prev_matt4 = [0]
        prev_matt5 = [0]

        for epoch in range(1, n_epochs+1):
            train_loss = 0
            shuffle = np.random.choice(user_tensor_train.shape[0], size=user_tensor_train.shape[0], replace=False)
            user_tensor_train = user_tensor_train[shuffle]

            for batch in range(num_batches):
                optimizer.zero_grad()

                input_tensor = user_tensor_train[batch * batch_size: (batch+1) * batch_size].to(device)
                target = target_train[batch * batch_size: (batch+1) * batch_size].to(device)

                output = model(input_tensor)
                target.require_grad = False # we don't use it in training

                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                train_loss += loss.data.item()

            scheduler.step()
            history.append(train_loss / num_batches)

            test_loss = 0
            scores = torch.zeros((testset_valid.userid.nunique(), data_description['n_items']))

            with torch.no_grad():
                model.eval()
                for batch in range(val_num_batches):
                    input_tensor = user_tensor_val[batch * batch_size: (batch+1) * batch_size].to(device)
                    target = target_val[batch * batch_size: (batch+1) * batch_size].to(device)

                    output = model(input_tensor)
                    target.require_grad = False

                    test_loss += criterion(output, target)
                    scores[batch * batch_size: (batch+1) * batch_size] = output
                model.train()

            scores = scores.detach().cpu().numpy()
            val_loss = test_loss / val_num_batches
            val_history.append(val_loss.item())

            downvote_seen_items(scores, testset_valid, data_description)

            prev_matt2 = predict_and_check(model, scores, holdout_valid, data_description, hrs2, mrrs2, cs2, ndcgs2, 2, prev_matt2, epoch, h)
            prev_matt3 = predict_and_check(model, scores, holdout_valid, data_description, hrs3, mrrs3, cs3, ndcgs3, 3, prev_matt3, epoch, h)
            prev_matt4 = predict_and_check(model, scores, holdout_valid, data_description, hrs4, mrrs4, cs4, ndcgs4, 4, prev_matt4, epoch, h)
            prev_matt5 = predict_and_check(model, scores, holdout_valid, data_description, hrs5, mrrs5, cs5, ndcgs5, 5, prev_matt5, epoch, h)

            # stop = epoch if epoch < early_stop else epoch-early_stop
            if len(prev_matt2) >= early_stop and len(prev_matt3) >= early_stop and len(prev_matt4) >= early_stop and len(prev_matt5) >= early_stop:
                print(f'Current epoch {epoch}')
                break

        # Testing the AE
        check_test(model, criterion, user_tensor_val, target_val, testset_valid, holdout_valid, data_description, val_num_batches, 2, h, batch_size=batch_size, dcg=True)
        check_test(model, criterion, user_tensor_val, target_val, testset_valid, holdout_valid, data_description, val_num_batches, 3, h, batch_size=batch_size, dcg=True)
        check_test(model, criterion, user_tensor_val, target_val, testset_valid, holdout_valid, data_description, val_num_batches, 4, h, batch_size=batch_size, dcg=True)
        check_test(model, criterion, user_tensor_val, target_val, testset_valid, holdout_valid, data_description, val_num_batches, 5, h, batch_size=batch_size, dcg=True)

        # our
        plt.figure(figsize=(10,6))
        plt.plot(history, label='train')
        plt.plot(val_history, label='val')
        plt.legend()
        plt.show()

        hrs = [hrs2, hrs3, hrs4, hrs5]
        mrrs = [mrrs2, mrrs3, mrrs4, mrrs5]
        cs = [cs2, cs3, cs4, cs5]
        ndcgs = [ndcgs2, ndcgs3, ndcgs4, ndcgs5]

        fig = plt.figure(figsize=(24,5))
        axes = fig.subplots(nrows=1, ncols=4)
        for i in range(4):
            axes[i].set_title(f'alpha={i+2}')
            axes[i].plot(hrs[i], label='HR@10')
            axes[i].plot(mrrs[i], label='MRR@10')
            axes[i].plot(cs[i], label='Matthews@10')
            axes[i].plot(ndcgs[i], label='NDCG@10')
            axes[i].legend()

        plt.show()

        print('Test loss:', val_history[-min(early_stop, epoch)])
        print('Train loss:', history[-min(early_stop, epoch)])

        print()
        print()

def tuning_pipeline_augment(training, testset_valid, holdout_valid, data_description, model_init, grid, device, MVDataset, batch_size=16, early_stop=50, n_epochs=1000):
    train_dataset = MVDataset(training, data_description, augment=True)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    user_tensor_val, target_val = prepare_tensor(testset_valid, data_description)    
    val_num_batches = int(np.ceil(target_val.shape[0] / batch_size))

    for h in tqdm(grid):
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

        prev_matt2 = [0]
        prev_matt3 = [0]
        prev_matt4 = [0]
        prev_matt5 = [0]

        for epoch in range(1, n_epochs+1):
            train_loss = 0
            shuffle = np.random.choice(user_tensor_train.shape[0], size=user_tensor_train.shape[0], replace=False)
            user_tensor_train = user_tensor_train[shuffle]

            for batch in train_dataloader:
                optimizer.zero_grad()

                input_tensor, target = batch
                input_tensor, target = input_tensor.to_dense().to(device), target.to_dense().to(device)

                output = model(input_tensor)
                target.require_grad = False # we don't use it in training

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
                    input_tensor = user_tensor_val[batch * batch_size: (batch+1) * batch_size].to(device)
                    target = target_val[batch * batch_size: (batch+1) * batch_size].to(device)

                    output = model(input_tensor)
                    target.require_grad = False

                    test_loss += criterion(output, target)
                    scores[batch * batch_size: (batch+1) * batch_size] = output
                model.train()

            scores = scores.detach().cpu().numpy()
            val_loss = test_loss / val_num_batches
            val_history.append(val_loss.item())

            downvote_seen_items(scores, testset_valid, data_description)

            prev_matt2 = predict_and_check(model, scores, holdout_valid, data_description, hrs2, mrrs2, cs2, ndcgs2, 2, prev_matt2, epoch)
            prev_matt3 = predict_and_check(model, scores, holdout_valid, data_description, hrs3, mrrs3, cs3, ndcgs3, 3, prev_matt3, epoch)
            prev_matt4 = predict_and_check(model, scores, holdout_valid, data_description, hrs4, mrrs4, cs4, ndcgs4, 4, prev_matt4, epoch)
            prev_matt5 = predict_and_check(model, scores, holdout_valid, data_description, hrs5, mrrs5, cs5, ndcgs5, 5, prev_matt5, epoch)

            # stop = epoch if epoch < early_stop else epoch-early_stop
            if len(prev_matt2) >= early_stop and len(prev_matt3) >= early_stop and len(prev_matt4) >= early_stop and len(prev_matt5) >= early_stop:
                print(f'Current epoch {epoch}')
                break

        # Testing the AE
        check_test(model, criterion, user_tensor_val, target_val, testset_valid, holdout_valid, data_description, val_num_batches, 2, batch_size=batch_size, dcg=True)
        check_test(model, criterion, user_tensor_val, target_val, testset_valid, holdout_valid, data_description, val_num_batches, 3, batch_size=batch_size, dcg=True)
        check_test(model, criterion, user_tensor_val, target_val, testset_valid, holdout_valid, data_description, val_num_batches, 4, batch_size=batch_size, dcg=True)
        check_test(model, criterion, user_tensor_val, target_val, testset_valid, holdout_valid, data_description, val_num_batches, 5, batch_size=batch_size, dcg=True)

        # our
        plt.figure(figsize=(10,6))
        plt.plot(history, label='train')
        plt.plot(val_history, label='val')
        plt.legend()
        plt.show()

        hrs = [hrs2, hrs3, hrs4, hrs5]
        mrrs = [mrrs2, mrrs3, mrrs4, mrrs5]
        cs = [cs2, cs3, cs4, cs5]
        ndcgs = [ndcgs2, ndcgs3, ndcgs4, ndcgs5]

        fig = plt.figure(figsize=(24,5))
        axes = fig.subplots(nrows=1, ncols=4)
        for i in range(4):
            axes[i].set_title(f'alpha={i+2}')
            axes[i].plot(hrs[i], label='HR@10')
            axes[i].plot(mrrs[i], label='MRR@10')
            axes[i].plot(cs[i], label='Matthews@10')
            axes[i].plot(ndcgs[i], label='NDCG@10')
            axes[i].legend()

        plt.show()

        print('Test loss:', val_history[-min(early_stop, epoch)])
        print('Train loss:', history[-min(early_stop, epoch)])

        print()
        print()

def training_testing_pipeline(training, testset_valid, holdout_valid, testset, holdout, data_description, model_init, h, device, batch_size=16, tensor_model=False, early_stop=50, n_epochs=1000):
    train_val = pd.concat((training, testset_valid, holdout_valid))
    user_tensor_train, target_train = prepare_tensor(train_val, data_description, tensor_model)
    user_tensor_val, target_val = prepare_tensor(testset, data_description)
    
    num_batches = int(np.ceil(user_tensor_train.shape[0] / batch_size))
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

    prev_matt2 = [0]
    prev_matt3 = [0]
    prev_matt4 = [0]
    prev_matt5 = [0]

    for epoch in range(1, n_epochs+1):
        train_loss = 0
        shuffle = np.random.choice(user_tensor_train.shape[0], size=user_tensor_train.shape[0], replace=False)
        user_tensor_train = user_tensor_train[shuffle]

        for batch in range(num_batches):
            optimizer.zero_grad()

            input_tensor = user_tensor_train[batch * batch_size: (batch+1) * batch_size].to(device)
            target = target_train[batch * batch_size: (batch+1) * batch_size].to(device)

            output = model(input_tensor)
            target.require_grad = False # we don't use it in training

            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.data.item()

        scheduler.step()
        history.append(train_loss / num_batches)

        test_loss = 0
        scores = torch.zeros((testset.userid.nunique(), data_description['n_items']))

        with torch.no_grad():
            model.eval()
            for batch in range(val_num_batches):
                input_tensor = user_tensor_val[batch * batch_size: (batch+1) * batch_size].to(device)
                target = target_val[batch * batch_size: (batch+1) * batch_size].to(device)

                output = model(input_tensor)
                target.require_grad = False

                test_loss += criterion(output, target)
                scores[batch * batch_size: (batch+1) * batch_size] = output
            model.train()

        scores = scores.detach().cpu().numpy()
        val_loss = test_loss / val_num_batches
        val_history.append(val_loss.item())

        downvote_seen_items(scores, testset, data_description)

        prev_matt2 = predict_and_check(model, scores, holdout, data_description, hrs2, mrrs2, cs2, ndcgs2, 2, prev_matt2, epoch)
        prev_matt3 = predict_and_check(model, scores, holdout, data_description, hrs3, mrrs3, cs3, ndcgs3, 3, prev_matt3, epoch)
        prev_matt4 = predict_and_check(model, scores, holdout, data_description, hrs4, mrrs4, cs4, ndcgs4, 4, prev_matt4, epoch)
        prev_matt5 = predict_and_check(model, scores, holdout, data_description, hrs5, mrrs5, cs5, ndcgs5, 5, prev_matt5, epoch)

        # stop = epoch if epoch < early_stop else epoch-early_stop
        if len(prev_matt2) >= early_stop and len(prev_matt3) >= early_stop and len(prev_matt4) >= early_stop and len(prev_matt5) >= early_stop:
            print(f'Current epoch {epoch}')
            break

    # Testing the AE
    check_test(model, criterion, user_tensor_val, target_val, testset, holdout, data_description, val_num_batches, 2, batch_size=batch_size, dcg=True)
    check_test(model, criterion, user_tensor_val, target_val, testset, holdout, data_description, val_num_batches, 3, batch_size=batch_size, dcg=True)
    check_test(model, criterion, user_tensor_val, target_val, testset, holdout, data_description, val_num_batches, 4, batch_size=batch_size, dcg=True)
    check_test(model, criterion, user_tensor_val, target_val, testset, holdout, data_description, val_num_batches, 5, batch_size=batch_size, dcg=True)

    # our
    plt.figure(figsize=(10,6))
    plt.plot(history, label='train')
    plt.plot(val_history, label='val')
    plt.legend()
    plt.show()

    hrs = [hrs2, hrs3, hrs4, hrs5]
    mrrs = [mrrs2, mrrs3, mrrs4, mrrs5]
    cs = [cs2, cs3, cs4, cs5]
    ndcgs = [ndcgs2, ndcgs3, ndcgs4, ndcgs5]

    fig = plt.figure(figsize=(24,5))
    axes = fig.subplots(nrows=1, ncols=4)
    for i in range(4):
        axes[i].set_title(f'alpha={i+2}')
        axes[i].plot(hrs[i], label='HR@10')
        axes[i].plot(mrrs[i], label='MRR@10')
        axes[i].plot(cs[i], label='Matthews@10')
        axes[i].plot(ndcgs[i], label='NDCG@10')
        axes[i].legend()

    plt.show()

    print('Test loss:', val_history[-min(early_stop, epoch)])
    print('Train loss:', history[-min(early_stop, epoch)])

    print()
    print()

def training_testing_pipeline_augment(training, testset_valid, holdout_valid, testset, holdout, data_description, model_init, h, device, MVDataset, batch_size=16, early_stop=50, n_epochs=1000):
    train_val = pd.concat((training, testset_valid, holdout_valid))
    train_dataset = MVDataset(train_val, data_description, augment=True)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    user_tensor_val, target_val = prepare_tensor(testset_valid, data_description)    
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

    prev_matt2 = [0]
    prev_matt3 = [0]
    prev_matt4 = [0]
    prev_matt5 = [0]

    for epoch in range(1, n_epochs+1):
        train_loss = 0
        shuffle = np.random.choice(user_tensor_train.shape[0], size=user_tensor_train.shape[0], replace=False)
        user_tensor_train = user_tensor_train[shuffle]

        for batch in train_dataloader:
            optimizer.zero_grad()

            input_tensor, target = batch
            input_tensor, target = input_tensor.to_dense().to(device), target.to_dense().to(device)

            output = model(input_tensor)
            target.require_grad = False # we don't use it in training

            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.data.item()

        scheduler.step()
        history.append(train_loss / len(train_dataloader))

        test_loss = 0
        scores = torch.zeros((testset.userid.nunique(), data_description['n_items']))

        with torch.no_grad():
            model.eval()
            for batch in range(val_num_batches):
                input_tensor = user_tensor_val[batch * batch_size: (batch+1) * batch_size].to(device)
                target = target_val[batch * batch_size: (batch+1) * batch_size].to(device)

                output = model(input_tensor)
                target.require_grad = False

                test_loss += criterion(output, target)
                scores[batch * batch_size: (batch+1) * batch_size] = output
            model.train()

        scores = scores.detach().cpu().numpy()
        val_loss = test_loss / val_num_batches
        val_history.append(val_loss.item())

        downvote_seen_items(scores, testset, data_description)

        prev_matt2 = predict_and_check(model, scores, holdout, data_description, hrs2, mrrs2, cs2, ndcgs2, 2, prev_matt2, epoch)
        prev_matt3 = predict_and_check(model, scores, holdout, data_description, hrs3, mrrs3, cs3, ndcgs3, 3, prev_matt3, epoch)
        prev_matt4 = predict_and_check(model, scores, holdout, data_description, hrs4, mrrs4, cs4, ndcgs4, 4, prev_matt4, epoch)
        prev_matt5 = predict_and_check(model, scores, holdout, data_description, hrs5, mrrs5, cs5, ndcgs5, 5, prev_matt5, epoch)

        # stop = epoch if epoch < early_stop else epoch-early_stop
        if len(prev_matt2) >= early_stop and len(prev_matt3) >= early_stop and len(prev_matt4) >= early_stop and len(prev_matt5) >= early_stop:
            print(f'Current epoch {epoch}')
            break

    # Testing the AE
    check_test(model, criterion, user_tensor_val, target_val, testset, holdout, data_description, val_num_batches, 2, batch_size=batch_size, dcg=True)
    check_test(model, criterion, user_tensor_val, target_val, testset, holdout, data_description, val_num_batches, 3, batch_size=batch_size, dcg=True)
    check_test(model, criterion, user_tensor_val, target_val, testset, holdout, data_description, val_num_batches, 4, batch_size=batch_size, dcg=True)
    check_test(model, criterion, user_tensor_val, target_val, testset, holdout, data_description, val_num_batches, 5, batch_size=batch_size, dcg=True)

    # our
    plt.figure(figsize=(10,6))
    plt.plot(history, label='train')
    plt.plot(val_history, label='val')
    plt.legend()
    plt.show()

    hrs = [hrs2, hrs3, hrs4, hrs5]
    mrrs = [mrrs2, mrrs3, mrrs4, mrrs5]
    cs = [cs2, cs3, cs4, cs5]
    ndcgs = [ndcgs2, ndcgs3, ndcgs4, ndcgs5]

    fig = plt.figure(figsize=(24,5))
    axes = fig.subplots(nrows=1, ncols=4)
    for i in range(4):
        axes[i].set_title(f'alpha={i+2}')
        axes[i].plot(hrs[i], label='HR@10')
        axes[i].plot(mrrs[i], label='MRR@10')
        axes[i].plot(cs[i], label='Matthews@10')
        axes[i].plot(ndcgs[i], label='NDCG@10')
        axes[i].legend()

    plt.show()

    print('Test loss:', val_history[-min(early_stop, epoch)])
    print('Train loss:', history[-min(early_stop, epoch)])

    print()
    print()
