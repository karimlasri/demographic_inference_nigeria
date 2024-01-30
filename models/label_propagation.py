import pandas as pd
from pathlib import Path
from loading_utils import load_labelled_df, load_names_df, load_user_ids, load_friendship_data
from data_formatting import preprocess_name_df, load_name_list, preprocess_label_df
from scipy.sparse import load_npz
from sklearn.preprocessing import normalize

from scipy.sparse import csr_matrix
import time
from sklearn.metrics import accuracy_score
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt

import sys

PATH_MATCHED_USERS = '../data/all_users_with_demographics.csv'
TEST_USERS_PATH = '../data/all_labels_final.csv' # TODO : Remove labeled_2000 file in data folder
NAME_LIST_PATH = '../data/labeled_name_list.csv'
PREDICTIONS_PATH = '../predictions'
CLASSES = {
    'ethnicity':('hausa', 'igbo', 'yoruba'),
    'gender':('m','f'),
    'religion':('christian', 'muslim')
         }
# The following are paths to the friendship network data
# ADJ_MATRIX_PATH = '../data/friendship_network/adjacency_attention_links_with_outnodes.npz'
# ADJ_NODES_PATH = '../data/friendship_network/nodes.csv'

def load_users_with_matched_names(force_load=False):
    """ Loads a dataframe of users matched with their names using the name list """
    df_list = list()
    for path in Path(PATH_MATCHED_USERS).glob('*.parquet'):
        df = pd.read_parquet(path)
        df_list.append(df)
    users_with_names_df = pd.concat(df_list)
    return users_with_names_df


def predict(feature, name_df, search_df):
    results_list = list()
    for i in range(search_df.shape[0]):
        x = search_df['matched_name'].iloc[i] + search_df['matched_screen_name'].iloc[i]
        x = [n.lower() for n in list(dict.fromkeys(x))]
        if len(x) > 0:
            if len(x) == 1:
                results_list.append(name_df[feature][x[0]])
            elif len(x) > 1:
                predicted_list = list()
                for name in x:
                    if name in name_df[feature]:
                        predicted_list.append(name_df[feature][name])
                if feature == 'gender':
                    predicted_list = [x for x in predicted_list if x != 'unisex']
                    if len(predicted_list) > 0:
                        results_list.append(predicted_list[0])
                    else:
                        results_list.append('None')
                elif feature == 'ethnicity':
                    if 'yoruba' in predicted_list:
                        results_list.append('yoruba')
                    elif 'hausa' in predicted_list:
                        results_list.append('hausa')
                    else:
                        results_list.append(predicted_list[0])
                elif feature == 'religion':
                    predicted_list = [x for x in predicted_list if not str(x) == 'nan']
                    if len(predicted_list) > 0:
                        if 'muslim' in predicted_list:
                            results_list.append('muslim')
                        else:
                            results_list.append(predicted_list[0])
                    else:
                        results_list.append('None')
            else:
                results_list.append('None')
        else:
            results_list.append('None')
    return results_list


def predict_from_names_label_df(label_df, name_df):
    label_df['ethnicity_name_predict'] = predict('ethnicity', name_df, label_df)
    label_df['gender_name_predict'] = predict('gender', name_df, label_df)
    label_df['religion_name_predict'] = predict('religion', name_df, label_df)
    return label_df


def load_friendship_data(matrix_path, nodes_path):
    friendship_matrix = load_npz(matrix_path)
    nodes = pd.read_csv(nodes_path).reset_index().rename(columns={'index':'user_index', '0':'user_id'})
    return friendship_matrix, nodes


def filter_matrix(matrix, nodes, user_ids):
    subset = nodes[nodes['user_id'].isin(user_ids)]
    return matrix[subset['user_index']][:, subset['user_index']], subset


def get_n_friends(indices, matrix):
    n_friends = []
    for i, idx in enumerate(indices):
        if idx is not None:
            n_friends.append(len(list(matrix.getrow(idx).indices)))
        else:
            n_friends.append(0)
    return n_friends


if __name__=='__main__':
    path_to_friendship_data = sys.argv[1]
    adj_matrix_path = path_to_friendship_data + 'adjacency_attention_links_with_outnodes.npz'
    adj_nodes_path = path_to_friendship_data + 'nodes.csv'
    
    # Load users with matched names
    users_with_names_df = pd.read_csv(PATH_MATCHED_USERS)
    # Load test set
#     labels_test = pd.read_csv(TEST_USERS_PATH).drop(['name', 'screen_name', 'link', 'comment'], axis=1)
    
#     # Preprocess test set
#     valid_id = labels_test['user_id'].apply(lambda x:'+' not in x)
#     labels_test = labels_test[valid_id]
#     labels_test['user_id'] = labels_test['user_id'].astype('int64')
#     labels_test['ethnicity'] = labels_test['ethnicity'].str.lower()
    
    label_df = pd.read_csv(TEST_USERS_PATH)
    # Load names labels
    name_df = load_names_df(NAME_LIST_PATH)
    # Preprocess names dataframe
    name_df = preprocess_name_df(name_df)

    # Get list of names
    name_list = load_name_list(NAME_LIST_PATH)
    
    # Preprocess test set
    label_df = preprocess_label_df(label_df)
    label_df = predict_from_names_label_df(label_df, name_df)
    label_df['valid_id'] = label_df['user_id'].apply(lambda x:'+' not in x)
    label_df = label_df[label_df['valid_id']]
    label_df['user_id'] = label_df['user_id'].astype('int64')
    # label_df = label_df[label_df['user_id'].isin(labels_test['user_id'])]
    label_df = label_df[['user_id', 'ethnicity_name_predict', 'gender_name_predict', 'religion_name_predict']]

    # Get all users with names
    all_users_with_name = pd.concat([users_with_names_df, label_df])
    all_users_with_name['user_id'] = all_users_with_name['user_id'].astype('int64')
    all_users_with_name = all_users_with_name.drop_duplicates('user_id')
    
    matched_names_users = all_users_with_name['user_id'].values
    
    # Load adjacency matrix
    friendship, nodes = load_friendship_data(ADJ_MATRIX_PATH, ADJ_NODES_PATH)
    user_ids = all_users_with_name['user_id']
    user_ids = user_ids.reset_index(drop=True)
    
    filtered_friendship, subset = filter_matrix(friendship, nodes, user_ids)
    subset = subset.reset_index(drop=True)
    
    subset2 = subset.join(user_ids.reset_index().set_index('user_id').rename({'index':'orig_index'}, axis=1), on='user_id', how='left').drop_duplicates('user_id') 
    subset_index_to_id = {row['user_id']:i for i, row in subset.iterrows()}
    
    # Get user ids positions from nodes list
    labels_test = pd.read_csv(TEST_USERS_PATH).drop(['name', 'screen_name', 'link'], axis=1) # pd.read_csv('label_df_test.csv')
    labels_test = labels_test[labels_test['org']=='0']
    labels_test = labels_test[labels_test['suspended']=='0']
    # print(labels_test.columns.values)

    # labels_test = pd.read_csv('labeled_2000.csv').drop(['name', 'screen_name', 'link', 'comment'], axis=1) # pd.read_csv('label_df_test.csv')
    valid_id = labels_test['user_id'].apply(lambda x:'+' not in x)
    labels_test = labels_test[valid_id]
    labels_test['user_id'] = labels_test['user_id'].astype('int64')
    labels_test['ethnicity'] = labels_test['ethnicity'].str.lower()
    
    indices_test = []
    for user_id in labels_test['user_id']:
        idx = subset_index_to_id.get(int(user_id), None)
        indices_test.append(idx)
        
    all_users_with_name2 = all_users_with_name.set_index('user_id')
    all_users_with_name2.index = all_users_with_name2.index.astype('int64')
    
    
    # Get friendship matrix
    symmetric = True

    if symmetric:
        symmetric_friendship = filtered_friendship.transpose() + filtered_friendship
        n_friends = get_n_friends(indices_test, symmetric_friendship)
        norm_filtered_friendship = normalize(symmetric_friendship.astype('float64'), norm='l1', axis=1)

    else:
        n_friends = get_n_friends(indices_test, filtered_friendship)
        norm_filtered_friendship = normalize(filtered_friendship.astype('float64'), norm='l1', axis=1)
    
    
    df = pd.DataFrame(subset['user_id'])
    for feature in ('ethnicity', 'religion', 'gender'):
        feature_df = pd.read_csv(f'{PREDICTIONS_PATH}/name_matching_scores_{feature}.csv').drop('Unnamed: 0', axis=1)
        feature_df = feature_df.rename({k:'{}_name_predict_{}'.format(feature, k) for k in feature_df.columns if k!='user_id'}, axis=1)
        df = df.merge(feature_df, on='user_id', how='left').fillna(0)
        
        
        
   # Perform Label Propagation
    alpha = 0.5
    num_generations = 5
    keep_init = True

    for label in ('ethnicity', 'gender', 'religion'):
        all_labels = users_with_names_df['{}_name_predict'.format(label)].unique()
        binary_df = pd.get_dummies(all_users_with_name2.loc[subset['user_id']].drop(['{}_name_predict'.format(l) for l in CLASSES if l != label], axis=1).replace(['cameroon','unisex','christian/muslim', 'muslim/christian'], 'None'), columns=['{}_name_predict'.format(label)])
    #     none_df = binary_df['{}_name_predict_None'.format(label)]
    #     init_df = binary_df[['{}_name_predict_{}'.format(label, e) for e in labels[label]]]
        init_df = df[['{}_name_predict_{}'.format(label, e) for e in CLASSES[label]]]
        scores_df = pd.DataFrame(init_df)
        scores_matrix = scores_df.values
        init_matrix = init_df.values
        idx_filter_matrix = [idx for i, idx in enumerate(indices_test) if idx is not None and labels_test.iloc[i][label] in CLASSES[label]]
        idx_filter_labels = [i for i, idx in enumerate(indices_test) if idx is not None and labels_test.iloc[i][label] in CLASSES[label]]
        print(len(idx_filter_labels))
        predictions = [CLASSES[label][p] for p in scores_matrix[idx_filter_matrix, :].argmax(axis=1)]
        acc = accuracy_score(predictions, labels_test.iloc[idx_filter_labels][label])
        print('Performing label propagation for {}.'.format(label))
        print('Initial accuracy : {}'.format(round(acc,2)))
        cov = len([idx for idx in indices_test if idx is not None and scores_matrix[idx].sum() > 0])/labels_test.shape[0]
        print('Initial coverage : {}'.format(round(cov,2)))
        accs = [acc]

        start_time = time.time()
        if keep_init:
            keep_vector = np.array([1 if (row.sum() == 0) else 0 for row in scores_matrix]).reshape(-1, 1)
            norm_adj_matrix = norm_filtered_friendship.multiply(keep_vector)
        else:
            norm_adj_matrix = norm_filtered_friendship
        for g in range(num_generations):
    #         print('Generation {}'.format(g))
            scores = []
            scores_matrix = alpha * csr_matrix.dot(norm_adj_matrix, scores_matrix) + (1-alpha)*init_matrix

    #         print('{} sec elapsed'.format(time.time()-start_time))
            predictions = [CLASSES[label][p] for p in scores_matrix[idx_filter_matrix, :].argmax(axis=1)]
            acc = accuracy_score(predictions, labels_test.iloc[idx_filter_labels][label])
            cov = len([idx for idx in indices_test if idx is not None and scores_matrix[idx].sum() > 0])/labels_test.shape[0]
            print('Acc. at generation {} : {}'.format(g+1, round(acc,2)))
            print('Cov. at generation {} : {}'.format(g+1, round(cov,2)))
            accs.append(acc)

        pkl.dump(predictions, open('data/lab_prop_predictions_{}_{}_20-03-03.npz'.format(label, alpha), 'wb'))
    #     scores_df.index.to_frame().to_csv('lab_prop_user_ids_{}.csv'.format(label))
        df['user_id'].to_csv('lab_prop_user_ids.csv')
        np.save('predictions/scores_matrix_{}_{}_20-03-03.npz'.format(label, alpha), scores_matrix)
    #     pkl.dump(accs, open('data/accuracies_label_propagation_{}_{}.pkl'.format(label, alpha), 'wb'))

        results = {}
        accs_thres = []
        coverages = []
        for min_friends in np.arange(0, 500, 10):
            predictions = [CLASSES[label][p] for p in scores_matrix[[idx for i, idx in enumerate(indices_test) if idx is not None and n_friends[i]<=min_friends+10 and labels_test.iloc[i][label] in CLASSES[label]], :].argmax(axis=1)]
            acc = accuracy_score(predictions, labels_test.iloc[[i for i,idx in enumerate(indices_test) if idx is not None and n_friends[i]<=min_friends+10 and labels_test.iloc[i][label] in CLASSES[label]]][label])
            accs_thres.append(acc)
    #         coverage = len([f for i, f in enumerate(n_friends) if f>=min_friends and labels_test.iloc[i][label] in labels[label]])/labels_test.shape[0]
            coverage = len([f for i, f in enumerate(n_friends) if f>=min_friends and labels_test.iloc[i][label] in labels[label]])/labels_test[labels_test[label].isin(CLASSES[label])].shape[0]

            coverages.append(coverage)
        print(coverages[0])
        plt.style.use('seaborn-darkgrid')
        fig, ax = plt.subplots()
        results['threshold'] = np.arange(0, 500, 10)
        results['accuracy'] = accs_thres
        results['cdf'] = [1-c for c in coverages]
        results['majority_baseline'] = nm_results[label]['maj']
        results['name_matching_cov'] = nm_results[label]['cov']
        results['name_matching_acc'] = nm_results[label]['acc']
        pkl.dump(results, open('results_{}.pkl'.format(label), 'wb'))
        ax.plot(np.arange(0, 500, 10), accs_thres, color='orange', marker='o')
        ax.set_ylabel('Accuracy on {}'.format(label), color='orange')
        ax.set_xlabel('Threshold on min # friends')
        ax.set_ylim(0,1)
        ax2 = ax.twinx()
        ax2.plot(np.arange(0, 500, 10), [1-c for c in coverages], color='blue', marker='o')
        ax2.axhline(y=nm_results[label]['acc'], color='orange', linestyle='--', label='Name Matching Accuracy')
        ax2.axhline(y=nm_results[label]['cov'], color='blue', linestyle='--', label='Name Matching Coverage')
        ax2.axhline(y=nm_results[label]['maj'], color='black', linestyle='--', label='Majority Baseline')
        ax2.set_ylabel('Cumulative distribution function kept users', color='blue')
        ax2.set_ylim(0,1)
        plt.legend()
        plt.show()