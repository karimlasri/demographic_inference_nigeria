import pandas as pd
from pathlib import Path
from match_names import predict_all_from_names
from loading_utils import load_labelled_df, load_name_mapping, load_user_ids, load_friendship_data, load_names_list
from format_data import preprocess_label_df, clean_user_ids
from scipy.sparse import load_npz
from sklearn.preprocessing import normalize

from scipy.sparse import csr_matrix
import time
from sklearn.metrics import accuracy_score
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import json
import argparse


ANNOTATIONS_PATH = 'data/annotations_demographics.csv' 
NAMES_TO_ATTRIBUTES_MAP = 'data/names_attribute_map.csv'
PREDICTIONS_PATH = 'predictions/'
CLASSES = {
    'ethnicity':('hausa', 'igbo', 'yoruba'),
    'gender':('m','f'),
    'religion':('christian', 'muslim')
         }

NM_RESULTS_PATH = 'data/name_matching_eval.json'
if os.path.exists()
with open(NM_RESULTS_PATH) as eval_file:
    NM_RESULTS = json.load(eval_file)

# NM_RESULTS = {
#     'ethnicity':{'cov':0.7, 'acc':0.86, 'maj':0.46494464944649444},
#     'religion':{'cov':0.57, 'acc': 0.91, 'maj':0.7652284263959391},
#     'gender':{'cov':0.65, 'acc':0.83, 'maj':0.7226993865030675}
# }
# The following are paths to the friendship network data
# ADJ_MATRIX_PATH = '../data/friendship_network/adjacency_attention_links_with_outnodes.npz'
# ADJ_NODES_PATH = '../data/friendship_network/nodes.csv'


def load_friendship_data(matrix_path, nodes_path):
    friendship_matrix = load_npz(matrix_path)
    nodes = pd.read_csv(nodes_path).reset_index().rename(columns={'index':'user_index', '0':'user_id'})
    return friendship_matrix, nodes


def filter_matrix(matrix, nodes, user_ids):
    target_nodes = nodes[nodes['user_id'].isin(user_ids)]
    target_nodes = target_nodes.reset_index(drop=True)
    return matrix[target_nodes['user_index']][:, target_nodes['user_index']], target_nodes


def update_scores(scores_matrix, norm_adj_matrix, init_matrix, alpha=0.5):
    """ Update the scores matrix using the label propagation formula. """
    scores_matrix = alpha * csr_matrix.dot(norm_adj_matrix, scores_matrix) + (1-alpha)*init_matrix
    return scores_matrix


def get_predictions(scores_matrix, test_score_idx, attr_classes):
    """ Get predictions for the current attribute using the scores matrix. """
    predictions = [attr_classes[p] for p in scores_matrix[test_score_idx, :].argmax(axis=1)]
    return predictions


def evaluate(predictions, ground_truth, scores_matrix, indices_test, labeled_profiles):
    acc = accuracy_score(predictions, ground_truth)
    cov = len([idx for idx in indices_test if idx is not None and scores_matrix[idx].sum() > 0])/labeled_profiles.shape[0]
    return acc, cov
    

def get_n_friends(indices, matrix):
    n_friends = []
    for i, idx in enumerate(indices):
        if idx is not None:
            n_friends.append(len(list(matrix.getrow(idx).indices)))
        else:
            n_friends.append(0)
    return n_friends


if __name__=='__main__':
    parser = argparse.ArgumentParser(
        description="Perform label propagation posterior to matching names of twitter users to infer demographic attributes for users that could not be matched using their names."
    )

    parser.add_argument("--name_matched_profiles", type=str, help="Path to the user profiles with name matching scores, i.e. the output of the name matching script.")
    parser.add_argument("--friendship_data", type=str, help="Path to the friendship data.")
    # parser.add_argument("--out_path", type=str, help="Path where user profiles with their name-based scores for each attribute will be saved.")
    parser.add_argument("--annotations_path", type=str, help="Path to labeled profiles.", default=None)
    parser.add_argument("--symmetrize", type=bool, help="Whether the followership matrix should be made symmetric or not", default=True)
    # parser.add_argument("--evaluate", type=str, help="Whether or not to evaluate name matching on a labeled test set.", default=False)
    
    args = parser.parse_args()

    ## Files loading
    # Load users with matched names
    name_matched_profiles = pd.read_csv(args.name_matched_profiles)
    # Load mapping from names to demographic attributes and name list
    names_mapping = load_name_mapping(NAMES_TO_ATTRIBUTES_MAP)
    name_list = load_names_list(NAMES_TO_ATTRIBUTES_MAP)

    ## Concatenate the labeled test set with matched names from the user set as those can also be used for label propagation
    if args.annotations_path is not None:
        # Load test set
        labeled_profiles = pd.read_csv(ANNOTATIONS_PATH)
        # Preprocess test set
        labeled_profiles = preprocess_label_df(labeled_profiles)
        labeled_profiles = predict_attrs_from_names(labeled_profiles, names_mapping)
        labeled_profiles = clean_user_ids(labeled_profiles)

        # Get all users with names
        columns_to_keep = ['user_id', 'ethnicity_name_predict', 'gender_name_predict', 'religion_name_predict']
        name_matched_profiles = pd.concat([name_matched_profiles[columns_to_keep], labeled_profiles[columns_to_keep]])
    
    name_matched_profiles['user_id'] = name_matched_profiles['user_id'].astype('int64')
    name_matched_profiles = name_matched_profiles.drop_duplicates('user_id')    
    name_matched_profiles = name_matched_profiles.set_index('user_id')
    print('all_users', name_matched_profiles.shape)
    
    ## Load followership data
    adj_matrix_path = args.path_to_friendship_data + 'adjacency_matrix.npz'
    adj_nodes_path = args.path_to_friendship_data + 'nodes.csv'
    adj_matrix, nodes = load_friendship_data(adj_matrix_path, adj_nodes_path)
    
    user_ids = name_matched_profiles['user_id'].reset_index(drop=True)
    target_adj_matrix, target_nodes = filter_matrix(adj_matrix, nodes, user_ids)

    # Normalize adjacency matrix
    if args.symmetrize:
        target_adj_matrix = target_adj_matrix.transpose() + target_adj_matrix
    norm_adj_matrix = normalize(target_adj_matrix.astype('float64'), norm='l1', axis=1)
        
    print('norm_adj_matrix', norm_adj_matrix.shape)
    print('target_nodes', target_nodes.shape)

    ## Get user ids positions from nodes list
    # Inverse index to user ids
    target_nodes_index_to_id = {row['user_id']:i for i, row in target_nodes.iterrows()}
    
    # Further clean the labeled profiles
    labeled_profiles = labeled_profiles.drop(['name', 'screen_name', 'link'], axis=1)
    if 'org' in labeled_profiles.columns.values:
        labeled_profiles = labeled_profiles[labeled_profiles['org']=='0']
    if 'suspended' in labeled_profiles.columns.values:
        labeled_profiles = labeled_profiles[labeled_profiles['suspended']=='0']
    
    indices_test = []
    for user_id in labeled_profiles['user_id']:
        idx = target_nodes_index_to_id.get(int(user_id), None)
        indices_test.append(idx)
    n_friends = get_n_friends(indices_test, norm_adj_matrix)

    # Load name matching scores to initialize label propagation, as this is a better signal than 
    nm_scores_df = pd.DataFrame(target_nodes['user_id'])
    for attr in ('ethnicity', 'religion', 'gender'):
        attr_scores = pd.read_csv(f'{PREDICTIONS_PATH}/name_matching_scores_{feature}.csv').drop('Unnamed: 0', axis=1)
        attr_scores = attr_scores.rename({k:'{}_name_predict_{}'.format(feature, k) for k in attr_scores.columns if k!='user_id'}, axis=1).drop_duplicates('user_id')
        nm_scores_df = nm_scores_df.merge(attr_scores, on='user_id', how='left').fillna(0)
        
    print('nm_scores_df', nm_scores_df.shape)
        
    # Perform Label Propagation
    alpha = 0.5
    num_generations = 10
    keep_init = True

    for attr in ('ethnicity', 'gender', 'religion'):
        print('Performing label propagation for {}.'.format(attr))
        
        attr_classes = CLASSES[attr]
        init_df = nm_scores_df[['{}_name_predict_{}'.format(attr, e) for e in attr_classes]]
        print('init_df', init_df.shape)
        scores_matrix = pd.DataFrame(init_df).values
        print('scores_matrix', scores_matrix.shape)
        init_matrix = init_df.values
        test_score_idx = [idx for i, idx in enumerate(indices_test) if idx is not None and labeled_profiles.iloc[i][attr] in attr_classes]
        idx_filter_labels = [i for i, idx in enumerate(indices_test) if idx is not None and labeled_profiles.iloc[i][attr] in attr_classes]
        print(len(idx_filter_labels))
        
        predictions = get_predictions(scores_matrix, test_score_idx, attr_classes)
        ground_truth = labeled_profiles.iloc[idx_filter_labels][attr]
        acc, cov = evaluate(predictions, ground_truth, scores_matrix, indices_test, labeled_profiles)
        
        print('Initial accuracy : {}'.format(round(acc,2)))
        print('Initial coverage : {}'.format(round(cov,2)))
        accs = [acc]

        print('norm_adj_matrix', norm_adj_matrix.shape)
        start_time = time.time()
        if keep_init:
            keep_vector = np.array([1 if (row.sum() == 0) else 0 for row in scores_matrix]).reshape(-1, 1)
            print('keep_vector', keep_vector.shape)
            norm_adj_matrix = norm_adj_matrix.multiply(keep_vector)

        for g in range(num_generations):
    #         print('Iteration {}'.format(g))
            scores_matrix = update_scores(scores_matrix, norm_adj_matrix, init_matrix, alpha)
            predictions = get_predictions(scores_matrix, test_score_idx, attr_classes)
            acc, cov = evaluate(predictions, ground_truth, scores_matrix, indices_test, labeled_profiles)
            print('Acc. at generation {} : {}'.format(g+1, round(acc,2)))
            print('Cov. at generation {} : {}'.format(g+1, round(cov,2)))
            accs.append(acc)
    #         print('{} sec elapsed'.format(time.time()-start_time))

        pkl.dump(predictions, open('../data/lab_prop_predictions_{}_{}_20-03-03.npz'.format(attr, alpha), 'wb'))
        nm_scores_df['user_id'].to_csv('lab_prop_user_ids.csv')
        np.save('../predictions/scores_matrix_{}_{}_20-03-03.npz'.format(attr, alpha), scores_matrix)

        results = {}
        accs_thres = []
        coverages = []
        for min_friends in np.arange(0, 500, 10):
            predictions = [attr_classes[p] for p in scores_matrix[[idx for i, idx in enumerate(indices_test) if idx is not None and n_friends[i]<=min_friends+10 and labeled_profiles.iloc[i][attr] in attr_classes], :].argmax(axis=1)]
            acc = accuracy_score(predictions, labeled_profiles.iloc[[i for i,idx in enumerate(indices_test) if idx is not None and n_friends[i]<=min_friends+10 and labeled_profiles.iloc[i][attr] in attr_classes]][attr])
            accs_thres.append(acc)
            coverage = len([f for i, f in enumerate(n_friends) if f>=min_friends and labeled_profiles.iloc[i][attr] in attr_classes])/labeled_profiles[labeled_profiles[attr].isin(attr_classes)].shape[0]

            coverages.append(coverage)
        print(coverages[0])
        plt.style.use('seaborn-darkgrid')
        fig, ax = plt.subplots()
        results['threshold'] = np.arange(0, 500, 10)
        results['accuracy'] = accs_thres
        results['cdf'] = [1-c for c in coverages]
        results['majority_baseline'] = NM_RESULTS[attr]['maj']
        results['name_matching_cov'] = NM_RESULTS[attr]['cov']
        results['name_matching_acc'] = NM_RESULTS[attr]['acc']
        pkl.dump(results, open('results_{}.pkl'.format(attr), 'wb'))
        ax.plot(np.arange(0, 500, 10), accs_thres, color='orange', marker='o')
        ax.set_ylabel('Accuracy on {}'.format(attr), color='orange')
        ax.set_xlabel('Threshold on min # friends')
        ax.set_ylim(0,1)
        ax2 = ax.twinx()
        ax2.plot(np.arange(0, 500, 10), [1-c for c in coverages], color='blue', marker='o')
        ax2.axhline(y=NM_RESULTS[attr]['acc'], color='orange', linestyle='--', label='Name Matching Accuracy')
        ax2.axhline(y=NM_RESULTS[attr]['cov'], color='blue', linestyle='--', label='Name Matching Coverage')
        ax2.axhline(y=NM_RESULTS[attr]['maj'], color='black', linestyle='--', label='Majority Baseline')
        ax2.set_ylabel('Cumulative distribution function kept users', color='blue')
        ax2.set_ylim(0,1)
        plt.legend()
        plt.show()