import pandas as pd
from pathlib import Path
from match_names import predict_all_from_names
from loading_utils import load_labelled_df, load_name_mapping, load_user_ids, load_friendship_data
from format_data import preprocess_label_df, keep_valid_ids, filter_test_set
from scipy.sparse import load_npz
from sklearn.preprocessing import normalize

from scipy.sparse import csr_matrix
from sklearn.metrics import accuracy_score
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import json
import argparse


NAMES_TO_ATTRIBUTES_MAP = 'data/names_attribute_map.csv'
SCORES_PATH = 'predictions/'
CLASSES = {
    'ethnicity':('hausa', 'igbo', 'yoruba'),
    'gender':('m','f'),
    'religion':('christian', 'muslim')
         }

NM_RESULTS_PATH = 'data/name_matching_eval.json'
if os.path.exists(NM_RESULTS_PATH):
    with open(NM_RESULTS_PATH) as eval_file:
        NM_RESULTS = json.load(eval_file)
        

EVAL_BY_CONNECTIONS_PARAMS = {
                'min_connections'=0
                'max_connections'=500
                'bin_size'=10
            }

LABEL_PROPAGATION_PARAMS = {
    'alpha' = 0.5
    'num_iterations' = 10
    'keep_init' = True
}


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


def get_predictions(scores_matrix, matrix_indices, attr_classes):
    """ Get predictions for the current attribute using the scores matrix. """
    predictions = [attr_classes[p] for p in scores_matrix[matrix_indices, :].argmax(axis=1)]
    return predictions


def get_test_matrix_indices(labeled_profiles, target_nodes):
    """ Get positions of test users in the adjacency matrix and nodes list. """
    # Inverse index from user ids to position in the matrix
    inverse_index = {row['user_id']:matrix_idx for matrix_idx, row in target_nodes.iterrows()}
    test_matrix_indices = []
    for user_id in labeled_profiles['user_id']:
        matrix_idx = inverse_index.get(int(user_id), None)
        test_matrix_indices.append(matrix_idx)
    return test_matrix_indices


def get_test_indices_for_attr(test_matrix_indices, labeled_profiles, attr, attr_classes):
    """ Get test indices in adjacency matrix and in labeled profiles set for current attribute. """
    attr_test_matrix_indices = [idx for i, idx in enumerate(test_matrix_indices) if idx is not None and labeled_profiles.iloc[i][attr] in attr_classes]
    attr_test_profile_indices = [i for i, idx in enumerate(test_matrix_indices) if idx is not None and labeled_profiles.iloc[i][attr] in attr_classes]
    return attr_test_matrix_indices, attr_test_profile_indices


def evaluate(predictions, ground_truth, scores_matrix, matrix_indices, labeled_profiles):
    """ Evaluate predictions against annotations, and compute coverage of the label propagation. """
    acc = accuracy_score(predictions, ground_truth)
    cov = len([idx for idx in matrix_indices if scores_matrix[idx].sum() > 0])/labeled_profiles.shape[0]
    return acc, cov
    

def get_n_connections(users_matrix_indices, adj_matrix):
    """ Get the number of connections for users given their indices in the adjacency matrix. """
    n_connections = []
    for i, idx in enumerate(users_matrix_indices):
        if idx is not None:
            n_connections.append(len(list(adj_matrix.getrow(idx).indices)))
        else:
            n_connections.append(0)
    return n_connections
    

def load_profiles_for_lp(matched_profiles_path, names_mapping_path, annotations_path):
    """ Load and preprocess user profiles along with matched names, and the labeled set. """
    # Load users with matched names
    name_matched_profiles = pd.read_csv(matched_profiles_path)
    # Load mapping from names to demographic attributes and name list
    names_mapping = load_name_mapping(names_mapping_path)

    # Load test set
    labeled_profiles = pd.read_csv(annotations_path)
    # Preprocess test set
    labeled_profiles = preprocess_label_df(labeled_profiles)
    labeled_profiles = predict_attrs_from_names(labeled_profiles, names_mapping)
    labeled_profiles = keep_valid_ids(labeled_profiles)

    # Concatenate the labeled test set with matched names from the user set as those can also be used for label propagation
    columns_to_keep = ['user_id'] + [f'{attr}_name_predict' for attr in CLASSES]
    name_matched_profiles = pd.concat([name_matched_profiles[columns_to_keep], labeled_profiles[columns_to_keep]])
    labeled_profiles = filter_test_set(labeled_profiles, columns_to_keep)
    
    # Further preprocess profiles
    name_matched_profiles['user_id'] = name_matched_profiles['user_id'].astype('int64')
    name_matched_profiles = name_matched_profiles.drop_duplicates('user_id')    
    name_matched_profiles = name_matched_profiles.set_index('user_id')
    return name_matched_profiles, labeled_profiles


def load_followership_for_lp(path_to_followership_data, name_matched_profiles, symmetrize=True):
    """ Load and prepare followership data for label propagation. """
    ## Load followership data
    adj_matrix_path = path_to_followership_data + 'adjacency_matrix.npz'
    adj_nodes_path = path_to_followership_data + 'nodes.csv'
    adj_matrix, nodes = load_friendship_data(adj_matrix_path, adj_nodes_path)
    # Filter data to only keep user ids present in the target profiles
    user_ids = name_matched_profiles['user_id'].reset_index(drop=True)
    target_adj_matrix, target_nodes = filter_matrix(adj_matrix, nodes, user_ids)
    # Normalize adjacency matrix, and symmetrize if required
    if symmetrize:
        target_adj_matrix = target_adj_matrix.transpose() + target_adj_matrix
    norm_adj_matrix = normalize(target_adj_matrix.astype('float64'), norm='l1', axis=1)
    
    return norm_adj_matrix, target_nodes


def load_nm_scores_for_lp(target_nodes):
    """ Load name matching scores to initialize label propagation, as this is a better signal than binary predictions. """
    nm_scores_df = pd.DataFrame(target_nodes['user_id'])
    for attr in CLASSES:
        attr_scores = pd.read_csv(f'{SCORES_PATH}/name_matching_scores_{feature}.csv').drop('Unnamed: 0', axis=1)
        attr_scores = attr_scores.rename({k:'{}_name_predict_{}'.format(feature, k) for k in attr_scores.columns if k!='user_id'}, axis=1).drop_duplicates('user_id')
        nm_scores_df = nm_scores_df.merge(attr_scores, on='user_id', how='left').fillna(0)
    return nm_scores_df


def get_eval_by_connections(labeled_profiles, attr, attr_classes, test_matrix_indices, scores_matrix, eval_by_connections_params):
    """ Get evaluation (accuracy and coverage) by number of connections for users in the test set. """
    accuracies = []
    coverages = []
    min_connections = eval_by_connections_params['min_connections']
    max_connections = eval_by_connections_params['max_connections']
    bin_size = eval_by_connections_params['bin_size']
    for min_connections in np.arange(0, 500, 10):
        predictions = [attr_classes[p] for p in scores_matrix[[idx for i, idx in enumerate(test_matrix_indices) if idx is not None and n_connections[i]<=min_connections+10 and labeled_profiles.iloc[i][attr] in attr_classes], :].argmax(axis=1)]
        acc = accuracy_score(predictions, labeled_profiles.iloc[[i for i,idx in enumerate(test_matrix_indices) if idx is not None and n_connections[i]<=min_connections+10 and labeled_profiles.iloc[i][attr] in attr_classes]][attr])
        accuracies.append(acc)
        coverage = len([f for i, f in enumerate(n_connections) if f>=min_connections and labeled_profiles.iloc[i][attr] in attr_classes])/labeled_profiles[labeled_profiles[attr].isin(attr_classes)].shape[0]
        coverages.append(coverage)
    return accuracies, coverages


def plot_by_connections_threshold(accuracies, coverages, attr, eval_by_connections_params):
    """ Plot the accuracy and coverage of label propagation against the number of connections for users in the test set. """
    min_connections = eval_by_connections_params['min_connections']
    max_connections = eval_by_connections_params['max_connections']
    # Produce plot
    bin_size = eval_by_connections_params['bin_size']
    plt.style.use('seaborn-darkgrid')
    fig, ax = plt.subplots()
    ax.plot(np.arange(min_connections, max_connections, bin_size), accuracies, color='orange', marker='o')
    ax.set_ylabel('Accuracy on {}'.format(attr), color='orange')
    ax.set_xlabel('Threshold on min # friends')
    ax.set_ylim(0,1)
    ax2 = ax.twinx()
    ax2.plot(np.arange(min_connections, max_connections, bin_size), [1-c for c in coverages], color='blue', marker='o')
    ax2.axhline(y=NM_RESULTS[attr]['acc'], color='orange', linestyle='--', label='Name Matching Accuracy')
    ax2.axhline(y=NM_RESULTS[attr]['cov'], color='blue', linestyle='--', label='Name Matching Coverage')
    ax2.axhline(y=NM_RESULTS[attr]['maj'], color='black', linestyle='--', label='Majority Baseline')
    ax2.set_ylabel('Cumulative distribution function kept users', color='blue')
    ax2.set_ylim(0,1)
    plt.legend()
    plt.savefig(f'plots/evaluation_by_connections_{attr}.png')
    plt.show()


def perform_lp_iteration(scores_matrix, norm_adj_matrix, init_matrix, alpha, attr_test_matrix_indices, attr_classes, iteration, print_bool=True):
    """ Perform one iteration of label propagation and evaluate on test set. """
    predictions = get_predictions(scores_matrix, attr_test_matrix_indices, attr_classes)
    acc, cov = evaluate(predictions, ground_truth, scores_matrix, attr_test_matrix_indices, labeled_profiles)
    if print_bool:
        print('Accucary at iteration {} : {}'.format(iteration, round(acc,2)))
        print('Coverage at iteration {} : {}'.format(iteration, round(cov,2)))
        print('Performing iteration {} of label propagation...'.format(iteration+1))
    scores_matrix = update_scores(scores_matrix, norm_adj_matrix, init_matrix, alpha)
    return scores_matrix, predictions


def initialize_matrices_for_lp(nm_scores_df, attr, attr_classes, keep_init, norm_adj_matrix):
    """ Initialize scores and adjacency matrices prior to performing label propagation for current attribute. """
    init_df = nm_scores_df[['{}_name_predict_{}'.format(attr, attr_class) for attr_class in attr_classes]]
    scores_matrix = pd.DataFrame(init_df).values
    init_matrix = init_df.values
    if keep_init:
        # <keep_init> encodes whether initial scores for matched users should remain unchanged 
        # (i.e. label propagation will only update other users)
        keep_vector = np.array([1 if (row.sum() == 0) else 0 for row in scores_matrix]).reshape(-1, 1)
        norm_adj_matrix = norm_adj_matrix.multiply(keep_vector)
    return init_matrix, scores_matrix, norm_adj_matrix
    

if __name__=='__main__':
    parser = argparse.ArgumentParser(
        description="Perform label propagation posterior to matching names of twitter users to infer demographic attributes for users that could not be matched using their names."
    )

    parser.add_argument("--name_matched_profiles", type=str, help="Path to the user profiles with name matching scores, i.e. the output of the name matching script.")
    parser.add_argument("--followership_path", type=str, help="Path to the followership data.")
    # parser.add_argument("--out_path", type=str, help="Path where user profiles with their name-based scores for each attribute will be saved.")
    parser.add_argument("--annotations_path", type=str, help="Path to labeled profiles.")
    parser.add_argument("--symmetrize", type=bool, help="Whether the followership matrix should be made symmetric or not", default=True)
    parser.add_argument("--plot_by_connections", type=bool, help="Whether the evaluation scores should be plotted against the number of connections.", default=True)
    
    args = parser.parse_args()

    ## Load params for Label Propagation
    alpha = LABEL_PROPAGATION_PARAMS['alpha']
    num_iterations = LABEL_PROPAGATION_PARAMS['num_iterations']
    keep_init = LABEL_PROPAGATION_PARAMS['keep_init']

    ## Files loading
    name_matched_profiles, labeled_profiles = load_profiles_for_lp(args.name_matched_profiles, NAMES_TO_ATTRIBUTES_MAP, args.annotations_path)
    
    ## Load followership data
    norm_adj_matrix, target_nodes = load_followership_for_lp(args.followership_path, name_matched_profiles, args.symmetrize)

    ## Get test user ids positions in nodes list
    test_matrix_indices = get_test_matrix_indices(labeled_profiles, target_nodes)

    if args.plot_by_connections:
        if not os.path.exists('plots/'):
            os.mkdir('plots/')
        n_connections = get_n_connections(test_matrix_indices, norm_adj_matrix)

    ## Load name matching scores to initialize label propagation, as this is a better signal than binary predictions
    nm_scores_df = load_nm_scores_for_lp(target_nodes)
    # Saving user ids for further processing of results
    nm_scores_df['user_id'].to_csv('{}/lab_prop_user_ids.csv'.format(SCORES_PATH))

    ## Perform Label Propagation
    for attr, attr_classes in CLASSES.items():
        print('Performing label propagation for {}.'.format(attr))
        
        # Initialize matrices for current demographic attribute
        init_matrix, scores_matrix, norm_adj_matrix = initialize_matrices_for_lp(nm_scores_df, attr, attr_classes, keep_init, norm_adj_matrix)
        
        # Get indices of test users for current attribute and their ground truth
        attr_test_matrix_indices, attr_test_profile_indices = get_test_indices_for_attr(test_matrix_indices, labeled_profiles, attr, attr_classes)
        ground_truth = labeled_profiles.iloc[attr_test_profile_indices][attr]
        
        # Perform label propagation
        for iteration in range(num_iterations):
            scores_matrix, predictions = perform_lp_iteration(scores_matrix, norm_adj_matrix, init_matrix, alpha, attr_test_matrix_indices, attr_classes, iteration)
        
        # Save results
        pkl.dump(predictions, open('{}/lab_prop_predictions_{}_{}.npz'.format(SCORES_PATH, attr, alpha), 'wb'))
        np.save('{}/lab_prop_scores_matrix_{}_{}.npz'.format(SCORES_PATH, attr, alpha), scores_matrix)

        if args.plot_by_connections:
            accuracies, coverages = get_eval_by_connections(labeled_profiles, attr, attr_classes, test_matrix_indices, scores_matrix, EVAL_BY_CONNECTIONS_PARAMS)
            plot_by_connections_threshold(accuracies, coverages, attr, EVAL_BY_CONNECTIONS_PARAMS)

