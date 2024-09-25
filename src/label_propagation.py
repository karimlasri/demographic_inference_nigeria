import pandas as pd
from loading_utils import (
     load_name_mapping, 
     load_user_ids, 
     load_profiles_for_lp, 
     load_followership_for_lp, 
     load_nm_scores_for_lp,
     load_nm_results
     )
from format_data import preprocess_label_df, keep_valid_ids, filter_test_set
from scipy.sparse import csr_matrix
from sklearn.metrics import accuracy_score
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import json
import argparse
from config import (
     NAMES_TO_ATTRIBUTES_MAP, 
     SCORES_PATH, 
     CLASSES, 
     EVAL_BY_CONNECTIONS_PARAMS, 
     LABEL_PROPAGATION_PARAMS
)


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
    nm_results = load_nm_results()
    if nm_results:
        ax2.axhline(y=nm_results[attr]['acc'], color='orange', linestyle='--', label='Name Matching Accuracy')
        ax2.axhline(y=nm_results[attr]['cov'], color='blue', linestyle='--', label='Name Matching Coverage')
        ax2.axhline(y=nm_results[attr]['maj'], color='black', linestyle='--', label='Majority Baseline')
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

