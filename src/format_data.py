import re
from loading_utils import *
from match_names import match_names, predict_attrs_from_name
from sklearn.model_selection import train_test_split
import pandas as pd
from scipy.sparse import coo_array, load_npz, save_npz
import pickle as pkl
import numpy as np
import time
import json


def keep_valid_ids(user_profiles):
    """ Removes malformated user ids."""
    user_profiles['valid_id'] = user_profiles['user_id'].apply(lambda x:'+' not in x)
    user_profiles = user_profiles[user_profiles['valid_id']]
    user_profiles['user_id'] = user_profiles['user_id'].astype('int64')
    return user_profiles


def filter_test_set(labeled_profiles, columns_to_keep):
    """ Further clean the labeled profiles to keep only active users. """
    if 'org' in labeled_profiles.columns.values:
        labeled_profiles = labeled_profiles[labeled_profiles['org']=='0']
    if 'suspended' in labeled_profiles.columns.values:
        labeled_profiles = labeled_profiles[labeled_profiles['suspended']=='0']
    labeled_profiles = labeled_profiles[columns_to_keep]
    return labeled_profiles


def modify_gender(x):
    """ We keep only the gender's first letter to represent the class. """
    if x:
        if x!='unisex':
            return x[0]


def preprocess_users_matched_names_df(users_with_names_df):
    users_with_names_df = users_with_names_df.drop_duplicates('user_id')
    return users_with_names_df


def preprocess_unigrams_df(unigrams_df):
    unigrams_df = unigrams_df.set_index(['user_id'])
    return unigrams_df


def join_dfs(unigrams_df, users_with_names_df):
    unigrams_df = unigrams_df.set_index(['user_id'])
    users_with_names_df = users_with_names_df.set_index(['user_id'])
    unig_with_names = unigrams_df.join(users_with_names_df)
    unig_with_names = unig_with_names.reset_index()
    return unig_with_names


def sample_fraction(unig_with_names, frac):
    print('Sampling {}% of users...'.format(round(frac*100)))
    unique_ids = unig_with_names['user_id'].unique()
    ids_to_keep = unig_with_names['user_id'].sample(int(len(unique_ids) * frac))
    unig_with_names = unig_with_names[unig_with_names['user_id'].isin(ids_to_keep)]
    print('Done.')
    return unig_with_names


def filter_by_threshold(count_df_train, idfs, min_n_users):
    # TODO : Change this to keep also total count before we pivot
    print('Filtering count dataframe by threshold of minimum {} users...'.format(min_n_users))
    kept_ids = idfs[idfs['n_users'] > min_n_users].index.tolist()
    count_df_train = count_df_train[count_df_train['token'].isin(kept_ids)]
    print('Done.')
    return count_df_train


def get_name_based_labels(filtered_count_df_train):
    name_labels_train = filtered_count_df_train[
        ['user_id', 'ethnicity_name_predict', 'religion_name_predict', 'gender_name_predict']]
    name_labels_train = name_labels_train.drop_duplicates('user_id')
    return name_labels_train


def get_X(count_df, idfs, feature_format='binary'):
    print('Extracting {} features dataframe...'.format(feature_format))
    assert feature_format in ['binary', 'count', 'tf', 'tf_idf']
    if feature_format.startswith('tf'): # tf is term_frequency
        count_df['tf'] = count_df.apply(lambda row: row['count']/row['total'], axis=1)
    if feature_format == 'tf_idf':
        n_users = len(count_df['user_id'].unique())
        count_df = count_df[count_df['token'].isin(idfs.index)]
        count_df['idf'] = count_df.apply(lambda row: np.log(n_users/idfs.loc[row['token']]['n_users']), axis=1)
        count_df['tf_idf'] = count_df.apply(lambda row: row['tf']*row['idf'], axis=1)
    X_train = count_df.pivot(index='user_id', columns='token', values=feature_format).fillna(0)
    print('Done')
    return X_train


def get_idfs(count_df):
    print('Getting inverse-document frequencies...')
    idfs = count_df.groupby('token').sum('binary_count')  # Number of users that used a given token
    idfs = idfs.rename({'binary_count': 'n_users'}, axis=1)
    print('Done.')
    return idfs


def print_elapsed_time(start_time):
    print('{} sec elapsed.'.format(round(time.time()-start_time)))


def get_Xy_train_name(feature_df, users_with_names_df, feature_format='binary', frac=1.0, save=True):
    start_time = time.time()

    # Join unigrams (train_df) with name-matched users
    unig_with_names = join_dfs(unigrams_df, users_with_names_df)
    print_elapsed_time(start_time)
    unig_with_names['binary_count'] = unig_with_names['count'].apply(lambda x: 1 if x > 0 else 0)  # Get binary count
    count_df_train = unig_with_names[
        ['user_id', 'token', 'count', 'total', 'binary_count', 'ethnicity_name_predict', 'gender_name_predict',
         'religion_name_predict']]

    # Keep only a portion of users to reduce dataset size
    if frac != 1.0:
        count_df_train = sample_fraction(count_df_train, frac)
    print_elapsed_time(start_time)
    idfs = get_idfs(count_df_train)
    print_elapsed_time(start_time)

    # Filter out infrequent tokens (used by too few users)
    # TODO : Use other criteria to filter tokens here ?
    min_n_users = 100
    count_df_train = filter_by_threshold(count_df_train, idfs, min_n_users) # Also gets binary count
    print_elapsed_time(start_time)

    # Get final X_train matrix, labels matrix, and feature_to_id dictionary
    name_labels_train = get_name_based_labels(count_df_train)
    print_elapsed_time(start_time)

    X_train = get_X(count_df_train, idfs, feature_format)
    print_elapsed_time(start_time)

    features_list = list(X_train.columns)
    X_train.columns = range(len(features_list))
    feature_to_id = {f: i for i, f in enumerate(features_list)}

    if save:
        user_ids = X_train.index.tolist()
        pkl.dump(user_ids, open('data/formatted_data/user_ids_train_{}_{}.pkl'.format(feature_format, frac), 'wb'))

        features_np = X_train.to_numpy() #drop('user_id', axis=1).to_numpy()
        sparse_features = coo_array(features_np)
        save_npz('sparse_X_train_{}_{}.npz'.format(feature_format, frac), sparse_features)

        json.dump(feature_to_id, open('data/formatted_data/feature_to_id_{}_{}.json'.format(feature_format, frac), 'w'))

        name_labels_train.to_csv('data/formatted_data/name_labels_train_{}_{}.csv'.format(feature_format, frac))

    return X_train, name_labels_train, feature_to_id, idfs


def get_Xy_test_name(label_df, features_df_train, feature_to_id, idfs, feature_format='binary', save=True):
    test_df = load_raw_unigrams(label_df)
    test_df = test_df.reset_index(drop=True)

    # binary feature indicating whether we have tweets for this user
    test_df['binary_count'] = test_df['count'].apply(lambda x: 1 if x > 0 else 0)
    count_df_test = test_df[['user_id', 'token', 'count', 'binary_count', 'total']]

    features_df_test = get_X(count_df_test, idfs, feature_format)

    label_df = label_df[['user_id', 'ethnicity', 'religion', 'gender']]
    label_df = label_df.set_index(['user_id'])

    X_test = features_df_test.rename(feature_to_id, axis=1)

    # Drop features that are not in train set
    cols_to_drop = [c for c in X_test.columns.values if c not in feature_to_id.values()]
    X_test = X_test.drop(cols_to_drop, axis=1)

    label_df = X_test.join(label_df)[['ethnicity', 'religion', 'gender']]
    # Get features df test by adding missing columns (words not appearing in test set)
    whole_df = pd.concat([features_df_train, X_test], axis=0)
    X_test = whole_df[-len(X_test):].fillna(0)

    if save:
        user_ids_test = X_test.index.tolist()
        pkl.dump(user_ids_test, open('data/formatted_data/user_ids_test_{}.pkl'.format(feature_format), 'wb'))

        features_np = X_test.to_numpy() # .drop('user_id', axis=1)
        sparse_features = coo_array(features_np)
        save_npz('data/formatted_data/sparse_X_test_{}.npz'.format(feature_format), sparse_features)

        label_df.to_csv('data/formatted_data/label_df_test_{}.csv'.format(feature_format))
    return X_test, label_df


def load_preprocessed_train_data(feature_type='unigram', feature_format='binary', frac=1.0):
    print('Loading previously preprocessed train data...')
    # Load X_train
    sparse_features = load_npz('data/formatted_data/sparse_X_train_{}_{}_{}.npz'.format(feature_type, feature_format, frac))
    X_train = pd.DataFrame(sparse_features.toarray())
    user_ids = pkl.load(open('data/formatted_data/user_ids_train_{}_{}_{}.pkl'.format(feature_type, feature_format, frac), 'rb'))
    X_train['user_id'] = user_ids
    # Load feature to id
    feature_to_id = json.load(open('data/formatted_data/feature_to_id_{}_{}_{}.json'.format(feature_type, feature_format, frac), 'r'))
    # Load name_labels_train
    name_labels_train = pd.read_csv('data/formatted_data/name_labels_train_{}_{}_{}.csv'.format(feature_type, feature_format, frac))
    print('Done.')
    return X_train, name_labels_train, feature_to_id


def load_preprocessed_test_data(feature_type='unigram', feature_format='binary', frac=1.0):
    print('Loading previously preprocessed train data...')
    # Load X_test
    sparse_features = load_npz('data/formatted_data/sparse_X_test_{}_{}_{}.npz'.format(feature_type, feature_format, frac))
    X_test= pd.DataFrame(sparse_features.toarray())
    user_ids_test = pkl.load(open('data/formatted_data/user_ids_test_{}_{}_{}.pkl'.format(feature_type, feature_format, frac), 'rb'))
    X_test['user_id'] = user_ids_test
    # Load label df test
    label_df = pd.read_csv('data/formatted_data/label_df_test_{}_{}_{}.csv'.format(feature_type, feature_format, frac))
    print('Done.')
    return X_test, label_df


def load_Xy_on_name_matched(label_df, feature_type='unigram', feature_format='binary', label='ethnicity', frac=0.1, load=True, save=False):
    print('Loading users with matched names...')
    users_with_names_df = load_users_with_matched_names()
    users_with_names_df = users_with_names_df.drop_duplicates('user_id')
    print('Done.')

    if load:
        X_train, name_labels_train, feature_to_id = load_preprocessed_train_data(feature_type, feature_format, frac)
    else:
        if feature_type=='unigram':
            feature_df = load_raw_unigrams(users_with_names_df)
        else:
            feature_df = load_descriptions(users_with_names_df)
        X_train, name_labels_train, feature_to_id, n_users_per_token = get_Xy_train_name(feature_df, users_with_names_df, feature_format, frac=frac, save=save)
    y_train = name_labels_train['{}_name_predict'.format(label)]

    if load:
        X_test, label_df = load_preprocessed_test_data(feature_type, feature_format, frac)
    else:
        X_test, label_df = get_Xy_test_name(label_df, X_train, feature_to_id, n_users_per_token, feature_format, save=save)
    y_test = label_df[label]

    return X_train, y_train, X_test, y_test


def load_Xy_on_labeled(label_df, name_df):
    df = load_raw_unigrams(label_df)
    df = df.reset_index(drop=True)
    # Total count of tweets per user
    total_count_user_df = df[['user_id', 'total']].drop_duplicates().reset_index(drop=True)
    total_count_user_df.head()

    # Create features_df
    # binary feature indicating whether we have tweets for this user
    df['binary_count'] = df['count'].apply(lambda x: 1 if x > 0 else 0)
    count_df = df[['user_id', 'token', 'binary_count']]
    # Matrice de comptage de chaque token pour chaque utilisateur
    features_df = count_df.pivot(index='user_id', columns='token', values='binary_count').fillna(0)

    features_list = list(features_df.columns)
    features_df.columns = range(len(features_list))
    label_df = label_df[
        ['user_id', 'ethnicity', 'religion', 'gender', 'ethnicity_name_predict', 'religion_name_predict',
         'religion_name_predict']]
    label_df = label_df.set_index(['user_id'])
    ethn_df = features_df.join(label_df)
    ethn_df['ethnicity'].value_counts(dropna=False)
    ethn_df = df.loc[df['ethnicity'].isin(['igbo', 'yoruba', 'hausa'])]
    all_voc = load_voc()
    voc_ids = [i for i, e in enumerate(features_list) if e in all_voc]

    X = ethn_df[list(range(len(features_list))) + ['ethnicity_name_predict', 'religion_name_predict',
                                                   'religion_name_predict']]
    y = ethn_df['ethnicity']
    X = X[voc_ids + ['ethnicity_name_predict', 'religion_name_predict', 'religion_name_predict']]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    X_name_train = X_train[['ethnicity_name_predict', 'religion_name_predict', 'religion_name_predict']]
    X_name_test = X_test[['ethnicity_name_predict', 'religion_name_predict', 'religion_name_predict']]
    X_train = X_train[voc_ids]
    X_test = X_test[voc_ids]

    return X_train, y_train, X_test, y_test, X_name_train, X_name_test


def filter_matrix(matrix, nodes, user_ids):
    subset = nodes[nodes['user_id'].isin(user_ids)]
    return matrix[subset['user_index']][:, subset['user_index']], subset


if __name__=='__main__':
    train_on_matched = True
    name_df = load_names_df()
    label_df = load_labelled_df()
    name_df = preprocess_name_df(name_df)
    names_list = load_names_list()
    label_df = match_names(label_df, names_list)
    label_df = predict_attrs_from_name(label_df, name_df)
    if train_on_matched:
        X_train, y_train, X_test, y_test = load_Xy_on_name_matched(label_df, frac=0.1)
    else:
        X_train, y_train, X_test, y_test, X_name_train, X_name_test = load_Xy_on_labeled(name_df, label_df)
