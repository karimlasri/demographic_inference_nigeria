import pandas as pd
import os
from pathlib import Path
import pickle as pkl
from scipy.sparse import load_npz


PATH_UNIGRAMS = '/scratch/spf248/twitter_social_cohesion/data/demographic_cls/unigrams/NG/tweet_text/' #'/scratch/spf248/twitter_social_cohesion/data/deprecated/unigrams/' # Path to tweets
PATH_DESC = '/scratch/spf248/twitter_social_cohesion/data/demographic_cls/unigrams/NG/user_description/' #'/scratch/spf248/twitter_social_cohesion/data/deprecated/unigrams/' # Path to tweets
PATH_LABELS = 'data/' # Path to labeled user data (ethnicity, religion and gender, along with names)
PATH_MATCHED_USERS = '/scratch/spf248/Karim/twitter_social_cohesion/data/matched_user_names'


def load_labelled_df(path_labels=os.path.join(PATH_LABELS, 'annotations_demographics.csv')):
    """ Loads the dataframe which includes users labelled with their ethnicity, gender and religion.
    This dataframe has 1000 entries so far. """
    label_df = pd.read_csv(path_labels)
    for column in ['ethnicity', 'religion', 'gender']:
        label_df[column] = label_df[column].str.lower()
    label_df['user_id'] = label_df['user_id'].astype(str)
    return label_df


def load_name_mapping(names_mapping_path='data/names_attribute_map.csv'):
    """ Loads the dataframe which includes names labelled with their ethnicity, gender and religion.
    This dataframe has 2813 rows so far. """
    names_mapping = pd.read_csv(names_mapping_path)
    names_mapping['gender'] = names_mapping['gender'].apply(modify_gender)
    names_mapping = names_mapping.drop_duplicates(subset=['name'])
    names_mapping = names_mapping.set_index(['name'])
    return names_mapping


def load_names_list(names_map_path='data/names_attributes_map.csv'):
    name_df = load_name_mapping(names_map_path=names_map_path)
    names_list = name_df.index.tolist()
    names_list.sort(key=len, reverse=True)
    names_list = [name for name in names_list if len(name) > 2]
    return names_list


def load_name_gender_prop(gender_proportions_path='data/names_gender_proportions.csv'):
    """ Loads a dataframe where scraped Nigerian names are assigned gender proportions in the population. """
    gender_proportions = pd.read_csv(gender_proportions_path)
    gender_proportions['Forename'] = gender_proportions['Forename'].str.lower()
    gender_proportions = gender_proportions.set_index('Forename').drop('Unnamed: 0', axis=1)
    return gender_proportions


def load_users_with_matched_names():
    """ Loads a dataframe of users matched with their names using the name list """
    df_list = list()
    for path in Path(PATH_MATCHED_USERS).glob('*.parquet'):
        df = pd.read_parquet(path)
        df_list.append(df)
    users_with_names_df = pd.concat(df_list)
    return users_with_names_df


def load_raw_unigrams(users_filter_df=None):
    """ Loads raw tweet unigrams as dataframe. It contains four columns :
    - user_id
    - token
    - count of that token for that user_id
    - total count of tokens for that user_id
    Optionally, unigrams are filtered on a set of users e.g. :
    - users who have been assigned a name (for training on matched names)
    - users who are in our labeled dataset (for training on labeled data only) """
    print('Loading raw unigrams...')
    df_list = list()
    for path in Path(PATH_UNIGRAMS).glob('*.parquet'):
        df = pd.read_parquet(path)
        if users_filter_df is not None: # Filters unigrams to users whose name has been matched
            df = df.loc[df['user_id'].isin(users_filter_df['user_id'].tolist())]
        df_list.append(df)
    unigrams_df = pd.concat(df_list)
    print('Done.')
    return unigrams_df


def load_descriptions(users_filter_df=None):
    """ Loads raw tweet unigrams as dataframe. It contains four columns :
    - user_id
    - token
    - count of that token for that user_id
    - total count of tokens for that user_id
    Optionally, unigrams are filtered on a set of users e.g. :
    - users who have been assigned a name (for training on matched names)
    - users who are in our labeled dataset (for training on labeled data only) """
    print('Loading raw unigrams...')
    df_list = list()
    for path in Path(PATH_DESC).glob('*.parquet'):
        df = pd.read_parquet(path)
        if users_filter_df is not None: # Filters unigrams to users whose name has been matched
            df = df.loc[df['user_id'].isin(users_filter_df['user_id'].tolist())]
        df_list.append(df)
    desc_df = pd.concat(df_list)
    print('Done.')
    return desc_df


def load_voc():
    languages = ['igbo', 'hausa', 'yoruba']
    voc_dfs = {}
    all_voc = []
    for language in languages:
        voc_df = pd.read_csv('data/vocab/{}_vocab.csv'.format(language))
        voc_dfs[language] = voc_df
        all_voc += voc_df[language.capitalize()].to_list()
    return all_voc


def load_stopwords():
    # From https://github.com/stopwords-iso/stopwords-en/blob/master/stopwords-en.txt
    with open('stopwords.txt', 'rt') as stop_fl:
        stopwords = [w.strip() for w in stop_fl.readlines()]
    return stopwords


def load_user_ids():
    unigram_users = pkl.load(open('data/all_unigram_user_ids.pkl', 'rb'))
    matched_names_users = pkl.load(open('data/all_name_matching_user_ids.pkl', 'rb'))
    return unigram_users, matched_names_users


def load_friendship_data():
    friendship_matrix = load_npz('data/friendship_network/adjacency_attention_links_with_outnodes.npz')
    nodes = pd.read_csv('data/friendship_network/nodes.csv').reset_index().rename(columns={'index':'user_index', '0':'user_id'})
    return friendship_matrix, nodes
    