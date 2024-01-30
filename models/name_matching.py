import re
import pandas as pd
import ast
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from unidecode import unidecode
from pathlib import Path
import os
import sys

NAME_LIST_PATH = '../data/labeled_name_list.csv'
TEST_USERS_PATH = '../data/all_labels_final.csv'
GENDER_SCRAPED_PATH = '../data/gender_scraped.csv' # TODO : Document this
# PROFILES_PATH = 'twitter_social_cohesion/data/profiles/NG' # PATH TO FOLDER CONTAINING LIST OF PARQUET FILES WITH USER PROFILES
OUTPUT_PATH = '../predictions'
LABELS = {
    'ethnicity':['yoruba', 'hausa', 'igbo'],
    'gender':['m', 'f'],
    'religion':['christian', 'muslim']
         }



def modify_gender(x):
    if x == 'male':
        return 'm'
    elif x == 'female':
        return 'f'
    else:
        return x


def load_name_df(path):
    name_df = pd.read_csv(path)
    name_df['gender'] = name_df['gender'].apply(modify_gender)
    # name_df['gender'].value_counts(dropna=False)
    name_df = name_df.drop_duplicates(subset=['name'])
    name_df = name_df.set_index(['name'])
    return name_df
    
    
def match_name(x, name_list):
    regex = re.compile(fr"\b(?:{'|'.join(name_list)})",re.IGNORECASE)
    match_list = [match.group(0) for match in regex.finditer(x)]
    if len(match_list) > 0:
        return match_list
    else:
        return list()
    

def all_equal(iterator):
    return len(set(iterator)) <= 1


def predict(feature, name_df, label_df):
    results_list = list()
    for i in range(label_df.shape[0]):
        x =  label_df['matched_name'][i] + label_df['matched_screen_name'][i]
        x = list(dict.fromkeys(x))
        if len(x) > 0:
            if len(x) == 1:
                results_list.append(name_df[feature][x[0]])
            elif len(x)>1:
                predicted_list = list()
                for name in x:
                    prediction = name_df[feature][name]
                    if '/' in prediction:
                        predicted_list += prediction.split('/')
                    else:
                        predicted_list.append(prediction)
                if feature == 'gender':
                    predicted_list = [x for x in predicted_list if x != 'unisex']
                    if len(predicted_list)> 0:
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
                    if len(predicted_list)> 0:
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


def score_for_name(name, name_df, feature):
    scores = {l:0 for l in LABELS[feature]}
    y = name_df[feature][name]
    if not pd.isnull(y):
        if feature=='ethnicity':
            if '/' in y:
                y = y.split('/')
                print(y)
            else:
                y = [y]
            if len(y)==1:
                scores[y[0]]=1
            elif len(y)>1:
                s = 1/len(y)
                scores = {l:s for l in scores}
        elif feature=='gender':
            if name in gender_df:
                for l in scores:
                    scores[l] = gender_df[name][l]
            else:
                if y=='unisex':
                    scores = {l:0.5 for l in scores}
                else:
                    scores[y]=1
        else:        
            if '/' in y:
                scores = {l:0.5 for l in scores}
            else:
                scores[y]=1
    return scores


def merge_scores(all_scores, weights, feature):
    merged_scores = {l:0 for l in LABELS[feature]}
    for i, scores in enumerate(all_scores):
        for l in merged_scores:
            merged_scores[l] += all_scores[i][l]*weights[i]
    return merged_scores


def score(feature, name_df, df):
    """ Compute name matching scores for all target labels of a given feature (e.g. ethnicity), and for 
    all observations of a given df """
    scores = {l:[] for l in LABELS[feature]}
    for i in range(df.shape[0]):
        x =  df['matched_name'].iloc[i] + df['matched_screen_name'].iloc[i]
#         print(x)
        x = list(set(x)) # We don't keep multiple occurrences of each name

        all_scores = [score_for_name(unidecode(n.lower()), name_df, feature) for n in x]
        # Compute a score vector for each name (e.g. if not ambiguous, 1 for target label, if ambiguous such as
        # muslim/christian, 0.5 for each), results in a vector list
        # Example : if 'Karim' is 'Yoruba' and 'Manu' is 'Yoruba/Igbo', returns [(0,1,0), (0,0.5,0.5)]
        
        sumlen = max(sum([len(n) for n in x]), 1)
        # Here we compute the sum of lengths for names
        
        weights = [len(n)/sumlen for n in x]
        # Weights for each name are its length
        
        merged = merge_scores(all_scores, weights, feature)
        # The merged vector is the mean of obtained vectors in "all_scores" variable, weighted by each name's length.
        # Example : with 'Karim' and 'Manu', sumlen=9, and weights=[5/9, 4/9] (the longer a name, the more reliable it is)
        
        for l in scores:
            scores[l].append(merged[l])
    return pd.DataFrame(scores)

    
if __name__=='__main__':
    profiles_path = sys.argv[1] # Path to folder containing list of parquet files with user profiles
    
    # Load and preprocess labeled list of names
    name_df = load_name_df(NAME_LIST_PATH)
    
    # Load and preprocess labeled test users list
    label_df = pd.read_csv(TEST_USERS_PATH)#'../labeled_2000.csv')
    label_df = label_df.dropna()

    label_df['ethnicity'] = label_df['ethnicity'].apply(str.lower)
    label_df['religion'] = label_df['religion'].apply(str.lower)
    label_df['gender'] = label_df['gender'].apply(str.lower)

    label_df = label_df.loc[label_df['gender'].isin(['m', 'f'])]
    label_df = label_df.loc[label_df['ethnicity'].isin(['igbo', 'yoruba', 'hausa'])]
    label_df = label_df.loc[label_df['religion']!='unsure']
    
    n_samples = label_df.shape[0]
    label_df = label_df.reset_index(drop=True)
    
    # Loading matching utils
    name_list = name_df.index.tolist()
    name_list.sort(key=len, reverse=True)
    name_list = [name for name in name_list if len(name)>2]
    
    label_df['matched_screen_name'] = label_df['screen_name'].apply(lambda x: match_name(x,name_list))
    label_df['matched_name'] = label_df['name'].apply(lambda x: match_name(x,name_list))
    
    gender_df = pd.read_csv('../data/gender_scraped.csv')
    gender_df['Forename'] = gender_df['Forename'].str.lower()
    gender_df = gender_df.set_index('Forename').drop('Unnamed: 0', axis=1)
    
    # Get scores for test_df
    scores_test = score('ethnicity', name_df, label_df)
    
    ## Get scores for all users
    # Load user profiles
#     df_list = []
#     for path in Path(profiles_path).glob('*.parquet'):
#         df = pd.read_parquet(path)

#         df['matched_screen_name'] = df['user_screen_name'].apply(lambda x: match_name(x,name_list))
#         df['matched_name'] = df['user_name'].apply(lambda x: match_name(x,name_list))
#         df_list.append(df)
#     df = pd.concat(df_list)
#     df = df.reset_index(drop=True)
    df = pd.read_parquet(profiles_path)
    df['matched_screen_name'] = df['user_screen_name'].apply(lambda x: match_name(x,name_list))
    df['matched_name'] = df['user_name'].apply(lambda x: match_name(x,name_list))    
    df = df.reset_index(drop=True)
    
    # Score each feature and save file
    if not os.path.exists(OUTPUT_PATH):
        os.mkdir(OUTPUT_PATH)
    for feature in ('ethnicity', 'religion', 'gender'):
        feature_scores = score(feature, name_df, df)
        feature_scores['user_id'] = df['user_id']
        feature_scores.to_csv(f'{OUTPUT_PATH}/name_matching_scores_{feature}.csv')
        
        feature_labels = LABELS[feature]
        predictions = feature_scores[feature_labels].apply(lambda x:feature_labels[np.argmax(x.values)], axis=1)

        df[f'{feature}_name_predict'] = predictions
    df.to_csv('../data/all_users_with_demographics.csv')