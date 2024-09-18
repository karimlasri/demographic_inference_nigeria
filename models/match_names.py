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
import argparse
from format_data import modify_gender
from loading_utils import load_name_mapping


NAMES_TO_ATTRIBUTES_MAP = 'data/names_attribute_map.csv'
ANNOTATIONS_PATH = 'data/annotations_demographics.csv'
GENDER_SCRAPED_PATH = 'data/names_gender_scraped.csv' # TODO : Document this
OUTPUT_PATH = 'predictions/'
CLASSES = {
    'ethnicity':['yoruba', 'hausa', 'igbo'],
    'gender':['m', 'f'],
    'religion':['christian', 'muslim']
         }

    
def match_name(x, names_list):
    regex = re.compile(fr"\b(?:{'|'.join(names_list)})",re.IGNORECASE)
    match_list = [match.group(0) for match in regex.finditer(x)]
    if len(match_list) > 0:
        return match_list
    else:
        return list()


def predict(attr, names_mapping, user_profiles):
    predictions = list()
    for i in range(user_profiles.shape[0]):
        x = user_profiles['matched_name'][i] + user_profiles['matched_screen_name'][i]
        x = [n.lower() for n in list(dict.fromkeys(x))]
        if len(x) > 0:
            if len(x) == 1:
                predictions.append(names_mapping[attr][x[0]])
            elif len(x) > 1:
                predicted_list = list()
                for name in x:
                    prediction = names_mapping[attr][name]
                    if '/' in prediction:
                        predicted_list += prediction.split('/')
                    else:
                        predicted_list.append(prediction)
                if attr == 'gender':
                    predicted_list = [x for x in predicted_list if x != 'unisex']
                    if len(predicted_list )> 0:
                        predictions.append(predicted_list[0])
                    else:
                        predictions.append('None')
                elif attr == 'ethnicity':
                    if 'yoruba' in predicted_list:
                        predictions.append('yoruba')
                    elif 'hausa' in predicted_list:
                        predictions.append('hausa')
                    else:
                        predictions.append(predicted_list[0])
                elif attr == 'religion':
                    predicted_list = [x for x in predicted_list if not str(x) == 'nan']
                    if len(predicted_list)> 0:
                        if 'muslim' in predicted_list:
                            predictions.append('muslim')
                        else:
                            predictions.append(predicted_list[0])
                    else:
                        predictions.append('None')
            else:
                predictions.append('None')
        else:
            predictions.append('None')
    return predictions


def score_for_name(name, names_mapping, attr):
    scores = {l:0 for l in CLASSES[attr]}
    y = names_mapping[attr][name]
    if not pd.isnull(y):
        if attr=='ethnicity':
            if '/' in y:
                y = y.split('/')
            else:
                y = [y]
            if len(y)==1:
                scores[y[0]]=1
            elif len(y)>1:
                s = 1/len(y)
                scores = {l:s for l in scores}
        elif attr=='gender':
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


def merge_scores(all_scores, weights, attr):
    merged_scores = {l:0 for l in CLASSES[attr]}
    for i, scores in enumerate(all_scores):
        for l in merged_scores:
            merged_scores[l] += all_scores[i][l]*weights[i]
    return merged_scores


def score(attr, names_mapping, df):
    """ Compute name matching scores for all target classes of a given demographic attribute (e.g. ethnicity), and for 
    all observations of a given df """
    scores = {l:[] for l in CLASSES[attr]}
    for i in range(df.shape[0]):
        x =  df['matched_name'].iloc[i] + df['matched_screen_name'].iloc[i]
        x = list(set(x)) # We keep only one instance of each name

        all_scores = [score_for_name(unidecode(n.lower()), names_mapping, attr) for n in x]
        # Compute a score vector for each name (e.g. if not ambiguous, 1 for target label, if ambiguous such as
        # muslim/christian, 0.5 for each), results in a vector list
        # Example : if 'Anu' is 'Yoruba' and 'Ife' is 'Yoruba/Igbo', returns [(0,1,0), (0,0.5,0.5)]
        
        sumlen = max(sum([len(n) for n in x]), 1)
        # Here we compute the sum of lengths for names
        
        weights = [len(n)/sumlen for n in x]
        # Weights for each name are its length
        
        merged = merge_scores(all_scores, weights, attr)
        # The merged vector is the mean of obtained vectors in "all_scores" variable, weighted by each name's length.
        # Example : with 'Karim' and 'Manu', sumlen=9, and weights=[5/9, 4/9] (the longer a name, the more reliable it is)
        
        for l in scores:
            scores[l].append(merged[l])
    return pd.DataFrame(scores)

    
if __name__=='__main__':
    parser = argparse.ArgumentParser(
        description="Assign name-based demographic attributes by matching names of twitter users using a database mapping names to demographic attributes."
    )

    parser.add_argument("--user_profiles", type=str, help="Path to the user profiles.")

    args = parser.parse_args()

    profiles_path = args.user_profiles # Path to folder containing list of parquet files with user profiles

    # Load and preprocess labeled list of names
    names_mapping = load_name_mapping(NAMES_TO_ATTRIBUTES_MAP)
    
    # Load and preprocess labeled test users list
    labeled_profiles = pd.read_csv(ANNOTATIONS_PATH)
    labeled_profiles = labeled_profiles.dropna()

    labeled_profiles['ethnicity'] = labeled_profiles['ethnicity'].apply(str.lower)
    labeled_profiles['religion'] = labeled_profiles['religion'].apply(str.lower)
    labeled_profiles['gender'] = labeled_profiles['gender'].apply(str.lower)

    labeled_profiles = labeled_profiles.loc[labeled_profiles['gender'].isin(['m', 'f'])]
    labeled_profiles = labeled_profiles.loc[labeled_profiles['ethnicity'].isin(['igbo', 'yoruba', 'hausa'])]
    labeled_profiles = labeled_profiles.loc[labeled_profiles['religion']!='unsure']
    
    n_samples = labeled_profiles.shape[0]
    labeled_profiles = labeled_profiles.reset_index(drop=True)
    
    # Loading matching utils
    name_list = names_mapping.index.tolist()
    name_list.sort(key=len, reverse=True)
    name_list = [name for name in name_list if len(name)>2]
    
    labeled_profiles['matched_screen_name'] = labeled_profiles['screen_name'].apply(lambda x: match_name(x,name_list))
    labeled_profiles['matched_name'] = labeled_profiles['name'].apply(lambda x: match_name(x,name_list))
    
    gender_df = pd.read_csv('data/names_gender_scraped.csv')
    gender_df['Forename'] = gender_df['Forename'].str.lower()
    gender_df = gender_df.set_index('Forename').drop('Unnamed: 0', axis=1)
    
    # Get scores for test_df
    scores_test = score('ethnicity', names_mapping, labeled_profiles)
    
    df = pd.read_parquet(profiles_path)
    df['matched_screen_name'] = df['user_screen_name'].apply(lambda x: match_name(x,name_list))
    df['matched_name'] = df['user_name'].apply(lambda x: match_name(x,name_list))    
    df = df.reset_index(drop=True)
    
    # Score each demographic attribute and save file
    if not os.path.exists(OUTPUT_PATH):
        os.mkdir(OUTPUT_PATH)
    for attr in ('ethnicity', 'religion', 'gender'):
        attr_scores = score(attr, names_mapping, df)
        attr_scores['user_id'] = df['user_id']
        attr_scores.to_csv(f'{OUTPUT_PATH}/name_matching_scores_{attr}.csv')
        
        attr_classes = CLASSES[attr]
        predictions = attr_scores[attr_classes].apply(lambda x : attr_classes[np.argmax(x.values)], axis=1)

        df[f'{attr}_name_predict'] = predictions
    df.to_csv('data/all_users_with_demographics.csv')