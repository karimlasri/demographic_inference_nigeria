import pandas as pd
from pathlib import Path
import re

PROFILES_PATH = 'twitter_social_cohesion/data/profiles/NG' # INSERT HERE PATH TO FOLDER CONTAINING LIST OF PARQUET FILES WITH USER PROFILES
NAME_LIST_PATH = '../data/labeled_name_list.csv'
OUTPUT_PATH = '../data/all_users_with_demographics.parquet'

def modify_gender(x):
    if x == 'male':
        return 'm'
    elif x == 'female':
        return 'f'
    else:
        return x
    
    
def match_name(x):
    regex = re.compile(fr"\b(?:{'|'.join(name_list)})",re.IGNORECASE)
    match_list = [match.group(0) for match in regex.finditer(x)]
    if len(match_list) > 0:
        return match_list
    else:
        return list()
    
    
def all_equal(iterator):
    return len(set(iterator)) <= 1


def predict(feature, name_df, user_df):
    results_list = list()
    for i in range(user_df.shape[0]):
        x =  user_df['matched_name'][i] + user_df['matched_screen_name'][i]
        x = list(dict.fromkeys(x))
        x = [element.replace('ı', 'i') for element in x]
        if len(x) > 0:
            if len(x) == 1:
                results_list.append(name_df[feature][x[0]])
            elif len(x)>1:
                predicted_list = list()
                for name in x:
                    predicted_list.append(name_df[feature][name])
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
    return results_list


def clean_religion_predict(x):
    if '/' in x:
        return 'christian/muslim'
    else:
        return x
    

if __name__=='__main__':
    # Load users df
    user_df = pd.concat([pd.read_parquet(path) for path in Path(PROFILES_PATH).glob('*.parquet')])
    user_df = user_df.reset_index(drop=True)
    #user_df.head()
    # Preprocess users df
    user_df['screen_name'] = user_df['user_screen_name'].apply(str.lower)
    user_df['name'] = user_df['user_name'].apply(str.lower)
    
    # Loading labeled name dataframe
    name_df = pd.read_csv(NAME_LIST_PATH)
    name_df['gender'] = name_df['gender'].apply(modify_gender)
    name_df = name_df.drop_duplicates(subset=['name'])
    name_df = name_df.set_index(['name'])
    # Getting list of names
    name_list = name_df.index.tolist()
    name_list.sort(key=len, reverse=True)
    name_list = [name for name in name_list if len(name)>2]
    
    # Get matched names and screen names
    user_df['matched_screen_name'] = user_df['screen_name'].apply(match_name)
    user_df['matched_name'] = user_df['name'].apply(match_name)
    
    # Predict demographic attributes
    user_df['ethnicity_name_predict'] = predict('ethnicity', name_df, user_df)
    #user_df['ethnicity_name_predict'].value_counts(dropna=False, normalize=True)
    #user_df.loc[~user_df['ethnicity_name_predict'].isin(['None', 'cameroon'])]['ethnicity_name_predict'].value_counts(dropna=False, normalize=True)
    user_df['gender_name_predict'] = predict('gender', name_df, user_df)
    #user_df['gender_name_predict'].value_counts(dropna=False, normalize=True)
    #user_df.loc[~user_df['gender_name_predict'].isin(['None', 'unisex'])]['gender_name_predict'].value_counts(dropna=False, normalize=True)
    user_df['religion_name_predict'] = predict('religion', name_df, user_df)
    #user_df['religion_name_predict'].value_counts(dropna=False, normalize=True)
    #user_df.loc[~user_df['religion_name_predict'].isin(['None', None, 'muslim/christian', 'christian/muslim', 'muslim/christain'])]['religion_name_predict'].value_counts(dropna=False, normalize=True)
   
    # Clean dataframe
    user_df['religion_name_predict'] = user_df['religion_name_predict'].fillna('None')
    user_df['religion_name_predict'] = user_df['religion_name_predict'].apply(clean_religion_predict)
    user_df['religion_name_predict'].value_counts(dropna=False, normalize=True)
    user_df = user_df[['user_id', 'ethnicity_name_predict', 'gender_name_predict', 'religion_name_predict']]
    user_df.to_parquet(OUTPUT_PATH)