import pandas as pd
import os
from scipy.sparse import load_npz
from sklearn.preprocessing import normalize
from config import SCORES_PATH, CLASSES, NM_RESULTS_PATH
import json
from format_data import (
    filter_matrix,
    filter_test_set,
    keep_valid_ids,
    modify_gender,
)
from match_names import predict_attrs_from_names


name_matched_profiles_path = 'data/profiles_with_name_matching.csv'
matched_profiles_path = 'data/profiles_with_name_matching.csv'
names_mapping_path = "data/names_attribute_map.csv"
def load_name_mapping(names_mapping_path="data/names_attribute_map.csv"):
    """Loads the dataframe which includes names labelled with their ethnicity, gender and religion.
    This dataframe has 2813 rows so far."""
    names_mapping = pd.read_csv(names_mapping_path)
    names_mapping["gender"] = names_mapping["gender"].apply(modify_gender)
    names_mapping = names_mapping.drop_duplicates(subset=["name"])
    names_mapping = names_mapping.set_index(["name"])
    return names_mapping


def load_names_list(names_map_path="data/names_attributes_map.csv"):
    """Loads the list of names found in the mapping from names to attributes."""
    name_df = load_name_mapping(names_map_path=names_map_path)
    names_list = name_df.index.tolist()
    names_list.sort(key=len, reverse=True)
    names_list = [name for name in names_list if len(name) > 2]
    return names_list


def load_name_gender_prop(gender_proportions_path="data/names_gender_proportions.csv"):
    """Loads a dataframe where scraped Nigerian names are assigned gender proportions in the population."""
    gender_proportions = pd.read_csv(gender_proportions_path)
    gender_proportions["Forename"] = gender_proportions["Forename"].str.lower()
    gender_proportions = gender_proportions.set_index("Forename").drop(
        "Unnamed: 0", axis=1
    )
    return gender_proportions


def load_followership_data(adj_matrix_path, nodes_path):
    """Loads the raw followership matrix and nodes list."""
    adj_matrix = load_npz(adj_matrix_path)
    nodes = (
        pd.read_csv(nodes_path)
        .reset_index()
        .rename(columns={"index": "user_index", "0": "user_id"})
    )
    return adj_matrix, nodes

names_mapping_path ='data/names_attribute_map.csv'
annotations_path = 'data/annotations_demographics.csv'

def load_profiles_for_lp(matched_profiles_path, names_mapping_path, annotations_path):
    """Load and preprocess user profiles along with matched names, and the labeled set."""
    # Load users with matched names
    name_matched_profiles = pd.read_csv(matched_profiles_path)
    # Load mapping from names to demographic attributes and name list
    names_mapping = load_name_mapping(names_mapping_path)
    #names_mapping = pd.merge(name_matched_profiles,names_mapping,on='user_id')
    # Load test set
    labeled_profiles = pd.read_csv(annotations_path)
    # Preprocess test set
    labeled_profiles = labeled_profiles.dropna()
    labeled_profiles = predict_attrs_from_names(labeled_profiles, names_mapping)
    labeled_profiles = keep_valid_ids(labeled_profiles)

    # Concatenate the labeled test set with matched names from the user set as those can also be used for label propagation
    columns_to_keep = ["user_id"] + [f"{attr}_name_predict" for attr in CLASSES]
    name_matched_profiles = pd.concat(
        [name_matched_profiles[columns_to_keep], labeled_profiles[columns_to_keep]]
    )
    labeled_profiles = filter_test_set(labeled_profiles, columns_to_keep)
    # Further preprocess profiles
    name_matched_profiles["user_id"] = name_matched_profiles["user_id"].astype("int64")
    name_matched_profiles = name_matched_profiles.drop_duplicates("user_id")
    name_matched_profiles = name_matched_profiles.set_index("user_id")
    return name_matched_profiles, labeled_profiles




def load_prodfvdsfvdfvdffiles_for_lp(matched_profiles_path, names_mapping_path, annotations_path):
    """Load and preprocess user profiles along with matched names, and the labeled set."""
    # Load users with matched names
    name_matched_profiles = pd.read_csv(matched_profiles_path)
    # Load mapping from names to demographic attributes and name list
    names_mapping = load_name_mapping(names_mapping_path)

    # Load test set
    labeled_profiles = pd.read_csv(annotations_path)
    # Preprocess test set
    labeled_profiles = labeled_profiles.dropna()
    labeled_profiles = predict_attrs_from_names(labeled_profiles, names_mapping)
    labeled_profiles = keep_valid_ids(labeled_profiles)

    # Concatenate the labeled test set with matched names from the user set as those can also be used for label propagation
    columns_to_keep = ["user_id"] + [f"{attr}_name_predict" for attr in CLASSES]
    name_matched_profiles = pd.concat(
        [name_matched_profiles[columns_to_keep], labeled_profiles[columns_to_keep]]
    )
    labeled_profiles = filter_test_set(labeled_profiles, columns_to_keep)

    # Further preprocess profiles
    name_matched_profiles["user_id"] = name_matched_profiles["user_id"].astype("int64")
    name_matched_profiles = name_matched_profiles.drop_duplicates("user_id")
    name_matched_profiles = name_matched_profiles.set_index("user_id")
    return name_matched_profiles, labeled_profiles


def load_followership_for_lp(
    path_to_followership_data, name_matched_profiles, symmetrize=True
):
    """Load and prepare followership data for label propagation."""
    ## Load followership data
    adj_matrix_path = path_to_followership_data + "adjacency_matrix.npz"
    adj_nodes_path = path_to_followership_data + "nodes.csv"
    adj_matrix, nodes = load_followership_data(adj_matrix_path, adj_nodes_path)
    # Filter data to only keep user ids present in the target profiles
    user_ids = name_matched_profiles["user_id"].reset_index(drop=True)
    target_adj_matrix, target_nodes = filter_matrix(adj_matrix, nodes, user_ids)
    # Normalize adjacency matrix, and symmetrize if required
    if symmetrize:
        target_adj_matrix = target_adj_matrix.transpose() + target_adj_matrix
    norm_adj_matrix = normalize(target_adj_matrix.astype("float64"), norm="l1", axis=1)

    return norm_adj_matrix, target_nodes


def load_nm_scores_for_lp(target_nodes):
    """Load name matching scores to initialize label propagation, as this is a better signal than binary predictions."""
    nm_scores_df = pd.DataFrame(target_nodes["user_id"])
    for attr in CLASSES:
        attr_scores = pd.read_csv(
            f"{SCORES_PATH}/name_matching_scores_{attr}.csv"
        ).drop("Unnamed: 0", axis=1)
        attr_scores = attr_scores.rename(
            {
                k: "{}_name_predict_{}".format(attr, k)
                for k in attr_scores.columns
                if k != "user_id"
            },
            axis=1,
        ).drop_duplicates("user_id")
        nm_scores_df = nm_scores_df.merge(attr_scores, on="user_id", how="left").fillna(
            0
        )
    return nm_scores_df


def load_nm_results():
    if os.path.exists(NM_RESULTS_PATH):
        with open(NM_RESULTS_PATH) as eval_file:
            nm_results = json.load(eval_file)
        return nm_results
    else:
        print(
            "Error : could not find name matching evaluation file at location {}.".format(
                NM_RESULTS_PATH
            )
        )
