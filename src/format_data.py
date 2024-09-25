# from loading_utils import *


def keep_valid_ids(user_profiles):
    """Removes malformated user ids."""
    user_profiles["valid_id"] = user_profiles["user_id"].apply(lambda x: "+" not in x)
    user_profiles = user_profiles[user_profiles["valid_id"]]
    user_profiles["user_id"] = user_profiles["user_id"].astype("int64")
    return user_profiles


def filter_test_set(labeled_profiles, columns_to_keep):
    """Further clean the labeled profiles to keep only active users."""
    if "org" in labeled_profiles.columns.values:
        labeled_profiles = labeled_profiles[labeled_profiles["org"] == "0"]
    if "suspended" in labeled_profiles.columns.values:
        labeled_profiles = labeled_profiles[labeled_profiles["suspended"] == "0"]
    labeled_profiles = labeled_profiles[columns_to_keep]
    return labeled_profiles


def filter_matrix(matrix, nodes, user_ids):
    """Keeps only a subset of the original matrix based on the position of user ids in the list of nodes."""
    target_nodes = nodes[nodes["user_id"].isin(user_ids)]
    target_nodes = target_nodes.reset_index(drop=True)
    return (
        matrix[target_nodes["user_index"]][:, target_nodes["user_index"]],
        target_nodes,
    )


def modify_gender(x):
    """We keep only the gender's first letter to represent the class."""
    if x:
        if x != "unisex":
            return x[0]


def preprocess_users_matched_names_df(users_with_names_df):
    users_with_names_df = users_with_names_df.drop_duplicates("user_id")
    return users_with_names_df


def preprocess_unigrams_df(unigrams_df):
    unigrams_df = unigrams_df.set_index(["user_id"])
    return unigrams_df


def join_dfs(unigrams_df, users_with_names_df):
    unigrams_df = unigrams_df.set_index(["user_id"])
    users_with_names_df = users_with_names_df.set_index(["user_id"])
    unig_with_names = unigrams_df.join(users_with_names_df)
    unig_with_names = unig_with_names.reset_index()
    return unig_with_names


def sample_fraction(unig_with_names, frac):
    print("Sampling {}% of users...".format(round(frac * 100)))
    unique_ids = unig_with_names["user_id"].unique()
    ids_to_keep = unig_with_names["user_id"].sample(int(len(unique_ids) * frac))
    unig_with_names = unig_with_names[unig_with_names["user_id"].isin(ids_to_keep)]
    print("Done.")
    return unig_with_names


def filter_by_threshold(count_df_train, idfs, min_n_users):
    # TODO : Change this to keep also total count before we pivot
    print(
        "Filtering count dataframe by threshold of minimum {} users...".format(
            min_n_users
        )
    )
    kept_ids = idfs[idfs["n_users"] > min_n_users].index.tolist()
    count_df_train = count_df_train[count_df_train["token"].isin(kept_ids)]
    print("Done.")
    return count_df_train
