import os
import re
import json
import argparse
import numpy as np
import pandas as pd
from unidecode import unidecode
from sklearn.metrics import accuracy_score
from loading_utils import load_name_mapping, load_name_gender_prop
from config import (
    NAMES_TO_ATTRIBUTES_MAP,
    GENDER_SCRAPED_PATH,
    SCORES_PATH,
    CLASSES,
    NM_RESULTS_PATH,
)


def match_name(user_name, names_list):
    """Match a given name (user name or screen name) against a list of names.
    A name will be matched if it is a substring of the input user name."""
    regex = re.compile(rf"\b(?:{'|'.join(names_list)})", re.IGNORECASE)
    match_list = [match.group(0) for match in regex.finditer(user_name)]
    if len(match_list) > 0:
        return match_list
    else:
        return list()


def match_names(user_profiles, names_list):
    user_profiles["matched_screen_name"] = user_profiles["screen_name"].apply(
        lambda x: match_name(x, names_list)
    )
    user_profiles["matched_name"] = user_profiles["name"].apply(
        lambda x: match_name(x, names_list)
    )
    return user_profiles


def predict_for_attr(attr, names_mapping, user_profiles):
    """Predict a demographic attribute for given users based on their matched name and screen name, and the name mapping to that attribute"""
    predictions = []
    for _, row in user_profiles.itterrows():
        matched_names = row["matched_name"] + row["matched_screen_name"]
        matched_names = [name.lower() for name in list(dict.fromkeys(matched_names))]
        if len(matched_names) > 0:
            if len(matched_names) == 1:
                predictions.append(names_mapping[attr][matched_names[0]])
            elif len(matched_names) > 1:
                predicted_list = []
                for name in matched_names:
                    prediction = names_mapping[attr][name]
                    if "/" in prediction:
                        predicted_list += prediction.split("/")
                    else:
                        predicted_list.append(prediction)
                if len(predicted_list) == 0:
                    predictions.append("None")
                else:
                    if attr == "gender":
                        predicted_list = [x for x in predicted_list if x != "unisex"]
                        predictions.append(predicted_list[0])
                    elif attr == "ethnicity":
                        if "yoruba" in predicted_list:
                            predictions.append("yoruba")
                        elif "hausa" in predicted_list:
                            predictions.append("hausa")
                        else:
                            predictions.append(predicted_list[0])
                    elif attr == "religion":
                        predicted_list = [x for x in predicted_list if str(x) != "nan"]
                        if "muslim" in predicted_list:
                            predictions.append("muslim")
                        else:
                            predictions.append(predicted_list[0])
            else:
                predictions.append("None")
        else:
            predictions.append("None")
    return predictions


def predict_attrs_from_names(labeled_profiles, names_mapping):
    for attr in CLASSES:
        labeled_profiles[f"{attr}_name_predict"] = predict_for_attr(
            attr, names_mapping, labeled_profiles
        )
    return labeled_profiles


def score_for_name(name, names_mapping, attr, gender_proportions):
    """Given a name, returns scores for each class of a given attribute based on
    the name mapping and an additional scraped gender score database."""
    scores = {c: 0 for c in CLASSES[attr]}
    y = names_mapping[attr][name]
    if not pd.isnull(y):
        if attr == "ethnicity":
            if "/" in y:
                y = y.split("/")
            else:
                y = [y]
            if len(y) == 1:
                scores[y[0]] = 1
            elif len(y) > 1:
                s = 1 / len(y)
                scores = {c: s for c in scores}
        elif attr == "gender":
            if name in gender_proportions:
                for c in scores:
                    scores[c] = gender_proportions[name][c]
            else:
                if y == "unisex":
                    scores = {c: 0.5 for c in scores}
                else:
                    scores[y] = 1
        else:
            if "/" in y:
                scores = {c: 0.5 for c in scores}
            else:
                scores[y] = 1
    return scores


def merge_scores(all_scores, weights, attr):
    """Merges all scores that were obtained from matched names by weighting scores based on the name lengths."""
    merged_scores = {c: 0 for c in CLASSES[attr]}
    for i, scores in enumerate(all_scores):
        for c in merged_scores:
            merged_scores[attr] += all_scores[i][c] * weights[i]
    return merged_scores


def score(attr, names_mapping, df, gender_proportions):
    """Compute name matching scores for all target classes of a given demographic attribute (e.g. ethnicity), and for
    all observations of a given df"""
    scores = {c: [] for c in CLASSES[attr]}
    for i in range(df.shape[0]):
        x = df["matched_name"].iloc[i] + df["matched_screen_name"].iloc[i]
        x = list(set(x))  # We keep only one instance of each name

        # Compute a score vector for each name (e.g. if not ambiguous, 1 for target label, if ambiguous such as
        # muslim/christian, 0.5 for each), results in a vector list
        # Example : if 'Anu' is 'Yoruba' and 'Ife' is 'Yoruba/Igbo', returns [(0,1,0), (0,0.5,0.5)]
        all_scores = [
            score_for_name(
                unidecode(n.lower()), names_mapping, attr, gender_proportions
            )
            for n in x
        ]

        # Here we compute the sum of lengths for names
        sumlen = max(sum([len(n) for n in x]), 1)

        # Weights for each name are its length
        weights = [len(n) / sumlen for n in x]

        # The merged vector is the mean of obtained vectors in "all_scores" variable, weighted by each name's length.
        # Example : with 'Abba' and 'Aashif', sumlen=10, and weights=[4/10, 6/10] (the longer a name, the more reliable it is)
        merged = merge_scores(all_scores, weights, attr)

        for c in scores:
            scores[c].append(merged[c])
    return pd.DataFrame(scores)


def evaluate(labeled_profiles):
    name_matching_eval = {}
    n_labeled_profiles = labeled_profiles.shape[0]
    for attr in CLASSES:
        attr_eval = {}
        n_predictions = labeled_profiles[
            labeled_profiles[f"{attr}_name_predict"] != "None"
        ].shape[0]
        # Compute coverage of name matching
        attr_eval["cov"] = n_predictions / n_labeled_profiles
        # Compute accuracy of name matching
        attr_eval["acc"] = accuracy_score(
            labeled_profiles[attr], labeled_profiles[f"{attr}_name_predict"]
        )
        # Compute majority class score
        attr_value_counts = labeled_profiles[
            labeled_profiles[f"{attr}_name_predict"] != "None"
        ].value_counts()
        majority_class = list(attr_value_counts.keys())[0]
        attr_eval["maj"] = accuracy_score(
            labeled_profiles[attr], [majority_class] * name_matching_eval
        )
        name_matching_eval[attr] = attr_eval
    return name_matching_eval


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Assign name-based demographic attributes by matching names of twitter users using a database mapping names to demographic attributes."
    )

    parser.add_argument("--user_profiles", type=str, help="Path to the user profiles.")
    parser.add_argument(
        "--out_path",
        type=str,
        help="Path where user profiles with their name-based scores for each attribute will be saved.",
    )
    parser.add_argument(
        "--annotations_path", type=str, help="Path to labeled profiles.", default=None
    )
    parser.add_argument(
        "--evaluate",
        type=bool,
        help="Whether or not to evaluate name matching on a labeled test set.",
        default=False,
    )

    args = parser.parse_args()

    profiles_path = (
        args.user_profiles
    )  # Path to folder containing list of parquet files with user profiles

    # Load and preprocess labeled list of names
    names_mapping = load_name_mapping(NAMES_TO_ATTRIBUTES_MAP)

    # Loading matching utils
    names_list = names_mapping.index.tolist()
    names_list.sort(key=len, reverse=True)
    names_list = [name for name in names_list if len(name) > 2]

    # Load additional scraped gender mapping
    gender_proportions = load_name_gender_prop(GENDER_SCRAPED_PATH)

    ## Name matching for user profiles without annotations
    # Load user profiles
    user_profiles = pd.read_parquet(profiles_path)
    user_profiles = match_names(user_profiles, names_list)
    user_profiles = user_profiles.reset_index(drop=True)

    # Score each demographic attribute and save file
    if not os.path.exists(SCORES_PATH):
        os.mkdir(SCORES_PATH)
    for attr in CLASSES:
        attr_scores = score(attr, names_mapping, user_profiles, gender_proportions)
        attr_scores["user_id"] = user_profiles["user_id"]
        attr_scores.to_csv(f"{SCORES_PATH}/name_matching_scores_{attr}.csv")

        attr_classes = CLASSES[attr]
        predictions = attr_scores[attr_classes].apply(
            lambda x: attr_classes[np.argmax(x.values)], axis=1
        )
        user_profiles[f"{attr}_name_predict"] = predictions

    out_path = args.out_path
    user_profiles.to_csv(out_path)

    ## Evaluate name matching on test set
    if args.evaluate:
        # Load and preprocess labeled test users list
        labeled_profiles = pd.read_csv(args.annotations_path)
        labeled_profiles = labeled_profiles.dropna()
        # Get predictions from name matching
        labeled_profiles = predict_attrs_from_names(labeled_profiles, names_mapping)

        labeled_profiles = labeled_profiles.loc[
            labeled_profiles["gender"].isin(["m", "f"])
        ]
        labeled_profiles = labeled_profiles.loc[
            labeled_profiles["ethnicity"].isin(["igbo", "yoruba", "hausa"])
        ]
        labeled_profiles = labeled_profiles.loc[
            labeled_profiles["religion"] != "unsure"
        ]
        labeled_profiles = labeled_profiles.reset_index(drop=True)

        # Match names for test set
        labeled_profiles = match_names(labeled_profiles, names_list)

        # Get predictions for test_df
        for attr in CLASSES:
            labeled_profiles[f"{attr}_name_predict"] = predict_for_attr(
                attr, names_mapping, labeled_profiles
            )
        name_matching_eval = evaluate(labeled_profiles)

        # Save the evaluation scores
        with open(NM_RESULTS_PATH, "w") as eval_file:
            json.dump(name_matching_eval, eval_file)
