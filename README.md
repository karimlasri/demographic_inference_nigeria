# Graph-Based Demographic Inference in Nigerian Twitter
This repository contains the implementation of [Large-Scale Demographic Inference of Social Media Users in a Low-Resource Scenario](https://ojs.aaai.org/index.php/ICWSM/article/view/22165/21944), accepted at ICWSM 2023.

# Data Format
## User profiles
User profiles can be stored as a csv or parquet file and should contain at least a column with users' twitter ID as `user_id`, one with the users' names as `name`, and one with the users' screen name as `screen_name`. It could be for example stored as `data/user_profiles.csv` along with other data files. 
## Annotated user profiles 
Annotations should have the same format as user profiles without annotations described above, and should have additional columns to encode the considered attributes. In our experiments, those fields correspond to `ethnicity`, `gender`, and `religion`, and their corresponding columns contain the annotations. Optionally, these accounts can be assigned labels for whether they belong to an organization, under an `org` column, or whether the account was suspended, under a `suspended` column, in which case they will still be treated as nodes for label propagation but discarded prior to evaluation for active individuals using the platform. 
## Followership data
Followership (or friendship) data consists in two files, that can be located in a subfolder of the `data/` folder, e.g. `data/followership/`. The first file is expected to be a dataframe containing a list of twitter user ids as `nodes.csv`, and an adjacency matrix, which is expected to be a sparse binary matrix as `adjacency_matrix.npz`, encoding followership edges among users represented in the first file. Hence, the second file is expected to contain a square matrix which has the same size as the users id list. 


# Usage
## Name Matching
First perform name matching by running performing the following command :
```
python3 src/match_names.py --user_profiles <path_to_user_profiles> --out_path <output_path> --annotations_path <annotations_path> --evaluate <bool_for_evaluation>
```
Where `annotations_path` and `evaluate` are optional arguments indicating the path for an annotated set of users, and a boolean indicating whether the name matching should be evaluated on that set. 

For example
```
python3 src/match_names.py --user_profiles data/user_profiles.csv --out_path data/profiles_with_name_matching.csv --annotations_path data/annotations_demographics.csv
```
This will save scores from name matching for each user in the data folder, at `data/all_users_with_demographics.csv`. If an annotation set is given and evaluation is performed, the evaluation will also be saved in the data folder at `data/name_matching_eval.json`. 

## Label Propagation
Then perform label propagation by running : 
```
python3 src/label_propagation.py --name_matched_profiles <path_to_profiles_with_matching_scores> --followership_path <path_to_followership_data> --annotations_path <path_to_annotations> --symmetrize <bool_for_symmetric_adj_matrix> --plot_by_connections <bool_for_plot_by_followership>
```
For example :
```
python3 src/label_propagation.py --name_matched_profiles data/all_users_with_demographics.csv --friendship_data data/friendship_network/ --annotations_path data/annotations_demographics.csv --symmetrize True --plot_by_connections True
```
Where `symmetrize` is an optional argument set to `True` and indicates whether the followership matrix should be made symmetric prior to performing label propagation, and `plot_by_connections` is also an optional bool set to `True` by default indicating whether the results (accuracy and coverage) should be plotted against the number of connections of users in the test set. If it is set to `True`, the plots will be saved in a `plots/` folder for each demographic attribute. 