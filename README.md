# Graph Based Demographic Inference in Nigerian Twitter
This repository contains the implementation of [Large-Scale Demographic Inference of Social Media Users in a Low-Resource Scenario](https://ojs.aaai.org/index.php/ICWSM/article/view/22165/21944), accepted at ICWSM 2023.

# Data Format
## User profiles
User profiles can be stored as a csv or parquet file and should contain at least a column with users' twitter ID as `user_id`, one with the users' names as `name`, and one with the users' screen name as `screen_name`. It could be for example stored as `data/user_profiles.csv` along with other data files. 
## Followership data


# Usage
First perform name matching by running performing the following command :
```
python3 models/match_names.py --user_profiles <path_to_user_profiles> --out_path <output_path>
```
For example
```
python3 models/match_names.py --user_profiles data/user_profiles.csv --out_path data/all_users_with_demographics.csv
```
This will save scores from name matching for each user in the predictions folder.

Then perform label propagation by running : 
```
python3 models/label_propagation.py <path_to_friendship_data>
```
For example :
```
python3 models/label_propagation.py data/friendship_network/
```
