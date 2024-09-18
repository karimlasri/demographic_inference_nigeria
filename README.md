# Graph Based Demographic Inference in Nigerian Twitter
This repository contains the implementation of [Large-Scale Demographic Inference of Social Media Users in a Low-Resource Scenario](https://ojs.aaai.org/index.php/ICWSM/article/view/22165/21944), accepted at ICWSM 2023.

# Data structure


# Usage
First perform name matching by running performing the following command :
```
python3 models/match_names.py <path_to_user_profiles>
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
