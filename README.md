# Graph Based Demographic Inference in Nigerian Twitter
This repository contains the implementation of [https://ojs.aaai.org/index.php/ICWSM/article/view/22165/21944](Large-Scale Demographic Inference of Social Media Users in a Low-Resource Scenario), accepted at ICWSM 2023.

# Usage
First run perform matching by running performing the following command :
```
cd models
python3 name_matching.py path_to_user_profiles
```
This will save scores from name matching for each user in the predictions folder.

Then perform label propagation by running : 
```
python3 label_propagation.py path_to_friendship_data
```
(e.g. python3 label_propagation.py ../data/friendship_network/)
