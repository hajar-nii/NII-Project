Code is based on the one for ACL 2022 paper:  [Schumann and Riezler, "Analyzing Generalization of Vision and Language Navigation to Unseen Outdoor Areas "](https://aclanthology.org/2022.acl-long.518.pdf)
Their GitHub repository is the following: https://github.com/raphael-sch/map2seq_vln

## MatterPort Dataset Download
The instructions to download the MatterPort dataset as well as the panoramic images used for the project can be found here: https://drive.google.com/drive/folders/1ZOS0i-XiBmWS0kWR2T0xu6P8_ShYXfv5?usp=drive_link
The R2R dataset is downloaded via the bash script in 'datasets/r2r/download.sh'.
### Panorama Preprocessing
After downloading the dataset images, use the `extract_features.py` script to extract the necessary features.

### Preparation
```
pip install -r requirements.txt
```

### Run code
```
python vln/main.py --test True --dataset r2r --img_feat_dir 'path_to_features_dir' --config link_to_config --exp_name 4th-to-last --resume SPD_best
```
The `path_to_features_dir` should contain the `resnet_fourth_layer.pickle` and `resnet_last_layer.pickle` file created in the pano preprocessing step.
Configurations are stored in configs/ and different parameters of the model can be adjusted there.
The splits used for training, validation and testing can be adjusted in the vln/main.py. 
After running the code, a test.json file is generated. It contains the model's predictions of the trajectories for each instruction in the test split of the R2R dataset.
This file is submitted to the R2Rchallenge on EvalAI: https://eval.ai/web/challenges/challenge-page/97/overview. The metrics are computed on the platform then returned in JSON format.
However, the number of submissions is limited.




