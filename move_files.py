import json
import os
import shutil

path = 'C:/Users/armaa/Desktop/deepfake_detection/deepfake-detection-challenge/train_sample_videos/'
json_path = path+'metadata.json'
fake = 'C:/Users/armaa/Desktop/deepfake_detection/data/FAKE/'
real= 'C:/Users/armaa/Desktop/deepfake_detection/data/REAL/'

with open(json_path, 'r') as f:
    labels = json.load(f)

for item in labels:
    if labels[item]['label'] == "FAKE":
        shutil.move(path+item, fake+item)
    elif labels[item]['label'] == "REAL":
        shutil.move(path+item, real+item)