import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import simplejson as json
from detect_blinks import *
import os
from subprocess import Popen, PIPE 
import subprocess
import shlex

#load labels from json file
json_path = "train_sample_videos/metadata.json"
with open(json_path, 'r') as f:
    labels = json.load(f)

#create labels array with 1 for deepfake and 0 for real video
y = []
train_videos = "train_sample_videos/"
for video_file in sorted(os.listdir(train_videos)):
    if ".mp4" in str(video_file):
        if labels[video_file]['label'] == "REAL":
            y.append(0)
        else:
            y.append(1)


#run detect_blinks.py script on all videos and store number of blinks
blinks = []
train_videos = "train_sample_videos/"
for video_file in sorted(os.listdir(train_videos)):
    if ".mp4" in str(video_file):
        output = get_ipython().getoutput('python detect_blinks.py train_sample_videos/$video_file shape_predictor_68_face_landmarks.dat')
        print(output)
        blinks.append(output)


# split data
X_train, X_test, y_train, y_test = train_test_split(blinks, y, test_size=0.75, random_state=42)

# k-NN classifier with gridsearch
from sklearn import neighbors
from sklearn.model_selection import GridSearchCV

base_clf = neighbors.KNeighborsClassifier()
parameters = {'n_neighbors': [1, 2, 5, 10, 15, 25], 'weights': ['uniform', 'distance']}

clf = GridSearchCV(base_clf, parameters, cv=3)
clf.fit(X_train, y_train)
print('Best Hyperparameters: ', clf.best_params_, '\n')

pred = clf.predict(X_test)
scores = clf.predict_proba(X_test)[:,1]   

print('Accuracy: ', accuracy_score(y_test, pred))
print('AUROC: ', roc_auc_score(y_test, scores))
print(classification_report(y_test, pred))


