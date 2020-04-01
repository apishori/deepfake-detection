#classification model using k-nearest neighbors
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

#create labels array
get_ipython().run_line_magic('pwd', '')
y = []
videos = "data/REAL"
for video_file in sorted(os.listdir(videos)):
    if ".mp4" in str(video_file):
            y.append(0)
            
videos = "data/FAKE"
for video_file in sorted(os.listdir(videos)):
    if ".mp4" in str(video_file):
            y.append(1)

#run detect_blinks.py script on all videos and store number of blinks
blinks = []
videos = "data/REAL"
for video_file in sorted(os.listdir(videos)):
    if ".mp4" in str(video_file):
        output = get_ipython().getoutput('python detect_blinks.py $videos/$video_file shape_predictor_68_face_landmarks.dat')
        print(output)
        blinks.append(output)
        
videos = "data/FAKE"
for video_file in sorted(os.listdir(videos)):
    if ".mp4" in str(video_file):
        output = get_ipython().getoutput('python detect_blinks.py $videos/$video_file shape_predictor_68_face_landmarks.dat')
        print(output)
        blinks.append(output)

# split data
X_train, X_test, y_train, y_test = train_test_split(blinks, y, test_size=0.75, random_state=42)

#KNN classifer
from sklearn import neighbors

clf = neighbors.KNeighborsClassifier(n_neighbors=10, weights='distance')
clf.fit(X_train, y_train)

pred = clf.predict(X_test)
scores = clf.predict_proba(X_test)[:,1]

print('Accuracy: ', accuracy_score(y_test, pred))
print('AUROC: ', roc_auc_score(y_test, scores))
print(classification_report(y_test, pred))
