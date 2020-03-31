#classification model using Gaussian Naive Bayes
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
videos = "deepfake_detection/videos/REAL"
for video_file in sorted(os.listdir(videos)):
    if ".mp4" in str(video_file):
            y.append(0)
            
videos = "deepfake_detection/videos/FAKE"
for video_file in sorted(os.listdir(videos)):
    if ".mp4" in str(video_file):
            y.append(1)

#run detect_blinks.py script on all videos and store number of blinks
blinks = []
videos = "/Users/nchatwani/Desktop/deepfake_detection/videos/REAL"
for video_file in sorted(os.listdir(videos)):
    if ".mp4" in str(video_file):
        output = get_ipython().getoutput('python detect_blinks.py $videos/$video_file shape_predictor_68_face_landmarks.dat')
        print(output)
        blinks.append(output)
        
videos = "/Users/nchatwani/Desktop/deepfake_detection/videos/FAKE"
for video_file in sorted(os.listdir(videos)):
    if ".mp4" in str(video_file):
        output = get_ipython().getoutput('python detect_blinks.py $videos/$video_file shape_predictor_68_face_landmarks.dat')
        print(output)
        blinks.append(output)
        
# split data
X_train, X_test, y_train, y_test = train_test_split(blinks, y, test_size=0.75, random_state=42)

#Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import numpy as np

X_train = np.array(X_train, dtype=np.float32)
y_train = np.array(y_train, dtype=np.float32)
X_test = np.array(X_test, dtype=np.float32)
y_test = np.array(y_test, dtype=np.float32)


y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

# Initialize our classifier
gnb = GaussianNB()

# Train our classifier
model = gnb.fit(X_train, y_train)

preds = gnb.predict(y_test)
scores = gnb.predict_proba(X_test)[:,1]
print(preds)

# Evaluate accuracy
print('Accuracy: ', accuracy_score(y_test, preds))
print('AUROC: ', roc_auc_score(y_test, scores))
print(classification_report(y_test, preds))

