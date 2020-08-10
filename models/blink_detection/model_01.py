import os
import shlex
import subprocess
from subprocess import PIPE, Popen

from detect_blinks import *
from sklearn.metrics import (accuracy_score, classification_report,
                             roc_auc_score)
from sklearn.model_selection import train_test_split

#create labels array
path = '../../data/train/'
real_videos = path + 'REAL'
fake_videos = path + 'FAKE'
y = [1] * len(sorted(os.listdir(real_videos)))
y.extend([0] * len(sorted(os.listdir(fake_videos))))
print(y)

#run detect_blinks.py script on all videos and store number of blinks
blinks = []
for video_file in sorted(os.listdir(real_videos)):
    if ".mp4" in str(video_file):
        output = get_ipython().getoutput('python detect_blinks.py $real_videos/$video_file shape_predictor_68_face_landmarks.dat')
        print('REAL', str(video_file), output)
        blinks.extend(output)

for video_file in sorted(os.listdir(fake_videos)):
    if ".mp4" in str(video_file):
        output = get_ipython().getoutput('python detect_blinks.py $fake_videos/$video_file shape_predictor_68_face_landmarks.dat')
        print('FAKE', str(video_file), output)
        blinks.extend(output)
        
# split data
X_train, X_test, y_train, y_test = train_test_split(blinks, y, test_size=0.75, random_state=42)

#Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB
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

# Save the model as a pickle in a file 
import joblib 
joblib.dump(model, '../output_models/bd_01.pkl') 
