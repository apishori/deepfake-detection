#classification model using Gaussian Naive Bayes w/ Gridsearch
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
from sklearn.naive_bayes import GaussianNB

path = 'C:/Users/armaa/Desktop/deepfake_detection/'

#create labels array
get_ipython().run_line_magic('pwd', '')
y = []
videos = path+"data/train/REAL"
for video_file in sorted(os.listdir(videos)):
    if ".mp4" in str(video_file):
            y.append(0)
            
videos = path+"data/train/FAKE"
for video_file in sorted(os.listdir(videos)):
    if ".mp4" in str(video_file):
            y.append(1)


#run detect_blinks.py script on all videos and store number of blinks
blinks = []
videos = path+"data/train/REAL"
for video_file in sorted(os.listdir(videos)):
    if ".mp4" in str(video_file):
        output = get_ipython().getoutput('python detect_blinks.py $videos/$video_file shape_predictor_68_face_landmarks.dat')
        print(output)
        blinks.append(output)
        
videos = path+"data/train/FAKE"
for video_file in sorted(os.listdir(videos)):
    if ".mp4" in str(video_file):
        output = get_ipython().getoutput('python detect_blinks.py $videos/$video_file shape_predictor_68_face_landmarks.dat')
        print(output)
        blinks.append(output)

# split data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(blinks, y, test_size=0.75, random_state=42)

#Gaussian Naive Bayes with Gridsearch
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
import numpy as np

X_train = np.array(X_train, dtype=np.float32)
y_train = np.array(y_train, dtype=np.float32)
X_test = np.array(X_test, dtype=np.float32)
y_test = np.array(y_test, dtype=np.float32)

base_clf = GaussianNB()
parameters = {'priors':[None], 'var_smoothing':[1e-09]}

clf = GridSearchCV(base_clf, parameters, cv=3)
clf.fit(X_train, y_train)
print('Best Hyperparameters: ', clf.best_params_, '\n')

pred = clf.predict(X_test)
scores = clf.predict_proba(X_test)[:,1]   

print('Accuracy: ', accuracy_score(y_test, pred))
print('AUROC: ', roc_auc_score(y_test, scores))
print(classification_report(y_test, pred))

# Save the model as a pickle in a file 
import joblib 
joblib.dump(clf, path+'output_model/my_model12.pkl') 