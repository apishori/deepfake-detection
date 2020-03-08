import json
import os
import shutil

json_path = 'metadata.json'

with open(json_path, 'r') as f:
    labels = json.load(f)

for item in labels:
    if labels[item]['label'] == "FAKE":
        shutil.move('data/' + item, 'data/FAKE/'+ item)
    elif labels[item]['label'] == "REAL":
        shutil.move('data/' + item, 'data/REAL/'+ item)