from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json
from shutil import copy2


##################Load config.json and correct path variable
with open('config.json','r') as f:
    config = json.load(f) 

model_path = os.path.join(config['output_model_path'])
dataset_csv_path = os.path.join(config['output_folder_path']) 
prod_deployment_path = os.path.join(config['prod_deployment_path']) 


####################function for deployment
def store_model_into_pickle():
    #copy the latest pickle file, the latestscore.txt value, and the ingestfiles.txt file into the deployment directory
    for i in ['ingestedfiles.txt',"trainedmodel.pkl", "encoder.pkl",'latestscore.txt']:

        if i in ['ingestedfiles.txt']:
            pth = os.path.join(dataset_csv_path,i)
        else:
            pth = os.path.join(model_path,i)
        
        new_path = os.path.join(prod_deployment_path, i)
        print(f'Copying {pth} to {new_path}')
        copy2(pth, new_path)
        
        
if __name__=="__main__":

    store_model_into_pickle()
