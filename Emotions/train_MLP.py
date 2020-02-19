# This code is used to train a MLPClassifier
# You need to pass workDir to train() function
# The workDir should contain reps.csv and labesl.csv files
# You can change the MLP layer structure in MLPClassifier()

import sklearn
from sklearn.neural_network import MLPClassifier
import pandas as pd
import pickle
import os
from operator import itemgetter
from sklearn.preprocessing import LabelEncoder
import argparse
import shutil
import time

path_to_features = './datas/faces/features'
path_to_aligned = './datas/faces/aligned'


def train(workDir):
    print("Loading embeddings.")
    fname = "{}/labels.csv".format(workDir)
    try :
        labels = pd.read_csv(fname, header=None).as_matrix()[:, 1]
    except :
        print("Error reading labels.csv")
        return
    labels = map(itemgetter(1),
                 map(os.path.split,
                     map(os.path.dirname, labels)))  # Get the directory.
    fname = "{}/reps.csv".format(workDir)
    try :
        embeddings = pd.read_csv(fname, header=None).as_matrix()
    except:
        print("Error reading reps.csv")
        return
    le = LabelEncoder().fit(labels)
    labelsNum = le.transform(labels)
    nClasses = len(le.classes_)
    print("Training for {} classes.".format(nClasses))

    clf =  MLPClassifier(hidden_layer_sizes=(100,), activation='logistic', max_iter=100, alpha=0.01,           #sigmoid fnc
                    solver='lbfgs', verbose=10, tol=1e-6, random_state=1,
                    learning_rate_init=.1)
    clf.fit(embeddings, labelsNum)

    fName = "{}/classifier.pkl".format(workDir)
    print("Saving classifier to '{}'".format(fName))
    with open(fName, 'w') as f:
        pickle.dump((le, clf), f)

# s = time.time()
# try:
#     os.remove(path_to_aligned + '/cache.t7')
# except OSError:
#     print 'Not found cache.t7, continue'
# list_person = os.listdir(path_to_aligned)
# for filename in list_person:
#     try:
#         if not os.listdir(path_to_aligned + '/' + filename):
#             shutil.rmtree(os.path.join(path_to_aligned, filename))
#     except OSError:
#         print 'non-folder'
# try:    #support CUDA - NVIDIA Graphics card
#     command = './../batch-represent/main.lua -outDir ' + path_to_features + ' -data ' + path_to_aligned + ' -cuda'
#     os.system(command)
# except: # without support CUDA
#     command = './../batch-represent/main.lua -outDir ' + path_to_features + ' -data ' + path_to_aligned
#     os.system(command) 
# train(path_to_features)
# print 'Training time: ' + str(round(time.time()-s,4))



# parser = argparse.ArgumentParser()
# parser.add_argument(
#         '--featureDir',
#         type=str,
#         help="Path to feature directory")
# args = parser.parse_args()
# if args.featureDir != None :
#     train(args.featureDir)
# else :
#     print("Enter feature directory")
