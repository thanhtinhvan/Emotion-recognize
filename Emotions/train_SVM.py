# This code is used to train a MLPClassifier
# You need to pass workDir to train() function
# The workDir should contain reps.csv and labesl.csv files
# You can change the MLP layer structure in MLPClassifier()

import sklearn
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import pandas as pd
import pickle
import os
from operator import itemgetter
from sklearn.preprocessing import LabelEncoder

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

    clf =  SVC(C=1, kernel = 'linear',probability=True)
    clf.fit(embeddings, labelsNum)

    fName = "{}/classifier.pkl".format(workDir)
    print("Saving classifier to '{}'".format(fName))
    with open(fName, 'w') as f:
        pickle.dump((le, clf), f)

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
