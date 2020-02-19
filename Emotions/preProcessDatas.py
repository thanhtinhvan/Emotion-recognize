import numpy as np
import pandas as pd
import os
import cv2
from scipy.io import loadmat

class processDatas(object):
    def __init__(self, dataset_name='imdb', path_to_data=None, img_size=(48,48)):
        self.dataset_name = dataset_name
        self.path_to_data = path_to_data
        self.img_size = img_size
        if self.path_to_data != None:
            self.path_to_data = path_to_data
        elif self.dataset_name == 'imdb':
            self.path_to_data = './datas/imdb_crop/imdb.mat'
        elif self.dataset_name == 'fer2013':
            self.path_to_data = './datas/fer2013/fer2013_5class.csv'
        elif self.dataset_name == 'KDEF':
            self.path_to_data = './datas/KDEF/'
        elif self.dataset_name == 'ferTest':
            self.path_to_data = './emotion_test.csv'
        else:
            raise Exception('Incorrect data')

    def get_data(self):
        if self.dataset_name == 'fer2013' or self.dataset_name == 'ferTest':
            data = self.get_fer()
        elif self.dataset_name == 'imdb':
            data = self.get_imdb()
        elif self.dataset_name == 'KDEF':
            data = self.load_KDEF()
        return data

    def get_fer(self):
        data = pd.read_csv(self.path_to_data)
        pixels = data['pixels'].tolist()
        width, height = 48, 48
        faces = []
        for pixel in pixels:
            face = [int(pix) for pix in pixel.split(' ')]
            face = np.asarray(face).reshape(width, height)
            face = cv2.resize(face.astype('uint8'), self.img_size)
            faces.append(face.astype('float32'))
        faces = np.asarray(faces)
        faces = np.expand_dims(faces, -1)
        emotions = pd.get_dummies(data['emotion']).as_matrix()
        return faces, emotions

    def get_imdb(self):
        faceTreshold = 3
        dataset = loadmat(self.path_to_data)
        img_array = dataset['imdb']['full_path'][0, 0][0]
        gender = dataset['imdb']['gender'][0, 0][0]
        faceScore = dataset['imdb']['face_score'][0, 0][0]
        faceScore2 = dataset['imdb']['second_face_score'][0, 0][0]
        faceScoreFilter = faceScore > faceTreshold
        faceScoreFilter2 = np.isnan(faceScore2)
        unGenderFilter = np.logical_not(np.isnan(gender))
        mask = np.logical_and(faceScoreFilter, faceScoreFilter2)
        mask = np.logical_and(mask, unGenderFilter)
        img_array = img_array[mask]
        gender = gender[mask].tolist()
        img_names = []
        for img_arg in range(img_array.shape[0]):
            img_name = img_array[img_arg][0]
            img_names.append(img_name)
        return dict(zip(img_names, gender))




def getLabels(dataName):
    if dataName == 'fer2013':
        return {0:'angry',1:'happy',2:'sad',3:'surprise',4:'neutral'}
    elif dataName == 'imdb':
        return {0:'woman', 1:'man'}
    elif dataName == 'KDEF':
        return {0:'AN', 1:'DI', 2:'AF', 3:'HA', 4:'SA', 5:'SU', 6:'NE'}
    else:
        raise Exception('Invalid dataset name')

def preProcessInput(x, v2=True):
    x = x.astype('float32')
    x = x / 255.0
    if v2:
        x = x - 0.5
        x = x * 2.0
    return x

def splitData(x, y, validation_split=.2):
    num_samples = len(x)
    num_train_samples = int((1 - validation_split)*num_samples)
    train_x = x[:num_train_samples]
    train_y = y[:num_train_samples]
    val_x = x[num_train_samples:]
    val_y = y[num_train_samples:]
    train_data = (train_x, train_y)
    val_data = (val_x, val_y)
    return train_data, val_data



