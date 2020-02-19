import cv2
import numpy as np
import os 
from preProcessDatas import getLabels, preProcessInput
from keras.models import load_model
from process import apply_offsets


emotions_model = './datas/emotions/fer2013_CNN.31-0.62.hdf5'
emotionLabels = getLabels('fer2013')
emotionClassifier = load_model(emotions_model, compile=False)
box_size  = emotionClassifier.input_shape[1:3]
emotions = []

def getEmotions(grayImg, face):
    x1, x2, y1, y2 = apply_offsets(face, 0)
    grayFace = grayImg[y1:y2, x1:x2]
    # cv2.imshow('gray', grayFace)
    # while True:
    #     if cv2.waitKey(1)&0xFF == ord('q'):
    #         break
    try: 
        grayFace = cv2.resize(grayFace, box_size)
    except:
        print 'not gray img'
    
    grayFace = preProcessInput(grayFace, True)
    grayFace = np.expand_dims(grayFace, 0)
    grayFace = np.expand_dims(grayFace, -1)
    emptionPredict = emotionClassifier.predict(grayFace)
    emotionIdex = np.argmax(emptionPredict)
    emotionText = emotionLabels[emotionIdex]
    if emotionText == 'neural':
        color = np.max(emptionPredict) * np.asarray((255,0,0))     #red (BGR)
    elif emotionText == 'sad':
        color = np.max(emptionPredict) * np.asarray((0, 0, 255))   #red
    elif emotionText == 'happy':
        color = np.max(emptionPredict) * np.asarray((0, 255, 0)) # green
    elif emotionText == 'surprise':
        color = np.max(emptionPredict) * np.asarray((0, 255, 255)) # green + red
    else:
        color = np.max(emptionPredict) * np.asarray((255, 0, 0))   #blue

    color = color.astype(int)
    color = color.tolist()
    return emotionText, color
    