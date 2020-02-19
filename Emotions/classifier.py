import face_recognition
import time
import cv2
import os
import pickle
import dlib
import numpy as np
import pandas as pd
import tkMessageBox
from preProcessDatas import getLabels, preProcessInput

from process import apply_offsets

# from keras.models import load_model
# emotions_model = './datas/emotions/full/fer2013_CNN.45-0.66.hdf5'
#emotions_model = './datas/emotions/fer2013_CNN.hdf5'
#emotions_model = './datas/emotions/full/fer2013_mini_XCEPTION.102-0.66.hdf5'
emotions_model = './datas/emotions/CNN_5Class2.hdf5'
emotionLabels = getLabels('fer2013')
# emotionClassifier = load_model(emotions_model, compile=False)
# box_size  = emotionClassifier.input_shape[1:3]
emotions = []

import openface

modelDir = './datas/faces/models'
classifierModel = './datas/faces/features/classifier.pkl'
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')
align = openface.AlignDlib(os.path.join(dlibModelDir, "shape_predictor_68_face_landmarks.dat"))
net = openface.TorchNeuralNet(os.path.join(openfaceModelDir, 'nn4.small2.v1.t7'), 96)

detector = dlib.get_frontal_face_detector()
        # self.cnn_face_detection_model = face_recognition_models.cnn_face_detector_model_location()
        # self.cnn_face_detector = dlib.cnn_face_detection_model_v1(self.cnn_face_detection_model)

class var:
    emotionClassifier = None

def getFaces(grayImg, multiple=False,scale=1):
    if grayImg is None or grayImg == []:
        return None, None, None
    # if scale<1:
    scale_grayImg = cv2.resize(grayImg, (0,0), fx=scale, fy=scale)
    #start = time.time()
    if multiple:
        #bbs = align.getAllFaceBoundingBoxes(scale_grayImg)
        bbs = detector(scale_grayImg,1)
        # import face_recognition
        # bbs = face_recognition.face_locations(scale_grayImg,number_of_times_to_upsample=0, model="cnn")
    else:
        
        bbs = [align.getLargestFaceBoundingBox(scale_grayImg)]
        print bbs
    #print 'Face Alignment took :'+str(time.time()-start)+'secs'
    if len(bbs) == 0 or bbs == [None]:
        return None, None, None

    reps = []
    alignedFace = None
    faces = []
    for bb in bbs:
    #print 'bb before',bb.left()
        us_factor = long((1/scale))
        # if multiple:
        #     bb = dlib.rectangle(bb[3]*us_factor, bb[0]*us_factor, bb[1]*us_factor, bb[2]*us_factor)
        # else:
        bb = dlib.rectangle(bb.left()*us_factor , bb.top()*us_factor, bb.right()*us_factor, bb.bottom()*us_factor)
        
        alignedFace = align.align(
                96,
                grayImg,
                bb,
                landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
        if alignedFace is None:
            continue
        rep = net.forward(alignedFace)
        reps.append((bb.center().x, rep))

        x = int(bb.left())
        y = int(bb.top())
        width = int(bb.right()-x)
        height = int(bb.bottom()-y)
        face = [x,y,width,height]
        for i in range(len(face)):
            if face[i]<0:
                face[i]=0
        faces.append(face)
    sreps = sorted(reps, key=lambda x: x[0])
    return faces, sreps, alignedFace  #face include (x,y,w,h), reps used for infer


def getEmotions(grayImg, face):
    if var.emotionClassifier is None:
        from keras.models import load_model
        var.emotionClassifier = load_model(emotions_model, compile=False)
        del load_model
        #emotions_model = './datas/emotions/fer2013_CNN.31-0.62.hdf5'
    box_size  = var.emotionClassifier.input_shape[1:3]
    

    #print face
    x1, x2, y1, y2 = apply_offsets(face, 0)
    grayFace = grayImg[y1:y2, x1:x2]
    try:
        grayFace = cv2.resize(grayFace, box_size)
    except:
        print 'not gray img'
        return None, None
    
    grayFace = preProcessInput(grayFace, True)
    grayFace = np.expand_dims(grayFace, 0)
    grayFace = np.expand_dims(grayFace, -1)
    emptionPredict = var.emotionClassifier.predict(grayFace)
    
    emotionIdex = np.argmax(emptionPredict)
    emotionConfidence= emptionPredict[0][emotionIdex]
    print '************'
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
    return emotionText, emotionConfidence, color
    
def testEmotions(grayImg, face):
    if var.emotionClassifier is None:
        from keras.models import load_model
        var.emotionClassifier = load_model(emotions_model, compile=False)
        del load_model
        #emotions_model = './datas/emotions/fer2013_CNN.31-0.62.hdf5'
    box_size  = var.emotionClassifier.input_shape[1:3]
    

    #print face
    x1, x2, y1, y2 = apply_offsets(face, 0)
    grayFace = grayImg[y1:y2, x1:x2]
    try:
        grayFace = cv2.resize(grayFace, box_size)
    except:
        print 'not gray img'
        return None, None
    
    grayFace = preProcessInput(grayFace, True)
    grayFace = np.expand_dims(grayFace, 0)
    grayFace = np.expand_dims(grayFace, -1)
    emptionPredict = var.emotionClassifier.predict(grayFace)
    emotionIdex = np.argmax(emptionPredict)
    emotionText = emotionLabels[emotionIdex]
    return emotionText, emotionIdex

def infer(reps, multiple=True):
    people=[]
    confidences = []

    # try :
    with open(classifierModel, 'r') as f:
        (le, clf) = pickle.load(f)
    # except :
    #     print 'error in opening classfier'

#     reps , alignedFace ,grayImg , bbs = self.getRep(img, multiple )
    rect = []
    if reps == None :
        return None,None        # name + confidence

    #for r in reps :
    rep = reps[1].reshape(1, -1)
    bbx = reps[0]
    try:
        predictions = clf.predict_proba(rep).ravel()        #dung ham softmax
    except :
        print('error in classifier prediction')
        people.append("Error")
        confidences.append(0)
        return people, confidences
        #continue

    maxI = np.argmax(predictions)
    person = le.inverse_transform(maxI)
    confidence = predictions[maxI]

    confidences.append(confidence)
    if confidence > 0.2:
        people.append(person)
    else:
        people.append('unknown')

    return people, confidences

def Train():
    if var.emotionClassifier is None:
        
        path_to_features = './datas/faces/features'
        path_to_aligned = './datas/faces/aligned'
        import train_MLP
        import train_SVM
        s = time.time()
        try:
            os.remove(path_to_aligned + '/cache.t7')
        except OSError:
            print 'Not found cache.t7, continue'
        list_person = os.listdir(path_to_aligned)
        for filename in list_person:
            try:
                if not os.listdir(path_to_aligned + '/' + filename):
                    shutil.rmtree(os.path.join(path_to_aligned, filename))
            except OSError:
                print 'non-folder'
        try:    #support CUDA - NVIDIA Graphics card
            command = './../batch-represent/main.lua -outDir ' + path_to_features + ' -data ' + path_to_aligned + ' -cuda'
            os.system(command)
        except: # without support CUDA
            command = './../batch-represent/main.lua -outDir ' + path_to_features + ' -data ' + path_to_aligned
            os.system(command) 
        train_MLP.train(path_to_features)
        # train_SVM.train(path_to_features)
        print 'Training time: ' + str(round(time.time()-s,4))
        tkMessageBox.showinfo(title="Notification", message='Training completed')
    else:
        #print var.emotionClassifier
        tkMessageBox.showinfo(title="Warning", message='Please restart this application for training!')
        #print 'Please restart this application for training!!'