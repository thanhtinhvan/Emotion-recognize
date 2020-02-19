import time
import cv2
import dlib
import numpy as np



def getFaces(bbs, multiple=False,scale=1):
    # if grayImg is None:
    #     return []
    # rgbImg = grayImg
    # if multiple:
    #     ds_grayImg = cv2.resize(grayImg, (0,0), fx=scale, fy=scale)
    #     s = time.time()
    #bbs = align.getAllFaceBoundingBoxes(ds_grayImg)
    #     #bbs = detector(ds_grayImg,1)
    #     #bbs = cnn_face_detector(ds_grayImg,1)
    #     # s = time.time()
    bbs = []
    import face_recognition
    bbs = face_recognition.face_locations(ds_grayImg,number_of_times_to_upsample=0, model="cnn")
    #     print 'detect: ' + str(time.time() - s)
    #     #bbs = face_recognition.batch_face_locations(ds_grayImg, number_of_times_to_upsample=0)
    #     #print bbs
    # else:
    #     scale = 0.5
    #     ds_grayImg = cv2.resize(grayImg, (0,0), fx=scale, fy=scale)
    #     bb1 = align.getLargestFaceBoundingBox(ds_grayImg)
    #     bbs = [bb1]

    if len(bbs) == 0 or (not multiple and bb1 is None):
        return []

    reps = []
    faces = []

    for bb in bbs:
    #print 'bb before',bb.left()
        us_factor = long((1/scale))
        if multiple:
            bb = dlib.rectangle(bb[3]*us_factor, bb[0]*us_factor, bb[1]*us_factor, bb[2]*us_factor)
        else:
            bb = dlib.rectangle(bb.left()*us_factor , bb.top()*us_factor, bb.right()*us_factor, bb.bottom()*us_factor)

        x = int(bb.left())
        y = int(bb.top())
        width = int(bb.right()-x)
        height = int(bb.bottom()-y)
        face = [x,y,width,height]
        faces.append(face)

    return faces            #x,y,width,height






































# import dlib

# dlib_model_path = './datas/dlib/shape_predictor_68_face_landmarks.dat'

# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor(dlib_model_path)

# def detectFaces(grayImg, multiple=True):   #multi: T(all faces), F(biggest face)
#     faces = []
#     if multiple:
#         #bbs = detector(grayImg, 0)
#         import face_recognition
#         bbs = face_recognition.face_locations(grayImg,number_of_times_to_upsample=0, model="cnn")
#     if len(bbs)==0:
#         return faces

#     for bb in bbs:
#         bb = dlib.rectangle(bb[3], bb[0], bb[1], bb[2])
#         x = int(bb.left())
#         y = int(bb.top())
#         width = int(bb.right()-x)
#         height = int(bb.bottom()-y)
#         face = [x,y,width,height]
#         faces.append(face)
#     return faces
