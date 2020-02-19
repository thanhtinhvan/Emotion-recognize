import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing import image

def loadImg(imgPath, grayScale=False, size=None):
    img = image.load_img(imgPath, grayScale, size)
    return image.img_to_array(img)

def apply_offsets(faceBox, offsets):
    x, y, width, height = faceBox
    x_off = offsets
    y_off = offsets
    return (x - x_off, x + width + x_off, y - y_off, y + height + y_off)

def drawBox(faceBox, img, color):
    x, y, w, h = faceBox
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 1)

def drawCircle(faceBox, img, color):
    x, y, w, h = faceBox
    centroid = (int((x+w/2)), int((y+h/2)))
    radius =  int(np.sqrt(w*w/4 + h*h/4) * 0.9)
    cv2.circle(img, centroid, radius, color, 1)

def drawElipse(faceBox, img, angle, color):
    x, y, w, h = faceBox
    centroid = (int((x+w/2)), int((y+h/2)))
    radius =  int(np.sqrt(w*w/4 + h*h/4) * 0.9)
    startAngle = -90
    endAngle = angle
    cv2.ellipse(img,centroid, (radius,radius),0, startAngle, endAngle, color,3)

def writeText(coordinates, image_array, text, color, x_offset=0, y_offset=0,
                                                font_scale=2, thickness=4):
    x, y, w, h = coordinates                                           
    #x, y = coordinates[:2]
    cv2.putText(image_array, text, (x + x_offset+int(w/2), y + y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, color, thickness)
    return image_array


def draw_box_label(img, faceBox, text, box_color=(0, 255, 255)):
    emotion=''
    name=''
    if len(text)>0:
        emotion = text[0]
        name = text[1]

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_size = 0.5
    font_color = (0, 0, 0)
    left, top, width, height = faceBox
    right = left+width
    bottom = top+height
    
    # Draw the bounding box
    cv2.rectangle(img, (left, top), (right, bottom), box_color, 2)
    
    # Draw a filled box on top of the bounding box (as the background for the labels)
    if name!='':
        cv2.rectangle(img, (left-2, top-40), (right+2, top), box_color, -1, 1)
    else:
        cv2.rectangle(img, (left-2, top-25), (right+2, top), box_color, -1, 1)
    # Output the labels that show the x and y coordinates of the bounding box center.
    # text_x= 'x='+str((left+right)/2)
    cv2.putText(img,name,(left,top-25), font, font_size, font_color, 1, cv2.CV_AA)
    cv2.putText(img,emotion,(left,top-5), font, font_size, font_color, 1, cv2.CV_AA)
    
