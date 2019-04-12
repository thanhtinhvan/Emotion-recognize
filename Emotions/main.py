import os
command = 'killall -9 luajit'
os.system(command)
import classifier
import cv2
import numpy as np
import threading

import shutil
from PIL import Image, ImageTk
import time
import multiprocessing
from tkFileDialog import askopenfilename
from Tkinter import *
import Tkinter
from PIL import Image, ImageTk
import tkMessageBox

from process import apply_offsets, drawBox, writeText, drawCircle, drawElipse, draw_box_label

#-------------------------------------------------------------------------constant variables 
font12 = "-family {DejaVu Sans} -size 12 -weight normal -slant "  \
            "roman -underline 0 -overstrike 0"
img_background = "#b1d9db"

path_to_features = './datas/faces/features'
path_to_aligned = './datas/faces/aligned'
durable_frame = 5
thresholdFace = 0.75
frameQueue = multiprocessing.Queue(maxsize=1)
resultQueue = multiprocessing.Queue(maxsize=1)

#-------------------------------------------------------------------------global variables
class variables:                                             
    cap = cv2.VideoCapture(0)
    recog_enable = False
    grayFrame = []
    recog_results = [None, None, None]
    RGBframe = []
    play = True
    cnt = 0     # each face detect will show in 10 frames (count = 10)
    color = (0,255,0)
    persons = None
    confidences = None
    faces = None
    reps = None
    alignedFace = None
    last_faces = []
    last_reps = []
    exit = False
    suppend = False
    add_name = 'tmp'
    inputFromTexbox = ''
    clean = False
    #mode
    mode_file = False
    mode_camera = True
    mode_add = False
    mode_recognize = False


    #add
    add_samples = 150
    add_cnt = 0
    add_percents = 0

    #browser
    browser_path = None
    browser_last_path = None
    #GUI
    top = Tk()
    PanelImg = LabelFrame(top)
    lbImg = Label(PanelImg)
    # ckGPU = Checkbutton(PanelImg)
    PanelControl = LabelFrame(top)
    btnStart = Button(PanelControl)
    btnBrowser = Button(PanelControl)
    btnAdd = Button(PanelControl)
    btnTrain = Button(PanelControl)
    btnRemove = Button(PanelControl)
    btnExit = Button(PanelControl)

    ww = top.winfo_screenwidth()
    wh = top.winfo_screenheight()
    top.title('Face and Emotions recognize')
    print 'resolution: ' + str(ww) + 'x' + str(wh)
    top.geometry( '%dx%d+%d+%d' % (ww,int(wh*0.95),ww,wh))

    w_panel_img = int(ww*3/4.3)
    h_panel_img = int(wh*0.9)
    w_img = int(w_panel_img*0.98)
    h_img = int(h_panel_img*0.9)
    x_ck = 0.1
    y_ck = 0.93
    w_panel = int(ww/4)
    h_panel = int(wh*0.9)
    w_btn = int(ww/12)
    x_btn = 0.3
    y_btn = 0.1
    y_btn_step = 0.1

def msgInput():
    def getInput(event=None):
            tmp = str(textBox.get())
            #print '---' + str(tmp)
            print len(tmp)
            if len(tmp) >1:
                print 'in------------'
                variables.inputFromTexbox = tmp#str(textBox.get())
            else:
                variables.inputFromTexbox = None
            txtBox.quit()

    font10 = "-family {DejaVu Sans} -size 0 -weight normal -slant "  \
        "roman -underline 0 -overstrike 0"
    txtBox = Toplevel()
    txtBox.title('Input name')
    txtBox.geometry("687x61+461+458")
    Label1 = Label(txtBox)
    Label1.place(relx=0.22, rely=0.16, height=38, width=116)
    Label1.configure(font=font10, text='''Input name:''')
    textBox = Entry(txtBox)
    textBox.place(relx=0.38, rely=0.16, relheight=0.66, relwidth=0.42)
    textBox.configure(background="white")
    textBox.configure(font="TkFixedFont")
    textBox.configure(width=286)
    textBox.bind('<Return>', getInput)
    txtBox.mainloop()
    txtBox.destroy()

def choose_mode(mode):
    variables.faces = None
    variables.last_faces = []
    variables.last_reps = []
    variables.cnt = 0
    if mode == 'file':
        variables.mode_camera = False
        variables.mode_file = True
        variables.mode_recognize = True
        variables.mode_add = False
    else:
        variables.mode_camera = True
        if mode == 'recognize':
            variables.mode_add = False
            variables.mode_recognize = True
        if mode == 'add':
            variables.mode_recognize = False
            variables.mode_add = True
        if mode == 'none':
            variables.mode_add = False
            variables.mode_recognize = False
    

def fncPlayback():
        #variables.play = True
        choose_mode('recognize')
        #variables.btnTrain['state'] = 'disabled'
    
def fncAdd():
    msgInput()
    if variables.inputFromTexbox is not None:
        variables.add_name = variables.inputFromTexbox
        variables.btnAdd.configure(text='Adding')
        variables.btnAdd['state'] = 'disabled'
        variables.btnBrowser['state'] = 'disabled'
        variables.btnStart['state'] = 'disabled'
        variables.btnRemove['state'] = 'disabled'
        variables.btnTrain['state'] = 'disabled'
        choose_mode('add')
    else:
        print 'Please input name'

def fncRemove():
        list_person = os.listdir(path_to_aligned)
        try:
            list_person.remove('unknown')
            list_person.remove('cache.t7')
        except ValueError:
            pass
        print list_person
        msgInput()
        if variables.inputFromTexbox is not None:
            try:
                shutil.rmtree(os.path.join(path_to_aligned,variables.inputFromTexbox))
                tkMessageBox.showinfo(title="Notification", message='Removed. Click [Train] button to update database.')
            except OSError:
                tkMessageBox.showinfo(title="Warning", message='Check again! This person don''t have in database!')

def fncClose():
    variables.exit = True

def fncBrowser():
    #variables.clean = True
    #variables.browser_path = askopenfilename() #'./test/cc.jpg' #askopenfilename()
    #print variables.browser_path
    variables.browser_path = './emotions test/none2.jpg'
    if variables.browser_path is not None:
        choose_mode('file')
        variables.browser_path = askopenfilename()

def fncTrain():
    classifier.Train()

def congig_GUI():
    variables.PanelImg.place(relx=0.01, rely=0.02, height=variables.h_panel_img, width=variables.w_panel_img)

    variables.lbImg.place(relx=0.01, rely=0.01, height=variables.h_img, width=variables.w_img)
    variables.lbImg.configure(background=img_background)

    # variables.ckGPU.place(relx=variables.x_ck, rely=variables.y_ck, width=variables.w_btn )
    # variables.ckGPU.configure(text='GPU', font=font12)

    variables.PanelControl.place(relx = 0.72, rely=0.02, height=variables.h_panel, width=variables.w_panel)

    variables.btnStart.place(relx=variables.x_btn, rely = variables.y_btn, width=variables.w_btn)
    variables.btnStart.configure(text='Cameras', font=font12, command=fncPlayback)
    variables.y_btn += variables.y_btn_step

    variables.btnBrowser.place(relx=variables.x_btn, rely = variables.y_btn, width=variables.w_btn)
    variables.btnBrowser.configure(text='Browser', font=font12, command=fncBrowser)
    variables.y_btn += variables.y_btn_step

    variables.btnAdd.place(relx=variables.x_btn, rely=variables.y_btn, width=variables.w_btn)
    variables.btnAdd.configure(text='Add person', font=font12, command=fncAdd)
    variables.y_btn += variables.y_btn_step

    variables.btnRemove.place(relx=variables.x_btn, rely=variables.y_btn, width=variables.w_btn)
    variables.btnRemove.configure(text='Remove person', font=font12, command=fncRemove)
    variables.y_btn += variables.y_btn_step

    variables.btnTrain.place(relx=variables.x_btn, rely=variables.y_btn, width=variables.w_btn)
    variables.btnTrain.configure(text='Train', font=font12, command=fncTrain)
    variables.y_btn += variables.y_btn_step

    variables.btnExit.place(relx=variables.x_btn, rely=variables.y_btn, width=variables.w_btn)
    variables.btnExit.configure(text='Exit', font=font12, command=fncClose)
    variables.y_btn += variables.y_btn_step

    variables.top.protocol('WM_DELETE_WINDOW', fncClose)


##################################################################################
#                                 MAIN PROGRAM                                   #
##################################################################################  

def recognize(frameQueue, resultQueue):     #return face 
    result = []
    resultQueue.put(result)
    while True:
        if not frameQueue.empty():
            # s = time.time()x
            data = frameQueue.get()
            recv = data[0]
            try:
                mode_add = data[1]
            except IndexError:
                pass
            if type(recv) == str:
                print 'exit'
                break
            else:
                while not frameQueue.empty():
                    print 'clean'
                    frameQueue.get()
                if mode_add:
                    print 'adding'
                    faces, reps, alignedFace = classifier.getFaces(recv,False, scale=0.25)
                else:
                    print 'mode add value: ' + str(mode_add)
                    faces, reps, alignedFace = classifier.getFaces(recv,True, scale=1)
                result = [faces, reps, alignedFace]
                if resultQueue.empty():
                    resultQueue.put(result)
                # print 'detect: ' + str(time.time() - s)
    print 'out process frames'

def add_person(alignedFace, name):
    path_to_trainDb = os.path.join(path_to_aligned,name)
    if not os.path.exists(path_to_trainDb):
        os.makedirs(path_to_trainDb)
    if alignedFace is not None:
        nameFile = path_to_trainDb + '/' + name + '_' + str(variables.add_cnt) + '.jpeg'
        cv2.imwrite(nameFile, alignedFace)
        variables.add_cnt += 1
    percents = (variables.add_cnt*100/variables.add_samples)
    if variables.add_cnt >= variables.add_samples:
        variables.add_cnt = 0
        choose_mode('none')
        print 'Add Done'
        variables.btnAdd.configure(text='Add person')
        variables.btnBrowser['state'] = 'active'
        variables.btnStart['state'] = 'active'
        variables.btnTrain['state'] = 'active'
        variables.btnAdd['state'] = 'active'
        variables.btnRemove['state'] = 'active'

    print percents
    return percents


# run multi process
recog = multiprocessing.Process(target=recognize ,args=(frameQueue,resultQueue,))
recog.start()

# main function
def playback():
    ret = None
    if variables.clean:
        while not resultQueue.empty():
            resultQueue.get()
        while not frameQueue.empty():
            frameQueue.get() 
        #variables.cnt = 0
        variables.clean = False 
        print 'this l'
    if variables.mode_camera:
        ret, frame = variables.cap.read()
        frame = cv2.flip(frame,1)
    elif variables.mode_file:# and (variables.browser_last_path != variables.browser_path):
        ret = True
        frame = cv2.imread(variables.browser_path)
        frame = cv2.resize(frame, (1280,768))
        #variables.browser_last_path = variables.browser_path

    if ret is not None:
        
        if not resultQueue.empty():
            result = resultQueue.get()
            try:
                variables.faces = result[0]
                variables.reps = result[1]
                variables.alignedFace = result[2]
            except IndexError:
                print 'list out of range'
            while not resultQueue.empty():
                resultQueue.get()
            #grayImg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            putData = [frame,variables.mode_add]
            frameQueue.put(putData)

        if variables.faces is not None:
            variables.last_faces = variables.faces
            variables.last_reps = variables.reps
            variables.cnt = durable_frame
        else:
            pass
            #print 'non face'
        #---------------------mode ---------------------------------

        if variables.mode_recognize:
            if variables.cnt>0:
                #i=0
                emotionConfidence = 0
                grayImg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                for i in range(len(variables.last_faces)):
                    #print face
                    face = variables.last_faces[i]
                    print face
                    emotionText,emotionConfidence, color = classifier.getEmotions(grayImg,face)
                    print emotionConfidence
                    emotion_info = emotionText.title()+' - ' + str(round(emotionConfidence,1))
                    #print last_reps[i][1]
                    if variables.mode_camera:
                        name, confidence = classifier.infer(variables.last_reps[i])
                    else:
                        name=['unknown']
                        confidence = [0]
                    #print name
                    if name[0]!= 'unknown' and confidence[0]>0.6:
                        face_text = name[0]  + ' - ' + str(round(confidence[0],3))
                        print face_text
                    else:
                        face_text=''
                        print name[0]  + ' - ' + str(round(confidence[0],3))
                    face_info = [emotion_info, face_text]
                    draw_box_label(frame, face, face_info,color)
                    # frame = writeText(face, frame, face_text, color, 0, -20, 0.5, 1)
                    # frame = writeText(face, frame, emotionText, color, 0, -5, 0.5, 1)
                    # drawBox(face, frame, color)
                    #i += 1
                print '----------------------------\n'
                

                variables.cnt -= 1

        elif variables.mode_add:
            print 'mode_add'
            if variables.cnt == durable_frame:
                variables.add_percents = add_person(variables.alignedFace, variables.add_name)
                print 'add***'
            if variables.cnt>0:
                for face in variables.last_faces:
                    drawCircle(face, frame, (255,255,255))
                    drawElipse(face, frame, int(variables.add_percents*3.6)-90, (0,255,0))
                variables.cnt -= 1



        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (variables.w_img,variables.h_img))
        img_array = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img_array)
        variables.lbImg.imgtk = imgtk
        variables.lbImg.configure(image=imgtk)
    


    if variables.exit == False:
        variables.lbImg.after(1, playback)
    else:
        while not resultQueue.empty():
            resultQueue.get()
        frameQueue.put('q')
        time.sleep(1)
        variables.top.quit()

congig_GUI()

playback()


variables.top.mainloop()
print 'Closed'




