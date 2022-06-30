import socket
import time
import sys
import tkinter
from tkinter import *
import math
import random
from threading import Thread 
from collections import defaultdict
from tkinter import ttk
from tkinter import filedialog
from multiprocessing import Queue
import matplotlib.pyplot as plt
import cv2
import socket
import struct
import time
import pickle
import zlib
import numpy as np
from PIL import Image

global mobile
global labels
global mobile_x
global mobile_y
global text, text1
global canvas
global mobile_list
global filename

encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]

queue = Queue()
click = False
shownOutput = False
files = []
global local_time
global offload_time
global extension_time

def calculateDistance(iot_x,iot_y,x1,y1):
    flag = False
    for i in range(len(iot_x)):
        dist = math.sqrt((iot_x[i] - x1)**2 + (iot_y[i] - y1)**2)
        if dist < 60:
            flag = True
            break
    return flag


def startMobileSimulation(mobile_x,mobile_y,canvas,text,mobile,labels):
    class SimulationThread(Thread):
        def __init__(self,mobile_x,mobile_y,canvas,text,mobile,labels): 
            Thread.__init__(self) 
            self.mobile_x = mobile_x
            self.mobile_y = mobile_y
            self.canvas = canvas
            self.text = text
            self.mobile = mobile
            self.labels = labels
 
        def run(self):
            while(True):
                for i in range(len(mobile_x)):
                    x = random.randint(10, 450)
                    y = random.randint(50, 600)
                    flag = calculateDistance(mobile_x,mobile_y,x,y)
                    if flag == False:
                        mobile_x[i] = x
                        mobile_y[i] = y
                        canvas.delete(mobile[i])
                        canvas.delete(labels[i])
                        name = canvas.create_rectangle(x,y,x+40,y+40, fill="blue")
                        lbl = canvas.create_text(x+20,y-10,fill="darkblue",font="Times 10 italic bold",text="Mobile "+str(i))
                        labels[i] = lbl
                        mobile[i] = name
                canvas.update()
                offload()
                if click == True and filename not in files:
                    print(filename)
                    #queue.put(filename)
                    files.append(filename)
                    
                time.sleep(4)
                   
                    
    newthread = SimulationThread(mobile_x,mobile_y,canvas,text,mobile,labels) 
    newthread.start()
    
    
def generate():
    global mobile
    global labels
    global mobile_x
    global mobile_y
    mobile = []
    mobile_x = []
    mobile_y = []
    labels = []

    for i in range(0,20):
        run = True
        while run == True:
            x = random.randint(10, 450)
            y = random.randint(50, 600)
            flag = calculateDistance(mobile_x,mobile_y,x,y)
            if flag == False:
                mobile_x.append(x)
                mobile_y.append(y)
                run = False
                name = canvas.create_rectangle(x,y,x+40,y+40, fill="blue")
                lbl = canvas.create_text(x+20,y-10,fill="darkblue",font="Times 10 italic bold",text="Mobile "+str(i))
                labels.append(lbl)
                mobile.append(name)
    startMobileSimulation(mobile_x,mobile_y,canvas,text,mobile,labels)

def offloadThread():
    global shownOutput
    global offload_time
    class OffloadThread(Thread):
        def __init__(self):
            Thread.__init__(self)
                        
        def run(self):
            global shownOutput
            global offload_time
            while not queue.empty():
                if shownOutput:
                    print(str(shownOutput))
                    filename = queue.get()
                    img = cv2.imread(filename)
                    result, img = cv2.imencode('.jpg', img, encode_param)
                    mid = int(mobile_list.get())
                    worker = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    worker.connect(('localhost', 8485))
                    data = pickle.dumps(img, 0)
                    size = len(data)
                    worker.sendall(struct.pack(">L", size) + data)
                    data = b""
                    payload_size = struct.calcsize(">L")
                    while len(data) < payload_size:
                        #print("Recv: {}".format(len(data)))
                        data += worker.recv(4096)
                    packed_msg_size = data[:payload_size]
                    data = data[payload_size:]
                    msg_size = struct.unpack(">L", packed_msg_size)[0]
                    print("msg_size: {}".format(msg_size))
                    start = time.time()
                    while len(data) < msg_size:
                        data += worker.recv(4096)
                    frame_data = data[:msg_size]
                    data = data[msg_size:]
                    frame=pickle.loads(frame_data, fix_imports=True, encoding="bytes")
                    frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
                    worker.close()
                    end = time.time()
                    click = False
                    files.clear()
                    offload_time = end - start
                    text1.insert(END,"Mobile Task Offloading execution time : "+str(offload_time)+"\n")
                    print("Mobile Task Offloading execution time : "+str(offload_time))
                    shownOutput = False
                    cv2.imshow("Detected Faces ", frame)
                    cv2.waitKey(0)
    if shownOutput:
        newthread = OffloadThread() 
        newthread.start()            

def offloadTask():
    global filename
    global click
    global shownOutput
    files.clear()
    filename = filedialog.askopenfilename(initialdir="images")
    queue.put(filename)
    click = True
    shownOutput = True
    offload()
    
def offload():
    global shownOutput
    global offload_time
    text.delete('1.0', END)
    offload_1 = 0
    offload_2 = 0
    selected = int(mobile_list.get())
    x = mobile_x[selected]
    y = mobile_y[selected]
    canvas.delete(labels[selected])
    lbl = canvas.create_text(x+20,y-10,fill="red",font="Times 10 italic bold",text="Mobile "+str(selected))
    labels[selected] = lbl                
    distance = 300
    for i in range(len(mobile_x)):
        if i != selected:
            x1 = mobile_x[i]
            y1 = mobile_y[i]
            dist = math.sqrt((x - x1)**2 + (y - y1)**2)
            if dist < distance:
                offload_2 = offload_1
                offload_1 = i
                distance = dist
                text.insert(END,"Mobile "+str(i)+" is in proximity of source "+str(selected)+"\n")
    text.insert(END,"\n\nCurrent Selected Mobile Worker to offload = "+str(offload_1))            
    if click:
        offloadThread() 


def localRun():
    global local_time
    text1.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir="images")
    local_time = time.time()
    cascPath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascPath)
    frame = cv2.imread(filename)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray,scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags = cv2.CASCADE_SCALE_IMAGE)
    print("Found {0} faces!".format(len(faces)))
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    local_time = time.time() - local_time
    text1.insert(END,"Local Task running time : "+str(local_time)+"\n")
    print("Local Task running time : "+str(local_time))
    cv2.imshow("Num Faces found ", frame)
    
    cv2.waitKey(0)



def extensionoffloadThread():
    global shownOutput
    global extension_time
    class OffloadThread(Thread):
        def __init__(self):
            Thread.__init__(self)
                        
        def run(self):
            global shownOutput
            global extension_time
            while not queue.empty():
                if shownOutput:
                    print(str(shownOutput))
                    filename = queue.get()
                    img = Image.open(filename)
                    img.save("test.jpg",optimize=True,quality=10)
                    img = cv2.imread("test.jpg")
                    result, img = cv2.imencode('.jpg', img, encode_param)
                    mid = int(mobile_list.get())
                    worker = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    worker.connect(('localhost', 8485))
                    data = pickle.dumps(img, 0)
                    size = len(data)
                    worker.sendall(struct.pack(">L", size) + data)
                    data = b""
                    payload_size = struct.calcsize(">L")
                    while len(data) < payload_size:
                        #print("Recv: {}".format(len(data)))
                        data += worker.recv(4096)
                    packed_msg_size = data[:payload_size]
                    data = data[payload_size:]
                    msg_size = struct.unpack(">L", packed_msg_size)[0]
                    print("msg_size: {}".format(msg_size))
                    start = time.time()
                    while len(data) < msg_size:
                        data += worker.recv(4096)
                    frame_data = data[:msg_size]
                    data = data[msg_size:]
                    frame=pickle.loads(frame_data, fix_imports=True, encoding="bytes")
                    frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
                    worker.close()
                    end = time.time()
                    click = False
                    files.clear()
                    extension_time = end - start
                    text1.insert(END,"Mobile Extension Compress Task Offloading execution time : "+str(extension_time)+"\n")
                    print("Mobile Extension Compress Task Offloading execution time : "+str(extension_time))
                    shownOutput = False
                    cv2.imshow("Detected Faces ", frame)
                    cv2.waitKey(0)
    if shownOutput:
        newthread = OffloadThread() 
        newthread.start()  

def extensionOffloadTask():
    global click
    global shownOutput
    global extension_time
    queue.put(filename)
    click = True
    shownOutput = True
    extensionoffloadThread()
    

def graph():
    print(str(local_time)+" "+str(offload_time)+" "+str( extension_time))
    height = [local_time,offload_time, extension_time]
    bars = ('Local Task Execution Time', 'Offload Task Execution Time','Extension Compress Task Time')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.show()

def Main():
    global text
    global text1
    global canvas
    global mobile_list
    root = tkinter.Tk()
    root.geometry("1300x1200")
    root.title("Computing with Nearby Mobile Devices: a Work Sharing Algorithm for Mobile Edge-Clouds")
    root.resizable(True,True)
    font1 = ('times', 12, 'bold')

    canvas = Canvas(root, width = 800, height = 700)
    canvas.pack()

    l1 = Label(root, text='Mobile ID:')
    l1.config(font=font1)
    l1.place(x=850,y=10)

    mid = []
    for i in range(0,20):
        mid.append(str(i))
    mobile_list = ttk.Combobox(root,values=mid,postcommand=lambda: mobile_list.configure(values=mid))
    mobile_list.place(x=1000,y=10)
    mobile_list.current(0)
    mobile_list.config(font=font1)

    createButton = Button(root, text="Generate Mobile Devices", command=generate)
    createButton.place(x=850,y=60)
    createButton.config(font=font1)

    localButton = Button(root, text="Local Run Task", command=localRun)
    localButton.place(x=850,y=110)
    localButton.config(font=font1)

    offloadButton = Button(root, text="Offload Task", command=offloadTask)
    offloadButton.place(x=1000,y=110)
    offloadButton.config(font=font1)

    extensionButton = Button(root, text="Extension Compress & Offload Task", command=extensionOffloadTask)
    extensionButton.place(x=850,y=160)
    extensionButton.config(font=font1)

    graphButton = Button(root, text="Task Execution Time Graph", command=graph)
    graphButton.place(x=1130,y=160)
    graphButton.config(font=font1)


    text=Text(root,height=15,width=60)
    scroll=Scrollbar(text)
    text.configure(yscrollcommand=scroll.set)
    text.place(x=750,y=200)

    text1=Text(root,height=10,width=70)
    scroll1=Scrollbar(text1)
    text1.configure(yscrollcommand=scroll1.set)
    text1.place(x=750,y=490)
    
    
    root.mainloop()
   
 
if __name__== '__main__' :
    Main ()
    
