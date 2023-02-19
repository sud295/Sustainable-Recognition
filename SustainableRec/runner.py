import cv2
from cv2 import *
import numpy as np
import time

picture = cv2.VideoCapture(0)
time.sleep(3)
a, inp = picture.read()
cv2.imwrite("image1.jpg",inp)

net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

classes = []
with open("coco.names", "r") as f:
    classes = f.read().splitlines()

# Load an image
img = cv2.imread("image1.jpg")
scaled_img = cv2.dnn.blobFromImage(img,1/255, (416,416), (0,0,0), swapRB = True, crop = False)

net.setInput(scaled_img)

output_layer_names = net.getUnconnectedOutLayersNames()
out_layer = net.forward(output_layer_names)

boxes = []
confidence_list = []
objects = []

for i in out_layer:
    for j in i:
        score = j[5:]
        class_id = np.argmax(score)
        confidence = score[class_id]
        if confidence > 0.75:
            center_x = int(j[0]*400)
            center_y = int(j[0]*400)
            x = int(center_x-center_x/2)
            y = int(center_y-center_y/2)
            boxes.append([x,y,center_x,center_y])
            confidence_list.append(float(confidence))
            objects.append(class_id)

print(classes[objects[0]])


maps = {"bicycle":0, "car":1, "motorbike":2, "aeroplane":3, "bus":4, "train":5, "truck":6, "boat":7, "banana":8, "laptop":9, "cell phone":10}
lis = []
f2 = open('cons.txt', 'r')
for i in f2:
    lis.append(i)
try:
    print(lis[maps[classes[objects[0]]]])
except:
    print("N/A")