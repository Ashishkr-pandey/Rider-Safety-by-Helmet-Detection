from time import sleep
import time
# from controller import doorAutomate
# from cvzone.SerialModule import SerialObject
from utils import postprocess
import cv2 as cv
# used in mainloop  where we're extracting images., and then to drawPred( called by post process)
frame_count = 0
# used in post process loop, to get the no of specified class value.
frame_count_out = 0
# Initialize the parameters
confThreshold = 0.5  # Confidence threshold
nmsThreshold = 0.4  # Non-maximum suppression threshold
inpWidth = 416  # Width of network's input image
inpHeight = 416  # Height of network's input image

# Load names of classes
classesFile = "obj.names"

with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

# Give the configuration and weight files for the model and load the network using them.
modelConfiguration = "yolov3-obj.cfg"
modelWeights = "yolov3-obj_2400.weights"

net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
output_layer = net.getLayerNames()
output_layer = [output_layer[i - 1] for i in net.getUnconnectedOutLayers()]
cap = cv.VideoCapture('test.mp4')
# # for fn in glob('images/*.jpg'):
ptime = 0
while True:
    ret, frame = cap.read()
    frame = cv.flip(frame, 1)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

    # Create a 4D blob from a frame.
    blob = cv.dnn.blobFromImage(
        frame, 1/255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)

    # Sets the input to the network
    net.setInput(blob)

    # Runs the forward pass to get output of the output layers
    outs = net.forward(output_layer)

    # Remove the bounding boxes with low confidence
    postprocess(frame, outs, confThreshold, nmsThreshold, classes)
    ctime = time.time()
    fps = 1/(ctime-ptime)
    ptime = 0
    print(fps)
    cv.putText(frame, str(int(fps)), (10, 50),
               cv.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)
    cv.imshow('img', frame)
    t, _ = net.getPerfProfile()
    # print(t)
    label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())
    # print(label)
    cv.putText(frame, label, (0, 15),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
    # print(label)