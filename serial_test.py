# USAGE
# python real_time_object_detection.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel

# import the necessary packages
# from imutils.video import VideoStream
from imutils.video import FPS
# from imutils.video import VideoStream
# from imutils.video import FileVideoStream
# from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2
import math
import os
import threading
from imutils import paths
import face_recognition
import pickle

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.2,
	help="minimum probability to filter weak detections")
ap.add_argument("-i", "--dataset", required=True,
	help="path to input directory of faces + images")
ap.add_argument("-e", "--encodings", required=True,
	help="path to serialized db of facial encodings")
ap.add_argument("-d", "--detection-method", type=str, default="cnn",
	help="face detection model to use: either `hog` or `cnn`")
# ap.add_argument("-c", "--cascade", required=True,
# 	help = "path to where the face cascade resides")
args = vars(ap.parse_args())
# For detecting the faces in each frame we will use Haarcascade Frontal Face default classifier of OpenCV
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
data = pickle.loads(open(args["encodings"], "rb").read())
detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# fps = FPS().start()

# tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'CSRT', 'MOSSE']
# tracker_type = tracker_types[5]
tracker_type = 'CSRT'
(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
font = cv2.FONT_HERSHEY_SIMPLEX

# Set unique id for each individual person
face_id = 7
# Variable for counting the no. of images
count = 0
# initialize the known distance from the camera to the object, which

# in this case is 24 inches
KNOWN_DISTANCE = 24.0

# initialize the known object width, which in this case, the piece of
# paper is 12 inches wide
KNOWN_WIDTH = 11.0
imagePaths = list(paths.list_images(args["dataset"]))
tracker = cv2.TrackerCSRT_create()
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
print("[INFO] starting video stream...")
vs= cv2.VideoCapture(0);
time.sleep(2.0)
xV=None
co = True
Interrup=True
xV=None
distance=0
newY=0
newX=0
storeStatus=False
newThredStatus=False
try:
	while True:
		_,frame = vs.read()
		frame =	cv2.resize(frame,(360,480))
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		(h, w) = frame.shape[:2]
		blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
			0.007843, (300, 300), 127.5)
		net.setInput(blob)
		detections = net.forward()
		for i in np.arange(0, detections.shape[2]):
			confidence = detections[0, 0, i, 2]
			if confidence > args["confidence"]:
				idx = int(detections[0, 0, i, 1])
				box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
				(startX, startY, endX, endY) = box.astype("int")
				if xV is not None and Interrup:
					(success, box) = tracker.update(frame)
					print(success)
					if success:
						(x, y, w, h) = [int(v) for v in box]
						newX=x
						newY=y
						distancei = (2*3.14 * 180)/(w+h*360)*1000 + 3
						distance = math.floor(distancei/2) * 2.54
				# print(CLASSES[idx])
				if Interrup and CLASSES[idx] =='person' :
					print('detected person')
					nameU='Unknown'
					if nameU =='Unknown' and storeStatus:
						print(nameU)
					# if(storeStatus):
						storeStatus=False
						cv2.putText(frame,'Please wait I\'m storing you' , (5,400),font,1,(255,255,255),2)
						# storeFaceDataset(10,frame,gray)
						# t.daemon = True
						# t.start()
						newThredStatus=True
					else:
						storeStatus=False
						xV = tuple(box)
						print(xV)
						# marker = find_marker(frame)
						# inches = distance_to_camera(KNOWN_WIDTH, focalLength, marker[1][0])
						# print(inches)
						tracker.init(frame, xV)

		if len(frame) != 0:
			cv2.imshow("Frame", frame)
			key = cv2.waitKey(1) & 0xFF

			# if the `q` key was pressed, break from the loop
			if key == ord("q"):
				break
			if key == ord("s"):
				Interrup=True
                # print('s')
			if key == ord("t"):
				storeStatus=True

			# update the FPS counter
			# fps.update()

except NameError:
  print(NameError)

# stop the timer and display FPS information
# fps.stop()
# # print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
# # print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
