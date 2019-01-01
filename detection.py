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
#
# def reconizeFace(frame,gray,rgb):
# 	# gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# 	# rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#
# 	# detect faces in the grayscale frame
# 	rects = detector.detectMultiScale(gray, scaleFactor=1.3,
# 	minNeighbors=5, minSize=(30, 30),
# 	flags=cv2.CASCADE_SCALE_IMAGE)
#
# 	# OpenCV returns bounding box coordinates in (x, y, w, h) order
# 	# but we need them in (top, right, bottom, left) order, so we
# 	# need to do a bit of reordering
# 	boxes = [(y, x + w, y + h, x) for (x, y, w, h) in rects]
#
# 	# compute the facial embeddings for each face bounding box
# 	encodings = face_recognition.face_encodings(rgb, boxes)
# 	names = []
#
# 	# loop over the facial embeddings
# 	for encoding in encodings:
# 		# attempt to match each face in the input image to our known
# 		# encodings
# 		matches = face_recognition.compare_faces(data["encodings"],encoding)
# 		name = "Unknown"
#
# 		# check to see if we have found a match
# 		if True in matches:
# 			# find the indexes of all matched faces then initialize a
# 			# dictionary to count the total number of times each face
# 			# was matched
# 			matchedIdxs = [i for (i, b) in enumerate(matches) if b]
# 			counts = {}
#
# 			# loop over the matched indexes and maintain a count for
# 			# each recognized face face
# 			for i in matchedIdxs:
# 				name = data["names"][i]
# 				counts[name] = counts.get(name, 0) + 1
#
# 			# determine the recognized face with the largest number
# 			# of votes (note: in the event of an unlikely tie Python
# 			# will select first entry in the dictionary)
# 			name = max(counts, key=counts.get)
#
# 		# update the list of names
# 		names.append(name)
#
# 	# loop over the recognized faces
# 	# for ((top, right, bottom, left), name) in zip(boxes, names):
# 	# 	# draw the predicted face name on the image
# 	# 	# cv2.rectangle(frame, (left, top), (right, bottom),
# 	# 	# 	(0, 255, 0), 2)
# 	# 	y = top - 15 if top - 15 > 15 else top + 15
# 	# 	cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
# 	# 		0.75, (0, 255, 0), 2)
# 		return name
# 	# display the image to our screen
# 	# cv2.imshow("Frame", frame)
# 	# key = cv2.waitKey(1) & 0xFF
#
# 	# if the `q` key was pressed, break from the loop
# 	# update the FPS counter
# 	# fps.update()
# 	# return name
#
# # stop the timer and display FPS information
# # fps.stop()
# # print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
# # print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# def createModel(rgb):
#
# 	print("[INFO] quantifying faces...")
# 	imagePaths = list(paths.list_images(args["dataset"]))
#
# 	# initialize the list of known encodings and known names
# 	knownEncodings = []
# 	knownNames = []
#
# 	# loop over the image paths
# 	for (i, imagePath) in enumerate(imagePaths):
# 		# extract the person name from the image path
# 		print("[INFO] processing image {}/{}".format(i + 1,
# 			len(imagePaths)))
# 		name = imagePath.split(os.path.sep)[-2]
#
# 		# load the input image and convert it from RGB (OpenCV ordering)
# 		# to dlib ordering (RGB)
# 		image = cv2.imread(imagePath)
# 		# rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#
# 		# detect the (x, y)-coordinates of the bounding boxes
# 		# corresponding to each face in the input image
# 		boxes = face_recognition.face_locations(rgb,
# 			model=args["detection_method"])
#
# 		# compute the facial embedding for the face
# 		encodings = face_recognition.face_encodings(rgb, boxes)
#
# 		# loop over the encodings
# 		for encoding in encodings:
# 			# add each encoding + name to our set of known names and
# 			# encodings
# 			knownEncodings.append(encoding)
# 			knownNames.append(name)
#
# 	# dump the facial encodings + names to disk
# 	print("[INFO] serializing encodings...")
# 	data = {"encodings": knownEncodings, "names": knownNames}
# 	f = open(args["encodings"], "wb")
# 	f.write(pickle.dumps(data))
# 	f.close()
# def runMethordInThred():
# 	t = threading.Thread(target=createModel,args = (rgb),name='name')
# 	t.daemon = True
# 	t.start()
# def storeFaceDataset(frameCountUser,image_frame,gray):
# 	# Looping starts here
# 	count=0
# 	frameCount=frameCountUser
# 	while(True):
# 		frameCount=frameCount-1
# 		# gray = cv2.cvtColor(image_frame, cv2.COLOR_BGR2GRAY)
#
# 	    # Detecting different faces
# 		faces = face_detector.detectMultiScale(gray, 1.3, 5)
#
# 	    # Looping through all the detected faces in the frame
# 		for (x,y,w,h) in faces:
#
# 	        # Crop the image frame into rectangle
# 			cv2.rectangle(image_frame, (x,y), (x+w,y+h), (255,0,0), 2)
#
# 	        # Increasing the no. of images by 1 since frame we captured
# 			count += 1
#
# 	        # Saving the captured image into the training_data folder
# 			if not os.path.exists("dataset/"+str(frameCountUser)):
# 				os.makedirs("dataset/"+str(frameCountUser))
# 			cv2.imwrite("dataset/"+str(frameCountUser)+"/Person." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])
#
# 	        # Displaying the frame with rectangular bounded box
# 			# cv2.imshow('frame', image_frame)
#
# 	    # press 'q' for at least 100ms to stop this capturing process
# 		if cv2.waitKey(100) & 0xFF == ord('q'):
# 			break
#
# 	    #We are taking 100 images for each person for the training data
# 	    # If image taken reach 100, stop taking video
# 		elif count>10:
# 			break
#

# def assure_path_exists(path):
#     dir = os.path.dirname(path)
#     if not os.path.exists(dir):
#         os.makedirs(dir)

# def find_marker(image,gray):
# 	# convert the image to grayscale, blur it, and detect edges
# 	# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# 	gray = cv2.GaussianBlur(gray, (5, 5), 0)
# 	edged = cv2.Canny(gray, 35, 125)
# 	T1 = tuple()
# 	# find the contours in the edged image and keep the largest one;
# 	# we'll assume that this is our piece of paper in the image
# 	cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
# 	cnts = imutils.grab_contours(cnts)
# 	c = max(cnts, key = cv2.contourArea) if cnts else T1
#
# 	try:
# 		print('c $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
# 		print(cv2.minAreaRect(c))
# 		return cv2.minAreaRect(c)
# 	except:
# 		return cv2.minAreaRect(T1)
#
# 	# print('c $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
# 	# compute the bounding box of the of the paper region and return it
# 	# return cv2.minAreaRect(c)

# def distance_to_camera(knownWidth, focalLength, perWidth):
# 	# compute and return the distance from the maker to the camera
# 	return (knownWidth * focalLength) / perWidth
if int(minor_ver) < 3:
	tracker = cv2.Tracker_create(tracker_type)
else:
	tracker = cv2.TrackerCSRT_create()
	# if tracker_type == 'BOOSTING':
	# 	tracker = cv2.TrackerBoosting_create()
	# if tracker_type == 'MIL':
	# 	tracker = cv2.TrackerMIL_create()
	# if tracker_type == 'KCF':
	# 	tracker = cv2.TrackerKCF_create()
	# if tracker_type == 'TLD':
	# 	tracker = cv2.TrackerTLD_create()
	# if tracker_type == 'MEDIANFLOW':
	# 	tracker = cv2.TrackerMedianFlow_create()
	# if tracker_type == 'CSRT':
	# 	tracker = cv2.TrackerCSRT_create()
	# if tracker_type == 'MOSSE':
	# 	tracker = cv2.TrackerMOSSE_create()
# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

#checking existence of path
# assure_path_exists("dataset/")

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# initialize the video stream, allow the cammera sensor to warmup,
# and initialize the FPS counter
print("[INFO] starting video stream...")
# vs = FileVideoStream(0).start()
vs= cv2.VideoCapture(0);
time.sleep(2.0)
# fps = FPS().start()

xV=None
co = True
Interrup=False
xV=None
distance=0
newY=0
newX=0
storeStatus=False
# t = threading.Thread(target=createModel,name='name')
newThredStatus=False
# loop over the frames from the video stream
try:
	while True:
		# grab the frame from the threaded video stream and resize it
		# to have a maximum width of 400 pixels
		_,frame = vs.read()
		# frame = imutils.resize(frame, width=300)
		frame =	cv2.resize(frame,(360,480))
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

		# if Interrup:
		# 	marker = find_marker(frame,gray)
		# 	focalLength = (marker[1][0] * KNOWN_DISTANCE) / KNOWN_WIDTH
		# grab the frame dimensions and convert it to a blob
		(h, w) = frame.shape[:2]
		blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
			0.007843, (300, 300), 127.5)

		# pass the blob through the network and obtain the detections and
		# predictions
		net.setInput(blob)
		detections = net.forward()

		# loop over the detections
		for i in np.arange(0, detections.shape[2]):
			# extract the confidence (i.e., probability) associated with
			# the prediction
			confidence = detections[0, 0, i, 2]

			# filter out weak detections by ensuring the `confidence` is
			# greater than the minimum confidence
			if confidence > args["confidence"]:
				# extract the index of the class label from the
				# `detections`, then compute the (x, y)-coordinates of
				# the bounding box for the object
				idx = int(detections[0, 0, i, 1])
				box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
				(startX, startY, endX, endY) = box.astype("int")

				# draw the prediction on the frame
				# label = "{}: {:.2f}%".format(CLASSES[idx],
				# 	confidence * 100)
				# # print(CLASSES[idx])
				# (success, box) = tracker.update(frame)
				if xV is not None and Interrup:
					(success, box) = tracker.update(frame)
					print(success)
					if success:
						# Interrup=False
						(x, y, w, h) = [int(v) for v in box]
						# cv2.rectangle(frame, (x, y), (x + w, y + h),
						# (0, 255, 0), 2)
						newX=x
						newY=y
						distancei = (2*3.14 * 180)/(w+h*360)*1000 + 3
						distance = math.floor(distancei/2) * 2.54
				print(CLASSES[idx])
				# print("thread status:--> " +str(t.isAlive()))
				# if t.isAlive() == False and newThredStatus:
				# 	print('data reloaded...**********************************')
				# 	newThredStatus=False
					# dataOriginal = pickle.loads(open(args["encodings"], "rb").read())
					# t._stop()
					# t = threading.Thread(target=createModel,name='name')
				if Interrup and CLASSES[idx] =='person' :
					print('detected person')

					# nameU = reconizeFace(frame,gray,rgb)
					nameU='Unknown'
					if nameU =='Unknown' and storeStatus:
						print(nameU)
					# if(storeStatus):
						storeStatus=False
						# cv2.putText(frame,'Please wait I\'m storing you' , (5,400),font,1,(255,255,255),2)
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
				# else:
					# xV = tuple(box)
					# print(xV)
					# cv2.rectangle(frame,(startX, startY),(endX, endY),COLORS[idx], 2)
					# y = startY - 15 if startY - 15 > 15 else startY + 15
					# cv2.putText(frame, distance, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
					# distance comment
					# cv2.putText(frame,'Distance = ' + str(newY), (5,100),font,1,(255,255,255),2)

		# show the output frame
		# print(frame)
		if len(frame) != 0:
			cv2.imshow("Frame", frame)
			key = cv2.waitKey(1) & 0xFF

			# if the `q` key was pressed, break from the loop
			if key == ord("q"):
				break
			if key == ord("s"):
				Interrup=True
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
