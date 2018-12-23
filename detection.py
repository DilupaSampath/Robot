# USAGE
# python real_time_object_detection.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel

# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2
import math
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.2,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())
tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'CSRT', 'MOSSE']
tracker_type = tracker_types[5]
(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
font = cv2.FONT_HERSHEY_SIMPLEX

# initialize the known distance from the camera to the object, which
# in this case is 24 inches
KNOWN_DISTANCE = 24.0

# initialize the known object width, which in this case, the piece of
# paper is 12 inches wide
KNOWN_WIDTH = 11.0

def find_marker(image):
	# convert the image to grayscale, blur it, and detect edges
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (5, 5), 0)
	edged = cv2.Canny(gray, 35, 125)

	# find the contours in the edged image and keep the largest one;
	# we'll assume that this is our piece of paper in the image
	cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	c = max(cnts, key = cv2.contourArea)

	# compute the bounding box of the of the paper region and return it
	return cv2.minAreaRect(c)

def distance_to_camera(knownWidth, focalLength, perWidth):
	# compute and return the distance from the maker to the camera
	return (knownWidth * focalLength) / perWidth
if int(minor_ver) < 3:
	tracker = cv2.Tracker_create(tracker_type)
else:
	if tracker_type == 'BOOSTING':
		tracker = cv2.TrackerBoosting_create()
	if tracker_type == 'MIL':
		tracker = cv2.TrackerMIL_create()
	if tracker_type == 'KCF':
		tracker = cv2.TrackerKCF_create()
	if tracker_type == 'TLD':
		tracker = cv2.TrackerTLD_create()
	if tracker_type == 'MEDIANFLOW':
		tracker = cv2.TrackerMedianFlow_create()
	if tracker_type == 'CSRT':
		tracker = cv2.TrackerCSRT_create()
	if tracker_type == 'MOSSE':
		tracker = cv2.TrackerMOSSE_create()
# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# initialize the video stream, allow the cammera sensor to warmup,
# and initialize the FPS counter
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)
fps = FPS().start()

xV=None
co = True
Interrup=False
xV=None
distance=0
newY=0
newX=0
# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	frame = vs.read()
	frame = imutils.resize(frame, width=400)
	marker = find_marker(frame)
	focalLength = (marker[1][0] * KNOWN_DISTANCE) / KNOWN_WIDTH
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
			label = "{}: {:.2f}%".format(CLASSES[idx],
				confidence * 100)
			# print(CLASSES[idx])
			# (success, box) = tracker.update(frame)
			if xV is not None and Interrup:
				(success, box) = tracker.update(frame)
				print(success)
				if success:
					# Interrup=False
					(x, y, w, h) = [int(v) for v in box]
					cv2.rectangle(frame, (x, y), (x + w, y + h),
					(0, 255, 0), 2)
					newX=x
					newY=y
					distancei = (2*3.14 * 180)/(w+h*360)*1000 + 3
					distance = math.floor(distancei/2) * 2.54
			print(CLASSES[idx])
			if Interrup and CLASSES[idx] =='person' :
				print('detected person')
				xV = tuple(box)
				print(xV)
				marker = find_marker(frame)
				inches = distance_to_camera(KNOWN_WIDTH, focalLength, marker[1][0])
				print(inches)
				tracker.init(frame, xV)
			# else:
				# xV = tuple(box)
				# print(xV)
				# cv2.rectangle(frame,(startX, startY),(endX, endY),COLORS[idx], 2)
				# y = startY - 15 if startY - 15 > 15 else startY + 15
				# cv2.putText(frame, distance, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
				cv2.putText(frame,'Distance = ' + str(newY), (5,100),font,1,(255,255,255),2)

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

		# update the FPS counter
		fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
