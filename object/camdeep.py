# USAGE
# python deep_learning_object_detection.py --image images/example_01.jpg \
#	--prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel

# import the necessary packages

from PIL import ImageGrab
import numpy as np
import argparse
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()

ap.add_argument("-c", "--confidence", type=float, default=0.7,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe("mprototxt.txt", "m.caffemodel")

faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")


while(True):
	img = ImageGrab.grab(bbox=(10,10,600,700)) #bbox specifies specific region (bbox= x,y,width,height)
	img_np = np.array(img)
	frame = np.array(img)
	
	
	#image = cv2.imread(img_np)
	(h, w) = img_np.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(img_np, (400, 400)), 0.007843, (300, 300), 127.5)
	#blob = cv2.dnn.blobFromImage(img_np)

	#print("[INFO] computing object detections...")
	net.setInput(blob)
	detections = net.forward()

	ll=0
	
	#ADD FACE DETECTION THROUGH HAAR
	gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
	faces = faceCascade.detectMultiScale(
		gray,
		scaleFactor=1.1,
		minNeighbors=5,
		minSize=(30, 30)
		#flags = cv2.CV_HAAR_SCALE_IMAGE
	)
	ll=0

	for i in np.arange(0, detections.shape[2]):
		confidence = detections[0, 0, i, 2]

		if confidence > args["confidence"]:
			ll=ll+1
			idx = int(detections[0, 0, i, 1])
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# display the prediction
			label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
			print("[INFO] no - {0} {1}".format(ll,label))
			cv2.rectangle(frame, (startX, startY), (endX, endY),
			COLORS[idx], 2)
			y = startY - 15 if startY - 15 > 15 else startY + 15
			cv2.putText(frame, label, (startX, y),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
	
	# show the output image
	for (x, y, w, h) in faces:
		cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
		cv2.putText(frame, "Face #{}".format(ll + 1), (x, y - 10),
		cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 0, 255), 2)
		ll=ll+1
	
	cv2.imshow("Output", frame)
	
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cv2.destroyAllWindows()
