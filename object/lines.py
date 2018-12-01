from PIL import ImageGrab
import numpy as np
import argparse
from matplotlib import pyplot as plt
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()

ap.add_argument("-c", "--confidence", type=float, default=0.7,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe("mprototxt.txt", "m.caffemodel")

faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")


while(True):
	img2 = ImageGrab.grab(bbox=(10,10,600,700)) #bbox specifies specific region (bbox= x,y,width,height)
	img_np = np.array(img2)
	img=img_np
	frame = np.array(img2)

	#img = cv2.imread(img_np.convert('RGB',0)
	
	
	#ADDING FACE RECONGNISTION MODULE
	gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
	faces = faceCascade.detectMultiScale(
		gray,
		scaleFactor=1.1,
		minNeighbors=5,
		minSize=(30, 30)
		#flags = cv2.CV_HAAR_SCALE_IMAGE
	)
	
	ll=0
	# show the output image
	
	
	for (x, y, w, h) in faces:
		cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
		cv2.putText(frame, "Face #{}".format(ll + 1), (x, y - 10),
		cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 0, 255), 2)
		ll=ll+1
	
	frame = cv2.Canny(frame,100,200)
	
	#SINGLE IMAGE USING MATLIB
	"""
	plt.subplot(121),plt.imshow(img,cmap = 'gray')
	plt.title('Original Image'), plt.xticks([]), plt.yticks([])
	plt.subplot(122),plt.imshow(edges,cmap = 'gray')
	plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
	"""
	#plt.show()
	cv2.imshow('houghlines3', frame)

	#ADD FACE DETECTION THROUGH HAAR,USING HOUGHLINES
	"""
	gray2 = cv2.cvtColor(img_np,cv2.COLOR_BGR2GRAY)
	edges = cv2.Canny(gray2,50,150,apertureSize = 3)
	lines = cv2.HoughLines(edges,1,np.pi/180, 200)
 
	# The below for loop runs till r and theta values 
	# are in the range of the 2d array
	for r,theta in lines[0]:
		a = np.cos(theta)
		b = np.sin(theta)
		x0 = a*r
		y0 = b*r
		x1 = int(x0 + 1000*(-b))
		y1 = int(y0 + 1000*(a))
		x2 = int(x0 - 1000*(-b))
		y2 = int(y0 - 1000*(a))
		cv2.line(frame,(x1,y1), (x2,y2), (0,0,255),2)
	
	cv2.imshow('houghlines3.jpg', frame)
	"""
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cv2.destroyAllWindows()
