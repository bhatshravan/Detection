# This script will detect faces via your webcam.
# Tested with OpenCV3

import cv2
import numpy as np
from PIL import ImageGrab

"""
cap = cv2.VideoCapture(0)

# Create the haar cascade
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

while(True):
	# Capture frame-by-frame
	ret, frame = cap.read()

	# Our operations on the frame come here
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# Detect faces in the image
	faces = faceCascade.detectMultiScale(
		gray,
		scaleFactor=1.1,
		minNeighbors=5,
		minSize=(30, 30)
		#flags = cv2.CV_HAAR_SCALE_IMAGE
	)

	print("Found {0} faces!".format(len(faces)))

	# Draw a rectangle around the faces
	for (x, y, w, h) in faces:
		cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)


	# Display the resulting frame
	cv2.imshow('frame', frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
"""

cap = cv2.VideoCapture(0)



# Create the haar cascade
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

while(True):
	# Capture frame-by-frame
	#ret, frame = cap.read()

	img = ImageGrab.grab(bbox=(100,10,400,700)) #bbox specifies specific region (bbox= x,y,width,height)
	img_np = np.array(img)
	frame = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
    
	# Our operations on the frame come here
	gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
	frame = np.array(img)

	# Detect faces in the image
	faces = faceCascade.detectMultiScale(
		gray,
		scaleFactor=1.1,
		minNeighbors=5,
		minSize=(30, 30)
		#flags = cv2.CV_HAAR_SCALE_IMAGE
	)
	#nbr_predicted, conf = recognizer.predict(predict_image[y: y + h, x: x + w])
    

	print("Found {0} faces!".format(len(faces)))

	# Draw a rectangle around the faces
	i=0
	for (x, y, w, h) in faces:
		cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
		cv2.putText(frame, "#{}".format(i + 1), (x, y - 10),
		cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 0, 255), 2)
		i=i+1


	# Display the resulting frame
	cv2.imshow('frame', frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# When everything done, release the capture
#cap.release()
cv2.destroyAllWindows()
