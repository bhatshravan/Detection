
import cv2
import numpy as np
from PIL import ImageGrab
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(0)
while(True):
	# Capture frame-by-frame
	ret, frame = cap.read()
	img_np=ret;

	"""
	img = ImageGrab.grab(bbox=(100,10,400,700)) #bbox specifies specific region (bbox= x,y,width,height)
	img_np = np.array(img)
	#frame = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
    
	# Our operations on the frame come here
	gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
	frame = np.array(img)
	"""
	
	img_filt = cv2.medianBlur(img_np, 5)
	img_th = cv2.adaptiveThreshold(img_filt,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
	contours, hierarchy = cv2.findContours(img_th, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	
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


