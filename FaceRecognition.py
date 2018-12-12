import cv2
import numpy
import os

font = cv2.FONT_HERSHEY_SIMPLEX

def drawFrameRate(img, frameRate):
	return cv2.putText(img,"FPS: "+str(int(frameRate)),(10,26),font,0.7,(255,0,255),2,cv2.LINE_AA)

def drawFaces(img, faces, eyes):
	for (xf,yf,wf,hf) in faces:
		for (xe,ye,we,he) in eyes:
			if (xe >= xf) and ((xe + we) <= (xf + wf)):
				if (ye >= yf) and ((ye + he) <= (yf + hf)):
					cv2.rectangle(img,(xf,yf),(xf+wf,yf+hf),(0,0,255),3)
					cv2.rectangle(img,(xe,ye),(xe+we,ye+he),(0,255,0),1)


freq = cv2.getTickFrequency()
frameRateCalc = 1

webcam = cv2.VideoCapture(0)
path = os.path.dirname(os.path.abspath(__file__)) + "/bdd/"

while True:

	t1 = cv2.getTickCount()

	img = webcam.read()[1]
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

	faceCascade = cv2.CascadeClassifier(path + 'face.xml')
	eyeCascade = cv2.CascadeClassifier(path + 'eye.xml')

	faces = faceCascade.detectMultiScale(gray, 1.1, 5)
	eyes = eyeCascade.detectMultiScale(gray, 1.1, 5)
	
	print "Found "+str(len(faces))+" face(s)"
	print "Found "+str(len(eyes))+" eye(s)"

	"""for (x,y,w,h) in faces:
	    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

	for (x,y,w,h) in eyes:
	    cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)"""

	drawFaces(img, faces, eyes)

	drawFrameRate(img, frameRateCalc)
	cv2.imshow("Face Recognition", img)

	t2 = cv2.getTickCount()
	time1 = (t2-t1)/freq
	frameRateCalc = 1/time1

	if cv2.waitKey(1) == 27:
		break  # esc to quit
	cv2.destroyAllWindows()