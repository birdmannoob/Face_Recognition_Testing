import numpy as np
import cv2
import pickle
import os
TP = 0
FP = 0
FN = 0
TN = 0  #parameters for ROC analysis
face_size = (300, 300)
face_cascade = cv2.CascadeClassifier('cascade/haarcascade_frontalface_alt2.xml')
num = 0 #counter of images
image_dir = 'dataset/test'
threshold = 0.7
if threshold <=1 and threshold >=0:
	tolerance = 20000*(1 - threshold)

recognizer = cv2.face.EigenFaceRecognizer_create()
recognizer.read("./recognizers/face-trainner.yml")

labels = {"person_name": 1}
with open("pickles/face-labels.pickle", 'rb') as f:
	og_labels = pickle.load(f)
	labels = {v:k for k,v in og_labels.items()}

cap = cv2.VideoCapture(0)

for files in os.listdir(image_dir):
	print("")
	print(files)
	for file in os.listdir(f'{image_dir}/{files}'):
		final_image = cv2.imread(f'{image_dir}/{files}/{file}')
		print("")
		print(file)
		#final_image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
		gray = cv2.cvtColor(final_image, cv2.COLOR_BGR2GRAY)
		image_array = np.array(gray, "uint8")
		faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.2, minNeighbors=5)
		for (x, y, w, h) in faces:
			roi_gray = gray[y:y+h, x:x+w] #(ycord_start, ycord_end)
			roi_color = final_image[y:y+h, x:x+w]

			resized = cv2.resize(roi_gray, face_size)
			id_, conf = recognizer.predict(resized)
			print(labels[id_], "conf = ", conf)
			if conf <= tolerance:
				#print(5: #id_)
				#print(labels[id_])
				name = labels[id_]
				output = name + "  " + str(round(conf))
			else:
				name = "unknown"
				output = name
			stroke = 2
			end_cord_x = x + w
			end_cord_y = y + h
			cv2.rectangle(final_image, (x, y), (end_cord_x, end_cord_y), (0, 255, 0), stroke)
			cv2.rectangle(final_image, (x, y - 15), (x + w, y + 3), (0, 255, 0), cv2.FILLED)
			cv2.putText(final_image, output, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
			if name == "unknown":
				if (files == "subject (10)untrained" or files == "subject(1)untrained"):
					TN += 1
				else:
					FN += 1
			else:
				if name == files:
					TP += 1
				else:
					FP +=1
		# Display the resulting frame
		cv2.imshow('Result',cv2.resize(final_image, (0, 0), fx=0.5, fy=0.5))
		num += 1
		filename = "Eigenoutput/" + str(num) + "-" + files + ".jpg"
		cv2.imwrite(filename, cv2.resize(final_image, (0, 0), fx=0.5, fy=0.5))
		cv2.waitKey(0)
print("total images: ", num)
print(TP)
print(FP)
print(FN)
print(TN)
cv2.destroyAllWindows()
