import pickle
import os
import cv2
import numpy as np
import imutils
image_dir = 'dataset/test'
num = 1
DetectFaceThreshold = 0.99
Threshold = 0.8    #threshold value for ROC
protoPath = ("face_detection_model/deploy.prototxt")
modelPath = ("face_detection_model/res10_300x300_ssd_iter_140000.caffemodel")
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
embedder = cv2.dnn.readNetFromTorch("openface_nn4.small2.v1.t7")
# load the actual face recognition model along with the label encoder
recognizer = pickle.loads(open('opencv-dnn-recognizer/recognizer.pickle', "rb").read())
le = pickle.loads(open('opencv-dnn-recognizer/le.pickle', "rb").read())

for files in os.listdir(image_dir):
	print(files)
	for file in os.listdir(f'{image_dir}/{files}'):
		final_image = cv2.imread(f'{image_dir}/{files}/{file}')
		# if (h < 300 or w < 300):
		# 	final_image = cv2.resize(final_image, (0, 0), fx=2.5, fy=2.5)
		# else:
		# 	final_image = cv2.resize(final_image, (0, 0), fx=0.5, fy=0.5)
		final_image = imutils.resize(final_image, width=600)
		print("")
		print(file)
		(h, w) = final_image.shape[:2]
		# construct a blob from the image
		imageBlob = cv2.dnn.blobFromImage(cv2.resize(final_image, (300, 300)), 1.0, (300, 300),(104.0, 177.0, 123.0), swapRB=False, crop=False)
		# apply OpenCV's deep learning-based face detector to localize
		# faces in the input image
		detector.setInput(imageBlob)
		detections = detector.forward()

		# loop over the detections
		for i in range(0, detections.shape[2]):
			# extract the confidence (i.e., probability) associated with
			# the prediction
			confidence = detections[0, 0, i, 2]
			# filter out weak detections
			if confidence > DetectFaceThreshold:
				# compute the (x, y)-coordinates of the bounding box for
				# the face
				box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
				(startX, startY, endX, endY) = box.astype("int")
				cv2.rectangle(final_image, (startX, startY), (endX, endY), (0, 255, 0), 2)
				# extract the face ROI
				face = final_image[startY:endY, startX:endX]
				(fH, fW) = face.shape[:2]

				# ensure the face width and height are sufficiently large
				if fW < 20 or fH < 20:
					continue

				# construct a blob for the face ROI, then pass the blob
				# through our face embedding model to obtain the 128-d
				# quantification of the face
				faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
				embedder.setInput(faceBlob)
				vec = embedder.forward()

				# perform classification to recognize the face
				preds = recognizer.predict_proba(vec)[0]
				j = np.argmax(preds)
				proba = preds[j]

				if proba >= Threshold:
					name = le.classes_[j]
					text = "{}: {:.2f}%".format(name, proba * 100)
				else:
					name = "unknown"
					text = name

				# draw the bounding box of the face along with the
				# associated probability

				y = startY - 10 if startY - 10 > 10 else startY + 10
				cv2.rectangle(final_image, (startX, startY), (endX, endY),
					(0, 255, 0), 2)
				cv2.putText(final_image, text, (startX, y),
					cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

		cv2.imshow("face", cv2.resize(final_image, (0, 0), fx=0.8, fy=0.8))
		num += 1
		filename = "opencv-dnn-output/" + str(num) + "-" + files + ".jpg"
		cv2.imwrite(filename, cv2.resize(final_image, (0, 0), fx=0.8, fy=0.8))
		cv2.waitKey(0)

print("total images: ", num)