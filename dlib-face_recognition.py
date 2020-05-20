
import os
import cv2
import numpy as np
import face_recognition
TP = 0
FP = 0
FN = 0
TN = 0
image_dir = 'dataset/test'
train_dir = 'dataset/train'
num = 0
MODEL = "hog"
threshold = 1
if threshold>=0 and threshold<=1:
    TOLERANCE = 1-threshold

filename = None
known_faces = []
known_names = []
FRAME_THICKNESS = 3
FONT_THICKNESS = 2
#<===========================================Face Training Algorithm===========================================>
print('Training with faces in directory...')
for name in os.listdir(train_dir):
    # Next we load every file of faces of known person
    for filename in os.listdir(f'{train_dir}/{name}'):
        # Load an image
        imagex = cv2.imread(f'{train_dir}/{name}/{filename}')
        (h, w) = imagex.shape[:2]
        if (h< 300 or w < 300):
            imagex = cv2.resize(imagex, (0, 0), fx= 2.5, fy= 2.5)
        rgbx = imagex[:, :, ::-1]

        image_arrayx = np.array(rgbx, "uint8")
        # Get 128-dimension face encoding
        # Always returns a list of found faces, for this purpose we take first face only (assuming one face per image as you can't be twice on one image)
        location = face_recognition.face_locations(image_arrayx, model=MODEL)
        encoding = face_recognition.face_encodings(image_arrayx, location)
        if len(encoding) > 0:
            finalencoding = encoding[0]
            # Append encodings and name
            known_faces.append(finalencoding)
            known_names.append(name)

for files in os.listdir(image_dir):
    print(" ")
    print(files)
    for file in os.listdir(f'{image_dir}/{files}'):
        final_image = cv2.imread(f'{image_dir}/{files}/{file}')
        [h, w] = final_image.shape[:2]
        print("")
        print(file)
        if h < 300 or w < 300:
            final_image = cv2.resize(final_image, (0, 0), fx=2.5, fy=2.5)
        rgb_image = final_image[:, :, ::-1]
        image_array = np.array(rgb_image, "uint8")
        # <===========================================faceDetectAndRecogAlgorithm===========================================>
        # Find all the faces and face encodings in the current frame of video
        locations = face_recognition.face_locations(image_array, model=MODEL)  # number_of_times_to_upsample=2,
        encodings = face_recognition.face_encodings(image_array, locations)
        for face_encoding, face_location in zip(encodings, locations):
            # We use compare_faces (but might use face_distance as well)
            # Returns array of True/False values in order of passed known_faces
            results = face_recognition.compare_faces(known_faces, face_encoding, TOLERANCE)
            # Since order is being preserved, we check if any face was found then grab index
            # then label (name) of first matching known face withing a tolerance
            # Each location contains positions in order: top, right, bottom, left
            face_top_left = (face_location[3], face_location[0])  # original face locations
            face_bottom_right = (face_location[1], face_location[2])  # Note that we shrinked down the frame previously
            # Draw rectangle on face
            color = [0, 255, 0]  # Green color
            cv2.rectangle(final_image, face_top_left, face_bottom_right, color, FRAME_THICKNESS)
            # Now we need smaller, filled box below for a name
            # This time we use bottom in both corners - to start from bottom and move 50 pixels down
            box_top_left = (face_location[3], face_location[2])
            box_bottom_right = (face_location[1], (face_location[2] + 22))
            # Draw everything below:
            # Paint frame
            cv2.rectangle(final_image, box_top_left, box_bottom_right, color, cv2.FILLED)
            match = None
            if True in results:  # If at least one is true, get a name of first of found labels
                match = known_names[results.index(True)]
                #match = known_names[len(results)-1]
                print(f"Match Found: {match}")

                # Write a name
                name = match
            else:
                # Write a name
                name = "unknown"
            if name == "unknown":
                if (files == "subject (10)untrained" or files == "subject(1)untrained"):
                    TN += 1
                else:
                    FN += 1
            else:
                if name == files:
                    TP += 1
                else:
                    FP += 1

            cv2.putText(final_image, name, ((face_location[3] + 10), (face_location[2] + 15)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 0), FONT_THICKNESS)

        # <===========================================faceDetectAndRecogAlgorithm===========================================>

        cv2.imshow('face', cv2.resize(final_image, (0, 0), fx=0.5, fy=0.5))
        num += 1
        output = "dliboutput/" + str(num) + "-" + files + ".jpg"
        cv2.imwrite(output, cv2.resize(final_image, (0, 0), fx=0.5, fy=0.5))


        cv2.waitKey(0)

print("total images: ", num)
print(TP)
print(FP)
print(FN)
print(TN)
cv2.destroyAllWindows()