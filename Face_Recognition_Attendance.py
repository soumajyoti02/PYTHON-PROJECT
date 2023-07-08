# pip install opencv-python
# pip install face-recognition
# Install Visual Studio Community from below link and inside that, install Desktop development with C++
# https://visualstudio.microsoft.com/thank-you-downloading-visual-studio/?sku=Community&channel=Release&version=VS2022&source=VSLandingPage&cid=2030&passive=false
# Create a folder named <Face_Recognition> and make a python file inside that. Paste this Code there.
# Make a folder inside Face_Recognition folder & name it as "faces". Add all the face images inside that.
# Modify the dotted Section according the image name and the person's name

import face_recognition
import cv2
import numpy as np
import csv
from datetime import datetime

# Initialize video capture from camera
video_capture = cv2.VideoCapture(0)

# ---------------------------------------------------------------------------
# Load known faces and their encodings
soumas_image = face_recognition.load_image_file("faces/souma.jpg")
souma_encoding = face_recognition.face_encodings(soumas_image)[0]

ahanas_image = face_recognition.load_image_file("faces/amy.jpg")
ahana_encoding = face_recognition.face_encodings(ahanas_image)[0]

known_face_encodings = [souma_encoding, ahana_encoding]
known_face_names = ["Soumajyoti", "Ahana"]
# ---------------------------------------------------------------------------

# Create a list of expected students
students = known_face_names.copy()

# Initialize variables for face locations and encodings
face_locations = []
face_encodings = []

# Get the current date and time
now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

# Create and open a CSV file for writing
f = open(f"{current_date}.csv", "w+", newline="")
lnwriter = csv.writer(f)

# Start the video capture loop
while True:
    # Read a frame from the video capture
    _, frame = video_capture.read()

    # Resize the frame for faster face recognition
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the frame from BGR to RGB
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Recognize faces in the frame
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    # Iterate over the detected face encodings
    for face_encoding in face_encodings:
        # Compare the face encoding with the known face encodings
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        # Calculate the face distance to find the best match
        face_distance = face_recognition.face_distance(
            known_face_encodings, face_encoding
        )
        best_match_index = np.argmin(face_distance)

        # If there is a match, assign the name
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        # Add text to the frame if a person is present
        if name in known_face_names:
            font = cv2.FONT_HERSHEY_SIMPLEX
            bottomLeftCornerOfText = (10, 100)
            fontScale = 1.5
            fontColor = (255, 0, 0)
            thickness = 3
            lineType = 2
            cv2.putText(
                frame,
                name + " Present",
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                thickness,
                lineType,
            )

            # If the person is in the student list, remove them and record the time
            if name in students:
                students.remove(name)
                current_time = now.strftime("%H-%M%S")
                lnwriter.writerow([name, current_time])

    # Show the frame with the attendance information
    cv2.imshow("Attendance", frame)

    # Check if the 'q' key is pressed to exit the loop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the video capture and close the CSV file
video_capture.release()
cv2.destroyAllWindows()
f.close()
