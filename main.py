# importing opencv-python to work with the video
import cv2

# Fetching the haar-cascade xml file
faceCascade = cv2.CascadeClassifier("haar_face.xml")

video_capt = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    # capturing live image feed from camera frame by frame
    ret, frame = video_capt.read()

    # convert video to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Applying face detection algorithm on the grayscaled video
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                                         flags=cv2.CASCADE_SCALE_IMAGE)
    # Draw a rectangular box around the detected face
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Displaying the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('a'):
        break

video_capt.release()
cv2.destroyAllWindows()
