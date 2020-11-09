# OpenCV Python program to detect cars in video frame
# import libraries of python OpenCV
import cv2
import face_recognition

# capture frames from a video
cap = cv2.VideoCapture( r"./000_M101.mp4",0)

# Initialize variables
face_locations = []
ret, frame = cap.read()
w, h, c = frame.shape
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

fourcc = cv2.VideoWriter_fourcc(*'X264')

out = cv2.VideoWriter('outpy.mp4',fourcc , 10, (frame_width, frame_height))

while cap.isOpened():
    # Grab a single frame of video
    ret, frame = cap.read()
    if ret == False:
            print("Frame doesn't Exist")
            break
    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_frame = frame[:, :, ::-1]

    # Find all the faces in the current frame of video
    face_locations = face_recognition.face_locations(rgb_frame)

    # Display the results
    for top, right, bottom, left in face_locations:
        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

    # Display the resulting image
    out.write(frame)
    cv2.imshow('Video', frame)
    
    # Wait for Enter key to stop
    if cv2.waitKey(25) == 13:
        break
    ret, frame = cap.read()
# Release everything if job is finished
cap.release()
out.release()

cv2.destroyAllWindows()