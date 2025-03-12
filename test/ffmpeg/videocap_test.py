import cv2
import os

cap = cv2.VideoCapture(0)

# Test video writer initialization
filename = "test.mp4"
fourcc = cv2.VideoWriter_fourcc(*"MP4V")
fps = 30
frame_size = (640, 480)

if os.path.exists(filename):
    os.remove(filename)

video_writer = cv2.VideoWriter(filename, fourcc, fps, frame_size)
if video_writer.isOpened():
    print("VideoWriter initialized successfully.")
else:
    print("Failed to initialize VideoWriter. Check your system setup.")

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
        frame = cv2.flip(frame,0)

        # write the flipped frame
        video_writer.write(frame)

        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()
video_writer.release()
cv2.destroyAllWindows()