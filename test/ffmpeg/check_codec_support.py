import cv2

# Create a VideoWriter object with a common codec
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
test_writer = cv2.VideoWriter("test.mp4", fourcc, 25, (640, 480))

if test_writer.isOpened():
    print("Codec is supported!")
else:
    print("Codec is NOT supported.")
test_writer.release()
