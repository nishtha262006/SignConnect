\
print("Python camera test started!")

import cv2

print("OpenCV version:", cv2.__version__)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

print("Camera opened:", cap.isOpened())

if cap.isOpened():
    ret, frame = cap.read()

    print("Frame captured:", ret)

    if ret:
        cv2.imshow("Camera Test", frame)
        cv2.waitKey(5000)

cap.release()
cv2.destroyAllWindows()

print("Camera test finished!")
