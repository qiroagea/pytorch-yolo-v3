import cv2


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 15)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 400)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 300)

ret, frame = cap.read()
cv2.imwrite("cap.png", frame)
