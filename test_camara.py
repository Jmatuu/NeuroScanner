import cv2
for i in range(4):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Índice {i} funciona")
        cap.release()
    else:
        print(f"Índice {i} no funciona")