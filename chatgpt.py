import cv2
import mediapipe as mp
import numpy as np

frame_shape = (480, 640, 3)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mphands = mp.solutions.hands

mask = np.zeros(frame_shape, dtype='uint8')
drawing_color = (125, 100, 140)
drawing_thickness = 4
eraser_thickness = 20
prevxy = None
drawing_mode = "line"

cap = cv2.VideoCapture(0)
hands = mphands.Hands()

while True:
    data, image = cap.read()
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGRA2BGR)  # Flip image
    results = hands.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mphands.HAND_CONNECTIONS)

            x = int(hand_landmarks.landmark[8].x * frame_shape[1])
            y = int(hand_landmarks.landmark[8].y * frame_shape[0])

            if prevxy is not None:
                if drawing_mode == "line":
                    cv2.line(mask, prevxy, (x, y), drawing_color, drawing_thickness)
                elif drawing_mode == "rectangle":
                    cv2.rectangle(mask, prevxy, (x, y), drawing_color, drawing_thickness)
                elif drawing_mode == "circle":
                    radius = int(np.sqrt((x - prevxy[0]) ** 2 + (y - prevxy[1]) ** 2))
                    cv2.circle(mask, prevxy, radius, drawing_color, drawing_thickness)

            prevxy = (x, y)

    image = np.where(mask, mask, image)
    cv2.imshow('Handtracker', image)

    key = cv2.waitKey(1)
    if key == 27 or key == ord('q'):
        break
    elif key == ord('1'):
        drawing_mode = "line"
    elif key == ord('2'):
        drawing_mode = "rectangle"
    elif key == ord('3'):
        drawing_mode = "circle"
    elif key == ord('e'):
        drawing_mode = "eraser"
    elif key == ord('r'):
        drawing_color = (0, 0, 255)  # Red
    elif key == ord('y'):
        drawing_color = (0, 255, 255)  # Yellow
    elif key == ord('b'):
        drawing_color = (255, 0, 0)  # Blue
    elif key == ord('g'):
        drawing_color = (0, 255, 0)  # Green
    elif key == ord('w'):
        drawing_color = (255, 255, 255)  # White
    elif key == ord('0'):
        mask = np.zeros(frame_shape, dtype='uint8')
        drawing_mode = "line"

cap.release()
cv2.destroyAllWindows()
