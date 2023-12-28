import cv2
import mediapipe as mp
import numpy as np

def drawrectangle(image, prevxy, index_finger_tip, frame_shape, drawing_color, drawing_thickness,hand_landmarks):
    middle_finger_tip = hand_landmarks.landmark[12]
    distance_between_fingers = int(
        np.sqrt((index_finger_tip.x - middle_finger_tip.x) ** 2 +
                (index_finger_tip.y - middle_finger_tip.y) ** 2) * frame_shape[1])

    rect_diagonal = distance_between_fingers
    rect_size = (int(rect_diagonal / np.sqrt(2)), int(rect_diagonal / np.sqrt(2)))
    rect_start = (int(prevxy[0] - rect_size[0] / 2), int(prevxy[1] - rect_size[1] / 2))
    cv2.rectangle(mask, rect_start, (int(index_finger_tip.x * frame_shape[1]), int(index_finger_tip.y * frame_shape[0])),
                  drawing_color, drawing_thickness)
    return mask

def drawcircle(image, prevxy, index_finger_tip, frame_shape, drawing_color, drawing_thickness,hand_landmarks):
    middle_finger_tip = hand_landmarks.landmark[12]
    distance_between_fingers = int(
        np.sqrt((index_finger_tip.x - middle_finger_tip.x) ** 2 +
                (index_finger_tip.y - middle_finger_tip.y) ** 2) * frame_shape[1])

    circle_radius = int(distance_between_fingers / 2)
    cv2.circle(mask, prevxy, circle_radius, drawing_color, drawing_thickness)
    return mask

frame_shape = (480, 640, 3)
mp_drawing = mp.solutions.drawing_utils
mphands = mp.solutions.hands
mask = np.zeros(frame_shape, dtype='uint8')
drawing_color = (125, 100, 140)
drawing_thickness = 4
drawing_mode = "draw"
drawing_shape = "line"
prevxy = None

cap = cv2.VideoCapture(0)
hands = mphands.Hands()

def main():
    global mask, prevxy, drawing_color, drawing_thickness, drawing_mode, drawing_shape

    while True:
        data, image = cap.read()
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGRA2BGR)
        results = hands.process(image)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image,
                                          hand_landmarks,
                                          mphands.HAND_CONNECTIONS)

                index_finger_tip = hand_landmarks.landmark[8]

                if prevxy is not None:
                    if drawing_mode == "draw":
                        if drawing_shape == "line":
                            cv2.line(mask, prevxy, (int(index_finger_tip.x * frame_shape[1]), int(index_finger_tip.y * frame_shape[0])),
                                     drawing_color, drawing_thickness)
                        elif drawing_shape == "rectangle":
                            mask = drawrectangle(image, prevxy, index_finger_tip, frame_shape, drawing_color, drawing_thickness,hand_landmarks)
                        elif drawing_shape == "circle":
                            mask = drawcircle(image, prevxy, index_finger_tip, frame_shape, drawing_color, drawing_thickness,hand_landmarks)
                    elif drawing_mode == "erase":
                        cv2.line(mask, prevxy, (int(index_finger_tip.x * frame_shape[1]), int(index_finger_tip.y * frame_shape[0])),
                                 (0, 0, 0), drawing_thickness)

                prevxy = (int(index_finger_tip.x * frame_shape[1]), int(index_finger_tip.y * frame_shape[0]))

        # Apply the drawing to the image
        image = np.where(mask, mask, image)

        # Display the image
        cv2.imshow('Handtracker', image)

        # Handle key events
        key = cv2.waitKey(1) & 0xFF
        if key == ord('1'):
            drawing_mode = "draw"
        elif key == ord('e'):
            drawing_mode = "erase"
        elif key == ord('2'):
            drawing_shape = "rectangle"
        elif key == ord('3'):
            drawing_shape = "circle"
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
            mask = np.zeros(frame_shape, dtype='uint8')  # Clear canvas
        elif key == 32:  # Spacebar key
            prevxy = None  # Do not draw anything

        # Exit the loop on 'q' key press
        elif key == ord('q') or key == 27:
            break

if __name__ == "__main__":
    main()

cap.release()
cv2.destroyAllWindows()
