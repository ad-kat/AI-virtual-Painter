import cv2
import numpy as np

# Global variables
drawing = False
mode = True
ix, iy = -1, -1

# Function to draw circle on the canvas
def draw_circle(event, x, y, flags, param):
    global ix, iy, drawing, mode

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            if mode:
                cv2.circle(img, (x, y), 5, (0, 0, 255), -1)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if mode:
            cv2.circle(img, (x, y), 5, (0, 0, 255), -1)

# Create a black image and a window
img = np.zeros((512, 512, 3), np.uint8)
cv2.namedWindow('Virtual Painter')
cv2.setMouseCallback('Virtual Painter', draw_circle)

while True:
    cv2.imshow('Virtual Painter', img)

    # Break the loop when 'esc' key is pressed
    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
