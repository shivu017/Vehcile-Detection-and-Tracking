# Vehcile-Detection-and-Tracking
import cv2
import numpy as np
import sys

# Video file path
video_path = "D:/GAMES/WhatsApp Video 2024-12-08 at 21.17.56_e0dce5e5.mp4"
cap = cv2.VideoCapture(video_path)

# Check if video is opened successfully
if not cap.isOpened():
    print("Error: Could not open video file.")
    sys.exit()  # Use sys.exit() properly

count_line_position = 550
min_width_rectangle = 80
min_height_rectangle = 80
algo = cv2.createBackgroundSubtractorMOG2()

def center_handle(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx, cy

detect = []
offset = 7  # Allowable error in pixels
counter = 0

while True:
    ret, frame1 = cap.read()
    if not ret:  # Indented properly
        print("End of video or cannot read frame.")
        break
    
    grey = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey, (3, 3), 5)
    img_sub = algo.apply(blur)
    dilat = cv2.dilate(img_sub, np.ones((5, 5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilatada = cv2.morphologyEx(dilat, cv2.MORPH_CLOSE, kernel)
    dilatada = cv2.morphologyEx(dilatada, cv2.MORPH_CLOSE, kernel)

    counterShap, _ = cv2.findContours(dilatada, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.line(frame1, (25, count_line_position), (1200, count_line_position), (255, 127, 0), 2)

    for c in counterShap:
        (x, y, w, h) = cv2.boundingRect(c)
        validate_counter = (w >= min_width_rectangle) and (h >= min_height_rectangle)

        if not validate_counter:
            continue
        
        cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)
        center = center_handle(x, y, w, h)
        detect.append(center)
        cv2.circle(frame1, center, 4, (0, 0, 255), -1)

    for (cx, cy) in detect:
        if count_line_position - offset < cy < count_line_position + offset:
            counter += 1
            detect.remove((cx, cy))
            print("Vehicle Counter:", counter)
            cv2.putText(frame1, "Vehicle Counter: " + str(counter), (450, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)

    cv2.imshow("Video Original", frame1)

    if cv2.waitKey(1) == 13:  # Press Enter to exit
        break

cap.release()
cv2.destroyAllWindows()

