import cv2
import numpy as np
import time
from datetime import datetime

face_ref = cv2.CascadeClassifier("face_ref.xml")
camera = cv2.VideoCapture(0)

def rounded_rectangle(image, pt1, pt2, color, thickness, radius, line_type):
    x1, y1 = pt1
    x2, y2 = pt2
    w = x2 - x1
    h = y2 - y1
    if w < 0 or h < 0:
        return

    if radius > min(w, h) / 2:
        radius = min(w, h) / 2

    # Top left
    cv2.ellipse(image, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness, line_type)
    # Top right
    cv2.ellipse(image, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness, line_type)
    # Bottom right
    cv2.ellipse(image, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness, line_type)
    # Bottom left
    cv2.ellipse(image, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness, line_type)

    # Draw four edges
    cv2.line(image, (x1 + radius, y1), (x2 - radius, y1), color, thickness, line_type)
    cv2.line(image, (x1 + radius, y2), (x2 - radius, y2), color, thickness, line_type)
    cv2.line(image, (x1, y1 + radius), (x1, y2 - radius), color, thickness, line_type)
    cv2.line(image, (x2, y1 + radius), (x2, y2 - radius), color, thickness, line_type)

def face_detection(frame):
  optimized_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
  faces = face_ref.detectMultiScale(optimized_frame, scaleFactor=1.1, minSize=(200,200), minNeighbors=5)
  return faces

def drawer_box(frame):
  for x, y, w, h in face_detection(frame):
    pt1 = (x, y)
    pt2 = (x + w, y + h)
    rounded_rectangle(frame, pt1, pt2, (0, 255, 0), 2, 20, cv2.LINE_AA)

def close_window():
  camera.release()
  cv2.destroyAllWindows()
  exit()

def main():
  while True:
    _, frame = camera.read()
    
    # flip camera
    frame = cv2.flip(frame, 1)
    
    drawer_box(frame)
    cv2.imshow("CharmFace AI", frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
      timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
      cv2.imwrite(f"deteksi_{timestamp}.jpg", frame)
      print(f"Frame telah disimpan sebagai deteksi_{timestamp}.jpg")
    elif key == ord('q'):
      break

if __name__ == '__main__':
  main()