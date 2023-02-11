import torch
import cv2
import numpy as np
import time


def POINTS(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        colorsBGR = [x, y]
        print(colorsBGR)


def find_polygon_area(pts):
    # pts should be a numpy array of shape (n, 2) containing the n vertices of the polygon
    x = pts[:, 0]
    y = pts[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


cv2.namedWindow('ROI')
cv2.setMouseCallback('ROI', POINTS)


model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
count = 0
cap = cv2.VideoCapture('video/9.mp4')

area1 = [(4, 492), (469, 492), (530, 296), (541, 272),
         (456, 272), (346, 310), (205, 367), (5, 471), (4, 492)]
area2 = [(637, 492), (1014, 492), (1014, 440), (668, 279),
         (635, 267), (563, 267), (563, 267), (563, 296), (637, 492)]

area1_s = find_polygon_area(np.array(area1, np.int32))
area2_s = find_polygon_area(np.array(area2, np.int32))

print("Area of left lane =", area1_s)
print("Area of right lane =", area2_s)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    count += 1
    if count % 3 != 0:
        continue

    frame = cv2.resize(frame, (1020, 500))
    results = model(frame)
    list1 = []
    list2 = []
    for index, row in results.pandas().xyxy[0].iterrows():
        x1 = int(row['xmin'])
        y1 = int(row['ymin'])
        x2 = int(row['xmax'])
        y2 = int(row['ymax'])
        d = (row['name'])
#        print(d)
        cx = int(x1+x2)//2
        cy = int(y1+y2)//2
        if "car" or "truck" or "bus" or "motorcycle" in d:
            result1 = cv2.pointPolygonTest(
                np.array(area1, np.int32), ((cx, cy)), False)
            result2 = cv2.pointPolygonTest(
                np.array(area2, np.int32), ((cx, cy)), False)
            if result1 >= 0: 
                # if result1 >= 0 :
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # cv2.putText(frame, str(d), (x1, y1),
                #         cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 2)  # text name of object
                cv2.circle(frame, (cx, cy), 2, (0, 255, 0), -1)
                list1.append([cx])
            elif result2 >= 0:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # cv2.putText(frame, str(d), (x1, y1),
                #         cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 2)  # text name of object
                cv2.circle(frame, (cx, cy), 2, (0, 255, 0), -1)
                list2.append([cx])
    cv2.polylines(frame, [np.array(area1, np.int32)], True, (0, 0, 255), 2)
    cv2.polylines(frame, [np.array(area2, np.int32)], True, (255, 0, 0), 2)
    a = (len(list1))
    b = (len(list2))
    cv2.putText(frame, str(a), (50, 50),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(frame, str(b), (965, 50),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)

    cv2.imshow("ROI", frame)
    time.sleep(0.1)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
# stream.release()
cv2.destroyAllWindows()
