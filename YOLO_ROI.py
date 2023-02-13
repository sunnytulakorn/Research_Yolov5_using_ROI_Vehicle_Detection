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
cap = cv2.VideoCapture('videoplayback11.mp4')

fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = frame_count/fps
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, fps, (1020, 500))
area1 = [(325, 491), (373, 323), (395, 192), (360, 65), (275, 66),
         (270, 120), (153, 263), (11, 413), (325, 491)]
area2 = [(417, 497), (446, 268), (427, 137), (400, 79), (443, 87),
         (514, 167), (562, 321), (575, 498), (417, 497)]

area1_s = find_polygon_area(np.array(area1, np.int32))
area2_s = find_polygon_area(np.array(area2, np.int32))

print("Area of left lane =", area1_s)
print("Area of right lane =", area2_s)

second = 0
set_second = 5  # change second here
density1 = 0
density2 = 0
list1 = []
list2 = []
while True:
    ret, frame = cap.read()
    if not ret:
        break
    count += 1
    if count == fps:
        second += 1
        if second == set_second:
            density1 = (len(list1)/(area1_s**0.5))/set_second
            density2 = (len(list2)/(area2_s**0.5))/set_second
            second = 0
            if density1 > density2:
                print("Now left lane is more density")
            elif density2 > density1:
                print("Now right lane is more density")
            else:
                print("Same density")
        count = 0
    cv2.putText(frame, "Density1 in "+str(set_second)+" second = "+str(density1), (25, 150),
                cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
    cv2.putText(frame, "Density2 in "+str(set_second)+" second = "+str(density2), (700, 150),
                cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
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
    a = "Vehicle in left lane  = " + str(len(list1))
    b = "Vehicle in right lane  = " + str(len(list2))
    cv2.putText(frame, str(a), (25, 50),
                cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, str(b), (550, 50),
                cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("ROI", frame)
    out.write(frame)
    # time.sleep(0.1)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
# stream.release()
cv2.destroyAllWindows()
