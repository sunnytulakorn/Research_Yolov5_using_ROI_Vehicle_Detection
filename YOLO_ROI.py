import csv
import os
import torch
import cv2
import numpy as np


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

num_video = int(input("Enter video number (1-10) :"))
video_name = str(num_video)
cap = cv2.VideoCapture(f'video/{video_name}.mp4')


fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = frame_count/fps
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(f'output_{video_name}.avi', fourcc, fps, (1020, 500))

area1_list = [
    [],
    [(325, 491), (373, 323), (395, 192), (360, 65), (338, 17),
     (281, 15), (270, 120), (153, 263), (11, 413), (325, 491)],
    [(389, 498), (485, 222), (503, 183), (449, 179),
     (401, 201), (4, 370), (3, 498), (389, 498)],
    [(2, 483), (1014, 486), (1014, 200), (765, 17),
     (471, 11), (281, 170), (5, 413), (2, 483)],
    [(2, 483), (396, 485), (482, 251), (505, 213), (520, 193),
     (481, 190), (435, 213), (233, 313), (4, 430), (2, 483)],
    [(38, 484), (470, 487), (513, 304), (515, 271),
     (517, 257), (470, 261), (378, 312), (38, 484)],
    [(6, 321), (6, 493), (71, 488), (338, 348), (534, 232), (602, 174),
     (612, 131), (532, 129), (462, 167), (276, 238), (6, 321)],
    [(72, 487), (1016, 487), (1016, 383),
     (683, 6), (560, 3), (552, 21), (72, 487)],
    [(4, 492), (436, 491), (508, 214), (469, 213), (442, 227), (4, 451), (4, 492)],
    [(4, 492), (469, 492), (530, 296), (541, 272), (562, 260),
     (488, 258), (450, 272), (346, 310), (205, 367), (5, 471), (4, 492)],
    [(142, 493), (287, 375), (336, 307), (455, 237),
     (516, 238), (500, 493), (142, 493)],
]
area2_list = [
    [],
    [(417, 497), (446, 268), (418, 135), (365, 59), (345, 17), (404, 19),
     (449, 51), (543, 142), (593, 234), (598, 499), (417, 497)],
    [(640, 498), (1018, 498), (1018, 385), (624, 206),
     (570, 175), (526, 175), (528, 204), (640, 498)],
    [],
    [(604, 487), (1014, 487), (664, 255), (585, 191),
     (539, 189), (538, 205), (545, 300), (604, 487)],
    [(604, 487), (1014, 487), (1014, 463), (606, 279),
     (564, 258), (526, 257), (533, 275), (559, 361), (604, 487)],
    [(183, 488), (714, 492), (797, 274), (771, 183), (710, 128),
     (624, 129), (618, 176), (541, 256), (310, 401), (183, 488)],
    [],
    [(605, 495), (1013, 495),  (1013, 475),
     (579, 211), (537, 210), (539, 230), (605, 473)],
    [(637, 492), (1014, 492), (1014, 440), (668, 279), (635, 267),
     (631, 259), (571, 257), (563, 267), (563, 267), (563, 296), (637, 492)],
    [(579, 493), (944, 492), (635, 265), (599, 238), (529, 237), (579, 493)]

]

area1 = area1_list[num_video]
if (num_video != 3 and num_video != 7):
    area2 = area2_list[num_video]

area1_s = find_polygon_area(np.array(area1, np.int32))
if (num_video != 3 and num_video != 7):
    area2_s = find_polygon_area(np.array(area2, np.int32))


if (num_video != 3 and num_video != 7):
    print("Area of left lane =", area1_s)
    print("Area of right lane =", area2_s)
else:
    print("Area of lane =", area1_s)

count = 0
second = 0
set_second = 2  # change second here
density1 = 0
density2 = 0
list1 = []
list2 = []

list_den1 = []
list_num1 = []
list_area1 = []
list_second1 = []

list_den2 = []
list_num2 = []
list_area2 = []
list_second2 = []

folder_name = f"image_folder_{video_name}"
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    count += 1
    if int(count % fps) == 0:
        second += 1

        if second % set_second == 0:
            density1 = ((len(list1)/(area1_s**0.5))/set_second)*100

            list_den1.append(density1)
            list_num1.append(len(list1))
            list_area1.append(area1_s**0.5)
            list_second1.append(second)

            if (num_video != 3 and num_video != 7):
                density2 = ((len(list2)/(area2_s**0.5))/set_second)*100
                list_den2.append(density2)
                list_num2.append(len(list2))
                list_area2.append(area2_s**0.5)
                list_second2.append(second)
                if density1 > density2:
                    print("Now left lane is more density")
                elif density2 > density1:
                    print("Now right lane is more density")
                else:
                    print("Same density")
    if (num_video != 3 and num_video != 7):
        cv2.putText(frame, "Density1 in "+str(set_second)+" second = "+str(density1), (25, 150),
                    cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
        cv2.putText(frame, "Density2 in "+str(set_second)+" second = "+str(density2), (700, 150),
                    cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
    else:
        cv2.putText(frame, "Density1 in "+str(set_second)+" second = "+str(density1), (25, 150),
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
            if (num_video != 3 and num_video != 7):
                result2 = cv2.pointPolygonTest(
                    np.array(area2, np.int32), ((cx, cy)), False)
                if result2 >= 0:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, str(d), (x1, y1),
                                cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 2)  # text name of object
                    cv2.circle(frame, (cx, cy), 2, (0, 255, 0), -1)
                    list2.append([cx])
            result1 = cv2.pointPolygonTest(
                np.array(area1, np.int32), ((cx, cy)), False)
            if result1 >= 0:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, str(d), (x1, y1),
                            cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 2)  # text name of object
                cv2.circle(frame, (cx, cy), 2, (0, 255, 0), -1)
                list1.append([cx])

    cv2.polylines(frame, [np.array(area1, np.int32)], True, (0, 0, 255), 2)
    if (num_video != 3 and num_video != 7):
        cv2.polylines(frame, [np.array(area2, np.int32)], True, (255, 0, 0), 2)
        a = "Vehicle in left lane  = " + str(len(list1))
        cv2.putText(frame, str(a), (25, 50),
                    cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)
        b = "Vehicle in right lane  = " + str(len(list2))
        cv2.putText(frame, str(b), (550, 50),
                    cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)
    else:
        a = "Vehicle in lane  = " + str(len(list1))
        cv2.putText(frame, str(a), (25, 50),
                    cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)

    if int(count % fps) == 0 and second % set_second == 0:
        image_file = os.path.join(folder_name, f"frame_at_{second}_second.jpg")
        cv2.imwrite(image_file, frame)

    cv2.imshow("ROI", frame)

    out.write(frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break


if (num_video != 3 and num_video != 7):
    with open(f'output_left_lane_{video_name}.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Second", 'Density', 'Number of vehicle', 'Area'])
        for row in zip(list_second1, list_den1, list_num1, list_area1,):
            writer.writerow(row)
    with open(f'output_right_lane_{video_name}.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Second", 'Density', 'Number of vehicle', 'Area'])
        for row in zip(list_second2, list_den2, list_num2, list_area2):
            writer.writerow(row)
else:
    with open(f'output_{video_name}.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Second", 'Density', 'Number of vehicle', 'Area'])
        for row in zip(list_second1, list_den1, list_num1, list_area1,):
            writer.writerow(row)
cap.release()
# stream.release()
cv2.destroyAllWindows()
