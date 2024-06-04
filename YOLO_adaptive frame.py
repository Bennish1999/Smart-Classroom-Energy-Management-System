import cv2
import pandas as pd
from ultralytics import YOLO
import cvzone
import numpy as np
import ctypes

user32 = ctypes.windll.user32
screen_width, screen_height = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)

model = YOLO('best.pt')

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:  
        point = [x, y]
        print(point)

cv2.namedWindow('RGB1', cv2.WINDOW_NORMAL)
cv2.namedWindow('RGB2', cv2.WINDOW_NORMAL)
cv2.setMouseCallback('RGB1', RGB)
cv2.setMouseCallback('RGB2', RGB)

videos = ['classroom1.mp4', 'classroom2.mp4']
caps = [cv2.VideoCapture(video) for video in videos]
window_names = ['RGB1', 'RGB2']

my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")

zone1_base_classroom1 = [(580,103),(290,234),(948,483),(1014,218)]
zone2_base_classroom1 = [(331,30),(113,114),(286,231),(574,99)]
zone3_base_classroom1 = [(700,24),(587,99),(1014,213),(1018,78)]
zone4_base_classroom1 = [(467,1),(338,27),(579,95),(694,23)]

zone1_base_classroom2 = [(521,131),(794,219),(980,120),(773,63)]
zone2_base_classroom2 = [(60,260),(138,494),(788,222),(515,135)]
zone3_base_classroom2 = [(375,48),(516,128),(771,60),(590,3)]
zone4_base_classroom2 = [(54,120),(59,255),(510,130),(368,50)]

# Set the minimum confidence score threshold (e.g., 0.1)
conf_threshold = 0.06

while True:
    for i, cap in enumerate(caps):
        window_name = window_names[i]
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        count = 0
        if count % 3 == 0:
            frame = cv2.resize(frame, (screen_width, screen_height))
            results = model.predict(frame, conf=conf_threshold)  # Set the confidence threshold
            a = results[0].boxes.data
            px = pd.DataFrame(a).astype("float")

            list1 = []
            list2 = []
            list3 = []
            list4 = []
            for index, row in px.iterrows():
                x1 = int(row[0])
                y1 = int(row[1])
                x2 = int(row[2])
                y2 = int(row[3])
                d = int(row[5])
                c = class_list[d]
                if c == 'person':
                    cx = int(x1 + x2) // 2
                    cy = int(y1 + y2) // 2
                    w, h = x2 - x1, y2 - y1
                    if i == 0:
                        zone1 = [(int(pt[0] * screen_width / 1020), int(pt[1] * screen_height / 500)) for pt in zone1_base_classroom1]
                        zone2 = [(int(pt[0] * screen_width / 1020), int(pt[1] * screen_height / 500)) for pt in zone2_base_classroom1]
                        zone3 = [(int(pt[0] * screen_width / 1020), int(pt[1] * screen_height / 500)) for pt in zone3_base_classroom1]
                        zone4 = [(int(pt[0] * screen_width / 1020), int(pt[1] * screen_height / 500)) for pt in zone4_base_classroom1]
                    else:
                        zone1 = [(int(pt[0] * screen_width / 1020), int(pt[1] * screen_height / 500)) for pt in zone1_base_classroom2]
                        zone2 = [(int(pt[0] * screen_width / 1020), int(pt[1] * screen_height / 500)) for pt in zone2_base_classroom2]
                        zone3 = [(int(pt[0] * screen_width / 1020), int(pt[1] * screen_height / 500)) for pt in zone3_base_classroom2]
                        zone4 = [(int(pt[0] * screen_width / 1020), int(pt[1] * screen_height / 500)) for pt in zone4_base_classroom2]
                    result1 = cv2.pointPolygonTest(np.array(zone1, np.int32), ((cx, cy)), False)
                    if result1 >= 0:
                        cvzone.cornerRect(frame, (x1, y1, w, h), 3, 2)
                        cv2.circle(frame, (cx, cy), 4, (255, 0, 0), -1)
                        cvzone.putTextRect(frame, f'person', (x1, y1), 1, 1)
                        list1.append(cx)
                    result2 = cv2.pointPolygonTest(np.array(zone2, np.int32), ((cx, cy)), False)
                    if result2 >= 0:
                        cvzone.cornerRect(frame, (x1, y1, w, h), 3, 2)
                        cv2.circle(frame, (cx, cy), 4, (255, 0, 0), -1)
                        cvzone.putTextRect(frame, f'person', (x1, y1), 1, 1)
                        list2.append(cx)
                    result3 = cv2.pointPolygonTest(np.array(zone3, np.int32), ((cx, cy)), False)
                    if result3 >= 0:
                        cvzone.cornerRect(frame, (x1, y1, w, h), 3, 2)
                        cv2.circle(frame, (cx, cy), 4, (255, 0, 0), -1)
                        cvzone.putTextRect(frame, f'person', (x1, y1), 1, 1)
                        list3.append(cx)
                    result4 = cv2.pointPolygonTest(np.array(zone4, np.int32), ((cx, cy)), False)
                    if result4 >= 0:
                        cvzone.cornerRect(frame, (x1, y1, w, h), 3, 2)
                        cv2.circle(frame, (cx, cy), 4, (255, 0, 0), -1)
                        cvzone.putTextRect(frame, f'person', (x1, y1), 1, 1)
                        list4.append(cx)

            cr1 = len(list1)
            print(f"Zone 1 for {videos[i]}:", cr1)
            cr2 = len(list2)
            print(f"Zone 2 for {videos[i]}:", cr2)
            cr3 = len(list3)
            print(f"Zone 3 for {videos[i]}:", cr3)
            cr4 = len(list4)
            print(f"Zone 4 for {videos[i]}:", cr4)

            cv2.polylines(frame, [np.array(zone1, np.int32)], True, (0, 0, 255), 2)
            cvzone.putTextRect(frame, f'zone 1 : {cr1}', (900, 75), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 1)
            cv2.polylines(frame, [np.array(zone2, np.int32)], True, (0, 0, 255), 2)
            cvzone.putTextRect(frame, f'zone 2 : {cr2}', (19, 457), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 1)
            cv2.polylines(frame, [np.array(zone3, np.int32)], True, (0, 0, 255), 2)
            cvzone.putTextRect(frame, f'zone 3 : {cr3}', (370, 20), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 1)
            cv2.polylines(frame, [np.array(zone4, np.int32)], True, (0, 0, 255), 2)
            cvzone.putTextRect(frame, f'zone 4 : {cr4}', (90, 70), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 1)

            cv2.imshow(window_name, frame)

        count += 1

    if cv2.waitKey(1) & 0xFF == 27:
        break

for cap in caps:
    cap.release()
cv2.destroyAllWindows()
                        
