import numpy as np
import cv2 as cv
import pyautogui

video = cv.VideoCapture(0)
sw,sh = pyautogui.size()

while video:
    r,frame = video.read()
    frame = cv.flip(frame, 1)
    hsv = cv.cvtColor(frame,cv.COLOR_BGR2HSV)
    lower_skin = np.array([0, 20, 20], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    mask = cv.inRange(hsv,lower_skin,upper_skin)
    mask = cv.dilate(mask,np.ones((7,7),np.uint8),iterations=1)
    c,h = cv.findContours(mask,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
    if r:
        max_contours = max(c,key=cv.contourArea)
        # print(max_contours)
        m = cv.moments(max_contours)
        x = int(m['m10']/m['m00'])
        y = int(m['m01']/m['m00'])
        hull = cv.convexHull(max_contours,returnPoints=False)
        defects = cv.convexityDefects(max_contours,hull)
        fingerCount = 0
        # print('Defects:',defects)
        for i in defects:
            start,end,farthest,distance = i[0]
            st = tuple(max_contours[start][0])
            en = tuple(max_contours[end][0])
            far = tuple(max_contours[farthest][0])

            a = np.linalg.norm(np.array(st) - np.array(far))
            b = np.linalg.norm(np.array(en) - np.array(far))
            c = np.linalg.norm(np.array(st) - np.array(en))

            angle = np.arccos((a**2 + b**2 - c**2)/(2*a*b))
            if angle < np.pi/2 and distance > 20:
                fingerCount += 1

        cv.drawContours(frame,[max_contours],-1,(0,0,255),3)
        fw,fh = video.get(3),video.get(4)
        # cv.circle(frame,(x,y),2,(0,255,0),3)

        mx = x * (sw/fw)
        my = y * (sh/fh)
        pyautogui.moveTo(mx,my)

        print('FingerCount:', fingerCount)
        if fingerCount == 0:
            pyautogui.leftClick(mx,my,duration=2.0)
            print("Mouse Left Click")

        cv.imshow('window',frame)
        if cv.waitKey(30) & 0xff == ord('q'):
            break
    else:
        break