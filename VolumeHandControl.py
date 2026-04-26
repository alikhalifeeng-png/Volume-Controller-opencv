import cv2
import time
import numpy as np
import HandTrackingModule as htm
import math
from pycaw.pycaw import AudioUtilities

###########################
wCam,hCam = 1280,720
###########################
cap = cv2.VideoCapture(0)
cap.set(3,wCam)
cap.set(4,hCam)
pTime = 0
detector = htm.HandDetector(detectionCon=0.8)

device = AudioUtilities.GetSpeakers()
volume = device.EndpointVolume
volRange = volume.GetVolumeRange()
minVol = volRange[0]
maxVol = volRange[1]
vol = 0
volBar = 400
volPer=0

while True:
    success, image = cap.read()
    if not success:
        break
    image = detector.findHands(image)
    lmList = detector.findPosition(image, draw=False)
    if len(lmList) != 0:
      #print(lmList[4],lmList[8])
      x1,y1 = lmList[4][1],lmList[4][2]
      x2,y2 = lmList[8][1],lmList[8][2]
      cx,cy = (x1 + x2)//2 ,(y1+y2)//2
      cv2.circle(image,(x1,y1),5,(0,0,255),2,cv2.FILLED)
      cv2.circle(image, (x2, y2), 5, (0, 0, 255), 3, cv2.FILLED)
      cv2.circle(image, (cx, cy), 5, (0, 0, 255), 3, cv2.FILLED)

      cv2.line(image,(x1,y1),(x2,y2),(0,0,255),3)
      length = math.hypot(x2-x1,y2-y1)
      print(length)
      # Hand Range 50 - 300
      # Volume range -65 to 0

      vol = np.interp(length, [50, 300], [minVol, maxVol])
      volBar = np.interp(length, [50, 300], [400,150])
      volPer = np.interp(length, [50, 300], [0,100])


      print(vol)
      volume.SetMasterVolumeLevel(vol, None)

      if length < 50:
          cv2.circle(image, (cx, cy), 5, (0, 255, 0), 3, cv2.FILLED)

    cv2.rectangle(image,(50,150),(85,400),(0,255,0),3)
    cv2.rectangle(image,(50,int(volBar)),(85,400),(0,255,0),cv2.FILLED)


    cTime = time.time()
    fps = 1 / (cTime - pTime) if pTime != 0 else 0
    pTime = cTime
    cv2.putText(image,f'FPS{int(fps)}',(40,70),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
    cv2.putText(image,f'{int(volPer)}%',(40,450),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

    cv2.imshow('img', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
