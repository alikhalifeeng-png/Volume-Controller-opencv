import cv2
import mediapipe as mp
import time

class HandDetector:
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        """
        HandDetector class using MediaPipe.
        - mode: static_image_mode
        - maxHands: maximum number of hands to detect
        - detectionCon: minimum detection confidence (0-1)
        - trackCon: minimum tracking confidence (0-1)
        """
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )
        self.mpDraw = mp.solutions.drawing_utils
        self.results = None

    def findHands(self, img, draw=True):
        """
        Detect hands in the frame and optionally draw landmarks.
        """
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(
                        img, handLms, self.mpHands.HAND_CONNECTIONS
                    )
        return img

    def findPosition(self, img, handNo=0, draw=True):
        """
        Returns a list of landmark positions for the specified hand.
        Each landmark is [id, x, y]
        """
        lmList = []

        if self.results and self.results.multi_hand_landmarks:
            if handNo < len(self.results.multi_hand_landmarks):
                myHand = self.results.multi_hand_landmarks[handNo]

                for id, lm in enumerate(myHand.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lmList.append([id, cx, cy])

                    # Draw a circle on the wrist or thumb tip (id 0)
                    if draw and id == 0:
                        cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

        return lmList


def main():
    cTime = 0
    pTime = 0

    cap = cv2.VideoCapture(0)
    detector = HandDetector(detectionCon=0.7, trackCon=0.7)  # slightly higher confidence

    while True:
        success, img = cap.read()
        if not success:
            continue

        # Detect hands and draw landmarks
        img = detector.findHands(img)

        # Get positions of landmarks
        lmList = detector.findPosition(img)

        # Print thumb tip coordinates (landmark 4)
        if len(lmList) > 4:  # ensure landmark 4 exists
            for i in range(21):
             id, x, y = lmList[i]
             print(f"[{id}, {x}, {y}]")

        # Calculate and display FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime) if (cTime - pTime) != 0 else 0
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Show the image
        cv2.imshow('Hand Tracking', img)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()