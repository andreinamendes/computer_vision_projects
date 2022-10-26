import cv2
import mediapipe as mp
import time

class FaceMeshDetector():
    def __init__(self, staticMode=False, maxFaces=2, minDetectionCon=0.5, minTrackCon=0.5):
        self.staticMode = staticMode
        self.maxFaces = maxFaces
        self.minDetectionCon = minDetectionCon
        self.minTrackCon = minTrackCon
        
        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        print(self.staticMode)
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.staticMode, self.maxFaces, refine_landmarks=True, self.minDetectionCon, self.minTrackCon)
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=1)

    def findFaceMesh(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.faceMesh.process(imgRGB)

        if results.multi_face_landmarks:
            for faceLm in results.multi_face_landmarks:
                self.mpDraw.draw_landmarks(img, faceLm, self.mpFaceMesh.FACEMESH_CONTOURS, self.drawSpec, self.drawSpec)

                # for l, in faceLm.landmark:
                #     ih, iw, ic = img.shape
                #     x, y = int(lm.x * iw), int(lm.y * ih)

        return img

def main():
    cap = cv2.VideoCapture("http://192.168.0.5:4747/video")
    pTime = 0
    detector = FaceMeshDetector()

    while True:
        success, img = cap.read()
        #img = detector.findFaceMesh(img=img)
        cTime = time.time()
        fps = 1/(cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
        cv2.imshow('image', img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()