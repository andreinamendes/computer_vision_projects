import cv2
import mediapipe as mp
import time

class FaceMeshDetector():
    def __init__(self, staticMode=False, maxFaces=2, minDetectionCon=0.5, minTrackCon=0.5):
        self.staticMode = staticMode
        self.maxFaces = maxFaces
        self.minDetectionCon = minDetectionCon
        self.minTrackCon = minTrackCon
        
        self.mpDraw = mp.solutions.drawing_utils  # type: ignore
        self.mpFaceMesh = mp.solutions.face_mesh  # type: ignore
        self.faceMesh = self.mpFaceMesh.FaceMesh(static_image_mode=self.staticMode, max_num_faces=self.maxFaces, min_detection_confidence=self.minDetectionCon, min_tracking_confidence=self.minTrackCon)
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=1)

    def findFaceMesh(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.faceMesh.process(imgRGB)
        faces = []
        
        if results.multi_face_landmarks:
            for faceLm in results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, faceLm, self.mpFaceMesh.FACEMESH_CONTOURS, self.drawSpec, self.drawSpec)

                face = []
                for id, lm in enumerate(faceLm.landmark):
                    ih, iw, ic = img.shape
                    x, y = int(lm.x * iw), int(lm.y * ih)
                    # cv2.putText(img, str(id), (x, y), cv2.FONT_HERSHEY_PLAIN, 0.3, (0, 255, 0), 1)
                    face.append([x, y])
                
                faces.append(face)

        return img, faces

def main():
    cap = cv2.VideoCapture("http://192.168.0.100:4747/video")
    pTime = 0.0
    detector = FaceMeshDetector()

    while True:
        success, img = cap.read()
        cTime = time.time()
        img, faces = detector.findFaceMesh(img, True)
        # if len(faces) != 0:
        #     print(faces[0])
        fps = 1/(cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
        cv2.imshow('image', img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()