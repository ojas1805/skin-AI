import cv2
import mediapipe as mp
import numpy as np

class FaceDetector:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )

    def get_landmarks(self, image_path):
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(image_rgb)

        if not results.multi_face_landmarks:
            return None, None

        landmarks = results.multi_face_landmarks[0]
        h, w = image.shape[:2]

        points = {}
        for idx, lm in enumerate(landmarks.landmark):
            points[idx] = (int(lm.x * w), int(lm.y * h))

        return image, points

    def get_cheek_roi(self, image, landmarks):
        left_cheek_indices  = [50, 101, 118, 117, 116]
        right_cheek_indices = [280, 330, 347, 346, 345]

        rois = []
        for indices in [left_cheek_indices, right_cheek_indices]:
            pts = np.array([landmarks[i] for i in indices])
            x, y, w, h = cv2.boundingRect(pts)
            pad = 10
            x, y = max(0, x-pad), max(0, y-pad)
            roi = image[y:y+h+pad, x:x+w+pad]
            if roi.size > 0:
                rois.append(roi)

        return rois