# ----------------------------------------------------------------------------------------------------
# 작성목적 : MediaPipe 기반 얼굴 랜드마크 감지
# 작성일 : 2024-06-09

# 변경사항 내역
# 2024-06-01 | 최초 구현 | MediaPipe Face Mesh 기반 랜드마크 감지 구현 | 이소미
# ----------------------------------------------------------------------------------------------------

import cv2
import mediapipe as mp

class FaceMeshDetector:
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def get_landmarks(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        if results.multi_face_landmarks:
            return results.multi_face_landmarks[0].landmark
        return None
