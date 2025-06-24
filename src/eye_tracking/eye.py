# ----------------------------------------------------------------------------------------------------
# 작성목적 : 눈 영역 분석 및 EAR(Eye Aspect Ratio) 계산
# 작성일 : 2024-06-09

# 변경사항 내역
# 2024-06-01 | 최초 구현 | 눈 영역 분석 및 EAR 계산 기능 구현 | 이소미
# ----------------------------------------------------------------------------------------------------

import numpy as np


class EyeAnalyzer:
    def __init__(self):
        # 시각화용 인덱스
        self.LEFT_EYE = [362, 263, 386, 374, 380, 385]  # 화면 기준 왼쪽 눈
        self.RIGHT_EYE = [33, 133, 159, 145, 153, 158]  # 화면 기준 오른쪽 눈

        # EAR 계산용 인덱스
        self.LEFT_EAR_INDICES = [33, 133, 160, 159, 158, 144]
        self.RIGHT_EAR_INDICES = [263, 362, 387, 386, 385, 373]

    def get_eye_info(self, landmarks, eye="left"):
        indices = self.LEFT_EYE if eye == "left" else self.RIGHT_EYE
        points = [(int(landmarks[i].x * 640), int(landmarks[i].y * 480)) for i in indices]
        center_x = int(np.mean([pt[0] for pt in points]))
        center_y = int(np.mean([pt[1] for pt in points]))
        return (center_x, center_y), points

    def compute_ear(self, landmarks, eye="left"):
        idx = self.LEFT_EAR_INDICES if eye == "left" else self.RIGHT_EAR_INDICES
        p1 = np.array([landmarks[idx[0]].x, landmarks[idx[0]].y])
        p2 = np.array([landmarks[idx[2]].x, landmarks[idx[2]].y])
        p3 = np.array([landmarks[idx[3]].x, landmarks[idx[3]].y])
        p4 = np.array([landmarks[idx[1]].x, landmarks[idx[1]].y])
        p5 = np.array([landmarks[idx[5]].x, landmarks[idx[5]].y])
        p6 = np.array([landmarks[idx[4]].x, landmarks[idx[4]].y])

        vertical1 = np.linalg.norm(p2 - p6)
        vertical2 = np.linalg.norm(p3 - p5)
        horizontal = np.linalg.norm(p1 - p4)

        ear = (vertical1 + vertical2) / (2.0 * horizontal)
        return ear
