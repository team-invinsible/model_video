# ----------------------------------------------------------------------------------------------------
# 작성목적 : 시각화 및 유틸리티 함수 모음
# 작성일 : 2024-06-09

# 변경사항 내역
# 2024-06-01 | 최초 구현 | 랜드마크 및 상태 시각화 기능 구현 | 이소미
# ----------------------------------------------------------------------------------------------------

import cv2
import numpy as np
import math

def draw_landmarks(frame, landmarks):
    """얼굴 랜드마크 시각화"""
    if landmarks is None:
        return
        
    h, w = frame.shape[:2]
    for landmark in landmarks:
        x = int(landmark.x * w)
        y = int(landmark.y * h)
        cv2.circle(frame, (x, y), 1, (255, 255, 255), -1)  # 흰색 점

def draw_eye_info(frame, center, outline, label="Eye", origin=(10, 30)):
    # 눈 테두리: 아주 작은 초록색 점 (반지름 1픽셀)
    for pt in outline:
        cv2.circle(frame, pt, 1, (0, 255, 0), -1)

def draw_iris_points(frame, landmarks, indices, color=(255, 255, 255)):
    h, w = frame.shape[:2]
    for idx in indices:
        pt = (int(landmarks[idx].x * w), int(landmarks[idx].y * h))
        cv2.circle(frame, pt, 2, color, -1)

def draw_head_pose_landmarks(frame, landmarks):
    h, w = frame.shape[:2]
    key_points = {
        "nose_tip": 1,
        "left_eye": 33,
        "right_eye": 263,
        "left_mouth": 61,
        "right_mouth": 291,
        "chin": 199
    }
    for label, idx in key_points.items():
        pt = (int(landmarks[idx].x * w), int(landmarks[idx].y * h))
        cv2.circle(frame, pt, 3, (0, 255, 255), -1)  # yellow
        cv2.putText(frame, label, (pt[0]+5, pt[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)

def draw_status(frame, gaze_direction, head_direction, is_calibrating):
    """상태 정보 시각화"""
    h, w = frame.shape[:2]
    
    # 상태 텍스트 색상
    color = (0, 0, 255) if is_calibrating else (0, 255, 0)
    
    # 보정 상태
    if is_calibrating:
        status = "보정 중..."
    else:
        status = f"시선: {gaze_direction}, 고개: {head_direction}"
    
    # 텍스트 그리기
    cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

# 눈 감았는지 판단 - 유클리드 거리로 판단, 좌표가 기울어져도 거리 자체가 좁아지면 감지
def are_eyes_closed_by_landmark_distance(landmarks, eye="left", threshold=0.015):
    if eye == "left":
        top_idx, bottom_idx = 159, 145
    else:
        top_idx, bottom_idx = 386, 374

    x_top, y_top = landmarks[top_idx].x, landmarks[top_idx].y
    x_bot, y_bot = landmarks[bottom_idx].x, landmarks[bottom_idx].y
    distance = math.sqrt((x_top - x_bot)**2 + (y_top - y_bot)**2)

    return distance < threshold