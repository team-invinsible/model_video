# ----------------------------------------------------------------------------------------------------
# 작성목적 : 시선 방향 분석 및 보정 로직 구현
# 작성일 : 2024-06-01

# 변경사항 내역 (날짜 | 변경목적 | 변경내용 | 작성자 순으로 기입)
# 2024-06-01 | 최초 구현 | 시선 방향 분석 및 자동 보정 기능 구현 | 이소미
# 2024-06-14 | 기능 개선 | 시선 상하 방향 정확도 향상, 눈 깜빡임과 아래 응시 구분 로직 개선 | 이소미
# 2024-06-15 | 기능 추가 | 재보정 로그 저장 기능 추가 | 이소미
# ----------------------------------------------------------------------------------------------------

import numpy as np
import time
import os
import json

class GazeAnalyzer:
    def __init__(self):
        # 얼굴 방향 기준점
        self.NOSE_TIP = 1
        self.CHIN = 199
        self.LEFT_EYE = 33    # 왼쪽 눈 외곽
        self.RIGHT_EYE = 263  # 오른쪽 눈 외곽
        
        # 목 관련 랜드마크 (양쪽 목 라인의 여러 점 사용)
        self.NECK_LEFT_POINTS = [149, 150, 136, 172, 58, 132]   # 왼쪽 목 라인의 여러 점
        self.NECK_RIGHT_POINTS = [378, 379, 365, 397, 288, 361] # 오른쪽 목 라인의 여러 점
        
        # 동공 중심점 랜드마크
        self.LEFT_IRIS_CENTER = 468   # 왼쪽 동공 중심
        self.RIGHT_IRIS_CENTER = 473  # 오른쪽 동공 중심
        
        # 눈 외곽선 랜드마크
        self.LEFT_EYE_CONTOUR = [362, 385, 387, 263, 373, 380]
        self.RIGHT_EYE_CONTOUR = [33, 158, 160, 133, 144, 153]
        
        # 보정 관련 변수
        self.calibration_start = None
        self.is_calibrated = False
        self.baseline_nose_pos = None
        self.baseline_left_iris = None   # 왼쪽 동공 기준 좌표
        self.baseline_right_iris = None  # 오른쪽 동공 기준 좌표
        self.calibration_count = 0
        self.baseline_neck_pos = None
        
        # 눈 영역 보정값 추가
        self.baseline_left_eye_height = None
        self.baseline_right_eye_height = None
        
        # 동공 위치 보정값 추가
        self.baseline_left_x_ratio = None
        self.baseline_right_x_ratio = None
        
        # 방향 감지 임계값
        self.head_x_threshold = 0.035  # 고개 좌우 방향 임계값
        self.head_y_threshold = 0.030  # 고개 상하 방향 임계값
        self.gaze_x_threshold = 0.15    # 시선 좌우 방향 임계값 (보정값 대비 변화율)
        self.gaze_y_threshold = 0.15   # 시선 상하 방향 임계값 (눈 영역 높이 변화 기준)
        
        # 얼굴 크기 관련 변수
        self.baseline_face_width = None
        self.face_width_tolerance = 0.15  # 얼굴 크기 변화 허용 범위 (15%)
        self.movement_count = 0
        self.max_movement_count = 5  # 연속된 움직임 감지 횟수
        
        # 이전 프레임들의 동공 위치 저장
        self.prev_left_positions = []
        self.prev_right_positions = []
        self.position_buffer_size = 5  # 이전 프레임 저장 개수
        
        # 눈 깜박임 관련 변수 추가
        self.blink_threshold = 0.25      # 눈 종횡비(EAR) 임계값
        self.blink_frames = []           # 최근 프레임의 눈 종횡비 저장
        self.blink_buffer_size = 5       # 분석할 프레임 수
        self.min_gaze_duration = 3       # 최소 시선 지속 프레임 수
        self.eye_closed_start = None     # 눈 감기 시작 시간
        self.max_blink_duration = 0.3    # 최대 깜박임 지속 시간 (초)
        
        # 아래를 응시하는 시간 체크를 위한 변수 추가
        self.looking_down_start = None   # 아래를 응시하기 시작한 시간
        self.min_looking_down_duration = 1.0  # 최소 아래 응시 시간 (초)
        
        # 깜빡임 타임스탬프 기록용 리스트 추가
        self.blink_timestamps = []  # 깜빡임 발생 시각(초) 저장
        
        self.recalib_log_path = os.path.join("logs", "recalib_log.jsonl")
        self._recalib_start_time = None  # 재보정 시작 임시 저장
        self.program_start_time = 0  # 프로그램 시작 시간 저장 (영상 시작 시간으로 변경)
        # 로그 파일 초기화
        with open(self.recalib_log_path, "w", encoding="utf-8") as f:
            pass
        
    def log_recalib_event(self, start_time, end_time):
        # 프로그램 시작 시간 기준으로 상대 시간 계산 (초 단위)
        relative_start = start_time - self.program_start_time
        relative_end = end_time - self.program_start_time
        
        log_entry = {
            "event": "recalib", 
            "start_time": round(relative_start, 3), 
            "end_time": round(relative_end, 3)
        }
        with open(self.recalib_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")

    def start_calibration(self, current_time=0):
        """보정 시작"""
        self.calibration_start = current_time  # time.time() 대신 current_time 사용
        self.is_calibrated = False
        self.calibration_count = 0
        self.movement_count = 0
        # 이전 위치 버퍼 초기화
        self.prev_left_positions = []
        self.prev_right_positions = []
        print("보정을 시작합니다. 1초간 정면을 바라봐주세요.")
        # 재보정 시작시간 임시 저장
        self._recalib_start_time = current_time  # time.time() 대신 current_time 사용
        
    def _calculate_face_width(self, landmarks):
        """얼굴 너비 계산 (3D 좌표 사용)"""
        # 양쪽 관자놀이 부근의 점들 사용 (고개 회전에 덜 민감한 위치)
        LEFT_POINTS = [447, 366, 401, 435, 367, 364, 394]  # 왼쪽 관자놀이 부근
        RIGHT_POINTS = [227, 137, 177, 215, 138, 135, 169]  # 오른쪽 관자놀이 부근
        
        # 3D 좌표로 각 점들의 평균 위치 계산
        left_points = np.array([[landmarks[idx].x, landmarks[idx].y, landmarks[idx].z] for idx in LEFT_POINTS])
        right_points = np.array([[landmarks[idx].x, landmarks[idx].y, landmarks[idx].z] for idx in RIGHT_POINTS])
        
        left_center = np.mean(left_points, axis=0)
        right_center = np.mean(right_points, axis=0)
        
        # 3D 공간에서의 거리 계산
        return np.linalg.norm(right_center - left_center)
        
    def _calculate_neck_position(self, landmarks):
        """목의 중심점 위치 계산 (3D 좌표 사용)"""
        # 양쪽 목 라인의 여러 점들의 평균 위치 계산
        left_points = np.array([[landmarks[idx].x, landmarks[idx].y, landmarks[idx].z] 
                               for idx in self.NECK_LEFT_POINTS])
        right_points = np.array([[landmarks[idx].x, landmarks[idx].y, landmarks[idx].z] 
                                for idx in self.NECK_RIGHT_POINTS])
        
        left_center = np.mean(left_points, axis=0)
        right_center = np.mean(right_points, axis=0)
        
        # 목의 중심점 반환 (x, y 좌표만 사용)
        center_3d = (left_center + right_center) / 2
        return center_3d[:2]  # x, y 좌표만 반환
        
    def _check_face_symmetry(self, landmarks):
        """얼굴의 좌우 대칭성 체크로 실제 회전 여부 확인"""
        left_eye = np.array([landmarks[self.LEFT_EYE].x, landmarks[self.LEFT_EYE].y])
        right_eye = np.array([landmarks[self.RIGHT_EYE].x, landmarks[self.RIGHT_EYE].y])
        nose = np.array([landmarks[self.NOSE_TIP].x, landmarks[self.NOSE_TIP].y])
        
        # 코가 양쪽 눈의 중앙에서 벗어난 정도 계산
        eye_center = (left_eye + right_eye) / 2
        nose_offset = abs(nose[0] - eye_center[0])
        
        # 눈 사이 거리 대비 코의 벗어남 정도로 대칭성 판단
        eye_distance = np.linalg.norm(right_eye - left_eye)
        symmetry_ratio = nose_offset / eye_distance
        
        return symmetry_ratio < 0.1  # 10% 이내면 대칭적
        
    def _is_looking_forward(self, landmarks):
        """사용자가 정면을 보고 있는지 확인"""
        # 1. 얼굴 대칭성 확인 (고개가 돌아가 있지 않은지)
        if not self._check_face_symmetry(landmarks):
            return False
            
        # 2. 동공 위치가 중앙에 가까운지 확인
        left_iris = np.array([landmarks[self.LEFT_IRIS_CENTER].x, landmarks[self.LEFT_IRIS_CENTER].y])
        right_iris = np.array([landmarks[self.RIGHT_IRIS_CENTER].x, landmarks[self.RIGHT_IRIS_CENTER].y])
        
        # x, y 좌표 모두 중앙(0.5)에서 ±0.2 이내에 있어야 함
        iris_threshold = 0.2
        
        # 양쪽 동공의 평균 위치로 판단
        avg_x = (left_iris[0] + right_iris[0]) / 2
        avg_y = (left_iris[1] + right_iris[1]) / 2
        
        # 중앙 기준 범위 체크 (y축도 여유 있게)
        x_centered = abs(avg_x - 0.5) < iris_threshold * 1.2  # x축은 20% 더 여유
        y_centered = abs(avg_y - 0.5) < iris_threshold * 1.5  # y축은 50% 더 여유
        
        return x_centered and y_centered
        
    def _check_movement(self, landmarks):
        """사람의 이동 여부 확인"""
        if (self.baseline_face_width is None or 
            self.baseline_neck_pos is None or 
            not isinstance(self.baseline_neck_pos, np.ndarray)):
            return False
            
        # 1. 전후 움직임 체크 (얼굴 크기 변화)
        current_width = self._calculate_face_width(landmarks)
        width_diff = abs(current_width - self.baseline_face_width) / self.baseline_face_width
        
        # 2. 목 위치로 좌우 이동 체크
        current_neck_pos = self._calculate_neck_position(landmarks)
        neck_x_diff = abs(current_neck_pos[0] - self.baseline_neck_pos[0])
        neck_threshold = 0.1  # 목 위치가 10% 이상 이동하면 움직임으로 판단
        
        # 움직임 판단 - 전후와 좌우 모두 체크
        has_forward_movement = width_diff > self.face_width_tolerance
        has_sideways_movement = neck_x_diff > neck_threshold
        
        if has_forward_movement or has_sideways_movement:
            self.movement_count += 1
            if self.movement_count >= self.max_movement_count:
                # 움직임 종류 판단
                movements = []
                if has_sideways_movement:
                    movements.append("좌우")
                if has_forward_movement:
                    movements.append("전후")
                reason = " 및 ".join(movements)
                print(f"{reason} 움직임이 감지되어 재보정을 시작합니다.")
                return True
        else:
            self.movement_count = max(0, self.movement_count - 1)
            
        return False
        
    def analyze_head_pose(self, landmarks, current_time):
        """고개 움직임 분석"""
        if not landmarks:
            return "center", False
            
        # 보정 상태 확인
        if not self.is_calibrated:
            if self.calibration_start is None:
                # 보정 시작 전 정면 응시 확인
                if not self._is_looking_forward(landmarks):
                    print("정면을 응시해주세요.")
                    return "not_ready", False
                    
                self.start_calibration(current_time)  # current_time 전달
                # 재보정 시작 시 이전 기준값 초기화
                self.baseline_nose_pos = None
                self.baseline_left_iris = None
                self.baseline_right_iris = None
                self.baseline_face_width = None
                self.baseline_neck_pos = None
                self.baseline_left_eye_height = None
                self.baseline_right_eye_height = None
                return "calibrating", False
                
            if current_time - self.calibration_start < 1.0:  # 영상 시간 기준
                # 보정 중 정면 응시 확인
                if not self._is_looking_forward(landmarks):
                    print("보정 중 정면에서 벗어났습니다. 다시 정면을 응시해주세요.")
                    self.start_calibration(current_time)  # current_time 전달
                    # 재보정 시작 시 이전 기준값 초기화
                    self.baseline_nose_pos = None
                    self.baseline_left_iris = None
                    self.baseline_right_iris = None
                    self.baseline_face_width = None
                    self.baseline_neck_pos = None
                    self.baseline_left_eye_height = None
                    self.baseline_right_eye_height = None
                    return "not_ready", False
                    
                # 보정 중에는 기준값 누적하여 평균 계산
                nose = np.array([landmarks[self.NOSE_TIP].x, landmarks[self.NOSE_TIP].y])
                left_iris = np.array([landmarks[self.LEFT_IRIS_CENTER].x, landmarks[self.LEFT_IRIS_CENTER].y])
                right_iris = np.array([landmarks[self.RIGHT_IRIS_CENTER].x, landmarks[self.RIGHT_IRIS_CENTER].y])
                face_width = self._calculate_face_width(landmarks)
                neck_pos = self._calculate_neck_position(landmarks)
                
                # 눈 영역 높이 계산 추가
                left_eye_height = self._calculate_eye_height(landmarks, "left")
                right_eye_height = self._calculate_eye_height(landmarks, "right")
                
                # 눈 영역 계산
                left_eye_left, left_eye_right, left_eye_top, left_eye_bottom = self._get_eye_region(landmarks, "left")
                right_eye_left, right_eye_right, right_eye_top, right_eye_bottom = self._get_eye_region(landmarks, "right")
                
                # 동공 상대 위치 계산
                left_x_ratio = (left_iris[0] - left_eye_left) / (left_eye_right - left_eye_left)
                right_x_ratio = (right_iris[0] - right_eye_left) / (right_eye_right - right_eye_left)
                
                self.calibration_count += 1
                
                if self.baseline_nose_pos is None:
                    self.baseline_nose_pos = nose
                    self.baseline_left_iris = left_iris
                    self.baseline_right_iris = right_iris
                    self.baseline_face_width = face_width
                    self.baseline_neck_pos = neck_pos
                    self.baseline_left_eye_height = left_eye_height
                    self.baseline_right_eye_height = right_eye_height
                    self.baseline_left_x_ratio = left_x_ratio
                    self.baseline_right_x_ratio = right_x_ratio
                else:
                    # 이동 평균 계산
                    self.baseline_nose_pos = (self.baseline_nose_pos * (self.calibration_count - 1) + nose) / self.calibration_count
                    self.baseline_left_iris = (self.baseline_left_iris * (self.calibration_count - 1) + left_iris) / self.calibration_count
                    self.baseline_right_iris = (self.baseline_right_iris * (self.calibration_count - 1) + right_iris) / self.calibration_count
                    self.baseline_face_width = (self.baseline_face_width * (self.calibration_count - 1) + face_width) / self.calibration_count
                    self.baseline_neck_pos = (self.baseline_neck_pos * (self.calibration_count - 1) + neck_pos) / self.calibration_count
                    self.baseline_left_eye_height = (self.baseline_left_eye_height * (self.calibration_count - 1) + left_eye_height) / self.calibration_count
                    self.baseline_right_eye_height = (self.baseline_right_eye_height * (self.calibration_count - 1) + right_eye_height) / self.calibration_count
                    self.baseline_left_x_ratio = (self.baseline_left_x_ratio * (self.calibration_count - 1) + left_x_ratio) / self.calibration_count
                    self.baseline_right_x_ratio = (self.baseline_right_x_ratio * (self.calibration_count - 1) + right_x_ratio) / self.calibration_count
                
                return "calibrating", False
            else:
                # 보정 완료 전에 모든 기준값이 제대로 설정되었는지 확인
                if (self.baseline_nose_pos is None or 
                    self.baseline_left_iris is None or 
                    self.baseline_right_iris is None or 
                    self.baseline_face_width is None or 
                    self.baseline_neck_pos is None or 
                    self.baseline_left_eye_height is None or 
                    self.baseline_right_eye_height is None or 
                    self.baseline_left_x_ratio is None or 
                    self.baseline_right_x_ratio is None):
                    print("보정값이 올바르게 설정되지 않았습니다. 재보정을 시작합니다.")
                    self.start_calibration(current_time)
                    return "recalibrating", False
                    
                self.is_calibrated = True
                print("보정이 완료되었습니다.")
                # 재보정 종료 로그 기록 (한 줄)
                self.finish_calibration(current_time)
                return "center", True
        
        # 움직임 감지 시 재보정 시작
        if self._check_movement(landmarks):
            self.start_calibration(current_time)
            return "recalibrating", False
                
        # 보정이 완료된 후의 분석
        # 보정값이 없으면 재보정 시작
        if (self.baseline_nose_pos is None or 
            self.baseline_left_iris is None or 
            self.baseline_right_iris is None or 
            self.baseline_face_width is None or 
            self.baseline_neck_pos is None):
            print("보정값이 유실되었습니다. 재보정을 시작합니다.")
            self.start_calibration(current_time)
            return "recalibrating", False
            
        nose = np.array([landmarks[self.NOSE_TIP].x, landmarks[self.NOSE_TIP].y])
        
        # 좌우 움직임 (x축)
        x_diff = nose[0] - self.baseline_nose_pos[0]
        # 상하 움직임 (y축)
        y_diff = nose[1] - self.baseline_nose_pos[1]
        
        direction = []
        
        # 좌우 방향은 대칭성 체크 필요
        is_symmetric = self._check_face_symmetry(landmarks)
        if not is_symmetric:
            if x_diff < -self.head_x_threshold:
                direction.append("left")
            elif x_diff > self.head_x_threshold:
                direction.append("right")
        
        # 상하 방향은 대칭성 체크와 무관하게 판단
        if y_diff < -self.head_y_threshold:
            direction.append("up")
        elif y_diff > self.head_y_threshold:
            direction.append("down")
            
        return " ".join(direction) if direction else "center", True
    
    def _get_eye_region(self, landmarks, eye="left"):
        """눈 영역의 경계 좌표 계산"""
        if eye == "left":
            # 눈꺼풀 위/아래 점 추가
            upper_points = [159, 160, 161, 246]  # 위쪽 눈꺼풀
            lower_points = [145, 144, 163, 7]    # 아래쪽 눈꺼풀
            contour_points = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7]
        else:
            upper_points = [386, 387, 388, 466]  # 위쪽 눈꺼풀
            lower_points = [374, 373, 390, 249]  # 아래쪽 눈꺼풀
            contour_points = [263, 466, 388, 387, 386, 385, 384, 398, 362, 382, 381, 380, 374, 373, 390, 249]
            
        # 눈꺼풀 위치 계산
        upper_y = np.mean([landmarks[idx].y for idx in upper_points])
        lower_y = np.mean([landmarks[idx].y for idx in lower_points])
        
        points = np.array([[landmarks[idx].x, landmarks[idx].y] for idx in contour_points])
        x_coords = points[:, 0]
        
        # 좌우 경계는 기존처럼 계산
        x_sorted = np.sort(x_coords)
        left = np.median(x_sorted[:len(x_sorted)//2])
        right = np.median(x_sorted[len(x_sorted)//2:])
        
        # 상하 경계는 눈꺼풀 위치 기반으로 계산
        top = upper_y
        bottom = lower_y
        
        # 눈 영역 확장 (상하 방향으로 더 넓게)
        width = right - left
        height = bottom - top
        left -= width * 0.1
        right += width * 0.1
        top -= height * 0.2    # 위쪽으로 더 넓게
        bottom += height * 0.1
        
        return left, right, top, bottom
        
    def _calculate_eye_aspect_ratio(self, landmarks, eye="left"):
        """눈의 종횡비(Eye Aspect Ratio) 계산"""
        if eye == "left":
            # 왼쪽 눈 6개 점
            p1, p2 = landmarks[159], landmarks[145]  # 세로 거리 1
            p3, p4 = landmarks[158], landmarks[153]  # 세로 거리 2
            p5, p6 = landmarks[33], landmarks[133]   # 가로 거리
        else:
            # 오른쪽 눈 6개 점
            p1, p2 = landmarks[386], landmarks[374]  # 세로 거리 1
            p3, p4 = landmarks[385], landmarks[380]  # 세로 거리 2
            p5, p6 = landmarks[263], landmarks[362]  # 가로 거리
            
        # 세로 거리 계산
        v1 = np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)
        v2 = np.sqrt((p3.x - p4.x)**2 + (p3.y - p4.y)**2)
        # 가로 거리 계산
        h = np.sqrt((p5.x - p6.x)**2 + (p5.y - p6.y)**2)
        
        # 눈 종횡비 = (세로1 + 세로2) / (2 * 가로)
        ear = (v1 + v2) / (2.0 * h)
        return ear
        
    def record_blink(self, current_time):
        """깜빡임 발생 시 타임스탬프 기록"""
        self.blink_timestamps.append(current_time)

    def get_blinks_per_minute(self):
        """분당 깜빡임 횟수 계산 (0초부터 시작, 마지막 깜빡임 시각 기준)"""
        if not self.blink_timestamps:
            return 0.0
        last_time = self.blink_timestamps[-1]
        duration_min = last_time / 60 if last_time > 0 else 1
        blink_count = len(self.blink_timestamps)
        return blink_count / duration_min if duration_min > 0 else blink_count
        
    def _is_blinking(self, landmarks):
        """눈 깜박임 여부 판단"""
        # 보정값이 없으면 깜빡임으로 처리하지 않음
        if (self.baseline_left_eye_height is None or 
            self.baseline_right_eye_height is None):
            return False

        # 양쪽 눈의 종횡비 계산
        left_ear = self._calculate_eye_aspect_ratio(landmarks, "left")
        right_ear = self._calculate_eye_aspect_ratio(landmarks, "right")
        
        # 양쪽 눈의 평균 종횡비
        avg_ear = (left_ear + right_ear) / 2
        current_time = time.time()
        
        # 최근 프레임의 종횡비 저장
        self.blink_frames.append(avg_ear)
        if len(self.blink_frames) > self.blink_buffer_size:
            self.blink_frames.pop(0)
            
        # 현재 눈 영역 높이 계산
        current_left_height = self._calculate_eye_height(landmarks, "left")
        current_right_height = self._calculate_eye_height(landmarks, "right")
        
        # 보정값이 없으면 깜빡임으로 처리하지 않음
        if current_left_height is None or current_right_height is None:
            return False
            
        # 보정값 대비 높이 변화율 계산
        try:
            left_height_ratio = current_left_height / self.baseline_left_eye_height
            right_height_ratio = current_right_height / self.baseline_right_eye_height
            avg_height_ratio = (left_height_ratio + right_height_ratio) / 2
        except (TypeError, ZeroDivisionError):
            # 보정값이 없거나 0인 경우
            return False
        
        # 눈 감김/뜸 상태 판단 (눈 영역 높이로 판단)
        is_closed = avg_height_ratio < 0.3  # 눈 영역이 70% 이상 감소하면 감은 것
        was_open = len(self.blink_frames) >= 2 and self.blink_frames[-2] > self.blink_threshold
        
        # 깜빡임 판단:
        # 1. 이전에 눈이 감겼다가
        # 2. 지금 눈을 떴고
        # 3. 감긴 시간이 max_blink_duration 이내인 경우
        if self.eye_closed_start is not None:
            if not is_closed and current_time - self.eye_closed_start < self.max_blink_duration:
                self.eye_closed_start = None  # 초기화
                # 깜빡임 발생 시 타임스탬프 기록
                self.record_blink(current_time)
                return True
            elif current_time - self.eye_closed_start >= self.max_blink_duration:
                self.eye_closed_start = None  # 시간 초과로 초기화
        elif is_closed and was_open:
            # 눈 감기 시작 시간 기록
            self.eye_closed_start = current_time
        
        return False
        
    def _calculate_eye_height(self, landmarks, eye="left"):
        """눈 영역의 높이 계산"""
        if eye == "left":
            upper_points = [159, 160, 161, 246]  # 위쪽 눈꺼풀
            lower_points = [145, 144, 163, 7]    # 아래쪽 눈꺼풀
        else:
            upper_points = [386, 387, 388, 466]  # 위쪽 눈꺼풀
            lower_points = [374, 373, 390, 249]  # 아래쪽 눈꺼풀
            
        # 눈꺼풀 위치 계산
        upper_y = np.mean([landmarks[idx].y for idx in upper_points])
        lower_y = np.mean([landmarks[idx].y for idx in lower_points])
        
        return lower_y - upper_y  # 눈 영역 높이 반환

    def analyze_gaze(self, landmarks):
        """양쪽 눈의 시선 방향을 통합 분석"""
        if not landmarks or not self.is_calibrated:
            return "center", None, None
            
        # 눈 깜박임 확인 - _is_blinking 메서드만 사용
        try:
            if self._is_blinking(landmarks):
                self.looking_down_start = None  # 깜빡임 시 아래 응시 시간 초기화
                return "blink", None, None
        except Exception as e:
            print(f"Warning: 깜빡임 감지 중 오류 발생 - {str(e)}")
            return "center", None, None
            
        # 현재 동공 좌표 계산
        current_left = np.array([landmarks[self.LEFT_IRIS_CENTER].x, landmarks[self.LEFT_IRIS_CENTER].y])
        current_right = np.array([landmarks[self.RIGHT_IRIS_CENTER].x, landmarks[self.RIGHT_IRIS_CENTER].y])
        
        # 눈 종횡비 계산
        left_ear = self._calculate_eye_aspect_ratio(landmarks, "left")
        right_ear = self._calculate_eye_aspect_ratio(landmarks, "right")
        avg_ear = (left_ear + right_ear) / 2
        
        # 현재 눈 영역 높이 계산
        current_left_height = self._calculate_eye_height(landmarks, "left")
        current_right_height = self._calculate_eye_height(landmarks, "right")
        
        # 보정값이 없으면 중앙으로 처리
        if (self.baseline_left_eye_height is None or 
            self.baseline_right_eye_height is None):
            return "center", None, None
            
        try:
            # 보정값 대비 높이 변화율 계산
            left_height_ratio = current_left_height / self.baseline_left_eye_height
            right_height_ratio = current_right_height / self.baseline_right_eye_height
            avg_height_ratio = (left_height_ratio + right_height_ratio) / 2
        except (TypeError, ZeroDivisionError):
            return "center", None, None

        # 눈 영역 계산 (시각화용)
        left_eye_left, left_eye_right, left_eye_top, left_eye_bottom = self._get_eye_region(landmarks, "left")
        right_eye_left, right_eye_right, right_eye_top, right_eye_bottom = self._get_eye_region(landmarks, "right")
        
        # 눈 영역 내에서의 동공 상대 위치 계산 (0~1 범위)
        left_x_ratio = (current_left[0] - left_eye_left) / (left_eye_right - left_eye_left)
        right_x_ratio = (current_right[0] - right_eye_left) / (right_eye_right - right_eye_left)
        
        # 보정값 대비 x축 변화율 계산
        left_x_ratio_change = (left_x_ratio - self.baseline_left_x_ratio) / self.baseline_left_x_ratio
        right_x_ratio_change = (right_x_ratio - self.baseline_right_x_ratio) / self.baseline_right_x_ratio
        avg_x_ratio_change = (left_x_ratio_change + right_x_ratio_change) / 2
        
        direction = []
        current_time = time.time()
        
        # 좌우 방향 (x축) - 보정값 대비 변화율로 판단
        if avg_x_ratio_change < -self.gaze_x_threshold:
            direction.append("left")
        elif avg_x_ratio_change > self.gaze_x_threshold:
            direction.append("right")
            
        # 상하 방향 (y축) - 눈 영역 높이 변화로 판단
        if avg_height_ratio > 1 + self.gaze_y_threshold:  # 눈 영역이 커지면 위를 보는 것
            self.looking_down_start = None  # 위를 보면 아래 응시 시간 초기화
            direction.append("up")
        elif avg_height_ratio < 1 - self.gaze_y_threshold:  # 눈 영역이 작아지면 아래를 보는 것
            if self.looking_down_start is None:
                self.looking_down_start = current_time
            elif current_time - self.looking_down_start >= self.min_looking_down_duration:
                direction.append("down")
        else:
            self.looking_down_start = None  # 중앙을 보면 아래 응시 시간 초기화
            
        # 시각화를 위한 정보 반환
        eye_regions = {
            "left": (left_eye_left, left_eye_right, left_eye_top, left_eye_bottom),
            "right": (right_eye_left, right_eye_right, right_eye_top, right_eye_bottom)
        }
        iris_positions = {
            "left": {"current": current_left, "ratio": (left_x_ratio, left_height_ratio)},
            "right": {"current": current_right, "ratio": (right_x_ratio, right_height_ratio)}
        }
            
        return " ".join(direction) if direction else "center", eye_regions, iris_positions

    def load_blink_log_and_calc(self, log_path):
        """로그 파일(jsonl)에서 time값을 읽어 blink_timestamps에 저장하고 분당 깜빡임 횟수 반환"""
        import json
        self.blink_timestamps = []
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        t = json.loads(line)["time"]
                        self.blink_timestamps.append(t)
                    except Exception:
                        continue
        return self.get_blinks_per_minute()

    def finish_calibration(self, timestamp):
        # 재보정 시작~끝 한 줄로 기록
        if self._recalib_start_time is not None:
            self.log_recalib_event(self._recalib_start_time, timestamp)
            self._recalib_start_time = None