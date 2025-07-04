# ----
# 작성목적 : 영상 파일에서 시선/눈깜빡임 로그 생성
# 작성일 : 2025-06-16

# 변경사항 내역 (날짜 | 변경목적 | 변경내용 | 작성자 순으로 기입)
# 2025-06-15 | 최초 구현 | 웹캠 기반 시선 추적 기능 구현 | 이재인
# 2025-06-15 | 기능 수정 | 웹캠 대신 webm 영상 처리 기능 추가, 프레임 스킵 옵션 추가 | 이재인
# 2025-06-15 | 기능 추가 | 평가 자동 계산 기능 추가 | 이재인
# ----------------------------------------------------------------------------------------------------

import cv2
import time
import argparse
import os
import sys
import json
import asyncio
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from collections import defaultdict

def resize_frame_for_speed(frame, scale=0.7):
    """프레임 크기를 줄여서 처리 속도 향상"""
    height, width = frame.shape[:2]
    new_width = int(width * scale)
    new_height = int(height * scale)
    return cv2.resize(frame, (new_width, new_height))

def calculate_basic_scores(blink_log_path: Path, gaze_log_path: Path, head_log_path: Path, 
                         anomaly_log_path: Path, total_duration: float) -> Dict[str, Any]:
    """첨부된 main.py 기반 평가 시스템 (40점 만점)"""
    try:
        # 깜빡임 로그 분석 (첨부된 main.py와 동일)
        blink_count = 0
        blink_timestamps = []
        if blink_log_path.exists():
            with open(blink_log_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            data = json.loads(line.strip())
                            if 'time' in data:
                                blink_timestamps.append(data['time'])
                                blink_count += 1
                        except:
                            blink_count += 1  # JSON 파싱 실패시에도 카운트
        
        # 시선 로그 분석 (첨부된 main.py와 동일)
        gaze_data = []
        if gaze_log_path.exists():
            with open(gaze_log_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            data = json.loads(line.strip())
                            gaze_data.append(data)
                        except:
                            continue
        
        # 1. 집중도 점수 계산 (15점 만점) - center 시선 비율 기반
        center_time = 0
        total_gaze_time = 0
        for gaze in gaze_data:
            if 'direction' in gaze and 'start_time' in gaze and 'end_time' in gaze:
                duration = gaze['end_time'] - gaze['start_time']
                total_gaze_time += duration
                if gaze['direction'] == 'center':
                    center_time += duration
        
        concentration_ratio = center_time / total_gaze_time if total_gaze_time > 0 else 0.8
        concentration_score = min(15, concentration_ratio * 15)  # 0~15점
        
        # 2. 안정성 점수 계산 (15점 만점) - 시선 변화 빈도 기반
        direction_changes = len(gaze_data)
        stability_ratio = max(0, 1 - (direction_changes / 100))  # 100회 변화를 기준으로 정규화
        stability_score = min(15, stability_ratio * 15)  # 0~15점
        
        # 3. 깜빡임 점수 계산 (10점 만점) - 분당 15-20회 기준
        blinks_per_minute = (blink_count / (total_duration / 60)) if total_duration > 0 else 0
        if 15 <= blinks_per_minute <= 20:
            blink_score = 10
        elif 10 <= blinks_per_minute <= 25:
            blink_score = 8
        else:
            blink_score = max(0, 10 - abs(blinks_per_minute - 17.5) * 0.5)
        
        # 총 시선 점수 (40점 만점)
        total_eye_score = concentration_score + stability_score + blink_score
        
        return {
            'concentration_score': round(concentration_score, 1),
            'stability_score': round(stability_score, 1),
            'blink_score': round(blink_score, 1),
            'total_eye_score': round(total_eye_score, 1),
            'blink_count': blink_count,
            'blinks_per_minute': round(blinks_per_minute, 1),
            'total_duration': round(total_duration, 1),
            'center_time_ratio': round((center_time / total_gaze_time * 100) if total_gaze_time > 0 else 80, 1),
            'concentration_ratio': round(concentration_ratio, 3),
            'stability_ratio': round(stability_ratio, 3),
            'direction_changes': direction_changes
        }
        
    except Exception as e:
        print(f"⚠️ 기본 점수 계산 오류: {e}")
        # 오류 시 기본값 (40점의 80%)
        return {
            'concentration_score': 12.0,  # 15점의 80%
            'stability_score': 12.0,     # 15점의 80%
            'blink_score': 8.0,          # 10점의 80%
            'total_eye_score': 32.0,     # 40점의 80%
            'blink_count': 0,
            'blinks_per_minute': 0.0,
            'total_duration': total_duration,
            'center_time_ratio': 80.0,
            'concentration_ratio': 0.8,
            'stability_ratio': 0.8,
            'direction_changes': 0
        }

# 상대 import와 절대 import 모두 지원 (원본 main.py와 동일)
try:
    from .face import FaceMeshDetector
    from .eye import EyeAnalyzer
    from .gaze_analyzer import GazeAnalyzer
    from .yolo_face import YOLOFaceDetector
    from .logger import BlinkLogger, GazeLogger, HeadLogger
    from .anomaly_logger import AnomalyLogger
    from .utils import draw_eye_info, draw_iris_points, draw_head_pose_landmarks, draw_status
except ImportError:
    # 직접 실행 시 절대 import 사용
    from face import FaceMeshDetector
    from eye import EyeAnalyzer
    from gaze_analyzer import GazeAnalyzer
    from yolo_face import YOLOFaceDetector
    from logger import BlinkLogger, GazeLogger, HeadLogger
    from anomaly_logger import AnomalyLogger
    from utils import draw_eye_info, draw_iris_points, draw_head_pose_landmarks, draw_status

class EyeTrackingAnalyzer:
    """시선 추적 분석을 수행하는 클래스 (API 호환성을 위한 래퍼)"""
    
    def __init__(self, yolo_model_path: str = None):
        """
        시선 추적 분석기 초기화
        
        Args:
            yolo_model_path: YOLO 얼굴 검출 모델 경로
        """
        # 기본 경로 설정
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.yolo_model_path = yolo_model_path or os.path.join(current_dir, 'yolov8n-face-lindevs.pt')
        
    def test_video_basic(self, video_path: str) -> Dict[str, Any]:
        """비디오 파일 기본 정보 테스트"""
        try:
            print(f"🔍 비디오 파일 기본 테스트: {video_path}")
            
            # 비디오 파일 존재 확인
            if not os.path.exists(video_path):
                print(f"❌ 비디오 파일이 존재하지 않습니다: {video_path}")
                return {"error": "File not found"}
            
            # OpenCV로 비디오 열기 테스트
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"❌ 비디오 파일을 열 수 없습니다: {video_path}")
                return {"error": "Cannot open video"}
            
            # 비디오 정보 확인
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = total_frames / fps if fps > 0 else 0
            
            print(f"✅ 비디오 정보:")
            print(f"   해상도: {width}x{height}")
            print(f"   FPS: {fps}")
            print(f"   총 프레임: {total_frames}")
            print(f"   재생시간: {duration:.2f}초")
            
            # 첫 프레임 읽기 테스트
            ret, frame = cap.read()
            if not ret:
                print(f"❌ 첫 프레임을 읽을 수 없습니다")
                cap.release()
                return {"error": "Cannot read first frame"}
            
            print(f"✅ 첫 프레임 읽기 성공: {frame.shape}")
            
            # YOLO 테스트
            try:
                face_detector = YOLOFaceDetector(self.yolo_model_path)
                faces = face_detector.detect_faces(frame)
                print(f"✅ YOLO 얼굴 감지 테스트: {len(faces)}개 얼굴 감지")
            except Exception as e:
                print(f"❌ YOLO 테스트 실패: {e}")
            
            # MediaPipe 테스트
            try:
                face_analyzer = FaceMeshDetector()
                landmarks = face_analyzer.get_landmarks(frame)
                print(f"✅ MediaPipe 테스트: {'랜드마크 감지 성공' if landmarks else '랜드마크 감지 실패'}")
            except Exception as e:
                print(f"❌ MediaPipe 테스트 실패: {e}")
            
            cap.release()
            
            return {
                "video_info": {
                    "width": width,
                    "height": height,
                    "fps": fps,
                    "total_frames": total_frames,
                    "duration": duration
                },
                "face_detection": len(faces) if 'faces' in locals() else 0,
                "landmarks_detected": landmarks is not None if 'landmarks' in locals() else False
            }
            
        except Exception as e:
            print(f"❌ 비디오 테스트 중 오류: {e}")
            return {"error": str(e)}
        
    async def analyze_video(self, video_path: str, show_window: bool = False, user_id: str = None, question_id: str = None, s3_key: str = None) -> Dict[str, Any]:
        """
        비디오 파일을 분석하여 시선 추적 결과를 반환합니다. (원본 main.py와 동일한 로직)
        
        Args:
            video_path: 분석할 비디오 파일 경로
            show_window: 시각화 창 표시 여부 (디버깅용)
            user_id: 사용자 ID (S3 키에서 추출하거나 직접 전달)
            question_id: 질문 ID (S3 키에서 추출하거나 직접 전달)  
            s3_key: S3 키 (user_id, question_id 추출용)
            
        Returns:
            Dict[str, Any]: 시선 추적 분석 결과
        """
        try:
            # S3 key에서 user_id와 question_id 추출 시도
            if s3_key:
                print(f"🔍 S3 키 파싱 시도: {s3_key}")
                # S3 키 형식: team12/interview_video/{user_id}/{question_id}/filename.mp4
                key_parts = s3_key.split('/')
                print(f"🔍 S3 키 분할: {key_parts}")
                
                if len(key_parts) >= 4 and 'interview_video' in key_parts:
                    video_index = key_parts.index('interview_video')
                    if video_index + 2 < len(key_parts):
                        extracted_user_id = key_parts[video_index + 1]
                        extracted_question_id = key_parts[video_index + 2]
                        print(f"🔍 S3 key에서 추출: user_id={extracted_user_id}, question_id={extracted_question_id}")
                        
                        # 기존 값이 없는 경우에만 사용
                        if not user_id:
                            user_id = extracted_user_id
                        if not question_id:
                            question_id = extracted_question_id
            
            # 기본값 설정
            if not user_id:
                import uuid
                temp_id = str(uuid.uuid4())[:8]
                user_id = f"api_user_{temp_id}"
            if not question_id:
                question_id = "Q1"
                
            print(f"🚀 시선 추적 분석 시작: {user_id}/{question_id}")
            
            # 비동기적으로 비디오 처리 (user_id, question_id 전달)
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, self._process_video_sync_with_window, video_path, show_window, user_id, question_id
            )
            return result
            
        except Exception as e:
            print(f"❌ 비디오 시선 추적 분석 실패: {str(e)}")
            raise Exception(f"비디오 시선 추적 분석 실패: {str(e)}")
    
    def _process_video_sync_with_window(self, video_path: str, show_window: bool = False, user_id: str = None, question_id: str = None) -> Dict[str, Any]:
        """시각화 옵션을 포함한 동기 비디오 처리"""
        try:
            # user_id와 question_id가 없으면 임시 생성
            if not user_id or not question_id:
                import uuid
                temp_id = str(uuid.uuid4())[:8]
                if not user_id:
                    user_id = f"api_user_{temp_id}"
                if not question_id:
                    question_id = "Q1"
            
            print(f"🎯 시선 추적 분석 시작: {user_id}/{question_id}")
            print(f"📹 비디오 파일: {video_path}")
            print(f"👁️ 시각화 창: {'ON' if show_window else 'OFF'}")
            
            # process_video 함수 호출 (속도 개선을 위해 frame_interval 증가)
            result = process_video(video_path, user_id, question_id, frame_interval=6, show_window=show_window)
            
            # 로그 파일에서 결과 읽기
            log_dir = Path("logs")
            blink_log = log_dir / f"{user_id}_{question_id}.jsonl"
            gaze_log = log_dir / f"{user_id}_{question_id}_gaze.jsonl"
            head_log = log_dir / f"{user_id}_{question_id}_head.jsonl"
            anomaly_log = log_dir / f"{user_id}_{question_id}_anomalies.jsonl"
            
            print(f"📊 로그 파일 생성 확인:")
            print(f"  - 깜빡임 로그: {blink_log.exists()} ({blink_log})")
            print(f"  - 시선 로그: {gaze_log.exists()} ({gaze_log})")
            print(f"  - 고개 로그: {head_log.exists()} ({head_log})")
            print(f"  - 이상 로그: {anomaly_log.exists()} ({anomaly_log})")
            
            # 분석 결과 구성
            analysis_result = self._build_analysis_result(
                blink_log, gaze_log, head_log, anomaly_log, video_path, user_id, question_id
            )
            
            print(f"✅ 시선 추적 분석 완료!")
            print(f"📈 분석 결과 요약:")
            print(f"  - 총 분석 시간: {analysis_result.get('total_duration', 0):.2f}초")
            print(f"  - 깜빡임 횟수: {analysis_result.get('blink_count', 0)}회")
            print(f"  - 집중도 점수: {analysis_result.get('attention_score', 0):.1f}")
            print(f"  - 시선 안정성: {analysis_result.get('gaze_stability', 0):.1f}")
            
            # 임시 로그 파일 정리
            for log_file in [blink_log, gaze_log, head_log, anomaly_log]:
                if log_file.exists():
                    log_file.unlink()
            
            return analysis_result
            
        except Exception as e:
            print(f"❌ 비디오 처리 중 오류: {str(e)}")
            raise Exception(f"비디오 처리 중 오류: {str(e)}")
    
    def _process_video_sync(self, video_path: str) -> Dict[str, Any]:
        """동기적으로 비디오를 처리합니다. (원본 main.py 로직 기반)"""
        try:
            # 임시 사용자 ID와 질문 ID 생성
            import uuid
            temp_id = str(uuid.uuid4())[:8]
            user_id = f"api_user_{temp_id}"
            question_id = "Q1"
            
            print(f"🎯 시선 추적 분석 시작: {user_id}")
            print(f"📹 비디오 파일: {video_path}")
            
            # process_video 함수 호출 (원본과 동일한 설정)
            result = process_video(video_path, user_id, question_id, frame_interval=2, show_window=False)
            
            # 로그 파일에서 결과 읽기
            log_dir = Path("logs")
            blink_log = log_dir / f"{user_id}_{question_id}.jsonl"
            gaze_log = log_dir / f"{user_id}_{question_id}_gaze.jsonl"
            head_log = log_dir / f"{user_id}_{question_id}_head.jsonl"
            anomaly_log = log_dir / f"{user_id}_{question_id}_anomalies.jsonl"
            
            print(f"📊 로그 파일 생성 확인:")
            print(f"  - 깜빡임 로그: {blink_log.exists()} ({blink_log})")
            print(f"  - 시선 로그: {gaze_log.exists()} ({gaze_log})")
            print(f"  - 고개 로그: {head_log.exists()} ({head_log})")
            print(f"  - 이상 로그: {anomaly_log.exists()} ({anomaly_log})")
            
            # 분석 결과 구성
            analysis_result = self._build_analysis_result(
                blink_log, gaze_log, head_log, anomaly_log, video_path, user_id, question_id
            )
            
            print(f"✅ 시선 추적 분석 완료!")
            print(f"📈 분석 결과 요약:")
            print(f"  - 총 분석 시간: {analysis_result.get('total_duration', 0):.2f}초")
            print(f"  - 깜빡임 횟수: {analysis_result.get('blink_count', 0)}회")
            print(f"  - 집중도 점수: {analysis_result.get('attention_score', 0):.1f}")
            print(f"  - 시선 안정성: {analysis_result.get('gaze_stability', 0):.1f}")
            
            # 임시 로그 파일 정리
            for log_file in [blink_log, gaze_log, head_log, anomaly_log]:
                if log_file.exists():
                    log_file.unlink()
            
            return analysis_result
            
        except Exception as e:
            print(f"❌ 비디오 처리 중 오류: {str(e)}")
            raise Exception(f"비디오 처리 중 오류: {str(e)}")
    
    def _build_analysis_result(self, blink_log: Path, gaze_log: Path, 
                              head_log: Path, anomaly_log: Path, video_path: str, user_id: str = None, question_id: str = None) -> Dict[str, Any]:
        """로그 파일들로부터 분석 결과를 구성합니다."""
        try:
            # 비디오 정보 가져오기
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            
            # 안전한 프레임 수 계산
            raw_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if raw_frame_count <= 0:
                total_frames = int(fps * 60)  # 최대 60초로 추정
            else:
                total_frames = raw_frame_count
                
            duration = total_frames / fps if fps > 0 else 0
            cap.release()
            
            # 부정행위 감지 결과 생성
            from .calc.cheat_cal import detect_cheating
            cheating_result = detect_cheating(
                str(head_log), str(anomaly_log), 
                user_id , question_id , video_path
            )
            
            # 부정행위 통계 추출
            total_violations = 0
            face_multiple_detected = False
            
            print(f"🔍 부정행위 감지 원본 결과: {cheating_result}")
            
            if cheating_result:
                # cheating_result는 {'user_id': ..., 'question_key': [...]} 형태
                if 'user_id' in cheating_result:
                    actual_user_id = cheating_result['user_id']
                    print(f"🔍 추출된 user_id: {actual_user_id}")
                    
                    # question_key 찾기 (user_id가 아닌 키)
                    for key, value in cheating_result.items():
                        if key != 'user_id' and isinstance(value, list):
                            print(f"🔍 검사 중인 키: {key}, 데이터: {value}")
                            cheating_data = value
                            
                            for item in cheating_data:
                                if item.get('category') == '부정행위':
                                    total_violations += 1
                                    comment = item.get('comments', '')
                                    if '2개 감지됨' in comment:
                                        face_multiple_detected = True
                                        print(f"🔍 다중얼굴 감지 확인: {comment}")
                            break
            
            print(f"🔍 부정행위 감지 결과: 총 {total_violations}회, 다중얼굴: {face_multiple_detected}")
            
            # 기본 점수 계산 사용
            basic_scores = calculate_basic_scores(blink_log, gaze_log, head_log, anomaly_log, duration)
            
            # 로그 파일 존재 확인
            log_files_exist = {
                'blink': blink_log.exists(),
                'gaze': gaze_log.exists(),
                'head': head_log.exists(),
                'anomaly': anomaly_log.exists()
            }
            
            # 기본 점수를 API 호환 형식으로 변환
            return {
                'total_duration': basic_scores['total_duration'],
                'blink_count': basic_scores['blink_count'],
                'blink_rate': basic_scores['blinks_per_minute'],
                'attention_score': basic_scores['concentration_score'],
                'gaze_stability': basic_scores['stability_score'],
                'focus_score': basic_scores['concentration_score'],  # API 호환성을 위한 별칭
                'video_info': {
                    'duration': duration,
                    'fps': fps,
                    'total_frames': total_frames
                },
                'basic_scores': basic_scores,  # 기본 점수 전체 포함
                'log_files_status': log_files_exist,
                'analysis_summary': {
                    'total_blinks': basic_scores['blink_count'],
                    'center_time_ratio': basic_scores['center_time_ratio'],
                    'concentration_score': basic_scores['concentration_score'],
                    'stability_score': basic_scores['stability_score'],
                    'blink_score': basic_scores['blink_score'],
                    'total_violations': total_violations,
                    'face_multiple_detected': face_multiple_detected
                }
            }
            
        except Exception as e:
            print(f"❌ 분석 결과 구성 오류: {e}")
            # 오류 발생 시 기본값 반환
            return {
                'total_duration': 0,
                'blink_count': 0,
                'blink_rate': 0,
                'attention_score': 0,
                'gaze_stability': 0,
                'focus_score': 0,
                'video_info': {
                    'duration': 0,
                    'fps': 0,
                    'total_frames': 0
                },
                'basic_scores': {
                    'concentration_score': 0,
                    'stability_score': 0,
                    'blink_score': 0,
                    'blink_count': 0,
                    'blinks_per_minute': 0.0,
                    'total_duration': 0,
                    'center_time_ratio': 0
                },
                'error': str(e)
            }



def process_video(video_path, user_id, question_id, frame_interval=3, show_window=False):
    """
    영상 처리 함수 (원본 main.py와 동일한 로직)
    frame_interval: 몇 프레임마다 처리할지 (예: 2면 2프레임마다 1번 처리)
    show_window: 시각화 창 표시 여부
    """
    # 비디오 파일 경로 처리 (수정됨 - API용)
    if not isinstance(video_path, Path):
        video_path = Path(video_path)
    
    # 파일이 이미 존재하면 그대로 사용
    if video_path.exists():
        print(f"✅ 비디오 파일 확인: {video_path}")
    elif not video_path.is_absolute():
        # 파일이 없고 상대 경로인 경우에만 videos 디렉토리 기준으로 처리
        video_dir = Path("videos")
        video_dir.mkdir(exist_ok=True)
        alternative_path = video_dir / video_path
        if alternative_path.exists():
            video_path = alternative_path
            print(f"✅ videos 디렉토리에서 발견: {video_path}")
        else:
            print(f"⚠️ 파일을 찾을 수 없습니다. 원본 경로 시도: {video_path}")
    
    if not video_path.exists():
        print(f"Error: Video file not found at {video_path}")
        return None
        
    # 비디오 파일 열기
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return None
    
    # 프레임 정보 (안전한 프레임 수 계산)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_time = frame_interval/fps if fps > 0 else frame_interval/30
    
    # 안전한 프레임 수 계산 - webm 파일의 음수 문제 해결
    raw_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if raw_frame_count <= 0:
        print("⚠️ 프레임 수를 자동으로 감지할 수 없습니다. 추정값을 사용합니다.")
        # FPS와 예상 길이로 추정 (최대 60초)
        total_frames = int(fps * 60) if fps > 0 else 1800
    else:
        total_frames = raw_frame_count
    
    print(f"원본 FPS: {fps}")
    print(f"처리 FPS: {fps/frame_interval}")
    print(f"총 프레임 수: {total_frames}")
    print(f"프레임 간격: {frame_interval}")
    
    # 로그 파일 경로 설정
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # 로그 파일 경로
    blink_log = log_dir / f"{user_id}_{question_id}.jsonl"
    gaze_log = log_dir / f"{user_id}_{question_id}_gaze.jsonl"
    head_log = log_dir / f"{user_id}_{question_id}_head.jsonl"
    anomaly_log = log_dir / f"{user_id}_{question_id}_anomalies.jsonl"
    
    # 기존 로그 파일 삭제 (새로 시작) - 원본과 동일
    for log_file in [blink_log, gaze_log, head_log, anomaly_log]:
        if log_file.exists():
            log_file.unlink()
    
    # 로거 초기화 (원본과 동일하게 Path 객체 전달)
    blink_logger = BlinkLogger(blink_log)
    gaze_logger = GazeLogger(gaze_log)
    head_logger = HeadLogger(head_log)
    anomaly_logger = AnomalyLogger(anomaly_log)
    
    # 분석기 초기화 (디버깅 로그 추가)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    yolo_model_path = os.path.join(current_dir, 'yolov8n-face-lindevs.pt')
    
    print(f"📦 YOLO 모델 경로: {yolo_model_path}")
    print(f"📦 YOLO 모델 존재 여부: {os.path.exists(yolo_model_path)}")
    
    try:
        face_detector = YOLOFaceDetector(yolo_model_path)
        print("✅ YOLO 얼굴 감지기 초기화 성공")
    except Exception as e:
        print(f"❌ YOLO 얼굴 감지기 초기화 실패: {e}")
        raise
    
    try:
        face_analyzer = FaceMeshDetector()
        print("✅ MediaPipe Face Mesh 초기화 성공")
    except Exception as e:
        print(f"❌ MediaPipe Face Mesh 초기화 실패: {e}")
        raise
        
    eye_analyzer = EyeAnalyzer()
    gaze_analyzer = GazeAnalyzer()
    print("✅ 모든 분석기 초기화 완료")
    
    # 시작 시간 기록
    start_time = time.time()
    frame_count = 0
    processed_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # 프레임 스킵
        if frame_count % frame_interval != 0:
            frame_count += 1
            continue
            
        # 현재 프레임 시간 계산
        current_time = processed_count * frame_time
        
        # 속도 개선을 위한 프레임 리사이징
        resized_frame = resize_frame_for_speed(frame, scale=0.7)
        
        # YOLO로 얼굴 감지 (리사이징된 프레임 사용)
        faces = face_detector.detect_faces(resized_frame)
        face_count = len(faces)
        
        # 디버깅: 첫 100프레임은 얼굴 감지 상태 출력
        if processed_count < 100 and processed_count % 10 == 0:
            print(f"[Frame {processed_count}] 감지된 얼굴 수: {face_count}")
        
        # 이상 상황 로깅
        anomaly_logger.update_state(current_time, face_count)
        
        if face_count != 1:
            frame_count += 1
            processed_count += 1
            continue
            
        # 얼굴 랜드마크 분석 (리사이징된 프레임 사용)
        face_landmarks = face_analyzer.get_landmarks(resized_frame)
        if face_landmarks is None:
            if processed_count < 100 and processed_count % 10 == 0:
                print(f"[Frame {processed_count}] MediaPipe 랜드마크 감지 실패")
            frame_count += 1
            processed_count += 1
            continue
        
        # 디버깅: 랜드마크가 감지되면 출력
        if processed_count < 100 and processed_count % 10 == 0:
            print(f"[Frame {processed_count}] 랜드마크 감지 성공! 분석 시작...")
            
        # 시선 방향 분석 및 기록 (디버깅 로그 추가)
        gaze_direction, eye_regions, iris_positions = gaze_analyzer.analyze_gaze(face_landmarks)
        
        # 디버깅: 시선 분석 결과 출력
        if processed_count < 100 and processed_count % 10 == 0:
            print(f"[Frame {processed_count}] 시선 방향: {gaze_direction}")
            
        if gaze_direction != "blink":
            gaze_logger.update_gaze(current_time, gaze_direction)
            # 디버깅: 시선 로깅 확인
            if processed_count < 100 and processed_count % 10 == 0:
                print(f"[Frame {processed_count}] 시선 로깅: {gaze_direction}")
        else:
            blink_logger.log_blink(current_time)
            # 디버깅: 깜빡임 로깅 확인
            if processed_count < 100 and processed_count % 10 == 0:
                print(f"[Frame {processed_count}] 깜빡임 감지!")
            
        # 고개 방향 분석 및 기록 (디버깅 로그 추가)
        head_direction, is_calibrated = gaze_analyzer.analyze_head_pose(face_landmarks, current_time)
        
        # 디버깅: 고개 방향 분석 결과 출력
        if processed_count < 100 and processed_count % 10 == 0:
            print(f"[Frame {processed_count}] 고개 방향: {head_direction}, 보정상태: {is_calibrated}")
            
        if is_calibrated and head_direction != "calibrating":
            head_logger.update_head(current_time, head_direction)
            # 디버깅: 고개 로깅 확인
            if processed_count < 100 and processed_count % 10 == 0:
                print(f"[Frame {processed_count}] 고개 로깅: {head_direction}")
        
        # 시각화 (원본과 동일하게 처리)
        if show_window:
            if eye_regions and iris_positions:
                draw_status(frame, gaze_direction, head_direction, not is_calibrated)
            cv2.imshow('Frame', frame)
            
            # 'q' 키로 종료
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # 진행률 표시 (안전한 프레임 수 사용)
        if frame_count % (frame_interval * 10) == 0 and total_frames > 0:
            progress = (frame_count / total_frames) * 100
            elapsed_time = time.time() - start_time
            processing_fps = processed_count / elapsed_time if elapsed_time > 0 else 0
            print(f"\r진행률: {progress:.1f}% ({frame_count}/{total_frames}) - 처리 속도: {processing_fps:.1f} FPS", end="")
            
        frame_count += 1
        processed_count += 1
    
    print("\n처리 완료!")
    print(f"총 처리 시간: {time.time() - start_time:.1f}초")
    print(f"평균 처리 속도: {processed_count / (time.time() - start_time):.1f} FPS")
    
    # 정리
    cap.release()
    if show_window:
        cv2.destroyAllWindows()
    
    # 로거 종료 (원본과 동일)
    current_time = processed_count * frame_time
    blink_logger.force_resolve(current_time)
    gaze_logger.force_resolve(current_time)
    head_logger.force_resolve(current_time)
    anomaly_logger.force_resolve(current_time)
    
    # 평가 계산 실행 (원본과 동일)
    try:
        print("\n평가 계산을 시작합니다...")
        
        # 평가 모듈 임포트 시도 (원본과 동일)
        sys.path.append(os.path.join(os.path.dirname(__file__), "calc"))
        from total_eval_calc import calc_blink_score, calc_eye_contact_score, save_total_eval
        from cheat_cal import detect_cheating
        
        # 1. 깜빡임과 아이컨택 평가 (원본과 동일)
        blink_result = calc_blink_score(str(blink_log), user_id)
        eye_contact_result = calc_eye_contact_score(str(gaze_log), user_id)
        
        # 통합 결과 저장 (S3 경로 기반 동적 설정)
        eval_result = save_total_eval(user_id, blink_result, eye_contact_result, question_id, str(video_path))
        print("\n[의사소통능력 및 면접태도 평가 결과]")
        print(json.dumps(eval_result, ensure_ascii=False, indent=2))
        
        # 2. 부정행위 감지 (S3 경로 기반 동적 설정)
        cheat_result = detect_cheating(str(head_log), str(anomaly_log), user_id, question_id, str(video_path))
        print("\n[부정행위 감지 결과]")
        print(json.dumps(cheat_result, ensure_ascii=False, indent=2))
        
        # 부정행위 결과 저장 (원본과 동일)
        cheat_log = Path("src/eye_tracking/calc") / "cheating_detected.jsonl"
        cheat_log.parent.mkdir(exist_ok=True)
        with open(cheat_log, "a", encoding="utf-8") as f:
            f.write(json.dumps(cheat_result, ensure_ascii=False, indent=2) + "\n\n")
            
        return {
            'blink_result': blink_result,
            'eye_contact_result': eye_contact_result,
            'eval_result': eval_result,
            'cheat_result': cheat_result
        }
        
    except ImportError as e:
        print(f"\n평가 모듈을 찾을 수 없습니다: {e}")
        print("로그 파일만 생성되었습니다.")
        
        # 기본 점수 계산으로 대체
        duration = processed_count * frame_time
        basic_scores = calculate_basic_scores(blink_log, gaze_log, head_log, anomaly_log, duration)
        
        print(f"\n📊 기본 점수 계산 결과:")
        print(f"  - 집중도: {basic_scores['concentration_score']}")
        print(f"  - 안정성: {basic_scores['stability_score']}")
        print(f"  - 깜빡임: {basic_scores['blink_score']}")
        
        return {
            'basic_scores': basic_scores,
            'duration': duration,
            'log_files_created': True
        }
    except Exception as e:
        print(f"\n평가 계산 중 오류 발생: {e}")
        return None

def main():
    """메인 함수 - 커맨드라인 인터페이스"""
    parser = argparse.ArgumentParser(description='Process video file for eye tracking analysis')
    parser.add_argument('video_path', type=str, help='Path to the webm video file (relative to videos/ directory or absolute path)')
    parser.add_argument('user_id', type=str, help='User ID (e.g., iv001)')
    parser.add_argument('question_id', type=str, help='Question ID (e.g., Q1)')
    parser.add_argument('--frame-interval', type=int, default=2, help='Process every N-th frame (default: 2)')
    parser.add_argument('--show-window', action='store_true', help='Show visualization window')
    
    args = parser.parse_args()
    
    # 비디오 파일 처리
    result = process_video(args.video_path, args.user_id, args.question_id, 
                          args.frame_interval, args.show_window)
    
    if result:
        print("\n모든 처리가 완료되었습니다.")
    else:
        print("\n처리 중 오류가 발생했습니다.")

if __name__ == "__main__":
    main() 