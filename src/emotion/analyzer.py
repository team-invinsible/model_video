# ----
# 작성목적 : 감정 분석 모듈 - last.py 방식 완전 반영
# 작성일 : 2025-06-15

# 변경사항 내역 (날짜 | 변경목적 | 변경내용 | 작성자 순으로 기입)
# 2025-06-15 | last.py 방식 반영 | 가중치 로딩, 점수 평가, 감정 분석 로직 완전 교체 | 이재인
# 2025-06-17 | 비디오 처리 오류 수정 | 얼굴 감지 영역이 비디오 프레임 경계를 벗어나지 않도록 수정 | 이재인
# ---- 

import cv2
import torch
from torchvision import transforms
from PIL import Image
import os
import time
import asyncio
from typing import Dict, Any, List, Optional
from collections import defaultdict
import numpy as np
import json

from .models import getModel

class EmotionAnalyzer:
    """감정 분석을 수행하는 클래스 """
    
    def __init__(self, 
                 model_path: str = None,
                 cascade_path: str = None,
                 model_name: str = 'efficientnet-b5',
                 image_size: int = 224):
        """
        감정 분석기 초기화 - last.py 방식
        
        Args:
            model_path: EfficientNet 모델 가중치 파일 경로
            cascade_path: Haar cascade 파일 경로
            model_name: 사용할 모델 이름
            image_size: 입력 이미지 크기
        """
        # 기본 경로 설정
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_path = model_path or os.path.join(current_dir, 'model_eff.pth')
        self.cascade_path = cascade_path or os.path.join(current_dir, 'face_classifier.xml')
        
        self.model_name = model_name
        self.image_size = image_size
        
        # 한글 라벨 (last.py와 동일)
        self.class_labels = ['기쁨', '당황', '분노', '불안', '상처', '슬픔', '중립']
        
        # 감정 매핑 (last.py와 동일)
        self.emotion_mapping = {
            '기쁨': 'happy',
            '당황': 'surprise', 
            '분노': 'angry',
            '불안': 'fear',
            '상처': 'disgust',
            '슬픔': 'sad',
            '중립': 'neutral'
        }
        
        # 면접 평가 기준 (last.py와 동일)
        self.positive_emotions = ['happy', 'neutral']
        self.negative_emotions = ['sad', 'angry', 'fear', 'surprise', 'disgust']
        
        # 속도 최적화 설정 (last.py와 동일)
        self.analysis_interval = 1  # 1초마다 1번 분석
        self.fast_face_detection = True  # 빠른 얼굴 검출 모드
        
        # 모델 및 변환기 초기화
        self.model = None
        self.transform = None
        self.face_cascade = None
        
        self._initialize_model()
    
    def _initialize_model(self):
        """모델과 관련 컴포넌트를 초기화합니다. (last.py 방식)"""
        try:
            # 모델 로드 (last.py의 build_model 방식)
            self.model = self._build_model(self.model_name, self.model_path)
            
            # 이미지 변환기 설정 (last.py와 동일)
            self.transform = transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                   [0.229, 0.224, 0.225])
            ])
            
            # 얼굴 검출기 로드 (last.py와 동일)
            self.face_cascade = cv2.CascadeClassifier(self.cascade_path)
            
            if self.face_cascade.empty():
                raise Exception(f"Haar cascade 파일을 로드할 수 없습니다: {self.cascade_path}")
                
        except Exception as e:
            raise Exception(f"모델 초기화 실패: {str(e)}")
    
    def _build_model(self, model_name: str, ckpt_path: str):
        """모델을 빌드하고 가중치를 로드합니다. (last.py와 완전 동일)"""
        # 모델 생성
        model = getModel(model_name)
        
        # 체크포인트 로드 (가중치 파일이 있는 경우만)
        try:
            if ckpt_path and os.path.exists(ckpt_path):
                ckpt = torch.load(ckpt_path, map_location='cpu')
                state = ckpt.get('model', ckpt)
                model.load_state_dict(state)
                print(f"가중치를 로드했습니다: {ckpt_path}")
            else:
                print(f"가중치 파일을 찾을 수 없습니다. 랜덤 초기화된 모델을 사용합니다.")
        except Exception as e:
            print(f"가중치 로딩 실패: {str(e)}")
            print("랜덤 초기화된 모델을 사용합니다.")
        
        model.eval()
        return model
    
    async def analyze_video(self, video_path: str) -> Dict[str, Any]:
        """
        비디오 파일을 분석하여 감정 분석 결과를 반환합니다. (last.py 방식)
        
        Args:
            video_path: 분석할 비디오 파일 경로
            
        Returns:
            Dict[str, Any]: 감정 분석 결과
        """
        try:
            # 비동기적으로 비디오 처리
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, self._process_video_sync, video_path
            )
            return result
            
        except Exception as e:
            raise Exception(f"비디오 감정 분석 실패: {str(e)}")
    
    def _process_video_sync(self, video_path: str) -> Dict[str, Any]:
        """동기적으로 비디오를 처리합니다. (last.py의 process_video_core 방식)"""
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                raise Exception(f"비디오 파일을 열 수 없습니다: {video_path}")
            
            # 비디오 정보
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0
            
            # last.py와 동일한 분석 설정
            frame_count = 0
            processed_frames = 0
            
            # 감정 데이터 수집
            emotion_data = []
            
            # FPS 측정을 위한 변수들 (last.py와 동일)
            start_time = time.time()
            
            # 1초에 해당하는 프레임 수 계산 (last.py와 동일)
            frames_per_interval = int(fps * self.analysis_interval)
            
            print(f"원본 비디오 FPS: {fps}")
            print(f"분석 간격: {self.analysis_interval}초 = {frames_per_interval} 프레임마다")
            print(f"이론적 처리 FPS: {fps / frames_per_interval:.1f}")
            print("-" * 50)
            
            # 얼굴 검출 설정 (last.py와 동일)
            if self.fast_face_detection:
                scale_factor = 1.2
                min_neighbors = 4
            else:
                scale_factor = 1.1
                min_neighbors = 5
            
            with torch.no_grad():
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame_count += 1
                    
                    # 1초 간격 프레임 스킵 적용 (last.py와 동일)
                    if frame_count % frames_per_interval != 0:
                        continue

                    processed_frames += 1
                    
                    # (웹캠용이라면 좌우 반전, 동영상이라면 주석 처리) - last.py와 동일
                    frame = cv2.flip(frame, 1)

                    # 얼굴 검출 (하나만 검출) - last.py와 동일
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    detected_faces = self.face_cascade.detectMultiScale(gray, scale_factor, min_neighbors)
                    
                    # 가장 큰 얼굴 하나만 선택 (last.py와 동일)
                    if len(detected_faces) > 0:
                        # 얼굴 크기(면적) 기준으로 가장 큰 얼굴 선택
                        largest_face = max(detected_faces, key=lambda face: face[2] * face[3])
                        x, y, w, h = largest_face
                        
                        # ROI 자르고 PIL→Tensor (last.py와 동일)
                        # 얼굴 영역이 이미지 경계를 벗어나지 않도록 클리핑
                        face = frame[max(0, y):min(frame.shape[0], y + h), max(0, x):min(frame.shape[1], x + w)]
                        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                        pil = Image.fromarray(cv2.resize(face, (self.image_size, self.image_size)))
                        inp = self.transform(pil).unsqueeze(0)

                        # 예측 (last.py와 동일)
                        logits = self.model(inp)
                        probs  = torch.softmax(logits, dim=1)[0]
                        p, idx = probs.max(0)
                        emotion_korean = self.class_labels[idx]
                        emotion_english = self.emotion_mapping[emotion_korean]
                        label = f"{emotion_korean} ({p*100:.1f}%)"

                        # 감정 데이터 저장 (last.py와 동일)
                        emotion_data.append({
                            'frame': frame_count,
                            'emotion': emotion_english,
                            'emotion_korean': emotion_korean,
                            'confidence': p.item()
                        })

                        # 콘솔에도 출력 (last.py와 동일)
                        print(f"[Frame {frame_count}] {label}")
            
            cap.release()
            
            # 최종 통계 (last.py와 동일)
            total_time = time.time() - start_time
            average_fps = processed_frames / total_time if total_time > 0 else 0
            
            print(f"✅ {os.path.basename(video_path)} 완료!")
            print(f"   처리시간: {total_time:.1f}초, 프레임: {processed_frames}/{frame_count}, FPS: {average_fps:.1f}")
            
            # 결과 분석
            if not emotion_data:
                # 얼굴이 감지되지 않은 경우 기본값 반환 (last.py 방식)
                print("⚠️ 얼굴이 감지되지 않았습니다. 기본값을 반환합니다.")
                analysis_result = {
                    'total_frames': 0,
                    'emotion_counts': {'neutral': 0},
                    'emotion_ratios': {'neutral': 0.0},
                    'dominant_emotion': 'neutral',
                    'confidence_scores': {'neutral': 0.5},
                    'interview_score': 0,  # 기본 점수
                    'grade': '보통',
                    'detailed_analysis': {
                        'total_frames': 0,
                        'emotion_counts': {'neutral': 0},
                        'emotion_ratios': {'neutral': 0},
                        'happy_ratio': 0.0,
                        'neutral_ratio': 0.0,
                        'negative_ratio': 0.0,
                        'happy_confidence': 0.0,
                        'scores': {
                            'positive_score': 0,
                            'negative_score': 0,
                            'happy_confidence_score': 0.0,
                            'total_score': 0
                        },
                        'improvement_suggestions': ['실제 얼굴이 포함된 비디오로 다시 테스트해주세요.']
                    },
                    'frame_by_frame_results': []
                }
            else:
                # 면접 평가 점수 계산 (last.py와 동일)
                interview_score, interview_analysis = self._calculate_interview_score(emotion_data)
                
                # 종합 분석 수행
                analysis_result = self._calculate_comprehensive_analysis(emotion_data, interview_score, interview_analysis)
            
            # 비디오 정보 추가
            analysis_result['video_info'] = {
                'duration': duration,
                'fps': fps,
                'total_frames': total_frames,
                'analyzed_frames': len(emotion_data),
                'processing_time': total_time,
                'average_fps': average_fps
            }
            
            return analysis_result
            
        except Exception as e:
            raise Exception(f"비디오 처리 중 오류: {str(e)}")
    
    def _calculate_comprehensive_analysis(self, emotion_data: List[Dict[str, Any]], interview_score: int, interview_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """종합적인 감정 분석 결과를 계산합니다. (last.py 방식)"""
        try:
            total_frames = len(emotion_data)
            emotion_counts = defaultdict(int)
            emotion_confidences = defaultdict(list)
            
            # 감정별 통계 계산
            for frame_data in emotion_data:
                emotion = frame_data['emotion']
                confidence = frame_data['confidence']
                
                emotion_counts[emotion] += 1
                emotion_confidences[emotion].append(confidence)
            
            # 비율 계산
            emotion_ratios = {emotion: count/total_frames 
                            for emotion, count in emotion_counts.items()}
            
            # 평균 신뢰도 계산
            confidence_scores = {}
            for emotion, confidences in emotion_confidences.items():
                confidence_scores[emotion] = sum(confidences) / len(confidences)
            
            # 지배적 감정
            dominant_emotion = max(emotion_counts.items(), key=lambda x: x[1])[0]
            
            # 등급 계산 (last.py와 동일)
            grade = self._get_grade(interview_score)
            
            return {
                'total_frames': total_frames,
                'emotion_counts': dict(emotion_counts),
                'emotion_ratios': emotion_ratios,
                'dominant_emotion': dominant_emotion,
                'confidence_scores': confidence_scores,
                'interview_score': interview_score,
                'grade': grade,
                'detailed_analysis': interview_analysis,
                'frame_by_frame_results': emotion_data
            }
            
        except Exception as e:
            raise Exception(f"종합 분석 계산 오류: {str(e)}")
    
    def _calculate_interview_score(self, emotion_data: List[Dict[str, Any]]) -> tuple:
        """면접 평가 점수 계산 함수 - last.py 버전"""
        if not emotion_data:
            return 0, {}

        total = len(emotion_data)
        cnt = defaultdict(int)
        conf = defaultdict(list)
        for d in emotion_data:
            e = d['emotion']
            c = d['confidence']
            cnt[e] += 1
            conf[e].append(c)

        happy = cnt.get('happy', 0)
        neutral = cnt.get('neutral', 0)
        happy_ratio = happy / total
        neutral_ratio = neutral / total
        positive_score = (happy_ratio + neutral_ratio) * 25

        neg_list = ['sad', 'angry', 'fear', 'surprise', 'disgust']
        negative = sum(cnt.get(e, 0) for e in neg_list)
        negative_ratio = negative / total
        negative_score = (1 - negative_ratio) * 15

        if conf.get('happy'):
            happy_conf = sum(conf['happy']) / len(conf['happy'])
            happy_conf_score = happy_conf * 20
        else:
            happy_conf = 0
            happy_conf_score = 0

        total_score = round(min(60, positive_score + negative_score + happy_conf_score))

        analysis = {
            'total_frames': total,
            'emotion_counts': dict(cnt),
            'emotion_ratios': {k: v/total for k, v in cnt.items()},
            'happy_ratio': happy_ratio,
            'neutral_ratio': neutral_ratio,
            'negative_ratio': negative_ratio,
            'happy_confidence': happy_conf,
            'scores': {
                'positive_score': positive_score,
                'negative_score': negative_score,
                'happy_confidence_score': happy_conf_score,
                'total_score': total_score
            },
            'improvement_suggestions': self._get_improvement_suggestions({
                'scores': {
                    'positive_score': positive_score,
                    'negative_score': negative_score,
                    'happy_confidence_score': happy_conf_score
                },
                'happy_ratio': happy_ratio,
                'neutral_ratio': neutral_ratio
            })
        }
        return total_score, analysis
    
    def _get_grade(self, score: float) -> str:
        """점수를 등급으로 변환 (last.py와 완전 동일)"""
        if score >= 55:  # 90% 이상
            return "우수"
        elif score >= 48:  # 80% 이상
            return "양호"
        elif score >= 36:  # 70% 이상
            return "보통"
        elif score >= 24:  # 60% 이상
            return "미흡"
        else:
            return "부족"
    
    def _get_improvement_suggestions(self, analysis: Dict[str, Any]) -> List[str]:
        """개선 제안 생성 - last.py 버전"""
        tips = []
        s = analysis['scores']
        if s['positive_score'] < 18:
            tips.append("긍정적인 표정(미소, 중립)을 더 자주 보여주세요")
        if s['negative_score'] < 12:
            tips.append("부정적인 감정 표현을 줄이고 안정적인 표정을 유지하세요")
        if s['happy_confidence_score'] < 15:
            tips.append("미소의 진정성을 높이고 자연스러운 표정을 연습하세요")
        if analysis['happy_ratio'] < 0.3:
            tips.append("면접에서는 적절한 미소를 보이는 것이 중요합니다")
        if analysis['neutral_ratio'] > 0.7:
            tips.append("무표정보다는 적극적인 표정 표현을 시도해보세요")
        if not tips:
            tips.append("전반적으로 우수한 표정 관리를 보여주었습니다!")
        return tips