# ----
# 작성목적 : YAML 기반 키워드 분석기 구현
# 작성일 : 2025-06-24

# 변경사항 내역 (날짜 | 변경목적 | 변경내용 | 작성자 순으로 기입)
# 2025-06-24 | 최초 구현 | YAML 기반 키워드 생성 시스템 구축 | 이재인
# ----

import yaml
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import logging

logger = logging.getLogger(__name__)

class KeywordAnalyzer:
    """YAML 설정 기반 키워드 분석기"""
    
    def __init__(self, config_path: str = None):
        """
        키워드 분석기 초기화
        
        Args:
            config_path: YAML 설정 파일 경로 (기본값: src/llm/interview_prompts.yaml)
        """
        if config_path is None:
            current_dir = Path(__file__).parent
            config_path = current_dir / "interview_prompts.yaml"
        
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """YAML 설정 파일 로드"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info(f"YAML 설정 파일 로드 완료: {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"YAML 설정 파일 로드 실패: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """기본 설정 반환 (YAML 파일 로드 실패 시)"""
        return {
            'keyword_limits': {
                'max_strength_keywords': 3,
                'max_weakness_keywords': 2,
                'min_total_keywords': 2
            },
            'scoring_rules': {
                'emotion_score': {
                    'excellent': {'threshold': 50, 'strength_keywords': ['안정적 표정'], 'weakness_keywords': []},
                    'poor': {'threshold': 0, 'strength_keywords': [], 'weakness_keywords': ['표정 개선 필요']}
                },
                'eye_score': {
                    'excellent': {'threshold': 32, 'strength_keywords': ['안정적 시선'], 'weakness_keywords': []},
                    'poor': {'threshold': 0, 'strength_keywords': [], 'weakness_keywords': ['시선 개선 필요']}
                }
            }
        }
    
    def analyze_keywords(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        분석 데이터를 바탕으로 강점/약점 키워드 생성
        
        Args:
            analysis_data: 분석 결과 데이터
            {
                'emotion_score': float,
                'eye_score': float, 
                'concentration_score': float,
                'stability_score': float,
                'blink_score': float,
                'total_violations': int,
                'face_multiple_detected': bool,
                'suspected_copying': bool,
                'suspected_impersonation': bool
            }
            
        Returns:
            Dict: {'strength_keyword': str, 'weakness_keyword': str, 'analysis_summary': str}
        """
        try:
            print(f"🔍 키워드 분석 시작: {analysis_data}")
            
            # 1. 점수 기반 키워드 추출
            strength_keywords = []
            weakness_keywords = []
            
            # 표정 점수 분석
            emotion_keywords = self._analyze_emotion_score(analysis_data.get('emotion_score', 0))
            strength_keywords.extend(emotion_keywords['strengths'])
            weakness_keywords.extend(emotion_keywords['weaknesses'])
            
            # 시선 점수 분석  
            eye_keywords = self._analyze_eye_score(analysis_data.get('eye_score', 0))
            strength_keywords.extend(eye_keywords['strengths'])
            weakness_keywords.extend(eye_keywords['weaknesses'])
            
            # 부정행위 분석
            cheating_keywords = self._analyze_cheating_detection(analysis_data)
            strength_keywords.extend(cheating_keywords['strengths'])
            weakness_keywords.extend(cheating_keywords['weaknesses'])
            
            # 2. 키워드 정제 및 제한 적용
            final_strength = self._filter_keywords(strength_keywords, 'strength')
            final_weakness = self._filter_keywords(weakness_keywords, 'weakness')
            
            # 3. 키워드 문자열 생성
            strength_str = ', '.join(final_strength) if final_strength else '성실한 태도'
            weakness_str = ', '.join(final_weakness) if final_weakness else ''
            
            # 4. 분석 요약 생성
            summary = self._generate_summary(analysis_data, final_strength, final_weakness)
            
            result = {
                'strength_keyword': strength_str,
                'weakness_keyword': weakness_str,
                'analysis_summary': summary
            }
            
            print(f"🔍 키워드 분석 완료: {result}")
            return result
            
        except Exception as e:
            logger.error(f"키워드 분석 실패: {e}")
            return {
                'strength_keyword': '성실한 태도',
                'weakness_keyword': '',
                'analysis_summary': '분석 완료'
            }
    
    def _analyze_emotion_score(self, emotion_score: float) -> Dict[str, List[str]]:
        """표정 점수 기반 키워드 추출"""
        scoring_rules = self.config.get('scoring_rules', {}).get('emotion_score', {})
        
        # 점수에 따른 등급 결정
        if emotion_score >= 50:
            rule = scoring_rules.get('excellent', {})
        elif emotion_score >= 40:
            rule = scoring_rules.get('good', {})
        elif emotion_score >= 30:
            rule = scoring_rules.get('fair', {})
        else:
            rule = scoring_rules.get('poor', {})
        
        return {
            'strengths': rule.get('strength_keywords', []),
            'weaknesses': rule.get('weakness_keywords', [])
        }
    
    def _analyze_eye_score(self, eye_score: float) -> Dict[str, List[str]]:
        """시선 점수 기반 키워드 추출"""
        scoring_rules = self.config.get('scoring_rules', {}).get('eye_score', {})
        
        # 점수에 따른 등급 결정
        if eye_score >= 32:
            rule = scoring_rules.get('excellent', {})
        elif eye_score >= 24:
            rule = scoring_rules.get('good', {})
        elif eye_score >= 16:
            rule = scoring_rules.get('fair', {})
        else:
            rule = scoring_rules.get('poor', {})
        
        return {
            'strengths': rule.get('strength_keywords', []),
            'weaknesses': rule.get('weakness_keywords', [])
        }
    
    def _analyze_cheating_detection(self, analysis_data: Dict[str, Any]) -> Dict[str, List[str]]:
        """부정행위 감지 기반 키워드 추출"""
        scoring_rules = self.config.get('scoring_rules', {}).get('cheating_detection', {})
        
        total_violations = analysis_data.get('total_violations', 0)
        face_multiple = analysis_data.get('face_multiple_detected', False)
        
        # 부정행위 유형별 키워드 결정
        if face_multiple:
            rule = scoring_rules.get('impersonation_risk', {})
        elif total_violations >= 5:
            rule = scoring_rules.get('major_violations', {})
        elif total_violations >= 1:
            rule = scoring_rules.get('minor_violations', {})
        else:
            rule = scoring_rules.get('clean', {})
        
        return {
            'strengths': rule.get('strength_keywords', []),
            'weaknesses': rule.get('weakness_keywords', [])
        }
    
    def _filter_keywords(self, keywords: List[str], keyword_type: str) -> List[str]:
        """키워드 필터링 및 우선순위 적용"""
        if not keywords:
            return []
        
        # 중복 제거
        unique_keywords = list(dict.fromkeys(keywords))
        
        # 우선순위 기반 정렬
        priorities = self.config.get('keyword_priorities', {}).get(keyword_type, {})
        unique_keywords.sort(key=lambda x: priorities.get(x, 0), reverse=True)
        
        # 개수 제한 적용
        limits = self.config.get('keyword_limits', {})
        if keyword_type == 'strength':
            max_count = limits.get('max_strength_keywords', 3)
        else:
            max_count = limits.get('max_weakness_keywords', 2)
        
        return unique_keywords[:max_count]
    
    def _generate_summary(self, analysis_data: Dict[str, Any], 
                         strengths: List[str], weaknesses: List[str]) -> str:
        """분석 요약 생성"""
        emotion_score = analysis_data.get('emotion_score', 0)
        eye_score = analysis_data.get('eye_score', 0)
        total_score = emotion_score + eye_score
        
        # 종합 평가
        if total_score >= 80:
            grade = "우수한"
        elif total_score >= 60:
            grade = "양호한"
        elif total_score >= 40:
            grade = "보통의"
        else:
            grade = "개선이 필요한"
        
        # 요약 문구 생성
        summary_parts = [f"{grade} 면접 태도를 보여주었습니다"]
        
        if strengths:
            summary_parts.append(f"강점: {', '.join(strengths[:2])}")
        
        if weaknesses:
            summary_parts.append(f"개선점: {', '.join(weaknesses[:2])}")
        
        return ". ".join(summary_parts) + "."
    
    def get_gpt_prompt(self, analysis_data: Dict[str, Any]) -> Tuple[str, str]:
        """GPT 분석용 프롬프트 생성"""
        try:
            config = self.config.get('interview_attitude_analysis', {})
            system_prompt = config.get('system_prompt', '')
            user_template = config.get('user_prompt_template', '')
            
            # 템플릿에 데이터 삽입
            user_prompt = user_template.format(
                emotion_score=analysis_data.get('emotion_score', 0),
                dominant_emotions=analysis_data.get('dominant_emotions', '중립'),
                emotion_stability=analysis_data.get('emotion_stability', '보통'),
                eye_score=analysis_data.get('eye_score', 0),
                concentration_score=analysis_data.get('concentration_score', 0),
                stability_score=analysis_data.get('stability_score', 0),
                blink_score=analysis_data.get('blink_score', 0),
                total_violations=analysis_data.get('total_violations', 0),
                face_multiple_detected=analysis_data.get('face_multiple_detected', False),
                suspected_copying=analysis_data.get('suspected_copying', False),
                suspected_impersonation=analysis_data.get('suspected_impersonation', False)
            )
            
            return system_prompt, user_prompt
            
        except Exception as e:
            logger.error(f"GPT 프롬프트 생성 실패: {e}")
            return "", "면접 태도를 분석해주세요."
    
    def get_detailed_gpt_prompt(self, emotion_result: Dict[str, Any], 
                               eye_tracking_result: Dict[str, Any]) -> Tuple[str, str]:
        """상세 GPT 분석용 프롬프트 생성 (레거시/fallback용)"""
        try:
            import json
            
            config = self.config.get('detailed_gpt_analysis', {})
            system_prompt = config.get('system_prompt', '')
            evaluation_guidelines = config.get('evaluation_guidelines', '')
            user_template = config.get('user_prompt_template', '')
            
            # 분석 데이터 구성
            analysis_data = self._prepare_analysis_data(emotion_result, eye_tracking_result)
            
            # 시스템 프롬프트에 가이드라인 추가
            full_system_prompt = f"{system_prompt}\n\n{evaluation_guidelines}"
            
            # 사용자 프롬프트에 데이터 삽입
            user_prompt = user_template.format(
                analysis_data=json.dumps(analysis_data, indent=2, ensure_ascii=False)
            )
            
            return full_system_prompt, user_prompt
            
        except Exception as e:
            logger.error(f"상세 GPT 프롬프트 생성 실패: {e}")
            return "", "면접 태도를 분석해주세요."
    
    def _prepare_analysis_data(self, emotion_result: Dict[str, Any], 
                              eye_tracking_result: Dict[str, Any]) -> Dict[str, Any]:
        """GPT 분석용 데이터 구조 준비"""
        
        # 감정 분석 데이터 추출
        emotion_details = emotion_result.get('detailed_analysis', {})
        emotion_total_frames = emotion_result.get('total_frames', 0)
        emotion_counts = emotion_result.get('emotion_counts', {})
        emotion_ratios = emotion_result.get('emotion_ratios', {})
        dominant_emotion = emotion_result.get('dominant_emotion', 'neutral')
        confidence_scores = emotion_result.get('confidence_scores', {})
        interview_score = emotion_result.get('interview_score', 0)
        grade = emotion_result.get('grade', 'C')
        
        # 시선 추적 데이터 추출
        basic_scores = eye_tracking_result.get('basic_scores', {})
        analysis_summary = eye_tracking_result.get('analysis_summary', {})
        
        return {
            "emotion_analysis": {
                "total_frames": emotion_total_frames,
                "emotion_counts": emotion_counts,
                "emotion_ratios": emotion_ratios,
                "dominant_emotion": dominant_emotion,
                "confidence_scores": confidence_scores,
                "interview_score": interview_score,
                "grade": grade,
                "happy_ratio": emotion_ratios.get('happy', 0.0),
                "neutral_ratio": emotion_ratios.get('neutral', 0.0),
                "negative_ratio": sum([
                    emotion_ratios.get('sad', 0.0),
                    emotion_ratios.get('angry', 0.0),
                    emotion_ratios.get('fear', 0.0),
                    emotion_ratios.get('surprise', 0.0),
                    emotion_ratios.get('disgust', 0.0)
                ]),
                "happy_confidence": confidence_scores.get('happy', 0.0),
                "scores": emotion_details.get('scores', {}),
                "improvement_suggestions": emotion_details.get('improvement_suggestions', [])
            },
            "eye_tracking_analysis": {
                "total_duration": eye_tracking_result.get('total_duration', 0),
                "blink_count": eye_tracking_result.get('blink_count', 0),
                "blink_rate": eye_tracking_result.get('blink_rate', 0),
                "attention_score": eye_tracking_result.get('attention_score', 0),
                "gaze_stability": eye_tracking_result.get('gaze_stability', 0),
                "focus_score": eye_tracking_result.get('focus_score', 0),
                "center_time_ratio": analysis_summary.get('center_time_ratio', 0.0),
                "scores": {
                    "concentration_score": basic_scores.get('concentration_score', 0),
                    "stability_score": basic_scores.get('stability_score', 0),
                    "blink_score": basic_scores.get('blink_score', 0),
                    "total_eye_score": basic_scores.get('total_eye_score', 0)
                },
                "improvement_suggestions": basic_scores.get('improvement_suggestions', [])
            }
        }
    
    def generate_dynamic_feedback(self, emotion_result: Dict[str, Any], 
                                 eye_tracking_result: Dict[str, Any]) -> str:
        """YAML 설정 기반 동적 피드백 생성"""
        try:
            # 점수 계산
            emotion_score = emotion_result.get('interview_score', 0)
            eye_score = eye_tracking_result.get('basic_scores', {}).get('total_eye_score', 0)
            total_score = emotion_score + eye_score
            
            # 부정행위 감지 여부
            analysis_summary = eye_tracking_result.get('analysis_summary', {})
            total_violations = analysis_summary.get('total_violations', 0)
            face_multiple_detected = analysis_summary.get('face_multiple_detected', False)
            cheating_detected = total_violations >= 5 or face_multiple_detected
            
            # YAML 설정에서 템플릿 가져오기
            feedback_config = self.config.get('dynamic_feedback', {})
            templates = feedback_config.get('templates', {})
            modifiers = feedback_config.get('modifiers', {})
            rules = self.config.get('feedback_generation_rules', {})
            
            # 점수 범위 기반 기본 템플릿 선택
            if total_score >= 80:
                base_templates = templates.get('excellent', [])
            elif total_score >= 60:
                base_templates = templates.get('good', [])
            else:
                base_templates = templates.get('needs_improvement', [])
            
            if not base_templates:
                return "면접 태도 분석이 완료되었습니다."
            
            # 랜덤하게 기본 템플릿 선택
            import random
            base_feedback = random.choice(base_templates)
            
            # 추가 수식어 적용
            enhancements = []
            enhancement_conditions = rules.get('enhancement_conditions', {})
            
            # 표정 점수 기반 수식어
            if emotion_score >= 50 and 'high_emotion_score' in enhancement_conditions:
                positive_traits = enhancement_conditions['high_emotion_score'].get('add_positive', [])
                enhancements.extend(positive_traits)
            elif emotion_score < 30 and 'low_emotion_score' in enhancement_conditions:
                improvement_areas = enhancement_conditions['low_emotion_score'].get('add_improvement', [])
                enhancements.extend([f"{area} 개선이 필요합니다" for area in improvement_areas])
            
            # 시선 점수 기반 수식어
            if eye_score >= 32 and 'high_eye_score' in enhancement_conditions:
                positive_traits = enhancement_conditions['high_eye_score'].get('add_positive', [])
                enhancements.extend(positive_traits)
            elif eye_score < 16 and 'low_eye_score' in enhancement_conditions:
                improvement_areas = enhancement_conditions['low_eye_score'].get('add_improvement', [])
                enhancements.extend([f"{area} 개선이 필요합니다" for area in improvement_areas])
            
            # 부정행위 감지 시 수식어
            if cheating_detected and 'cheating_detected' in enhancement_conditions:
                improvement_areas = enhancement_conditions['cheating_detected'].get('add_improvement', [])
                enhancements.extend([f"{area}가 필요합니다" for area in improvement_areas])
            
            # 최종 피드백 조합
            if enhancements:
                enhancement_text = " 특히 " + ", ".join(enhancements[:2]) + "이 돋보였습니다."
                base_feedback += enhancement_text
            
            return base_feedback
            
        except Exception as e:
            logger.error(f"동적 피드백 생성 실패: {e}")
            return "면접 태도 분석이 완료되었습니다. 전반적으로 성실한 자세를 보여주었습니다."
    
    def reload_config(self) -> bool:
        """설정 파일 다시 로드"""
        try:
            self.config = self._load_config()
            logger.info("YAML 설정 파일 다시 로드 완료")
            return True
        except Exception as e:
            logger.error(f"설정 파일 다시 로드 실패: {e}")
            return False

# 전역 키워드 분석기 인스턴스
keyword_analyzer = KeywordAnalyzer() 