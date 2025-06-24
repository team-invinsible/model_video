# ----
# 작성목적 : GPT를 활용한 면접 분석 및 평가 생성 모듈
# 작성일 : 2025-06-17

# 변경사항 내역 (날짜 | 변경목적 | 변경내용 | 작성자 순으로 기입)
# 2025-06-15 | 최초 구현 | FastAPI 베스트 프랙티스에 따른 구조로 재구성 | 이재인
# 2025-06-17 | 프롬프트 개선 | 면접 태도 평가 프롬프트 구체화 및 세분화 | 이재인
# 2025-06-24 | YAML 기반 전환 | 모든 하드코딩된 프롬프트를 YAML 파일 기반으로 전환 | 이재인
# ----

from openai import AsyncOpenAI
import os
import json
import asyncio
import time
import random
from typing import Dict, Any, Optional
from datetime import datetime
from src.db.models import LLMComment
from dotenv import load_dotenv

# .env 파일을 명시적 경로로 로드 (프로젝트 루트에서)
import pathlib
project_root = pathlib.Path(__file__).parent.parent.parent
load_dotenv(project_root / '.env', override=True)

class GPTAnalyzer:
    """GPT API를 사용하여 면접 분석 결과를 평가하고 피드백을 생성하는 클래스"""
    
    def __init__(self, api_key: str = None, model: str = "gpt-4"):
        """GPT 분석기 초기화"""
        # GPT 분석 활성화 여부 확인
        self.enabled = os.getenv('OPENAI_ENABLED', 'true').lower() == 'true'
        
        if not self.enabled:
            print("⚠️ OpenAI GPT 분석이 비활성화되어 있습니다.")
            self.api_key = None
            self.client = None
            return
        
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.model = model
        
        if not self.api_key:
            print("❌ OpenAI API 키가 설정되지 않았습니다.")
            self.enabled = False
            self.client = None
            return
        
        # API 키 유효성 검증
        print("🔐 OpenAI API 키 유효성 검증 중...")
        if not self._validate_api_key():
            print("❌ OpenAI API 키가 유효하지 않습니다. Fallback 모드로 전환합니다.")
            self.enabled = False
            self.client = None
            return
        
        # AsyncOpenAI 클라이언트 생성 (자동 재시도 비활성화)
        self.client = AsyncOpenAI(api_key=self.api_key, max_retries=0)
        
        # 모델별 설정
        self._configure_model_settings()
        
        print(f"🤖 GPT 분석기 초기화 완료")
        print(f"📋 모델: {self.model}")
        print(f"⏱️ 요청 간격: {self.request_interval}초")
        print(f"🔄 최대 재시도: {self.max_retries}회")
        print(f"⏳ 타임아웃: {self.timeout}초")
        print(f"🚀 병렬 처리 모드 활성화 (세마포어 제거)")
    
    def _configure_model_settings(self):
        """모델별 설정 값 구성"""
        if "gpt-4" in self.model.lower():
            self.request_interval = float(os.getenv('OPENAI_GPT4_INTERVAL', '2.0'))  # 병렬 처리를 위해 간격 단축
            self.max_retries = int(os.getenv('OPENAI_GPT4_RETRIES', '6'))
            self.base_delay = float(os.getenv('OPENAI_GPT4_DELAY', '3.0'))  # 재시도 간격 단축
            self.timeout = 90
            self.max_tokens = 1500
        else:
            self.request_interval = float(os.getenv('OPENAI_GPT35_INTERVAL', '1.0'))  # 병렬 처리를 위해 간격 단축
            self.max_retries = int(os.getenv('OPENAI_GPT35_RETRIES', '4'))
            self.base_delay = float(os.getenv('OPENAI_GPT35_DELAY', '2.0'))  # 재시도 간격 단축
            self.timeout = 60
            self.max_tokens = 1200
        
        self.last_request_time = 0
    
    def _validate_api_key(self) -> bool:
        """OpenAI API 키 유효성을 동기적으로 검증"""
        try:
            import openai
            from openai import OpenAI
            
            # 동기 클라이언트로 간단한 테스트 요청
            test_client = OpenAI(api_key=self.api_key, max_retries=0)
            
            # 1단계: 모델 목록 요청으로 API 키 기본 유효성 확인
            test_client.models.list()
            print("✅ API 키 기본 인증 성공")
            
            # 2단계: 사용량 정보 확인 (가능한 경우)
            try:
                print("🔎 사용량 정보 확인 중...")
                # OpenAI의 usage API는 특정 조건에서만 사용 가능
                # 조직 계정이나 특정 플랜에서만 지원됨
                usage = test_client.usage.retrieve()
                print("🔎 사용량 정보:", usage)
            except Exception as usage_e:
                print("ℹ️ 사용량 정보를 가져올 수 없습니다 (개인 계정 또는 권한 제한)")
            
            # 3단계: 실제 GPT 요청으로 사용량 한도 확인
            print("🔍 GPT 사용량 한도 확인 중...")
            response = test_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "test"}],
                max_tokens=5
            )
            
            print("✅ OpenAI API 키가 유효하고 사용 가능합니다.")
            return True
            
        except openai.AuthenticationError:
            print("❌ API 키 인증 오류: 유효하지 않은 API 키입니다.")
            return False
        except openai.RateLimitError:
            print("⚠️ 유효한 키지만 Rate Limit에 도달했습니다.")
            print("   - 분당/시간당 요청 한도 초과")
            print("   - Fallback 모드로 전환합니다.")
            return False
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg:
                print(f"⚠️ API 사용량 한도 초과 (429 에러)")
                print(f"   - 분당/시간당 요청 한도 초과 또는 크레딧 소진")
                print(f"   - Fallback 모드로 전환합니다.")
            elif "quota" in error_msg.lower() or "insufficient" in error_msg.lower():
                print(f"⚠️ API 크레딧 완전 소진")
                print(f"   - OpenAI 대시보드에서 크레딧 충전 필요")
                print(f"   - Fallback 모드로 전환합니다.")
            else:
                print(f"❓ 알 수 없는 오류 발생: {error_msg}")
            return False

    async def analyze_interview_results(self, 
                                      emotion_result: Dict[str, Any], 
                                      eye_tracking_result: Dict[str, Any],
                                      user_id: str, 
                                      question_num: str) -> LLMComment:
        """사용자별 질문별 면접 결과 종합 분석"""
        analysis_id = f"{user_id}_{question_num}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # GPT 분석이 비활성화된 경우 즉시 fallback 사용
        if not self.enabled or not self.client:
            print(f"📝 GPT 분석 비활성화됨, fallback 사용: {analysis_id}")
            return await self._create_fallback_comment(emotion_result, eye_tracking_result, user_id, question_num, analysis_id)
        
        try:
            # 프롬프트 생성
            prompt = self._create_prompt(emotion_result, eye_tracking_result, user_id, question_num)
            
            # GPT 호출 (병렬 처리)
            response = await self._call_gpt_with_retry(prompt)
            
            # 응답 파싱
            comment = await self._parse_response(response, emotion_result, eye_tracking_result, analysis_id)
            comment.user_id = user_id
            # question_num은 analysis_id에 포함되어 있음
            
            return comment
            
        except Exception as e:
            print(f"⚠️ GPT 분석 실패, fallback 사용: {str(e)}")
            return await self._create_fallback_comment(emotion_result, eye_tracking_result, user_id, question_num, analysis_id)
    
    async def generate_comment(self, 
                             emotion_result: Dict[str, Any], 
                             eye_tracking_result: Dict[str, Any],
                             analysis_id: str) -> LLMComment:
        """일반 분석 결과 GPT 코멘트 생성"""
        # GPT 분석이 비활성화된 경우 즉시 fallback 사용
        if not self.enabled or not self.client:
            print(f"📝 GPT 분석 비활성화됨, fallback 사용: {analysis_id}")
            return await self._create_fallback_comment(emotion_result, eye_tracking_result, analysis_id=analysis_id)
        
        try:
            # 프롬프트 생성
            prompt = self._create_prompt(emotion_result, eye_tracking_result)
            
            # GPT 호출 (병렬 처리)
            response = await self._call_gpt_with_retry(prompt)
            
            # 응답 파싱
            return await self._parse_response(response, emotion_result, eye_tracking_result, analysis_id)
            
        except Exception as e:
            print(f"⚠️ GPT 분석 실패, fallback 사용: {str(e)}")
            return await self._create_fallback_comment(emotion_result, eye_tracking_result, analysis_id=analysis_id)
    
    async def _call_gpt_with_retry(self, prompt: str) -> str:
        """GPT API 호출 (재시도 로직 포함)"""
        print(f"🚀 GPT API 호출 시작 - 모델: {self.model}")
        print(f"📤 전송할 프롬프트 길이: {len(prompt)} 문자")
        
        for attempt in range(self.max_retries):
            try:
                await self._apply_rate_limiting()
                
                print(f"🔄 시도 {attempt + 1}/{self.max_retries}")
                response = await self._make_api_call(prompt, self.model)
                
                if response:
                    print(f"✅ GPT API 응답 성공 - 응답 길이: {len(response)} 문자")
                    return response
                else:
                    print(f"⚠️ 빈 응답 받음 (시도 {attempt + 1})")
                    
            except Exception as e:
                print(f"❌ API 호출 실패 (시도 {attempt + 1}): {str(e)}")
                if attempt < self.max_retries - 1:  # 마지막 시도가 아니면
                    wait_time = (2 ** attempt) * self.base_delay
                    print(f"⏳ {wait_time}초 후 재시도...")
                    await asyncio.sleep(wait_time)
                else:
                    print(f"💥 모든 재시도 실패, 최종 오류: {str(e)}")
                    
        print("❌ GPT API 호출 최종 실패")
        return ""
    
    async def _apply_rate_limiting(self):
        """요청 간격 제한 적용 (병렬 처리를 위해 최소한으로)"""
        current_time = time.time()
        elapsed = current_time - self.last_request_time
        
        if elapsed < self.request_interval:
            wait_time = self.request_interval - elapsed
            print(f"⏱️ Rate limiting: {wait_time:.1f}초 대기")
            await asyncio.sleep(wait_time)
        
        self.last_request_time = time.time()
    
    async def _make_api_call(self, prompt: str, model: str) -> str:
        """실제 OpenAI API 호출"""
        print(f"🤖 API 호출 중... (모델: {model})")
        
        # prompt가 이미 시스템과 사용자 프롬프트가 결합된 형태라고 가정
        # YAML에서 생성된 프롬프트를 사용자 메시지로 전달
        response = await self.client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.7,
            max_tokens=self.max_tokens,
            timeout=self.timeout
        )
        
        return response.choices[0].message.content
    
    def _create_prompt(self, emotion_result: Dict[str, Any], eye_tracking_result: Dict[str, Any], 
                      user_id: str = None, question_num: str = None) -> str:
        """YAML 설정 기반 GPT 프롬프트 생성"""
        
        # YAML 기반 키워드 분석기 import
        try:
            from .keyword_analyzer import keyword_analyzer
            
            # 분석 데이터 준비
            analysis_data = {
                'emotion_score': emotion_result.get('interview_score', 0),
                'eye_score': eye_tracking_result.get('basic_scores', {}).get('total_eye_score', 0),
                'concentration_score': eye_tracking_result.get('basic_scores', {}).get('concentration_score', 0),
                'stability_score': eye_tracking_result.get('basic_scores', {}).get('stability_score', 0),
                'blink_score': eye_tracking_result.get('basic_scores', {}).get('blink_score', 0),
                'total_violations': eye_tracking_result.get('analysis_summary', {}).get('total_violations', 0),
                'face_multiple_detected': eye_tracking_result.get('analysis_summary', {}).get('face_multiple_detected', False),
                'suspected_copying': eye_tracking_result.get('analysis_summary', {}).get('total_violations', 0) >= 5,
                'suspected_impersonation': eye_tracking_result.get('analysis_summary', {}).get('face_multiple_detected', False),
                'dominant_emotions': emotion_result.get('dominant_emotion', '중립'),
                'emotion_stability': '높음' if emotion_result.get('interview_score', 0) >= 50 else '보통'
            }
            
            # YAML 기반 프롬프트 생성
            system_prompt, user_prompt = keyword_analyzer.get_gpt_prompt(analysis_data)
            
            # system_prompt와 user_prompt를 결합
            full_prompt = f"{system_prompt}\n\n{user_prompt}"
            print(f"🔍 YAML 기반 프롬프트 생성 완료")
            return full_prompt
            
        except Exception as e:
            print(f"⚠️ YAML 기반 프롬프트 생성 실패, 기존 방식 사용: {e}")
            return self._create_legacy_prompt(emotion_result, eye_tracking_result, user_id, question_num)
    
    def _create_legacy_prompt(self, emotion_result: Dict[str, Any], eye_tracking_result: Dict[str, Any], 
                             user_id: str = None, question_num: str = None) -> str:
        """YAML 기반 상세 프롬프트 생성 (fallback)"""
        
        # YAML 기반 키워드 분석기 사용
        try:
            from .keyword_analyzer import keyword_analyzer
            
            # YAML 기반 상세 GPT 프롬프트 생성
            system_prompt, user_prompt = keyword_analyzer.get_detailed_gpt_prompt(
                emotion_result, eye_tracking_result
            )
            
            # system_prompt와 user_prompt를 결합
            full_prompt = f"{system_prompt}\n\n{user_prompt}"
            print(f"🔍 YAML 기반 레거시 프롬프트 생성 완료")
            return full_prompt
            
        except Exception as e:
            print(f"⚠️ YAML 기반 레거시 프롬프트 생성 실패: {e}")
            # 최소한의 fallback
            return "면접 영상 분석 결과를 바탕으로 지원자의 면접 태도를 4줄(300자 이내)로 평가해주세요."
    
    async def _parse_response(self, response: str, emotion_result: Dict[str, Any], 
                             eye_tracking_result: Dict[str, Any], analysis_id: str) -> LLMComment:
        """GPT 응답을 LLMComment 객체로 파싱"""
        try:
            print(f"🔍 GPT 원본 응답: {response[:500]}...")  # 응답 내용 확인
            
            overall_feedback = response.strip()
            
            # JSON 형태로 응답한 경우 텍스트만 추출
            if overall_feedback.startswith('{') and overall_feedback.endswith('}'):
                try:
                    import json
                    data = json.loads(overall_feedback)
                    # evaluation 키에서 텍스트 추출
                    if 'evaluation' in data:
                        overall_feedback = data['evaluation'].strip()
                        print("✅ JSON에서 evaluation 텍스트 추출 성공")
                    elif 'overall_feedback' in data:
                        overall_feedback = data['overall_feedback'].strip()
                        print("✅ JSON에서 overall_feedback 텍스트 추출 성공")
                    else:
                        # JSON의 첫 번째 값 사용
                        overall_feedback = list(data.values())[0].strip()
                        print("✅ JSON에서 첫 번째 값 추출 성공")
                except json.JSONDecodeError:
                    print("⚠️ JSON 파싱 실패, 원본 텍스트 사용")
                    pass
            
            # 응답이 비어있거나 너무 짧으면 기본값 사용
            if not overall_feedback or len(overall_feedback) < 20:
                print("⚠️ GPT 응답이 비어있거나 너무 짧음, fallback 사용")
                return await self._create_fallback_comment(emotion_result, eye_tracking_result, analysis_id=analysis_id)
            
            print(f"📝 최종 overall_feedback: {overall_feedback}")
            
            return LLMComment(
                analysis_id=analysis_id,
                overall_score=0.0,
                emotion_feedback="",
                attention_feedback="",
                overall_feedback=overall_feedback,
                improvement_suggestions=[],
                strengths=[],
                weaknesses=[],
                emotion_score=0.0,
                attention_score=0.0,
                stability_score=0.0
            )
            
        except Exception as e:
            print(f"❌ _parse_response 오류: {str(e)}")
            print(f"❌ 응답 내용: {response}")
            return await self._create_fallback_comment(emotion_result, eye_tracking_result, analysis_id=analysis_id)
    
    async def _create_fallback_comment(self, emotion_result: Dict[str, Any], 
                                     eye_tracking_result: Dict[str, Any],
                                     user_id: str = None, question_num: str = None,
                                     analysis_id: str = None) -> LLMComment:
        """Fallback LLMComment 생성"""
        if not analysis_id:
            analysis_id = f"fallback_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # LLM 기반 동적 평가 생성 시도
        try:
            fallback_comment = await self._generate_fallback_with_llm(emotion_result, eye_tracking_result, user_id, question_num)
            if fallback_comment:
                return fallback_comment
        except Exception as e:
            print(f"⚠️ LLM 기반 fallback 생성 실패: {str(e)}")
        
        # 최종 fallback: YAML 기반 동적 피드백 생성
        overall_feedback = self._generate_dynamic_feedback(emotion_result, eye_tracking_result)
        
        comment = LLMComment(
            analysis_id=analysis_id,
            overall_score=0.0,
            emotion_feedback="",
            attention_feedback="",
            overall_feedback=overall_feedback,
            improvement_suggestions=[],
            strengths=[],
            weaknesses=[],
            emotion_score=0.0,
            attention_score=0.0,
            stability_score=0.0
        )
        
        if user_id:
            comment.user_id = user_id
            
        return comment

    async def _generate_fallback_with_llm(self, emotion_result: Dict[str, Any], 
                                         eye_tracking_result: Dict[str, Any],
                                         user_id: str = None, question_num: str = None) -> Optional[LLMComment]:
        """LLM을 사용한 fallback 코멘트 생성"""
        try:
            # 간단한 프롬프트로 바로 텍스트 피드백 생성
            prompt = f"""면접 분석 결과를 바탕으로 4줄로 구성된 면접 태도 평가를 작성해주세요.

1. 전반적인 면접 태도 평가
2. 표정 평가
3. 시선움직임 평가  
4. 눈깜빡임 평가

과거형으로 서술하고 자연스러운 한국어로 작성해주세요."""

            response = await self._call_gpt_with_retry(prompt)
            if response and response.strip():
                overall_feedback = response.strip()
                
                return LLMComment(
                    analysis_id=f"fallback_llm_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    overall_score=0.0,
                    emotion_feedback="",
                    attention_feedback="",
                    overall_feedback=overall_feedback,
                    improvement_suggestions=[],
                    strengths=[],
                    weaknesses=[],
                    emotion_score=0.0,
                    attention_score=0.0,
                    stability_score=0.0
                )
                
        except Exception as e:
            print(f"⚠️ LLM fallback 생성 중 오류: {str(e)}")
            
        return None

    def _generate_dynamic_feedback(self, emotion_result: Dict[str, Any], 
                                  eye_tracking_result: Dict[str, Any]):
        """YAML 기반 동적 피드백 생성 (최종 fallback)"""
        try:
            from .keyword_analyzer import keyword_analyzer
            
            # YAML 기반 동적 피드백 생성
            feedback = keyword_analyzer.generate_dynamic_feedback(emotion_result, eye_tracking_result)
            print(f"🔍 YAML 기반 동적 피드백 생성 완료")
            return feedback
            
        except Exception as e:
            print(f"⚠️ YAML 기반 동적 피드백 생성 실패: {e}")
            # 최소한의 하드코딩된 fallback
            return "면접 태도 분석이 완료되었습니다. 전반적으로 성실한 자세를 보여주었습니다."

# 헬퍼 함수
def create_gpt_analyzer_from_env() -> GPTAnalyzer:
    """환경변수에서 설정을 로드하여 GPTAnalyzer 생성"""
    api_key = os.getenv('OPENAI_API_KEY')
    model = os.getenv('OPENAI_MODEL', 'gpt-4')
    return GPTAnalyzer(api_key=api_key, model=model) 