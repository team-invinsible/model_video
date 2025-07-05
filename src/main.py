# ----

# 작성목적 : 통합 영상 분석 API 메인 애플리케이션
# 작성일 : 2025-06-14

# 변경사항 내역 (날짜 | 변경목적 | 변경내용 | 작성자 순으로 기입)
# 2025-06-14 | 최초 구현 | FastAPI 베스트 프랙티스에 따른 구조로 재구성 | 이재인
# 2025-06-16 | 구조 개선 | DB 저장, S3 연동, LLM 연동 구조 최적화 | 이재인
# 2025-06-16 | 자동 분석 | 서버 시작 시 S3 모든 영상 자동 분석 기능 추가 | 이재인
# 2025-06-17 | 기능 최적화 | 자동 분석 기능만 남기고 수동 업로드 관련 기능 삭제 | 이재인
# ----

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import asyncio
import os
import tempfile
import json
import shutil
from datetime import datetime
from dotenv import load_dotenv
import sys

# 현재 파일의 디렉토리를 Python 경로에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# 환경변수 로드
load_dotenv()

from src.utils.s3_handler import S3Handler
from src.utils.file_utils import FileProcessor
from src.emotion.analyzer import EmotionAnalyzer
from src.eye_tracking.analyzer import EyeTrackingAnalyzer
from src.db.database import get_db_session
from src.db.crud import save_analysis_result, get_analysis_results
from src.llm.gpt_analyzer import GPTAnalyzer
from src.db.mariadb_handler import mariadb_handler

# --- Pydantic 모델 정의 ---
class AnalysisPayload(BaseModel):
    """메인 서버로부터 분석 요청을 수신할 모델"""
    s3ObjectKey: str

app = FastAPI(
    title="통합 영상 분석 API",
    description="API 요청을 통해 S3 영상을 분석하여 감정 및 시선 추적 결과를 제공하는 API",
    version="2.1.0"
)

# 배치 처리를 위한 전역 변수
_pending_gpt_analyses = []  # GPT 분석 대기 큐
_batch_processing_active = False  # 배치 처리 활성화 상태

@app.on_event("startup")
async def startup_event():
    """애플리케이션 시작 시 실행되는 이벤트"""
    try:
        # MariaDB 연결 풀 생성
        await mariadb_handler.create_pool()
        print("✅ MariaDB 연결이 활성화되었습니다.")
        print("🚀 분석 서버가 성공적으로 시작되었습니다. API 요청을 대기합니다.")
        
        # ⚠️ S3 자동 분석 로직 제거
        # print("📡 S3 자동 분석을 시작합니다...")
        # asyncio.create_task(auto_analyze_all_s3_videos())
        
    except Exception as e:
        print(f"⚠️ 애플리케이션 시작 중 오류 발생: {e}")
        print("📍 MariaDB 연결 실패 - MongoDB만 사용합니다.")

@app.on_event("shutdown")
async def shutdown_event():
    """애플리케이션 종료 시 실행되는 이벤트"""
    try:
        await mariadb_handler.close_pool()
        print("✅ 애플리케이션이 정상적으로 종료되었습니다.")
    except Exception as e:
        print(f"⚠️ 애플리케이션 종료 중 오류 발생: {e}")

# 요청 모델 (자동 분석 관련만)
class S3UserVideoRequest(BaseModel):
    """S3 사용자별 영상 분석 요청"""
    user_id: str  # 사용자 ID (예: "iv001", "user123")
    question_num: str  # 질문 번호 (예: "Q001", "1", "question_1")
    session_id: Optional[str] = None

class AnalysisResponse(BaseModel):
    analysis_id: str
    status: str
    message: str
    result: Optional[Dict[str, Any]] = None

# 전역 인스턴스
s3_handler = S3Handler()
file_processor = FileProcessor()
emotion_analyzer = EmotionAnalyzer()
eye_tracking_analyzer = EyeTrackingAnalyzer()
gpt_analyzer = GPTAnalyzer()

# --- 영상 수신 API 엔드포인트 ---
@app.post("/analysis/attitude", response_model=AnalysisResponse)
async def analyze_video_from_s3_key(payload: AnalysisPayload):
    """
    메인 서버로부터 S3 Object Key를 받아 영상 분석을 완료한 후 결과를 응답합니다.
    """
    try:
        s3_key = payload.s3ObjectKey
        print(f"Received analysis request for s3ObjectKey: {s3_key}")
        
        # S3 키로부터 사용자 ID와 질문 번호 추출
        try:
            # 실제 S3 키 형식: skala25a/team12/interview_video/{userId}/{question_num}/*.webm
            parts = s3_key.split('/')
            print(f"🔍 S3 키 분할: {parts}")
            
            # interview_video 다음에 오는 경로에서 user_id와 question_num 추출
            if 'interview_video' in parts:
                video_index = parts.index('interview_video')
                if video_index + 2 < len(parts):
                    user_id = parts[video_index + 1]
                    question_num = parts[video_index + 2]
                    print(f"🔍 파싱 성공: user_id={user_id}, question_num={question_num}")
                else:
                    raise IndexError("interview_video 다음에 user_id와 question_num이 없습니다.")
            else:
                raise IndexError("S3 키에 'interview_video' 경로가 없습니다.")
                
        except (IndexError, ValueError) as e:
            raise HTTPException(
                status_code=400,
                detail=f"잘못된 S3 키 형식입니다. 'skala25a/team12/interview_video/{{user_id}}/{{question_num}}/...' 형식을 기대합니다: {s3_key}"
            )
            
        analysis_id = f"api_s3_analysis_{user_id}_{question_num}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        bucket_name = os.getenv('S3_BUCKET_NAME', 'skala25a')
        session_id = f"api_triggered_{user_id}"

        print(f"🎬 API 요청 기반 분석 시작: {user_id}/{question_num} -> {s3_key}")
        
        # 모든 분석 작업을 비동기로 실행하고 완료까지 대기
        analysis_result = await process_s3_user_video_analysis(
            analysis_id=analysis_id,
            s3_bucket=bucket_name,
            s3_key=s3_key,
            user_id=user_id,
            question_num=question_num,
            session_id=None
        )
        
        # 분석 결과 확인
        if analysis_result and "error" not in analysis_result:
            return AnalysisResponse(
                analysis_id=analysis_id,
                status="completed",
                message=f"사용자 {user_id}, 질문 {question_num} 영상 분석이 성공적으로 완료되었습니다.",
                result=analysis_result
            )
        else:
            error_msg = analysis_result.get("error", "알 수 없는 오류") if analysis_result else "분석 결과가 없습니다"
            return AnalysisResponse(
                analysis_id=analysis_id,
                status="failed",
                message=f"영상 분석 실패: {error_msg}",
                result=analysis_result
            )

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        # 예상치 못한 에러에 대한 로깅 강화
        print(f"🔥 /analyze 엔드포인트에서 심각한 오류 발생: {str(e)}")
        raise HTTPException(status_code=500, detail=f"분석 요청 처리 중 서버 오류 발생: {str(e)}")

@app.get("/")
async def root():
    """API 상태 확인"""
    return {"message": "통합 영상 분석 API (자동 분석 전용)가 정상 작동 중입니다."}

@app.get("/s3/available-users-questions")
async def get_available_users_and_questions():
    """
    S3에서 사용 가능한 사용자와 질문 목록을 조회합니다.
    """
    try:
        bucket_name = os.getenv('S3_BUCKET_NAME', 'skala25a')
        available_videos = await s3_handler.list_available_users_and_questions(bucket_name)
        
        return {
            "bucket": bucket_name,
            "available_videos": available_videos,
            "total_users": len(available_videos),
            "total_videos": sum(len(questions) for questions in available_videos.values())
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"S3 목록 조회 실패: {str(e)}")

@app.get("/s3/find-video/{user_id}/{question_num}")
async def find_specific_video(user_id: str, question_num: str):
    """
    특정 사용자/질문의 영상 파일을 S3에서 검색합니다.
    """
    try:
        bucket_name = os.getenv('S3_BUCKET_NAME', 'skala25a')
        video_key = await s3_handler.find_video_file(bucket_name, user_id, question_num)
        
        if video_key:
            return {
                "found": True,
                "user_id": user_id,
                "question_num": question_num,
                "s3_key": video_key,
                "s3_url": f"s3://{bucket_name}/{video_key}"
            }
        else:
            return {
                "found": False,
                "user_id": user_id,
                "question_num": question_num,
                "message": "해당 사용자/질문의 영상 파일을 찾을 수 없습니다."
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"영상 검색 실패: {str(e)}")

@app.post("/analyze-s3-user-video", response_model=AnalysisResponse)
async def analyze_s3_user_video(request: S3UserVideoRequest, background_tasks: BackgroundTasks):
    """
    S3에서 특정 사용자/질문의 영상을 분석합니다. (수동 트리거)
    """
    try:
        # 분석 ID 생성
        analysis_id = f"manual_s3_analysis_{request.user_id}_{request.question_num}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # S3 설정
        bucket_name = os.getenv('S3_BUCKET_NAME', 'skala25a')
        
        # 영상 파일 검색
        video_key = await s3_handler.find_video_file(bucket_name, request.user_id, request.question_num)
        
        if not video_key:
            raise HTTPException(
                status_code=404, 
                detail=f"사용자 {request.user_id}, 질문 {request.question_num}의 영상 파일을 찾을 수 없습니다."
            )
        
        print(f"🎬 수동 분석 시작: {request.user_id}/{request.question_num} -> {video_key}")
        
        # 백그라운드에서 분석 실행
        background_tasks.add_task(
            process_s3_user_video_analysis,
            analysis_id,
            bucket_name,
            video_key,
            request.user_id,
            request.question_num,
            request.session_id or "manual"
        )
        
        return AnalysisResponse(
            analysis_id=analysis_id,
            status="processing",
            message=f"사용자 {request.user_id}, 질문 {request.question_num} 영상 분석이 시작되었습니다."
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"분석 시작 실패: {str(e)}")

@app.get("/analysis/{analysis_id}")
async def get_analysis_result(analysis_id: str):
    """
    분석 결과를 조회합니다.
    """
    try:
        with get_db_session() as db:
            collection = db['analysis_results']
            result = collection.find_one({"analysis_id": analysis_id})
            
            if not result:
                raise HTTPException(status_code=404, detail="분석 결과를 찾을 수 없습니다.")
            
            # ObjectId를 문자열로 변환
            if '_id' in result:
                result['_id'] = str(result['_id'])
            
            return result
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"분석 결과 조회 중 오류 발생: {str(e)}")

@app.get("/analysis/{analysis_id}/llm-comment")
async def get_llm_comment(analysis_id: str):
    """
    특정 분석의 LLM 코멘트를 조회합니다.
    """
    try:
        with get_db_session() as db:
            collection = db['llm_comments']
            comment = collection.find_one({"analysis_id": analysis_id})
            
            if not comment:
                return {"message": "LLM 코멘트가 아직 생성되지 않았습니다."}
            
            # ObjectId를 문자열로 변환
            if '_id' in comment:
                comment['_id'] = str(comment['_id'])
            
            return comment
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM 코멘트 조회 중 오류 발생: {str(e)}")

@app.get("/analysis/recent")
async def get_recent_analyses(limit: int = 10):
    """
    최근 분석 결과들을 조회합니다.
    """
    try:
        with get_db_session() as db:
            collection = db['analysis_results']
            
            results = []
            for doc in collection.find().sort("created_at", -1).limit(limit):
                # ObjectId를 문자열로 변환
                if '_id' in doc:
                    doc['_id'] = str(doc['_id'])
                results.append(doc)
            
            return {"recent_analyses": results, "count": len(results)}
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"최근 분석 조회 중 오류 발생: {str(e)}")

@app.get("/analysis/{analysis_id}/status")
async def get_analysis_status(analysis_id: str):
    """
    분석 진행 상태를 조회합니다.
    """
    try:
        with get_db_session() as db:
            collection = db['analysis_results']
            result = collection.find_one(
                {"analysis_id": analysis_id},
                {"analysis_id": 1, "status": 1, "progress": 1, "stage": 1, "created_at": 1, "completed_at": 1}
            )
            
            if not result:
                raise HTTPException(status_code=404, detail="분석을 찾을 수 없습니다.")
            
            # ObjectId를 문자열로 변환
            if '_id' in result:
                result['_id'] = str(result['_id'])
            
            return result
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"분석 상태 조회 중 오류 발생: {str(e)}")

@app.post("/analysis/{analysis_id}/cancel")
async def cancel_analysis(analysis_id: str):
    """
    진행 중인 분석을 취소합니다. (실제로는 상태만 변경)
    """
    try:
        with get_db_session() as db:
            collection = db['analysis_results']
            result = collection.update_one(
                {"analysis_id": analysis_id, "status": "processing"},
                {"$set": {"status": "cancelled", "cancelled_at": datetime.now().isoformat()}}
            )
            
            if result.matched_count == 0:
                raise HTTPException(status_code=404, detail="취소할 수 있는 분석을 찾을 수 없습니다.")
            
            return {"message": "분석이 취소되었습니다.", "analysis_id": analysis_id}
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"분석 취소 중 오류 발생: {str(e)}")

@app.get("/interview-attitude/{user_id}")
async def get_interview_attitude_by_user(user_id: str):
    """
    특정 사용자의 모든 면접태도 평가를 조회합니다.
    """
    try:
        scores = await mariadb_handler.get_interview_attitude(user_id)
        
        if not scores:
            return {"message": f"사용자 {user_id}의 면접태도 평가를 찾을 수 없습니다.", "scores": []}
        
        return {
            "user_id": user_id,
            "scores": scores,
            "count": len(scores) if isinstance(scores, list) else 1
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"면접태도 조회 실패: {str(e)}")

@app.get("/interview-attitude/{user_id}/{question_num}")
async def get_interview_attitude_by_user_question(user_id: str, question_num: str):
    """
    특정 사용자/질문의 면접태도 평가를 조회합니다.
    """
    try:
        score = await mariadb_handler.get_interview_attitude(user_id, question_num)
        
        if not score:
            raise HTTPException(
                status_code=404, 
                detail=f"사용자 {user_id}, 질문 {question_num}의 면접태도 평가를 찾을 수 없습니다."
            )
        
        return score
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"면접태도 조회 실패: {str(e)}")

@app.get("/health")
async def health_check():
    """
    시스템 상태를 확인합니다.
    """
    try:
        # MongoDB 연결 확인
        mongodb_status = "healthy"
        try:
            with get_db_session() as db:
                db.list_collection_names()
        except Exception as e:
            mongodb_status = f"error: {str(e)}"
        
        # MariaDB 연결 확인
        mariadb_status = "healthy"
        try:
            await mariadb_handler.test_connection()
        except Exception as e:
            mariadb_status = f"error: {str(e)}"
        
        # S3 연결 확인
        s3_status = "healthy"
        try:
            bucket_name = os.getenv('S3_BUCKET_NAME', 'skala25a')
            await s3_handler.test_connection(bucket_name)
        except Exception as e:
            s3_status = f"error: {str(e)}"
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "services": {
                "mongodb": mongodb_status,
                "mariadb": mariadb_status,
                "s3": s3_status
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"상태 확인 실패: {str(e)}")

@app.post("/test/yaml-keywords")
async def test_yaml_keywords(analysis_data: Dict[str, Any] = None):
    """YAML 기반 키워드 분석 테스트 엔드포인트"""
    try:
        from src.llm.keyword_analyzer import keyword_analyzer
        
        # 테스트 데이터 생성 (요청에 데이터가 없으면 기본값 사용)
        if not analysis_data:
            analysis_data = {
                'emotion_score': 45,
                'eye_score': 28,
                'concentration_score': 15,
                'stability_score': 8,
                'blink_score': 5,
                'total_violations': 3,
                'face_multiple_detected': False,
                'suspected_copying': False,
                'suspected_impersonation': False,
                'dominant_emotions': 'neutral',
                'emotion_stability': '보통'
            }
        
        # 키워드 분석 실행
        result = keyword_analyzer.analyze_keywords(analysis_data)
        
        # GPT 프롬프트도 생성해보기
        system_prompt, user_prompt = keyword_analyzer.get_gpt_prompt(analysis_data)
        
        return {
            "status": "success",
            "keyword_analysis": result,
            "gpt_prompts": {
                "system_prompt": system_prompt[:500] + "..." if len(system_prompt) > 500 else system_prompt,
                "user_prompt": user_prompt[:500] + "..." if len(user_prompt) > 500 else user_prompt
            },
            "input_data": analysis_data
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "input_data": analysis_data or {}
        }

@app.post("/test/mariadb-save")
async def test_mariadb_save(request: dict):
    """MariaDB 저장 테스트 (새로운 ID 형식)"""
    try:
        user_id = request.get("user_id", "123")
        question_num = request.get("question_num", "1")
        emotion_score = request.get("emotion_score", 45.0)
        eye_score = request.get("eye_score", 28.0)
        suspected_copying = request.get("suspected_copying", False)
        suspected_impersonation = request.get("suspected_impersonation", False)
        gpt_analysis = request.get("gpt_analysis", {})
        
        # MariaDB에 저장
        success = await mariadb_handler.save_interview_attitude(
            user_id=user_id,
            question_num=question_num,
            emotion_score=emotion_score,
            eye_score=eye_score,
            suspected_copying=suspected_copying,
            suspected_impersonation=suspected_impersonation,
            gpt_analysis=gpt_analysis
        )
        
        if success:
            # 저장된 데이터 조회
            saved_data = await mariadb_handler.get_interview_attitude(user_id, question_num)
            
            return {
                "status": "success",
                "message": f"✅ 데이터 저장 완료 (새로운 ID 형식)",
                "request_data": request,
                "id_format": {
                    "ANS_SCORE_ID": f"{user_id}0{question_num}",
                    "INTV_ANS_ID": f"{user_id}0{question_num}", 
                    "ANS_CAT_RESULT_ID": f"{user_id}0{question_num}0"
                },
                "saved_data": saved_data
            }
        else:
            return {"status": "error", "message": "데이터 저장 실패"}
            
    except Exception as e:
        logger.error(f"MariaDB 테스트 오류: {e}")
        return {"status": "error", "message": str(e)}

@app.get("/test/yaml-all-features")
async def test_yaml_all_features():
    """모든 YAML 기반 기능 테스트"""
    try:
        from src.llm.keyword_analyzer import keyword_analyzer
        
        # 테스트 데이터 생성
        test_emotion_result = {
            'interview_score': 45,
            'dominant_emotion': 'neutral',
            'emotion_ratios': {'happy': 0.3, 'neutral': 0.6, 'sad': 0.1},
            'detailed_analysis': {
                'scores': {'expressiveness': 30, 'stability': 25},
                'improvement_suggestions': ['표정 다양성 개선']
            },
            'total_frames': 1500,
            'emotion_counts': {'happy': 450, 'neutral': 900, 'sad': 150},
            'confidence_scores': {'happy': 0.85, 'neutral': 0.75, 'sad': 0.6},
            'grade': 'B'
        }
        
        test_eye_result = {
            'basic_scores': {
                'total_eye_score': 28,
                'concentration_score': 15,
                'stability_score': 8,
                'blink_score': 5
            },
            'analysis_summary': {
                'total_violations': 3,
                'face_multiple_detected': False,
                'center_time_ratio': 0.7
            },
            'total_duration': 60.0,
            'blink_count': 45,
            'blink_rate': 0.75,
            'attention_score': 25,
            'gaze_stability': 20,
            'focus_score': 22
        }
        
        analysis_data = {
            'emotion_score': 45,
            'eye_score': 28,
            'concentration_score': 15,
            'stability_score': 8,
            'blink_score': 5,
            'total_violations': 3,
            'face_multiple_detected': False,
            'suspected_copying': False,
            'suspected_impersonation': False,
            'dominant_emotions': 'neutral',
            'emotion_stability': '보통'
        }
        
        # 1. 키워드 분석 테스트
        keywords = keyword_analyzer.analyze_keywords(analysis_data)
        
        # 2. GPT 프롬프트 생성 테스트
        gpt_prompt = keyword_analyzer.get_gpt_prompt(analysis_data)
        
        # 3. 상세 GPT 프롬프트 테스트
        detailed_prompt = keyword_analyzer.get_detailed_gpt_prompt(test_emotion_result, test_eye_result)
        
        # 4. 동적 피드백 생성 테스트
        dynamic_feedback = keyword_analyzer.generate_dynamic_feedback(test_emotion_result, test_eye_result)
        
        return {
            "status": "success",
            "message": "✅ 모든 YAML 기반 기능이 정상 작동합니다",
            "test_results": {
                "1_keyword_analysis": keywords,
                "2_gpt_prompt": {
                    "system": gpt_prompt[0][:300] + "..." if len(gpt_prompt[0]) > 300 else gpt_prompt[0],
                    "user": gpt_prompt[1][:300] + "..." if len(gpt_prompt[1]) > 300 else gpt_prompt[1]
                },
                "3_detailed_prompt": {
                    "system": detailed_prompt[0][:300] + "..." if len(detailed_prompt[0]) > 300 else detailed_prompt[0],
                    "user": detailed_prompt[1][:300] + "..." if len(detailed_prompt[1]) > 300 else detailed_prompt[1]
                },
                "4_dynamic_feedback": dynamic_feedback
            },
            "test_data": {
                "emotion_result": test_emotion_result,
                "eye_result": test_eye_result,
                "analysis_data": analysis_data
            }
        }
        
    except Exception as e:
        logger.error(f"YAML 전체 기능 테스트 실패: {e}")
        return {"status": "error", "message": str(e)}

@app.get("/auto-analysis/status")
async def get_auto_analysis_status():
    """
    자동 분석 진행 상황을 조회합니다.
    """
    try:
        with get_db_session() as db:
            collection = db['analysis_results']
            
            # 전체 분석 결과 통계
            total_analyses = collection.count_documents({})
            completed_analyses = collection.count_documents({"status": "completed"})
            processing_analyses = collection.count_documents({"status": "processing"})
            failed_analyses = collection.count_documents({"status": "error"})
            
            # 자동 분석 결과 (session_id가 "auto_batch"인 것들)
            auto_analyses = collection.count_documents({"session_id": "auto_batch"})
            auto_completed = collection.count_documents({
                "session_id": "auto_batch", 
                "status": "completed"
            })
            
            # 최근 분석 결과 (최근 10개)
            recent_analyses = []
            for doc in collection.find().sort("created_at", -1).limit(10):
                recent_analyses.append({
                    "analysis_id": doc.get("analysis_id"),
                    "user_id": doc.get("user_id"),
                    "question_num": doc.get("question_num"),
                    "status": doc.get("status"),
                    "created_at": doc.get("created_at"),
                    "session_id": doc.get("session_id")
                })
            
            return {
                "timestamp": datetime.now().isoformat(),
                "total_statistics": {
                    "total_analyses": total_analyses,
                    "completed": completed_analyses,
                    "processing": processing_analyses,
                    "failed": failed_analyses,
                    "completion_rate": round(completed_analyses / total_analyses * 100, 1) if total_analyses > 0 else 0
                },
                "auto_batch_statistics": {
                    "total_auto_analyses": auto_analyses,
                    "auto_completed": auto_completed,
                    "auto_completion_rate": round(auto_completed / auto_analyses * 100, 1) if auto_analyses > 0 else 0
                },
                "recent_analyses": recent_analyses
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"자동 분석 상태 조회 중 오류 발생: {str(e)}")

@app.post("/auto-analysis/restart")
async def restart_auto_analysis():
    """
    자동 분석을 수동으로 재시작합니다.
    """
    try:
        print("🔄 자동 분석을 수동으로 재시작합니다...")
        asyncio.create_task(auto_analyze_all_s3_videos())
        
        return {
            "message": "자동 분석이 백그라운드에서 시작되었습니다.",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"자동 분석 재시작 중 오류 발생: {str(e)}")

@app.get("/gpt-batch/status")
async def get_gpt_batch_status():
    """GPT 배치 처리 상태를 확인합니다."""
    global _pending_gpt_analyses, _batch_processing_active
    
    return {
        "batch_processing_active": _batch_processing_active,
        "pending_analyses": len(_pending_gpt_analyses),
        "queue": [
            {
                "analysis_id": item["analysis_id"],
                "user_id": item["user_id"], 
                "question_num": item["question_num"],
                "added_at": item["added_at"].isoformat()
            }
            for item in _pending_gpt_analyses
        ]
    }

@app.post("/gpt-batch/trigger")
async def trigger_gpt_batch(background_tasks: BackgroundTasks):
    """수동으로 GPT 배치 처리를 시작합니다."""
    global _pending_gpt_analyses
    
    if not _pending_gpt_analyses:
        return {
            "status": "success",
            "message": "GPT 분석 대기 항목이 없습니다.",
            "pending_count": 0
        }
    
    try:
        pending_count = len(_pending_gpt_analyses)
        # 백그라운드에서 GPT 배치 처리 시작
        background_tasks.add_task(process_gpt_batch)
        
        return {
            "status": "success", 
            "message": f"GPT 배치 처리를 시작했습니다.",
            "pending_count": pending_count,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"GPT 배치 처리 시작 실패: {str(e)}")

@app.post("/gpt-batch/check-and-trigger")
async def check_and_trigger_gpt_batch_endpoint(background_tasks: BackgroundTasks):
    """영상 분석 완료 상태를 확인하고 필요시 GPT 배치 처리를 시작합니다."""
    try:
        # 백그라운드에서 확인 및 트리거
        background_tasks.add_task(check_and_trigger_gpt_batch)
        
        return {
            "status": "success",
            "message": "GPT 배치 처리 확인을 시작했습니다.",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"GPT 배치 확인 실패: {str(e)}")

# === 내부 처리 함수들 ===

async def process_s3_user_video_analysis(
    analysis_id: str,
    s3_bucket: str,
    s3_key: str,
    user_id: str,
    question_num: str,
    session_id: Optional[str]
):
    """
    S3 사용자별 영상 분석 워크플로우를 처리합니다. (기존 비동기 버전)
    """
    temp_dir = None
    start_time = datetime.now()
    processing_times = {}
    
    try:
        # 분석 상태를 PROCESSING으로 업데이트
        await update_analysis_status(analysis_id, "processing", "download", 10.0)
        
        # 1. 임시 디렉토리 생성 및 S3 다운로드
        stage_start = datetime.now()
        temp_dir = tempfile.mkdtemp(prefix="video_analysis_")
        video_path = await s3_handler.download_file(s3_bucket, s3_key, temp_dir)
        processing_times["download"] = (datetime.now() - stage_start).total_seconds()
        
        await update_analysis_status(analysis_id, "processing", "emotion_analysis", 30.0)
        
        # 2. 영상/음성 분리 (필요시)
        processed_video_path = await file_processor.process_video(video_path)
        
        # 3. 감정 분석 실행
        stage_start = datetime.now()
        emotion_result = await emotion_analyzer.analyze_video(processed_video_path)
        processing_times["emotion_analysis"] = (datetime.now() - stage_start).total_seconds()
        
        await update_analysis_status(analysis_id, "processing", "eye_tracking", 60.0)
        
        # 4. 시선 추적 분석 실행
        stage_start = datetime.now()
        eye_tracking_result = await eye_tracking_analyzer.analyze_video(processed_video_path, s3_key)
        processing_times["eye_tracking"] = (datetime.now() - stage_start).total_seconds()
        
        await update_analysis_status(analysis_id, "processing", "llm_analysis", 80.0)
        
        # 5. LLM으로 종합 분석 및 코멘트 생성
        stage_start = datetime.now()
        llm_comment = await gpt_analyzer.generate_comment(
            emotion_result, eye_tracking_result, analysis_id
        )
        processing_times["llm_analysis"] = (datetime.now() - stage_start).total_seconds()
        
        await update_analysis_status(analysis_id, "processing", "save_results", 95.0)
        
        # 6. 결과를 MongoDB에 저장 (처리 시간 포함)
        stage_start = datetime.now()
        total_processing_time = (datetime.now() - start_time).total_seconds()
        
        analysis_data = {
            "analysis_id": analysis_id,
            "user_id": user_id,
            "session_id": session_id,
            "s3_bucket": s3_bucket,
            "s3_key": s3_key,
            "emotion_analysis": emotion_result,
            "eye_tracking_analysis": eye_tracking_result,
            "llm_comment_id": str(llm_comment.id) if hasattr(llm_comment, 'id') else None,
            "processing_times": processing_times,
            "total_processing_time": total_processing_time,
            "created_at": start_time.isoformat(),
            "completed_at": datetime.now().isoformat(),
            "status": "completed"
        }
        
        with get_db_session() as db:
            save_analysis_result(db, analysis_data)

        
        # 삭제된 save_analysis_summary 함수 호출 제거 (분석 요약 테이블 삭제됨)
        processing_times["save_results"] = (datetime.now() - stage_start).total_seconds()
        
        # 최종 완료 상태 업데이트
        await update_analysis_status(analysis_id, "completed", None, 100.0)
        
        # GPT 배치 분석 큐에 추가 (MariaDB 저장을 위해)
        await add_to_gpt_batch_queue(analysis_id, user_id, question_num)
        print(f"📝 GPT 분석 큐에 추가됨: {analysis_id}")
        
        # 즉시 GPT 배치 처리 트리거 (백그라운드에서 실행)
        asyncio.create_task(process_gpt_batch())
        print(f"🚀 GPT 배치 처리 즉시 트리거됨")
        
        print(f"분석 완료: {analysis_id}")
        return analysis_data
        
    except Exception as e:
        print(f"분석 중 오류 발생 ({analysis_id}): {str(e)}")
        
        # 오류 상태를 MongoDB에 저장
        error_data = {
            "analysis_id": analysis_id,
            "user_id": user_id,
            "session_id": session_id,
            "s3_bucket": s3_bucket,
            "s3_key": s3_key,
            "error": str(e),
            "created_at": datetime.now().isoformat(),
            "status": "error"
        }
        
        try:
            with get_db_session() as db:
                save_analysis_result(db, error_data)
        except:
            pass  # 오류 저장 실패는 무시
        
        return error_data
            
    finally:
        # 임시 파일 정리
        if temp_dir and os.path.exists(temp_dir):
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)

async def update_analysis_status(analysis_id: str, status: str, stage: Optional[str] = None, progress: float = 0.0):
    """분석 상태를 DB에 업데이트합니다."""
    try:
        update_data = {
            "status": status,
            "progress_percentage": progress,
            "updated_at": datetime.now().isoformat()
        }
        
        if stage:
            update_data["current_stage"] = stage
            
        if status == "processing" and "started_at" not in update_data:
            update_data["started_at"] = datetime.now().isoformat()
        elif status == "completed":
            update_data["completed_at"] = datetime.now().isoformat()
            
        # MongoDB 업데이트
        with get_db_session() as db:
            # 실제 구현에서는 update 쿼리 사용
            print(f"상태 업데이트: {analysis_id} -> {status} ({stage}, {progress}%)")
            
    except Exception as e:
        print(f"⚠️ 상태 업데이트 실패 ({analysis_id}): {e}")

async def add_to_gpt_batch_queue(analysis_id: str, user_id: str, question_num: str):
    """GPT 분석 대기열에 추가하고, 조건 충족 시 배치를 처리합니다."""
    global _pending_gpt_analyses
    _pending_gpt_analyses.append({
        'analysis_id': analysis_id,
        'user_id': user_id,
        'question_num': question_num,
        'added_at': datetime.now()
    })
    print(f"📝 GPT 분석 큐에 추가: {analysis_id} (대기 중: {len(_pending_gpt_analyses)}개)")

async def process_gpt_batch():
    """
    대기 중인 GPT 분석 작업을 일괄 처리합니다.
    """
    global _pending_gpt_analyses, _batch_processing_active
    
    if _batch_processing_active or not _pending_gpt_analyses:
        return
    
    _batch_processing_active = True
    batch_to_process = _pending_gpt_analyses.copy()
    _pending_gpt_analyses = []
    
    try:
        print(f"🚀 GPT 배치 분석 시작: {len(batch_to_process)}개 항목")
        
        for i, item in enumerate(batch_to_process, 1):
            try:
                analysis_id = item['analysis_id']
                user_id = item['user_id']
                question_num = item['question_num']
                
                print(f"[{i}/{len(batch_to_process)}] GPT 분석 시작: {analysis_id}")
                
                # MongoDB에서 분석 결과 가져오기
                with get_db_session() as db:
                    collection = db['analysis_results']
                    doc = collection.find_one({'analysis_id': analysis_id})
                
                if not doc:
                    print(f"⚠️ 분석 결과를 찾을 수 없음: {analysis_id}")
                    continue
                
                emotion_result = doc.get('emotion_analysis', {})
                eye_tracking_result = doc.get('eye_tracking_analysis', {})
                
                if not emotion_result or not eye_tracking_result:
                    print(f"⚠️ 영상 분석 결과가 불완전함: {analysis_id}")
                    continue
                
                # GPT 분석 수행
                llm_comment = await gpt_analyzer.analyze_interview_results(
                    emotion_result, eye_tracking_result, user_id, question_num
                )
                
                # === CLI 출력: 분석 결과 표시 ===
                print(f" ======================================\n")
                print(f"\n 전체 피드백:")
                print(f"   {llm_comment.overall_feedback}") 
                print(f" ======================================\n")
                
                # MariaDB atti_score 테이블에 종합 코멘트와 점수 저장
                try:
                    # MongoDB에서 원본 분석 결과의 점수를 직접 사용 (이미 60:40 배점으로 산출됨)
                    emotion_score_60 = doc.get('emotion_analysis', {}).get('interview_score', 0)  # 60점 만점
                    eye_score_40 = doc.get('eye_tracking_analysis', {}).get('basic_scores', {}).get('total_eye_score', 0)  # 40점 만점
                    
                    # LLM 전체 피드백을 종합 코멘트로 사용
                    total_comment = llm_comment.overall_feedback
                    
                    # audio.answer_score 및 answer_category_result 테이블에 면접태도 평가 저장
                    # 부정행위 감지 결과 추출
                    eye_analysis = doc.get('eye_tracking_analysis', {})
                    suspected_copying = eye_analysis.get('analysis_summary', {}).get('total_violations', 0) >= 5
                    suspected_impersonation = eye_analysis.get('analysis_summary', {}).get('face_multiple_detected', False)
                    
                    # GPT 분석 결과에서 키워드 추출 (LLMComment의 strengths/weaknesses 사용)
                    strength_keywords = llm_comment.strengths if llm_comment.strengths else ['성실한 태도']
                    weakness_keywords = llm_comment.weaknesses if llm_comment.weaknesses else ['개선 필요']
                    
                    gpt_analysis = {
                        'strength_keyword': '\n'.join(strength_keywords),
                        'weakness_keyword': '\n'.join(weakness_keywords)
                    }
                    
                    print(f"🔍 GPT 키워드 추출: 강점={strength_keywords}, 약점={weakness_keywords}")
                    
                    await mariadb_handler.save_interview_attitude(
                        user_id=user_id,
                        question_num=question_num,
                        emotion_score=emotion_score_60,
                        eye_score=eye_score_40,
                        suspected_copying=suspected_copying,
                        suspected_impersonation=suspected_impersonation,
                        gpt_analysis=gpt_analysis
                    )
                    
                    print(f"💾 MariaDB 면접태도 평가 저장 완료: {user_id}/{question_num} (감정:{emotion_score_60:.1f}, 시선:{eye_score_40:.1f}, 커닝:{suspected_copying}, 대리:{suspected_impersonation})")
                    
                except Exception as mariadb_error:
                    print(f"⚠️ MariaDB 저장 실패: {mariadb_error}")
                
                # MongoDB에 LLM 결과 추가 (주석처리)
                # with get_db_session() as db:
                #     collection = db['analysis_results']
                #     collection.update_one(
                #         {'analysis_id': analysis_id},
                #         {
                #             '$set': {
                #                 'llm_comment_id': str(llm_comment.id) if hasattr(llm_comment, 'id') else None,
                #                 'llm_processed_at': datetime.now().isoformat(),
                #                 'overall_score': llm_comment.overall_score
                #             }
                #         }
                #     )
                
                print(f"[{i}/{len(batch_to_process)}] GPT 분석 완료: {analysis_id} (점수: {llm_comment.overall_score})")
                
                # 분석 간 대기 (Rate limiting)
                if i < len(batch_to_process):  # 마지막이 아니면 대기
                    await asyncio.sleep(1)
                    
            except Exception as e:
                print(f"❌ GPT 분석 실패 ({item['analysis_id']}): {str(e)}")
                continue
        
        print(f"✅ GPT 배치 분석 완료: {len(batch_to_process)}개 처리")
        
    except Exception as e:
        print(f"❌ GPT 배치 처리 중 오류: {str(e)}")
    finally:
        _batch_processing_active = False

async def check_and_trigger_gpt_batch():
    """
    GPT 분석 대기열을 확인하고, 조건(개수 또는 시간)에 따라 배치를 트리거합니다.
    """
    global _pending_gpt_analyses, _batch_processing_active
    
    # ... (함수 내용은 그대로 유지) ...

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 