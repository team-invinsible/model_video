# ----
# 작성목적 : 데이터베이스 모델 정의
# 작성일 : 2025-06-14

# 변경사항 내역 (날짜 | 변경목적 | 변경내용 | 작성자 순으로 기입)
# 2025-06-14 | 최초 구현 | FastAPI 베스트 프랙티스에 따른 구조로 재구성 | 이재인
# 2025-06-16 | 구조 개선 | 분석 상태 관리 및 LLM 연동 구조 최적화 | 이재인
# ----

from typing import Optional, Dict, Any, List
from datetime import datetime
from pydantic import BaseModel, Field, GetJsonSchemaHandler
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import core_schema
from bson import ObjectId
from enum import Enum

class AnalysisStatus(str, Enum):
    """분석 상태 열거형"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    ERROR = "error"
    CANCELLED = "cancelled"

class ProcessingStage(str, Enum):
    """처리 단계 열거형"""
    DOWNLOAD = "download"
    EMOTION_ANALYSIS = "emotion_analysis"
    EYE_TRACKING = "eye_tracking"
    LLM_ANALYSIS = "llm_analysis"
    SAVE_RESULTS = "save_results"

class PyObjectId(ObjectId):
    """MongoDB ObjectId를 위한 커스텀 타입"""
    
    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: Any
    ) -> core_schema.CoreSchema:
        return core_schema.json_or_python_schema(
            json_schema=core_schema.str_schema(),
            python_schema=core_schema.union_schema([
                core_schema.is_instance_schema(ObjectId),
                core_schema.chain_schema([
                    core_schema.str_schema(),
                    core_schema.no_info_plain_validator_function(cls.validate),
                ])
            ]),
            serialization=core_schema.plain_serializer_function_ser_schema(
                lambda x: str(x)
            ),
        )

    @classmethod
    def validate(cls, v):
        if isinstance(v, ObjectId):
            return v
        if isinstance(v, str) and ObjectId.is_valid(v):
            return ObjectId(v)
        raise ValueError("Invalid ObjectId")

    @classmethod
    def __get_pydantic_json_schema__(
        cls, schema: core_schema.CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        return {"type": "string"}

class EmotionAnalysisResult(BaseModel):
    """감정 분석 결과 모델"""
    total_frames: int
    emotion_counts: Dict[str, int]
    emotion_ratios: Dict[str, float]
    dominant_emotion: str
    confidence_scores: Dict[str, float]
    interview_score: float
    grade: str
    detailed_analysis: Dict[str, Any]
    frame_by_frame_results: List[Dict[str, Any]]
    
    model_config = {"arbitrary_types_allowed": True}

class EyeTrackingResult(BaseModel):
    """시선 추적 결과 모델"""
    total_duration: float
    blink_count: int
    blink_rate: float
    gaze_directions: Dict[str, float]  # center, left, right, up, down 비율
    attention_score: float
    anomaly_events: List[Dict[str, Any]]  # 다중 얼굴 감지 등
    gaze_stability: float
    detailed_tracking: List[Dict[str, Any]]
    
    model_config = {"arbitrary_types_allowed": True}

class AnalysisResult(BaseModel):
    """전체 분석 결과 모델"""
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    analysis_id: str = Field(..., description="고유 분석 ID")
    user_id: Optional[str] = Field(None, description="사용자 ID")
    session_id: Optional[str] = Field(None, description="세션 ID")
    
    # S3 정보 (S3 파일인 경우)
    s3_bucket: Optional[str] = Field(None, description="S3 버킷 이름")
    s3_key: Optional[str] = Field(None, description="S3 객체 키")
    
    # 로컬 파일 정보 (로컬 파일인 경우)
    video_path: Optional[str] = Field(None, description="비디오 파일 경로")
    video_filename: Optional[str] = Field(None, description="비디오 파일명")
    
    # 분석 결과
    emotion_analysis: Optional[EmotionAnalysisResult] = None
    eye_tracking_analysis: Optional[EyeTrackingResult] = None
    
    # 메타데이터
    status: AnalysisStatus = Field(default=AnalysisStatus.PENDING, description="분석 상태")
    current_stage: Optional[ProcessingStage] = Field(None, description="현재 처리 단계")
    progress_percentage: float = Field(default=0.0, description="진행률 (0-100)")
    created_at: datetime = Field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    
    # 처리 시간 추적
    processing_times: Optional[Dict[str, float]] = Field(default_factory=dict, description="각 단계별 처리 시간")
    
    # 비디오 정보
    video_info: Optional[Dict[str, Any]] = None
    
    # LLM 분석 결과 연결
    llm_comment_id: Optional[str] = Field(None, description="LLM 코멘트 ID (MariaDB)")
    
    # S3 업로드 정보 (결과 파일)
    result_s3_bucket: Optional[str] = Field(None, description="결과 파일 S3 버킷")
    result_s3_key: Optional[str] = Field(None, description="결과 파일 S3 키")
    
    model_config = {
        "populate_by_name": True,
        "arbitrary_types_allowed": True,
        "json_encoders": {ObjectId: str}
    }

class LLMComment(BaseModel):
    """LLM 생성 코멘트 모델 (MariaDB용)"""
    id: Optional[int] = None
    analysis_id: str = Field(..., description="분석 ID")
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    
    # LLM 분석 결과
    overall_score: float = Field(..., description="전체 점수")
    emotion_feedback: str = Field(..., description="감정 분석 피드백")
    attention_feedback: str = Field(..., description="집중도 피드백")
    overall_feedback: str = Field(..., description="전체 피드백")
    improvement_suggestions: List[str] = Field(..., description="개선 제안사항")
    strengths: List[str] = Field(..., description="강점")
    weaknesses: List[str] = Field(..., description="약점")
    
    # 세부 점수
    emotion_score: float = Field(..., description="감정 점수")
    attention_score: float = Field(..., description="집중도 점수")
    stability_score: float = Field(..., description="안정성 점수")
    
    # 메타데이터
    llm_model: str = Field(default="gpt-4", description="사용된 LLM 모델")
    generated_at: datetime = Field(default_factory=datetime.now)
    
    model_config = {"arbitrary_types_allowed": True}

# MongoDB 컬렉션 스키마 검증을 위한 JSON 스키마
ANALYSIS_RESULT_SCHEMA = {
    "$jsonSchema": {
        "bsonType": "object",
        "required": ["analysis_id", "status", "created_at"],
        "properties": {
            "analysis_id": {
                "bsonType": "string",
                "description": "고유 분석 ID"
            },
            "user_id": {
                "bsonType": ["string", "null"],
                "description": "사용자 ID"
            },
            "session_id": {
                "bsonType": ["string", "null"],
                "description": "세션 ID"
            },
            "s3_bucket": {
                "bsonType": ["string", "null"],
                "description": "S3 버킷 이름"
            },
            "s3_key": {
                "bsonType": ["string", "null"],
                "description": "S3 객체 키"
            },
            "video_path": {
                "bsonType": ["string", "null"],
                "description": "비디오 파일 경로"
            },
            "video_filename": {
                "bsonType": ["string", "null"],
                "description": "비디오 파일명"
            },
            "status": {
                "bsonType": "string",
                "enum": ["processing", "completed", "error"],
                "description": "분석 상태"
            },
            "created_at": {
                "bsonType": "date",
                "description": "생성 시간"
            },
            "completed_at": {
                "bsonType": ["date", "null"],
                "description": "완료 시간"
            },
            "emotion_analysis": {
                "bsonType": ["object", "null"],
                "description": "감정 분석 결과"
            },
            "eye_tracking_analysis": {
                "bsonType": ["object", "null"],
                "description": "시선 추적 결과"
            }
        }
    }
}

def create_analysis_result_document(data: Dict[str, Any]) -> Dict[str, Any]:
    """분석 결과 문서를 생성합니다."""
    document = {
        "analysis_id": data["analysis_id"],
        "user_id": data.get("user_id"),
        "session_id": data.get("session_id"),
        "status": data["status"],
        "created_at": data.get("created_at", datetime.now()),
        "completed_at": data.get("completed_at"),
        "error_message": data.get("error_message"),
        "emotion_analysis": data.get("emotion_analysis"),
        "eye_tracking_analysis": data.get("eye_tracking_analysis"),
        "video_info": data.get("video_info")
    }
    
    # S3 정보 (있는 경우에만 추가)
    if "s3_bucket" in data:
        document["s3_bucket"] = data["s3_bucket"]
    if "s3_key" in data:
        document["s3_key"] = data["s3_key"]
    
    # 로컬 파일 정보 (있는 경우에만 추가)
    if "video_path" in data:
        document["video_path"] = data["video_path"]
    if "video_filename" in data:
        document["video_filename"] = data["video_filename"]
    
    # None 값 제거
    return {k: v for k, v in document.items() if v is not None}

def parse_analysis_result(document: Dict[str, Any]) -> AnalysisResult:
    """MongoDB 문서를 AnalysisResult 모델로 변환합니다."""
    try:
        return AnalysisResult(**document)
    except Exception as e:
        raise ValueError(f"분석 결과 파싱 오류: {str(e)}") 