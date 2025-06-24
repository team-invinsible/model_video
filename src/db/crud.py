from typing import Optional, Dict, Any, List
from datetime import datetime
from pymongo.database import Database
from pymongo.collection import Collection
from pymongo.errors import DuplicateKeyError
import logging
import numpy as np

from .models import create_analysis_result_document, parse_analysis_result, AnalysisResult

logger = logging.getLogger(__name__)

def convert_numpy_types(obj):
    """numpy 타입을 Python 기본 타입으로 변환합니다."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

def save_analysis_result(db: Database, analysis_data: Dict[str, Any]) -> str:
    """
    분석 결과를 MongoDB에 저장합니다.
    
    Args:
        db: MongoDB 데이터베이스 인스턴스
        analysis_data: 저장할 분석 데이터
        
    Returns:
        str: 저장된 문서의 ObjectId
    """
    try:
        collection: Collection = db['analysis_results']
        
        # numpy 타입을 Python 기본 타입으로 변환
        cleaned_data = convert_numpy_types(analysis_data)
        
        # 문서 생성
        document = create_analysis_result_document(cleaned_data)
        
        # 중복 확인 및 업데이트 또는 삽입
        existing = collection.find_one({"analysis_id": document["analysis_id"]})
        
        if existing:
            # 기존 문서 업데이트
            result = collection.update_one(
                {"analysis_id": document["analysis_id"]},
                {"$set": document}
            )
            logger.info(f"분석 결과 업데이트: {document['analysis_id']}")
            return str(existing["_id"])
        else:
            # 새 문서 삽입
            result = collection.insert_one(document)
            logger.info(f"분석 결과 저장: {document['analysis_id']}")
            return str(result.inserted_id)
            
    except Exception as e:
        logger.error(f"분석 결과 저장 오류: {str(e)}")
        raise Exception(f"분석 결과 저장 실패: {str(e)}")

def get_analysis_results(db: Database, analysis_id: str) -> Optional[Dict[str, Any]]:
    """
    분석 ID로 결과를 조회합니다.
    
    Args:
        db: MongoDB 데이터베이스 인스턴스
        analysis_id: 조회할 분석 ID
        
    Returns:
        Optional[Dict[str, Any]]: 분석 결과 또는 None
    """
    try:
        collection: Collection = db['analysis_results']
        document = collection.find_one({"analysis_id": analysis_id})
        
        if document:
            # ObjectId를 문자열로 변환
            document["_id"] = str(document["_id"])
            return document
        
        return None
        
    except Exception as e:
        logger.error(f"분석 결과 조회 오류: {str(e)}")
        raise Exception(f"분석 결과 조회 실패: {str(e)}")

def get_analysis_results_by_user(db: Database, user_id: str, 
                                limit: int = 10, skip: int = 0) -> List[Dict[str, Any]]:
    """
    사용자 ID로 분석 결과 목록을 조회합니다.
    
    Args:
        db: MongoDB 데이터베이스 인스턴스
        user_id: 사용자 ID
        limit: 조회할 최대 개수
        skip: 건너뛸 개수
        
    Returns:
        List[Dict[str, Any]]: 분석 결과 목록
    """
    try:
        collection: Collection = db['analysis_results']
        
        cursor = collection.find(
            {"user_id": user_id}
        ).sort("created_at", -1).skip(skip).limit(limit)
        
        results = []
        for document in cursor:
            document["_id"] = str(document["_id"])
            results.append(document)
        
        return results
        
    except Exception as e:
        logger.error(f"사용자별 분석 결과 조회 오류: {str(e)}")
        raise Exception(f"사용자별 분석 결과 조회 실패: {str(e)}")

def get_analysis_results_by_session(db: Database, session_id: str) -> List[Dict[str, Any]]:
    """
    세션 ID로 분석 결과 목록을 조회합니다.
    
    Args:
        db: MongoDB 데이터베이스 인스턴스
        session_id: 세션 ID
        
    Returns:
        List[Dict[str, Any]]: 분석 결과 목록
    """
    try:
        collection: Collection = db['analysis_results']
        
        cursor = collection.find(
            {"session_id": session_id}
        ).sort("created_at", -1)
        
        results = []
        for document in cursor:
            document["_id"] = str(document["_id"])
            results.append(document)
        
        return results
        
    except Exception as e:
        logger.error(f"세션별 분석 결과 조회 오류: {str(e)}")
        raise Exception(f"세션별 분석 결과 조회 실패: {str(e)}")

def update_analysis_status(db: Database, analysis_id: str, 
                          status: str, error_message: Optional[str] = None) -> bool:
    """
    분석 상태를 업데이트합니다.
    
    Args:
        db: MongoDB 데이터베이스 인스턴스
        analysis_id: 분석 ID
        status: 새로운 상태
        error_message: 오류 메시지 (선택사항)
        
    Returns:
        bool: 업데이트 성공 여부
    """
    try:
        collection: Collection = db['analysis_results']
        
        update_data = {
            "status": status,
            "updated_at": datetime.now()
        }
        
        if status == "completed":
            update_data["completed_at"] = datetime.now()
        
        if error_message:
            update_data["error_message"] = error_message
        
        result = collection.update_one(
            {"analysis_id": analysis_id},
            {"$set": update_data}
        )
        
        return result.modified_count > 0
        
    except Exception as e:
        logger.error(f"분석 상태 업데이트 오류: {str(e)}")
        return False

def delete_analysis_result(db: Database, analysis_id: str) -> bool:
    """
    분석 결과를 삭제합니다.
    
    Args:
        db: MongoDB 데이터베이스 인스턴스
        analysis_id: 삭제할 분석 ID
        
    Returns:
        bool: 삭제 성공 여부
    """
    try:
        collection: Collection = db['analysis_results']
        
        result = collection.delete_one({"analysis_id": analysis_id})
        
        if result.deleted_count > 0:
            logger.info(f"분석 결과 삭제: {analysis_id}")
            return True
        
        return False
        
    except Exception as e:
        logger.error(f"분석 결과 삭제 오류: {str(e)}")
        return False

def get_analysis_statistics(db: Database, user_id: Optional[str] = None) -> Dict[str, Any]:
    """
    분석 통계를 조회합니다.
    
    Args:
        db: MongoDB 데이터베이스 인스턴스
        user_id: 특정 사용자의 통계 (선택사항)
        
    Returns:
        Dict[str, Any]: 통계 정보
    """
    try:
        collection: Collection = db['analysis_results']
        
        # 기본 필터
        match_filter = {}
        if user_id:
            match_filter["user_id"] = user_id
        
        # 집계 파이프라인
        pipeline = [
            {"$match": match_filter},
            {
                "$group": {
                    "_id": None,
                    "total_analyses": {"$sum": 1},
                    "completed_analyses": {
                        "$sum": {"$cond": [{"$eq": ["$status", "completed"]}, 1, 0]}
                    },
                    "error_analyses": {
                        "$sum": {"$cond": [{"$eq": ["$status", "error"]}, 1, 0]}
                    },
                    "processing_analyses": {
                        "$sum": {"$cond": [{"$eq": ["$status", "processing"]}, 1, 0]}
                    },
                    "avg_emotion_score": {
                        "$avg": "$emotion_analysis.interview_score"
                    },
                    "avg_attention_score": {
                        "$avg": "$eye_tracking_analysis.attention_score"
                    }
                }
            }
        ]
        
        result = list(collection.aggregate(pipeline))
        
        if result:
            stats = result[0]
            stats.pop("_id", None)  # _id 필드 제거
            return stats
        
        return {
            "total_analyses": 0,
            "completed_analyses": 0,
            "error_analyses": 0,
            "processing_analyses": 0,
            "avg_emotion_score": 0,
            "avg_attention_score": 0
        }
        
    except Exception as e:
        logger.error(f"분석 통계 조회 오류: {str(e)}")
        raise Exception(f"분석 통계 조회 실패: {str(e)}")

def cleanup_old_analyses(db: Database, days_old: int = 30) -> int:
    """
    오래된 분석 결과를 정리합니다.
    
    Args:
        db: MongoDB 데이터베이스 인스턴스
        days_old: 삭제할 분석 결과의 기준 일수
        
    Returns:
        int: 삭제된 문서 수
    """
    try:
        from datetime import timedelta
        
        collection: Collection = db['analysis_results']
        
        cutoff_date = datetime.now() - timedelta(days=days_old)
        
        result = collection.delete_many({
            "created_at": {"$lt": cutoff_date},
            "status": {"$in": ["completed", "error"]}  # 처리 중인 것은 제외
        })
        
        logger.info(f"오래된 분석 결과 {result.deleted_count}개 삭제")
        return result.deleted_count
        
    except Exception as e:
        logger.error(f"오래된 분석 결과 정리 오류: {str(e)}")
        return 0 