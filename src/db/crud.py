# ----------------------------------------------------------------------------------------------------
# 작성목적 : MongoDB CRUD 연산 정의 (연결 실패 시 스킵 기능 포함)
# 작성일 : 2025-01-27

# 변경사항 내역 (날짜 | 변경목적 | 변경내용 | 작성자 순으로 기입)
# 2025-01-27 | 최초 구현 | FastAPI 베스트 프랙티스에 따른 구조로 재구성 | 구동빈
# 2025-01-27 | 안정성 개선 | MongoDB 연결 실패 시 스킵 기능 추가 | 구동빈
# ----------------------------------------------------------------------------------------------------

from typing import Optional, Dict, Any, List
from datetime import datetime
from pymongo.database import Database
from pymongo.collection import Collection
from pymongo.errors import DuplicateKeyError
import logging
import numpy as np

from .models import create_analysis_result_document, parse_analysis_result, AnalysisResult
from .database import safe_mongodb_operation, get_db_session

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

def _save_analysis_result_impl(db: Database, analysis_data: Dict[str, Any]) -> str:
    """
    분석 결과를 MongoDB에 저장하는 내부 구현
    """
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
        logger.info(f"✅ 분석 결과 업데이트: {document['analysis_id']}")
        return str(existing["_id"])
    else:
        # 새 문서 삽입
        result = collection.insert_one(document)
        logger.info(f"✅ 분석 결과 저장: {document['analysis_id']}")
        return str(result.inserted_id)

def save_analysis_result(analysis_data: Dict[str, Any]) -> Optional[str]:
    """
    분석 결과를 MongoDB에 저장합니다. (연결 실패 시 스킵)
    
    Args:
        analysis_data: 저장할 분석 데이터
        
    Returns:
        Optional[str]: 저장된 문서의 ObjectId 또는 None (실패 시)
    """
    return safe_mongodb_operation(
        "분석 결과 저장",
        _save_analysis_result_impl,
        analysis_data
    )

def _get_analysis_results_impl(db: Database, analysis_id: str) -> Optional[Dict[str, Any]]:
    """
    분석 ID로 결과를 조회하는 내부 구현
    """
    collection: Collection = db['analysis_results']
    document = collection.find_one({"analysis_id": analysis_id})
    
    if document:
        # ObjectId를 문자열로 변환
        document["_id"] = str(document["_id"])
        return document
    
    return None

def get_analysis_results(analysis_id: str) -> Optional[Dict[str, Any]]:
    """
    분석 ID로 결과를 조회합니다. (연결 실패 시 None 반환)
    
    Args:
        analysis_id: 조회할 분석 ID
        
    Returns:
        Optional[Dict[str, Any]]: 분석 결과 또는 None
    """
    return safe_mongodb_operation(
        "분석 결과 조회",
        _get_analysis_results_impl,
        analysis_id
    )

def _get_analysis_results_by_user_impl(db: Database, user_id: str, 
                                     limit: int = 10, skip: int = 0) -> List[Dict[str, Any]]:
    """
    사용자 ID로 분석 결과 목록을 조회하는 내부 구현
    """
    collection: Collection = db['analysis_results']
    
    cursor = collection.find(
        {"user_id": user_id}
    ).sort("created_at", -1).skip(skip).limit(limit)
    
    results = []
    for document in cursor:
        document["_id"] = str(document["_id"])
        results.append(document)
    
    return results

def get_analysis_results_by_user(user_id: str, limit: int = 10, skip: int = 0) -> List[Dict[str, Any]]:
    """
    사용자 ID로 분석 결과 목록을 조회합니다. (연결 실패 시 빈 리스트 반환)
    
    Args:
        user_id: 사용자 ID
        limit: 조회할 최대 개수
        skip: 건너뛸 개수
        
    Returns:
        List[Dict[str, Any]]: 분석 결과 목록
    """
    result = safe_mongodb_operation(
        "사용자별 분석 결과 조회",
        _get_analysis_results_by_user_impl,
        user_id, limit, skip
    )
    return result if result is not None else []

def _get_analysis_results_by_session_impl(db: Database, session_id: str) -> List[Dict[str, Any]]:
    """
    세션 ID로 분석 결과 목록을 조회하는 내부 구현
    """
    collection: Collection = db['analysis_results']
    
    cursor = collection.find(
        {"session_id": session_id}
    ).sort("created_at", -1)
    
    results = []
    for document in cursor:
        document["_id"] = str(document["_id"])
        results.append(document)
    
    return results

def get_analysis_results_by_session(session_id: str) -> List[Dict[str, Any]]:
    """
    세션 ID로 분석 결과 목록을 조회합니다. (연결 실패 시 빈 리스트 반환)
    
    Args:
        session_id: 세션 ID
        
    Returns:
        List[Dict[str, Any]]: 분석 결과 목록
    """
    result = safe_mongodb_operation(
        "세션별 분석 결과 조회",
        _get_analysis_results_by_session_impl,
        session_id
    )
    return result if result is not None else []

def _update_analysis_status_impl(db: Database, analysis_id: str, 
                                status: str, error_message: Optional[str] = None) -> bool:
    """
    분석 상태를 업데이트하는 내부 구현
    """
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

def update_analysis_status(analysis_id: str, status: str, error_message: Optional[str] = None) -> bool:
    """
    분석 상태를 업데이트합니다. (연결 실패 시 False 반환)
    
    Args:
        analysis_id: 분석 ID
        status: 새로운 상태
        error_message: 오류 메시지 (선택사항)
        
    Returns:
        bool: 업데이트 성공 여부
    """
    result = safe_mongodb_operation(
        "분석 상태 업데이트",
        _update_analysis_status_impl,
        analysis_id, status, error_message
    )
    return result if result is not None else False

def _delete_analysis_result_impl(db: Database, analysis_id: str) -> bool:
    """
    분석 결과를 삭제하는 내부 구현
    """
    collection: Collection = db['analysis_results']
    
    result = collection.delete_one({"analysis_id": analysis_id})
    
    if result.deleted_count > 0:
        logger.info(f"✅ 분석 결과 삭제: {analysis_id}")
        return True
    
    return False

def delete_analysis_result(analysis_id: str) -> bool:
    """
    분석 결과를 삭제합니다. (연결 실패 시 False 반환)
    
    Args:
        analysis_id: 삭제할 분석 ID
        
    Returns:
        bool: 삭제 성공 여부
    """
    result = safe_mongodb_operation(
        "분석 결과 삭제",
        _delete_analysis_result_impl,
        analysis_id
    )
    return result if result is not None else False

def _get_recent_analysis_results_impl(db: Database, limit: int = 10) -> List[Dict[str, Any]]:
    """
    최근 분석 결과를 조회하는 내부 구현
    """
    collection: Collection = db['analysis_results']
    
    cursor = collection.find().sort("created_at", -1).limit(limit)
    
    results = []
    for document in cursor:
        document["_id"] = str(document["_id"])
        results.append(document)
    
    return results

def get_recent_analysis_results(limit: int = 10) -> List[Dict[str, Any]]:
    """
    최근 분석 결과를 조회합니다. (연결 실패 시 빈 리스트 반환)
    
    Args:
        limit: 조회할 최대 개수
        
    Returns:
        List[Dict[str, Any]]: 분석 결과 목록
    """
    result = safe_mongodb_operation(
        "최근 분석 결과 조회",
        _get_recent_analysis_results_impl,
        limit
    )
    return result if result is not None else []

def _get_analysis_statistics_impl(db: Database) -> Dict[str, Any]:
    """
    분석 통계를 조회하는 내부 구현
    """
    collection: Collection = db['analysis_results']
    
    # 전체 분석 수
    total_count = collection.count_documents({})
    
    # 상태별 분석 수
    status_counts = {}
    status_pipeline = [
        {"$group": {"_id": "$status", "count": {"$sum": 1}}}
    ]
    
    for result in collection.aggregate(status_pipeline):
        status_counts[result["_id"]] = result["count"]
    
    # 오늘 분석 수
    today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    today_count = collection.count_documents({
        "created_at": {"$gte": today_start}
    })
    
    return {
        "total_analyses": total_count,
        "status_breakdown": status_counts,
        "today_analyses": today_count
    }

def get_analysis_statistics() -> Dict[str, Any]:
    """
    분석 통계를 조회합니다. (연결 실패 시 기본값 반환)
    
    Returns:
        Dict[str, Any]: 분석 통계
    """
    result = safe_mongodb_operation(
        "분석 통계 조회",
        _get_analysis_statistics_impl
    )
    
    if result is not None:
        return result
    else:
        # MongoDB 연결 실패 시 기본값 반환
        return {
            "total_analyses": 0,
            "status_breakdown": {},
            "today_analyses": 0,
            "note": "MongoDB 연결 없음 - 통계 정보 사용 불가"
        }

def _cleanup_old_analyses_impl(db: Database, days_old: int = 30) -> int:
    """
    오래된 분석 결과를 정리하는 내부 구현
    """
    from datetime import timedelta
    
    collection: Collection = db['analysis_results']
    
    cutoff_date = datetime.now() - timedelta(days=days_old)
    
    result = collection.delete_many({
        "created_at": {"$lt": cutoff_date},
        "status": {"$in": ["completed", "error"]}  # 처리 중인 것은 제외
    })
    
    logger.info(f"✅ 오래된 분석 결과 {result.deleted_count}개 삭제")
    return result.deleted_count

def cleanup_old_analyses(days_old: int = 30) -> int:
    """
    오래된 분석 결과를 정리합니다. (연결 실패 시 0 반환)
    
    Args:
        days_old: 삭제할 분석 결과의 기준 일수
        
    Returns:
        int: 삭제된 문서 수
    """
    result = safe_mongodb_operation(
        "오래된 분석 결과 정리",
        _cleanup_old_analyses_impl,
        days_old
    )
    return result if result is not None else 0

# 안전한 MongoDB 작업을 위한 헬퍼 함수들
def safe_save_analysis_result(analysis_data: Dict[str, Any]) -> bool:
    """
    분석 결과를 안전하게 저장합니다. 실패해도 예외를 발생시키지 않습니다.
    
    Args:
        analysis_data: 저장할 분석 데이터
        
    Returns:
        bool: 저장 성공 여부
    """
    try:
        result = save_analysis_result(analysis_data)
        return result is not None
    except Exception as e:
        logger.warning(f"⚠️ 분석 결과 저장 실패 (무시됨): {e}")
        return False

def safe_get_analysis_results(analysis_id: str) -> Optional[Dict[str, Any]]:
    """
    분석 결과를 안전하게 조회합니다. 실패해도 예외를 발생시키지 않습니다.
    
    Args:
        analysis_id: 조회할 분석 ID
        
    Returns:
        Optional[Dict[str, Any]]: 분석 결과 또는 None
    """
    try:
        return get_analysis_results(analysis_id)
    except Exception as e:
        logger.warning(f"⚠️ 분석 결과 조회 실패 (무시됨): {e}")
        return None

def safe_update_analysis_status(analysis_id: str, status: str, error_message: Optional[str] = None) -> bool:
    """
    분석 상태를 안전하게 업데이트합니다. 실패해도 예외를 발생시키지 않습니다.
    
    Args:
        analysis_id: 분석 ID
        status: 새로운 상태
        error_message: 오류 메시지 (선택사항)
        
    Returns:
        bool: 업데이트 성공 여부
    """
    try:
        return update_analysis_status(analysis_id, status, error_message)
    except Exception as e:
        logger.warning(f"⚠️ 분석 상태 업데이트 실패 (무시됨): {e}")
        return False 