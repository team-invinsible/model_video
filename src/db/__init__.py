# ----------------------------------------------------------------------------------------------------
# 작성목적 : MongoDB 패키지 초기화 (연결 실패 시 스킵 기능 포함)
# 작성일 : 2025-01-27

# 변경사항 내역 (날짜 | 변경목적 | 변경내용 | 작성자 순으로 기입)
# 2025-01-27 | 최초 구현 | 안전한 MongoDB 연결 및 스킵 기능 제공 | 구동빈
# ----------------------------------------------------------------------------------------------------

from .database import (
    get_db_session,
    get_mongodb_handler,
    safe_mongodb_operation,
    setup_database,
    check_database_connection
)

from .crud import (
    safe_save_analysis_result,
    safe_get_analysis_results,
    safe_update_analysis_status,
    get_analysis_results_by_user,
    get_recent_analysis_results,
    get_analysis_statistics,
    cleanup_old_analyses
)

from .models import (
    AnalysisResult,
    EmotionAnalysisResult,
    EyeTrackingResult,
    LLMComment,
    AnalysisStatus,
    ProcessingStage
)

__all__ = [
    # Database connection
    'get_db_session',
    'get_mongodb_handler', 
    'safe_mongodb_operation',
    'setup_database',
    'check_database_connection',
    
    # CRUD operations
    'safe_save_analysis_result',
    'safe_get_analysis_results',
    'safe_update_analysis_status',
    'get_analysis_results_by_user',
    'get_recent_analysis_results',
    'get_analysis_statistics',
    'cleanup_old_analyses',
    
    # Models
    'AnalysisResult',
    'EmotionAnalysisResult',
    'EyeTrackingResult',
    'LLMComment',
    'AnalysisStatus',
    'ProcessingStage'
] 