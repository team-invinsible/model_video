# ----------------------------------------------------------------------------------------------------
# 작성목적 : MongoDB 연결 및 관리 (연결 실패 시 스킵 기능 포함)
# 작성일 : 2025-01-27

# 변경사항 내역 (날짜 | 변경목적 | 변경내용 | 작성자 순으로 기입)
# 2025-01-27 | 최초 구현 | FastAPI 베스트 프랙티스에 따른 구조로 재구성 | 구동빈
# 2025-01-27 | 안정성 개선 | MongoDB 연결 실패 시 스킵 기능 추가 | 구동빈
# ----------------------------------------------------------------------------------------------------

import os
from contextlib import contextmanager
from typing import Generator, Optional
import pymongo
from pymongo import MongoClient
from pymongo.database import Database
from pymongo.collection import Collection
import logging
from datetime import datetime, timedelta

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MongoDBHandler:
    """MongoDB 연결 및 관리 클래스 (연결 실패 시 스킵 기능 포함)"""
    
    def __init__(self, connection_string: str = None, database_name: str = "video_analysis"):
        """
        MongoDB 핸들러 초기화
        
        Args:
            connection_string: MongoDB 연결 문자열
            database_name: 사용할 데이터베이스 이름
        """
        self.connection_string = connection_string or os.getenv(
            'MONGODB_CONNECTION_STRING', 
            'mongodb://localhost:27017/'
        )
        self.database_name = database_name
        self.client = None
        self.database = None
        self.is_connected = False
        self.last_connection_attempt = None
        self.connection_retry_interval = 300  # 5분마다 재시도
        
    def connect(self) -> bool:
        """
        MongoDB에 연결합니다.
        
        Returns:
            bool: 연결 성공 여부
        """
        try:
            self.client = MongoClient(
                self.connection_string,
                serverSelectionTimeoutMS=5000,  # 5초 타임아웃
                connectTimeoutMS=5000,
                socketTimeoutMS=5000
            )
            self.database = self.client[self.database_name]
            
            # 연결 테스트
            self.client.admin.command('ping')
            self.is_connected = True
            self.last_connection_attempt = datetime.now()
            logger.info(f"✅ MongoDB 연결 성공: {self.database_name}")
            return True
            
        except Exception as e:
            self.is_connected = False
            self.last_connection_attempt = datetime.now()
            logger.warning(f"⚠️ MongoDB 연결 실패 (스킵됨): {str(e)}")
            return False
    
    def disconnect(self):
        """MongoDB 연결을 종료합니다."""
        if self.client:
            try:
                self.client.close()
                logger.info("MongoDB 연결 종료")
            except Exception as e:
                logger.warning(f"MongoDB 연결 종료 중 오류: {e}")
        
        self.client = None
        self.database = None
        self.is_connected = False
    
    def should_retry_connection(self) -> bool:
        """연결 재시도 여부를 판단합니다."""
        if self.is_connected:
            return False
            
        if self.last_connection_attempt is None:
            return True
            
        time_since_last_attempt = datetime.now() - self.last_connection_attempt
        return time_since_last_attempt.total_seconds() > self.connection_retry_interval
    
    def get_database(self) -> Optional[Database]:
        """
        데이터베이스 인스턴스를 반환합니다.
        
        Returns:
            Optional[Database]: 연결 성공 시 Database 인스턴스, 실패 시 None
        """
        if not self.is_connected:
            if self.should_retry_connection():
                self.connect()
        
        return self.database if self.is_connected else None
    
    def get_collection(self, collection_name: str) -> Optional[Collection]:
        """
        컬렉션 인스턴스를 반환합니다.
        
        Returns:
            Optional[Collection]: 연결 성공 시 Collection 인스턴스, 실패 시 None
        """
        database = self.get_database()
        return database[collection_name] if database is not None else None

# 전역 MongoDB 핸들러 인스턴스
_mongodb_handler = None

def get_mongodb_handler() -> MongoDBHandler:
    """전역 MongoDB 핸들러를 반환합니다."""
    global _mongodb_handler
    if _mongodb_handler is None:
        _mongodb_handler = MongoDBHandler()
    return _mongodb_handler

@contextmanager
def get_db_session() -> Generator[Optional[Database], None, None]:
    """
    MongoDB 데이터베이스 세션을 제공하는 컨텍스트 매니저 (연결 실패 시 None 반환)
    
    Usage:
        with get_db_session() as db:
            if db is not None:
                collection = db['analysis_results']
                collection.insert_one(data)
            else:
                print("MongoDB 연결 실패 - 저장 스킵")
    """
    handler = get_mongodb_handler()
    database = None
    
    try:
        database = handler.get_database()
        if database is None:
            logger.warning("⚠️ MongoDB 세션 가져오기 실패 - 연결 없음")
        yield database
        
    except Exception as e:
        logger.warning(f"⚠️ MongoDB 세션 오류 (스킵됨): {str(e)}")
        yield None
    finally:
        # MongoDB는 연결 풀을 사용하므로 명시적으로 닫을 필요 없음
        pass

def safe_mongodb_operation(operation_name: str, operation_func, *args, **kwargs):
    """
    MongoDB 작업을 안전하게 실행합니다. 실패 시 스킵하고 계속 진행합니다.
    
    Args:
        operation_name: 작업 이름 (로깅용)
        operation_func: 실행할 함수
        *args, **kwargs: 함수에 전달할 인자들
        
    Returns:
        작업 결과 또는 None (실패 시)
    """
    try:
        with get_db_session() as db:
            if db is not None:
                return operation_func(db, *args, **kwargs)
            else:
                logger.warning(f"⚠️ {operation_name} 스킵 - MongoDB 연결 없음")
                return None
    except Exception as e:
        logger.warning(f"⚠️ {operation_name} 실패 (스킵됨): {str(e)}")
        return None

def init_database():
    """데이터베이스 초기화 및 인덱스 생성 (실패 시 스킵)"""
    def _init_db(db):
        # analysis_results 컬렉션 인덱스 생성
        analysis_collection = db['analysis_results']
        
        # 인덱스 생성
        analysis_collection.create_index("analysis_id", unique=True)
        analysis_collection.create_index("user_id")
        analysis_collection.create_index("session_id")
        analysis_collection.create_index("created_at")
        analysis_collection.create_index("status")
        
        logger.info("✅ 데이터베이스 초기화 완료")
        return True
    
    result = safe_mongodb_operation("데이터베이스 초기화", _init_db)
    if result is None:
        logger.warning("⚠️ 데이터베이스 초기화 스킵됨")

def check_database_connection() -> bool:
    """데이터베이스 연결 상태를 확인합니다."""
    try:
        handler = get_mongodb_handler()
        database = handler.get_database()
        if database is None:
            return False
            
        # ping 명령으로 연결 확인
        database.client.admin.command('ping')
        return True
    except Exception as e:
        logger.warning(f"⚠️ 데이터베이스 연결 확인 실패: {str(e)}")
        return False

def setup_database():
    """애플리케이션 시작 시 데이터베이스 설정 (실패 시 스킵)"""
    try:
        if check_database_connection():
            init_database()
            logger.info("✅ 데이터베이스 설정 완료")
        else:
            logger.warning("⚠️ 데이터베이스 연결 실패 - 애플리케이션은 MongoDB 없이 계속 실행됩니다")
    except Exception as e:
        logger.warning(f"⚠️ 데이터베이스 설정 오류 (스킵됨): {str(e)}")

# 환경변수 설정 예시
"""
환경변수 설정:
export MONGODB_CONNECTION_STRING="mongodb://username:password@localhost:27017/"
또는
export MONGODB_CONNECTION_STRING="mongodb+srv://username:password@cluster.mongodb.net/"

MongoDB가 없어도 애플리케이션이 정상 동작하도록 설계되었습니다.
연결 실패 시 관련 작업은 자동으로 스킵됩니다.
""" 