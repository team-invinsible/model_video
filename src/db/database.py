import os
from contextlib import contextmanager
from typing import Generator
import pymongo
from pymongo import MongoClient
from pymongo.database import Database
from pymongo.collection import Collection
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MongoDBHandler:
    """MongoDB 연결 및 관리 클래스"""
    
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
        
    def connect(self):
        """MongoDB에 연결합니다."""
        try:
            self.client = MongoClient(self.connection_string)
            self.database = self.client[self.database_name]
            
            # 연결 테스트
            self.client.admin.command('ping')
            logger.info(f"MongoDB 연결 성공: {self.database_name}")
            
        except Exception as e:
            logger.error(f"MongoDB 연결 실패: {str(e)}")
            raise Exception(f"MongoDB 연결 실패: {str(e)}")
    
    def disconnect(self):
        """MongoDB 연결을 종료합니다."""
        if self.client:
            self.client.close()
            logger.info("MongoDB 연결 종료")
    
    def get_database(self) -> Database:
        """데이터베이스 인스턴스를 반환합니다."""
        if self.database is None:
            self.connect()
        return self.database
    
    def get_collection(self, collection_name: str) -> Collection:
        """컬렉션 인스턴스를 반환합니다."""
        database = self.get_database()
        return database[collection_name]

# 전역 MongoDB 핸들러 인스턴스
_mongodb_handler = None

def get_mongodb_handler() -> MongoDBHandler:
    """전역 MongoDB 핸들러를 반환합니다."""
    global _mongodb_handler
    if _mongodb_handler is None:
        _mongodb_handler = MongoDBHandler()
    return _mongodb_handler

@contextmanager
def get_db_session() -> Generator[Database, None, None]:
    """
    MongoDB 데이터베이스 세션을 제공하는 컨텍스트 매니저
    
    Usage:
        with get_db_session() as db:
            collection = db['analysis_results']
            collection.insert_one(data)
    """
    handler = get_mongodb_handler()
    try:
        database = handler.get_database()
        if database is None:
            raise Exception("데이터베이스 연결을 가져올 수 없습니다")
        yield database
    except Exception as e:
        logger.error(f"데이터베이스 세션 오류: {str(e)}")
        raise
    finally:
        # MongoDB는 연결 풀을 사용하므로 명시적으로 닫을 필요 없음
        pass

def init_database():
    """데이터베이스 초기화 및 인덱스 생성"""
    try:
        with get_db_session() as db:
            # analysis_results 컬렉션 인덱스 생성
            analysis_collection = db['analysis_results']
            
            # 인덱스 생성
            analysis_collection.create_index("analysis_id", unique=True)
            analysis_collection.create_index("user_id")
            analysis_collection.create_index("session_id")
            analysis_collection.create_index("created_at")
            analysis_collection.create_index("status")
            
            logger.info("데이터베이스 초기화 완료")
            
    except Exception as e:
        logger.error(f"데이터베이스 초기화 실패: {str(e)}")
        raise

def check_database_connection() -> bool:
    """데이터베이스 연결 상태를 확인합니다."""
    try:
        handler = get_mongodb_handler()
        database = handler.get_database()
        # ping 명령으로 연결 확인
        database.client.admin.command('ping')
        return True
    except Exception as e:
        logger.error(f"데이터베이스 연결 확인 실패: {str(e)}")
        return False

# 애플리케이션 시작 시 데이터베이스 초기화
def setup_database():
    """애플리케이션 시작 시 데이터베이스 설정"""
    try:
        if check_database_connection():
            init_database()
            logger.info("데이터베이스 설정 완료")
        else:
            logger.warning("데이터베이스 연결 실패 - 나중에 재시도됩니다")
    except Exception as e:
        logger.error(f"데이터베이스 설정 오류: {str(e)}")

# 환경변수 설정 예시
"""
환경변수 설정:
export MONGODB_CONNECTION_STRING="mongodb://username:password@localhost:27017/"
또는
export MONGODB_CONNECTION_STRING="mongodb+srv://username:password@cluster.mongodb.net/"
""" 