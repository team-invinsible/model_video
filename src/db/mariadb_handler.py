# ----
# 작성목적 : MariaDB 연결 및 테이블 관리
# 작성일 : 2025-06-12

# 변경사항 내역 (날짜 | 변경목적 | 변경내용 | 작성자 순으로 기입)
# 2025-06-14 | 최초 구현 | FastAPI 베스트 프랙티스에 따른 구조로 재구성 | 이재인
# 2025-06-14 | 테이블 리팩터링 | audio.answer_score, answer_category_result 테이블 구조로 변경 | 이재인
# 2025-06-24 | 면접태도 전용 | INTERVIEW_ATTITUDE 카테고리 전용 저장 시스템 구현 | 이재인
# 2025-06-24 | ID 형식 변경 | INTV_ANS_ID를 userId0questionNum, ANS_CAT_RESULT_ID를 userId0questionNum0 형식으로 변경 | 이재인
# ----

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any, List
import aiomysql
from datetime import datetime
import json
import os
from dotenv import load_dotenv
from .models import LLMComment

# 환경변수 로드
load_dotenv()

logger = logging.getLogger(__name__)

class MariaDBHandler:
    """MariaDB 연결 및 audio 데이터베이스 answer_score, answer_category_result 테이블 관리"""
    
    def __init__(self):
        self.pool: Optional[aiomysql.Pool] = None
        self.host = os.getenv("MARIADB_HOST", "localhost")
        self.port = int(os.getenv("MARIADB_PORT", "3306"))
        self.user = os.getenv("MARIADB_USER", "root")
        self.password = os.getenv("MARIADB_PASSWORD", "")
        self.database = os.getenv("MARIADB_DATABASE", "audio")
        
    async def create_pool(self):
        """MariaDB 연결 풀 생성"""
        try:
            self.pool = await aiomysql.create_pool(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                db=self.database,
                charset='utf8mb4',
                autocommit=True,
                minsize=1,
                maxsize=10
            )
            logger.info("MariaDB 연결 풀이 생성되었습니다.")
            
            # 테이블 생성
            await self._create_tables()
            
        except Exception as e:
            logger.error(f"MariaDB 연결 풀 생성 실패: {e}")
            raise
    
    async def close_pool(self):
        """MariaDB 연결 풀 종료"""
        if self.pool:
            self.pool.close()
            await self.pool.wait_closed()
            logger.info("MariaDB 연결 풀이 종료되었습니다.")
    
    async def _create_tables(self):
        """audio 데이터베이스에 필요한 테이블들을 생성합니다."""
        # 기존 불필요한 테이블들 삭제

        
        # 1. answer_score 테이블 생성
        create_answer_score_table = """
        CREATE TABLE IF NOT EXISTS answer_score (
            ANS_SCORE_ID BIGINT PRIMARY KEY NOT NULL COMMENT '답변 평가 ID (userId0questionNum 형식)',
            INTV_ANS_ID BIGINT NOT NULL COMMENT '면접 답변 ID (userId)',
            ANS_SUMMARY TEXT NULL COMMENT '답변 요약',
            EVAL_SUMMARY TEXT NULL COMMENT '전체 평가 요약',
            INCOMPLETE_ANSWER BOOLEAN NULL DEFAULT FALSE COMMENT '미완료 여부',
            INSUFFICIENT_CONTENT BOOLEAN NULL DEFAULT FALSE COMMENT '내용 부족 여부',
            SUSPECTED_COPYING BOOLEAN NULL DEFAULT FALSE COMMENT '커닝 의심 여부 (시선 분산)',
            SUSPECTED_IMPERSONATION BOOLEAN NULL DEFAULT FALSE COMMENT '대리 시험 의심 여부 (다중 얼굴)',
            RGS_DTM TIMESTAMP NULL DEFAULT CURRENT_TIMESTAMP COMMENT '등록 일시',
            UPD_DTM TIMESTAMP NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '수정 일시',
            
            INDEX idx_intv_ans_id (INTV_ANS_ID),
            INDEX idx_suspected_copying (SUSPECTED_COPYING),
            INDEX idx_suspected_impersonation (SUSPECTED_IMPERSONATION),
            INDEX idx_rgs_dtm (RGS_DTM)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci 
        COMMENT='답변 평가 테이블 (면접태도 부정행위 감지)';
        """
        
        # 2. answer_category_result 테이블 생성
        create_answer_category_result_table = """
        CREATE TABLE IF NOT EXISTS answer_category_result (
            ANS_CAT_RESULT_ID BIGINT PRIMARY KEY NOT NULL COMMENT '답변 항목별 평가 ID (userId0questionNum0 형식)',
            EVAL_CAT_CD VARCHAR(20) NOT NULL COMMENT '평가 항목 코드 (INTERVIEW_ATTITUDE)',
            ANS_SCORE_ID BIGINT NOT NULL COMMENT '답변 평가 ID',
            ANS_CAT_SCORE DOUBLE NULL COMMENT '항목별 점수 (표정+시선 총합)',
            STRENGTH_KEYWORD TEXT NULL COMMENT '강점 키워드 (GPT 분석)',
            WEAKNESS_KEYWORD TEXT NULL COMMENT '약점 키워드 (GPT 분석)',
            RGS_DTM TIMESTAMP NULL DEFAULT CURRENT_TIMESTAMP COMMENT '등록 일시',
            
            FOREIGN KEY (ANS_SCORE_ID) REFERENCES answer_score(ANS_SCORE_ID) ON DELETE CASCADE,
            INDEX idx_eval_cat_cd (EVAL_CAT_CD),
            INDEX idx_ans_score_id (ANS_SCORE_ID),
            INDEX idx_ans_cat_score (ANS_CAT_SCORE),
            INDEX idx_rgs_dtm (RGS_DTM),
            UNIQUE KEY unique_score_category (ANS_SCORE_ID, EVAL_CAT_CD)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci 
        COMMENT='답변 항목별 평가 결과 테이블 (면접태도 전용)';
        """
        
        async with self.pool.acquire() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute(create_answer_score_table)
                await cursor.execute(create_answer_category_result_table)
                logger.info("audio 데이터베이스 테이블이 생성되었습니다.")
    
    # async def _drop_unnecessary_tables(self):
    #     """기존 불필요한 테이블들을 삭제합니다."""
    #     tables_to_drop = [
    #         'atti_score',  # 기존 테이블 삭제
    #     ]
        
    #     try:
    #         async with self.pool.acquire() as conn:
    #             async with conn.cursor() as cursor:
    #                 for table in tables_to_drop:
    #                     try:
    #                         await cursor.execute(f"DROP TABLE IF EXISTS {table}")
    #                         logger.info(f"기존 테이블 {table} 삭제됨")
    #     except Exception as e:
    #         logger.warning(f"테이블 삭제 중 오류 (무시 가능): {e}")
    
    @asynccontextmanager
    async def get_connection(self):
        """MariaDB 연결을 가져오는 컨텍스트 매니저"""
        if not self.pool:
            await self.create_pool()
        
        async with self.pool.acquire() as conn:
            try:
                yield conn
            except Exception as e:
                await conn.rollback()
                logger.error(f"MariaDB 트랜잭션 롤백: {e}")
                raise
    
    async def get_analysis_summary(self, analysis_id: str) -> Optional[Dict[str, Any]]:
        """분석 요약 정보 조회"""
        try:
            async with self.get_connection() as conn:
                async with conn.cursor(aiomysql.DictCursor) as cursor:
                    query = """
                    SELECT * FROM analysis_summary 
                    WHERE analysis_id = %s
                    """
                    await cursor.execute(query, (analysis_id,))
                    result = await cursor.fetchone()
                    return dict(result) if result else None
                    
        except Exception as e:
            logger.error(f"분석 요약 조회 실패: {e}")
            return None
    
    async def get_recent_analyses(self, limit: int = 10) -> List[Dict[str, Any]]:
        """최근 분석 결과 목록 조회"""
        try:
            async with self.get_connection() as conn:
                async with conn.cursor(aiomysql.DictCursor) as cursor:
                    query = """
                    SELECT * FROM analysis_summary 
                    ORDER BY created_at DESC 
                    LIMIT %s
                    """
                    await cursor.execute(query, (limit,))
                    results = await cursor.fetchall()
                    return [dict(result) for result in results]
                    
        except Exception as e:
            logger.error(f"최근 분석 목록 조회 실패: {e}")
            return []
    
    async def update_analysis_status(self, analysis_id: str, status: str, 
                                   current_stage: str = None, progress: float = None) -> bool:
        """분석 상태 업데이트"""
        try:
            async with self.get_connection() as conn:
                async with conn.cursor() as cursor:
                    # 동적 쿼리 생성
                    update_fields = ["analysis_status = %s", "updated_at = CURRENT_TIMESTAMP"]
                    params = [status]
                    
                    if current_stage:
                        update_fields.append("current_stage = %s")
                        params.append(current_stage)
                    
                    if progress is not None:
                        update_fields.append("progress_percentage = %s")
                        params.append(progress)
                    
                    # 시작 시간 설정
                    if status == 'processing' and current_stage == 'download':
                        update_fields.append("started_at = CURRENT_TIMESTAMP")
                    
                    params.append(analysis_id)
                    
                    query = f"""
                    UPDATE analysis_summary 
                    SET {', '.join(update_fields)}
                    WHERE analysis_id = %s
                    """
                    
                    await cursor.execute(query, params)
                    return True
                    
        except Exception as e:
            logger.error(f"분석 상태 업데이트 실패: {e}")
            return False
    
    async def create_analysis_record(self, analysis_id: str, user_id: str = None, 
                                   session_id: str = None, question_id: str = 'Q1',
                                   video_filename: str = None, video_path: str = None,
                                   file_size: int = None) -> bool:
        """새로운 분석 레코드 생성"""
        try:
            async with self.get_connection() as conn:
                async with conn.cursor() as cursor:
                    insert_query = """
                    INSERT INTO analysis_summary (
                        analysis_id, user_id, session_id, question_id,
                        video_filename, video_path, file_size,
                        analysis_status, progress_percentage
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, 'pending', 0)
                    """
                    
                    await cursor.execute(insert_query, (
                        analysis_id, user_id, session_id, question_id,
                        video_filename, video_path, file_size
                    ))
                    
                    logger.info(f"분석 레코드 생성 완료: {analysis_id}")
                    return True
                    
        except Exception as e:
            logger.error(f"분석 레코드 생성 실패: {e}")
            return False

    async def save_interview_attitude(self, user_id: str, question_num: str, 
                                     emotion_score: float, eye_score: float,
                                     suspected_copying: bool = False, 
                                     suspected_impersonation: bool = False,
                                     gpt_analysis: Dict[str, str] = None) -> bool:
        """audio.answer_score 및 answer_category_result 테이블에 면접태도 평가 저장"""
        try:
            # ANS_SCORE_ID 생성: {userId}0{question_num}
            ans_score_id = int(f"{user_id}0{question_num}")
            # INTV_ANS_ID 생성: {userId}0{questionNum} 형식
            intv_ans_id = int(f"{user_id}0{question_num}")
            # ANS_CAT_RESULT_ID 생성: {userId}0{questionNum}0 형식  
            ans_cat_result_id = int(f"{user_id}0{question_num}0")
            total_score = emotion_score + eye_score
            
            # GPT 분석 결과에서 키워드 추출
            strength_keyword = ""
            weakness_keyword = ""
            if gpt_analysis:
                strength_keyword = gpt_analysis.get('strength_keyword', '')
                weakness_keyword = gpt_analysis.get('weakness_keyword', '')
            
            async with self.get_connection() as conn:
                async with conn.cursor() as cursor:
                    # 1. answer_score 테이블에 UPSERT
                    answer_score_query = """
                    INSERT INTO answer_score (
                        ANS_SCORE_ID, INTV_ANS_ID, SUSPECTED_COPYING, SUSPECTED_IMPERSONATION
                    ) VALUES (%s, %s, %s, %s)
                    ON DUPLICATE KEY UPDATE
                        SUSPECTED_COPYING = VALUES(SUSPECTED_COPYING),
                        SUSPECTED_IMPERSONATION = VALUES(SUSPECTED_IMPERSONATION),
                        UPD_DTM = CURRENT_TIMESTAMP
                    """
                    
                    await cursor.execute(answer_score_query, (
                        ans_score_id, intv_ans_id, suspected_copying, suspected_impersonation
                    ))
                    
                    # 2. answer_category_result 테이블에 UPSERT (면접태도만)
                    category_result_query = """
                    INSERT INTO answer_category_result (
                        ANS_CAT_RESULT_ID, EVAL_CAT_CD, ANS_SCORE_ID, ANS_CAT_SCORE, 
                        STRENGTH_KEYWORD, WEAKNESS_KEYWORD
                    ) VALUES (%s, %s, %s, %s, %s, %s)
                    ON DUPLICATE KEY UPDATE
                        ANS_CAT_SCORE = VALUES(ANS_CAT_SCORE),
                        STRENGTH_KEYWORD = VALUES(STRENGTH_KEYWORD),
                        WEAKNESS_KEYWORD = VALUES(WEAKNESS_KEYWORD),
                        RGS_DTM = CURRENT_TIMESTAMP
                    """
                    
                    await cursor.execute(category_result_query, (
                        ans_cat_result_id, 'INTERVIEW_ATTITUDE', ans_score_id, total_score,
                        strength_keyword, weakness_keyword
                    ))
                    
                    logger.info(f"면접태도 저장 완료: ANS_SCORE_ID={ans_score_id}, ANS_CAT_RESULT_ID={ans_cat_result_id} - 표정:{emotion_score}, 시선:{eye_score}, 총합:{total_score}")
                    logger.info(f"부정행위 감지: 커닝={suspected_copying}, 대리시험={suspected_impersonation}")
                    print(f"🔍 MariaDB 저장: user_id={user_id}, question_num={question_num}")
                    print(f"🔍 ID 생성: ANS_SCORE_ID={ans_score_id}, INTV_ANS_ID={intv_ans_id}, ANS_CAT_RESULT_ID={ans_cat_result_id}")
                    print(f"🔍 부정행위 결과: 커닝={suspected_copying}, 대리시험={suspected_impersonation}")
                    return True
                    
        except Exception as e:
            logger.error(f"면접태도 저장 실패: {e}")
            return False

    async def get_interview_attitude(self, user_id: str, question_num: str = None) -> Optional[Dict]:
        """면접태도 평가 결과 조회"""
        try:
            async with self.get_connection() as conn:
                async with conn.cursor(aiomysql.DictCursor) as cursor:
                    if question_num:
                        ans_score_id = int(f"{user_id}0{question_num}")
                        query = """
                        SELECT 
                            a.ANS_SCORE_ID, a.INTV_ANS_ID, a.SUSPECTED_COPYING, a.SUSPECTED_IMPERSONATION,
                            c.ANS_CAT_RESULT_ID, c.ANS_CAT_SCORE, c.STRENGTH_KEYWORD, c.WEAKNESS_KEYWORD, c.RGS_DTM
                        FROM answer_score a
                        LEFT JOIN answer_category_result c ON a.ANS_SCORE_ID = c.ANS_SCORE_ID 
                        WHERE a.ANS_SCORE_ID = %s AND c.EVAL_CAT_CD = 'INTERVIEW_ATTITUDE'
                        """
                        await cursor.execute(query, (ans_score_id,))
                        return await cursor.fetchone()
                    else:
                        # 특정 사용자의 모든 질문 조회 (INTV_ANS_ID가 userId0으로 시작하는 것들)
                        user_pattern = f"{user_id}0%"
                        query = """
                        SELECT 
                            a.ANS_SCORE_ID, a.INTV_ANS_ID, a.SUSPECTED_COPYING, a.SUSPECTED_IMPERSONATION,
                            c.ANS_CAT_RESULT_ID, c.ANS_CAT_SCORE, c.STRENGTH_KEYWORD, c.WEAKNESS_KEYWORD, c.RGS_DTM
                        FROM answer_score a
                        LEFT JOIN answer_category_result c ON a.ANS_SCORE_ID = c.ANS_SCORE_ID 
                        WHERE CAST(a.INTV_ANS_ID AS CHAR) LIKE %s AND c.EVAL_CAT_CD = 'INTERVIEW_ATTITUDE'
                        ORDER BY a.ANS_SCORE_ID
                        """
                        await cursor.execute(query, (user_pattern,))
                        return await cursor.fetchall()
                        
        except Exception as e:
            logger.error(f"면접태도 조회 실패: {e}")
            return None

# 전역 MariaDB 핸들러 인스턴스
mariadb_handler = MariaDBHandler() 