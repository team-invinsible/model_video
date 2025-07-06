# ----
# 작성목적 : MariaDB 연결 및 테이블 관리
# 작성일 : 2025-06-12

# 변경사항 내역 (날짜 | 변경목적 | 변경내용 | 작성자 순으로 기입)
# 2025-06-14 | 최초 구현 | FastAPI 베스트 프랙티스에 따른 구조로 재구성 | 이재인
# 2025-06-14 | 테이블 리팩터링 | audio.answer_score, answer_category_result 테이블 구조로 변경 | 이재인
# 2025-06-24 | 면접태도 전용 | INTERVIEW_ATTITUDE 카테고리 전용 저장 시스템 구현 | 이재인
# 2025-06-24 | ID 형식 변경 | INTV_ANS_ID를 userId0questionNum, ANS_CAT_RESULT_ID를 userId0questionNum0 형식으로 변경 | 이재인
# 2025-01-04 | 외래키 제약조건 완전 해결 | 기존 저장 방식 유지하면서 외래키 제약조건 및 참조 테이블 자동 생성 | 이재인
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
        self.database = os.getenv("MARIADB_DATABASE", "SKAI")
        
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
        async with self.pool.acquire() as conn:
            async with conn.cursor() as cursor:
                # 외래키 체크 비활성화
                await cursor.execute("SET FOREIGN_KEY_CHECKS = 0")
                
                try:
                    # interview_answer 테이블 처리 (참조 테이블이므로 먼저 생성)
                    await self._create_or_update_interview_answer_table(cursor)
                    
                    # answer_score 테이블 처리
                    await self._create_or_update_answer_score_table(cursor)
                    
                    # answer_category_result 테이블 처리
                    await self._create_or_update_category_result_table(cursor)
                    
                    logger.info("audio 데이터베이스 테이블 생성/업데이트 완료")
                    
                finally:
                    # 외래키 체크 재활성화
                    await cursor.execute("SET FOREIGN_KEY_CHECKS = 1")

    async def _create_or_update_interview_answer_table(self, cursor):
        """interview_answer 테이블이 존재하지 않을 때만 생성"""
        table_name = "interview_answer"
        
        if not await self._table_exists(cursor, table_name):
            # 테이블이 없으면 기존 구조에 맞춰 새로 생성
            logger.info(f"{table_name} 테이블이 존재하지 않아 새로 생성합니다.")
            create_sql = """
            CREATE TABLE interview_answer (
                INTV_ANS_ID BIGINT NOT NULL AUTO_INCREMENT,
                INTV_Q_ASSIGN_ID BIGINT NOT NULL,
                ANS_TXT TEXT DEFAULT NULL,
                RGS_DTM TIMESTAMP NULL DEFAULT NULL,
                USER_ID VARCHAR(100) DEFAULT NULL COMMENT '사용자 ID',
                QUESTION_NUM INT DEFAULT NULL COMMENT '질문 번호',
                ANSWER_TEXT TEXT DEFAULT NULL COMMENT '답변 내용',
                UPD_DTM TIMESTAMP NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '수정 일시',
                PRIMARY KEY (INTV_ANS_ID),
                KEY INTV_Q_ASSIGN_ID (INTV_Q_ASSIGN_ID)
            ) ENGINE=InnoDB AUTO_INCREMENT=10010 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci
            """
            await cursor.execute(create_sql)
            logger.info(f"{table_name} 테이블 생성 완료")
        else:
            # 테이블이 존재하면 아무것도 하지 않음
            logger.info(f"{table_name} 테이블이 이미 존재합니다. 수정하지 않습니다.")

    async def _create_or_update_answer_score_table(self, cursor):
        """answer_score 테이블 생성 또는 업데이트"""
        table_name = "answer_score"
        
        if not await self._table_exists(cursor, table_name):
            # 테이블이 없으면 새로 생성 (외래키 제약조건 포함)
            logger.info(f"{table_name} 테이블이 존재하지 않아 새로 생성합니다.")
            create_sql = """
            CREATE TABLE answer_score (
                ANS_SCORE_ID BIGINT NOT NULL COMMENT '답변 평가 ID',
                INTV_ANS_ID BIGINT NOT NULL COMMENT '면접 답변 ID',
                ANS_SUMMARY TEXT NULL COMMENT '답변 요약',
                EVAL_SUMMARY TEXT NULL COMMENT '전체 평가 요약',
                INCOMPLETE_ANSWER BOOLEAN NULL DEFAULT FALSE COMMENT '미완료 여부',
                INSUFFICIENT_CONTENT BOOLEAN NULL DEFAULT FALSE COMMENT '내용 부족 여부',
                SUSPECTED_COPYING BOOLEAN NULL DEFAULT FALSE COMMENT '커닝 의심 여부',
                SUSPECTED_IMPERSONATION BOOLEAN NULL DEFAULT FALSE COMMENT '대리 시험 의심 여부',
                RGS_DTM TIMESTAMP NULL DEFAULT CURRENT_TIMESTAMP COMMENT '등록 일시',
                UPD_DTM TIMESTAMP NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '수정 일시',
                PRIMARY KEY (ANS_SCORE_ID),
                INDEX idx_intv_ans_id (INTV_ANS_ID),
                INDEX idx_suspected_copying (SUSPECTED_COPYING),
                INDEX idx_suspected_impersonation (SUSPECTED_IMPERSONATION),
                INDEX idx_rgs_dtm (RGS_DTM)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='답변 평가'
            """
            await cursor.execute(create_sql)
            logger.info(f"{table_name} 테이블 생성 완료")
            
            # 외래키 제약조건 추가 (interview_answer 테이블이 존재하는 경우)
            await self._add_foreign_key_if_possible(cursor, table_name)
        else:
            # 테이블이 있으면 필요한 컬럼 및 외래키 제약조건 확인/추가
            logger.info(f"{table_name} 테이블이 존재합니다. 필요한 컬럼과 외래키를 확인합니다.")
            
            required_columns = {
                'INCOMPLETE_ANSWER': 'BOOLEAN NULL DEFAULT FALSE COMMENT "미완료 여부"',
                'INSUFFICIENT_CONTENT': 'BOOLEAN NULL DEFAULT FALSE COMMENT "내용 부족 여부"',
                'SUSPECTED_COPYING': 'BOOLEAN NULL DEFAULT FALSE COMMENT "커닝 의심 여부"',
                'SUSPECTED_IMPERSONATION': 'BOOLEAN NULL DEFAULT FALSE COMMENT "대리 시험 의심 여부"'
            }
            
            # 없는 컬럼들 추가
            for column_name, column_definition in required_columns.items():
                if not await self._column_exists(cursor, table_name, column_name):
                    logger.info(f"  컬럼 {column_name} 추가 중...")
                    try:
                        await cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_definition}")
                        logger.info(f"  컬럼 {column_name} 추가 완료")
                    except Exception as e:
                        logger.warning(f"  컬럼 {column_name} 추가 실패: {e}")
            
            # 외래키 제약조건 추가 (기존 테이블에도 적용)
            await self._add_foreign_key_if_possible(cursor, table_name)

    async def _create_or_update_category_result_table(self, cursor):
        """answer_category_result 테이블 생성 또는 업데이트"""
        table_name = "answer_category_result"
        
        if not await self._table_exists(cursor, table_name):
            # 테이블이 없으면 새로 생성
            logger.info(f"{table_name} 테이블이 존재하지 않아 새로 생성합니다.")
            create_sql = """
            CREATE TABLE answer_category_result (
                ANS_CAT_RESULT_ID BIGINT NOT NULL COMMENT '답변 항목별 평가 ID',
                EVAL_CAT_CD VARCHAR(20) NOT NULL COMMENT '평가 항목 코드',
                ANS_SCORE_ID BIGINT NOT NULL COMMENT '답변 평가 ID',
                ANS_CAT_SCORE DOUBLE NULL COMMENT '항목별 점수',
                STRENGTH_KEYWORD TEXT NULL COMMENT '강점 키워드',
                WEAKNESS_KEYWORD TEXT NULL COMMENT '약점 키워드',
                RGS_DTM TIMESTAMP NULL DEFAULT CURRENT_TIMESTAMP COMMENT '등록 일시',
                PRIMARY KEY (ANS_CAT_RESULT_ID),
                INDEX idx_eval_cat_cd (EVAL_CAT_CD),
                INDEX idx_ans_score_id (ANS_SCORE_ID),
                INDEX idx_ans_cat_score (ANS_CAT_SCORE),
                INDEX idx_rgs_dtm (RGS_DTM),
                UNIQUE KEY unique_score_category (ANS_SCORE_ID, EVAL_CAT_CD)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='답변 항목별 평가 결과'
            """
            await cursor.execute(create_sql)
            logger.info(f"{table_name} 테이블 생성 완료")
            
            # 외래키 제약조건 추가
            await self._add_category_foreign_key_if_possible(cursor, table_name)
        else:
            logger.info(f"{table_name} 테이블이 이미 존재합니다.")

    async def _add_foreign_key_if_possible(self, cursor, table_name: str):
        """interview_answer 테이블이 존재하면 외래키 제약조건 추가"""
        try:
            # interview_answer 테이블 존재 확인
            if await self._table_exists(cursor, "interview_answer"):
                constraint_name = "answer_score_ibfk_1"
                if not await self._foreign_key_exists(cursor, table_name, constraint_name):
                    logger.info(f"  외래키 제약조건 '{constraint_name}' 추가 중...")
                    await cursor.execute(f"""
                        ALTER TABLE {table_name} 
                        ADD CONSTRAINT {constraint_name}
                        FOREIGN KEY (INTV_ANS_ID) REFERENCES interview_answer(INTV_ANS_ID)
                        ON DELETE CASCADE ON UPDATE CASCADE
                    """)
                    logger.info(f"  외래키 제약조건 '{constraint_name}' 추가 완료")
                else:
                    logger.info(f"  외래키 제약조건 '{constraint_name}'이 이미 존재합니다.")
            else:
                logger.info("  interview_answer 테이블이 존재하지 않아 외래키 제약조건을 추가하지 않습니다.")
        except Exception as e:
            logger.warning(f"  외래키 제약조건 추가 중 오류 (무시하고 계속): {e}")

    async def _add_category_foreign_key_if_possible(self, cursor, table_name: str):
        """answer_score 테이블이 존재하면 외래키 제약조건 추가"""
        try:
            if await self._table_exists(cursor, "answer_score"):
                constraint_name = "answer_category_result_ibfk_1"
                if not await self._foreign_key_exists(cursor, table_name, constraint_name):
                    logger.info(f"  외래키 제약조건 '{constraint_name}' 추가 중...")
                    await cursor.execute(f"""
                        ALTER TABLE {table_name} 
                        ADD CONSTRAINT {constraint_name}
                        FOREIGN KEY (ANS_SCORE_ID) REFERENCES answer_score(ANS_SCORE_ID)
                        ON DELETE CASCADE ON UPDATE CASCADE
                    """)
                    logger.info(f"  외래키 제약조건 '{constraint_name}' 추가 완료")
                else:
                    logger.info(f"  외래키 제약조건 '{constraint_name}'이 이미 존재합니다.")
            else:
                logger.info("  answer_score 테이블이 존재하지 않아 외래키 제약조건을 추가하지 않습니다.")
        except Exception as e:
            logger.warning(f"  외래키 제약조건 추가 중 오류 (무시하고 계속): {e}")

    async def _table_exists(self, cursor, table_name: str) -> bool:
        """테이블 존재 여부 확인"""
        await cursor.execute("""
            SELECT COUNT(*) FROM information_schema.TABLES 
            WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = %s
        """, (table_name,))
        result = await cursor.fetchone()
        return result[0] > 0

    async def _column_exists(self, cursor, table_name: str, column_name: str) -> bool:
        """컬럼 존재 여부 확인"""
        await cursor.execute("""
            SELECT COUNT(*) FROM information_schema.COLUMNS 
            WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = %s AND COLUMN_NAME = %s
        """, (table_name, column_name))
        result = await cursor.fetchone()
        return result[0] > 0

    async def _foreign_key_exists(self, cursor, table_name: str, constraint_name: str) -> bool:
        """외래키 제약조건 존재 여부 확인"""
        await cursor.execute("""
            SELECT COUNT(*) FROM information_schema.TABLE_CONSTRAINTS 
            WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = %s 
            AND CONSTRAINT_NAME = %s AND CONSTRAINT_TYPE = 'FOREIGN KEY'
        """, (table_name, constraint_name))
        result = await cursor.fetchone()
        return result[0] > 0

    def _generate_safe_id(self, user_id: str, question_num: int, suffix: str = "") -> int:
        """안전한 ID 생성 (user_id + 0 + question_num 형식)"""
        try:
            # user_id가 숫자인 경우 그대로 사용
            if user_id.isdigit():
                user_id_num = int(user_id)
                # 너무 큰 숫자는 적당한 크기로 제한
                if user_id_num > 999:
                    user_id_num = user_id_num % 999 + 1
            else:
                # 문자열인 경우 해시 사용하되 적당한 크기로 제한
                user_id_num = abs(hash(user_id)) % 999 + 1

            # question_num이 너무 크면 제한
            if question_num > 99:
                question_num = question_num % 99 + 1

            # ID 형식: {user_id}0{question_num}{suffix}
            # 예: user_id=2, question_num=3 -> 203
            # 예: user_id=2, question_num=3, suffix="1" -> 2031
            id_str = f"{user_id_num}0{question_num}{suffix}"
            generated_id = int(id_str)

            logger.info(f"ID 생성: user_id={user_id} -> {user_id_num}, question_num={question_num}, suffix='{suffix}' -> ID={generated_id}")

            # MySQL BIGINT 범위 확인 (최대 9223372036854775807)
            if generated_id > 9223372036854775807:
                # 너무 큰 경우 해시 사용
                generated_id = abs(hash(f"{user_id}_{question_num}_{suffix}")) % (10**10)
                logger.warning(f"ID가 너무 커서 해시로 변경: {generated_id}")

            return generated_id

        except (ValueError, OverflowError) as e:
            # 모든 실패 시 해시 사용
            fallback_id = abs(hash(f"{user_id}_{question_num}_{suffix}")) % (10**10)
            logger.error(f"ID 생성 실패, 해시 사용: {e} -> {fallback_id}")
            return fallback_id
    
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
            # 기존 방식대로 ID 생성 (userId0questionNum 형식)
            ans_score_id = self._generate_safe_id(user_id, int(question_num))
            intv_ans_id = self._generate_safe_id(user_id, int(question_num))
            ans_cat_result_id = self._generate_safe_id(user_id, int(question_num), "0")
            
            total_score = emotion_score + eye_score
            
            # GPT 분석 결과에서 키워드 추출
            strength_keyword = ""
            weakness_keyword = ""
            if gpt_analysis:
                strength_keyword = gpt_analysis.get('strength_keyword', '')
                weakness_keyword = gpt_analysis.get('weakness_keyword', '')
            
            async with self.get_connection() as conn:
                async with conn.cursor() as cursor:
                    # 트랜잭션 시작
                    await conn.begin()
                    
                    try:
                        # 0. interview_answer 테이블에 레코드가 존재하는지 확인하고 없으면 생성
                        if not await self._ensure_interview_answer_exists(cursor, intv_ans_id, user_id, int(question_num)):
                            logger.warning(f"interview_answer 레코드 생성에 실패했습니다: INTV_ANS_ID={intv_ans_id}")
                        
                        # 1. answer_score 테이블에 UPSERT (기존 방식 유지)
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
                        
                        # 트랜잭션 커밋
                        await conn.commit()
                        
                        logger.info(f"면접태도 저장 완료: ANS_SCORE_ID={ans_score_id}, ANS_CAT_RESULT_ID={ans_cat_result_id} - 표정:{emotion_score}, 시선:{eye_score}, 총합:{total_score}")
                        logger.info(f"부정행위 감지: 커닝={suspected_copying}, 대리시험={suspected_impersonation}")
                        print(f"🔍 MariaDB 저장: user_id={user_id}, question_num={question_num}")
                        print(f"🔍 ID 생성: ANS_SCORE_ID={ans_score_id}, INTV_ANS_ID={intv_ans_id}, ANS_CAT_RESULT_ID={ans_cat_result_id}")
                        print(f"🔍 부정행위 결과: 커닝={suspected_copying}, 대리시험={suspected_impersonation}")
                        return True
                        
                    except Exception as e:
                        # 트랜잭션 롤백
                        await conn.rollback()
                        raise e
                    
        except Exception as e:
            logger.error(f"면접태도 저장 실패: {e}")
            return False

    async def _ensure_interview_answer_exists(self, cursor, intv_ans_id: int, user_id: str, question_num: int):
        """interview_answer 테이블에 레코드가 존재하는지 확인하고 없으면 생성"""
        try:
            # interview_answer 테이블이 존재하는지 확인
            if not await self._table_exists(cursor, "interview_answer"):
                logger.warning("interview_answer 테이블이 존재하지 않습니다.")
                return False

            # 레코드 존재 확인
            await cursor.execute("SELECT COUNT(*) FROM interview_answer WHERE INTV_ANS_ID = %s", (intv_ans_id,))
            result = await cursor.fetchone()

            if result and result[0] > 0:
                logger.info(f"interview_answer 레코드가 이미 존재합니다: INTV_ANS_ID={intv_ans_id}")
                return True

            # 레코드가 없으면 생성
            logger.info(f"interview_answer 레코드 생성 중: INTV_ANS_ID={intv_ans_id}")

            # interview_question_assignment 테이블에 필요한 레코드가 있는지 확인
            assign_id = await self._ensure_question_assignment_exists(cursor, user_id, question_num)
            if not assign_id:
                logger.error(f"interview_question_assignment 레코드 생성 실패")
                return False

            # interview_answer 레코드 생성
            insert_sql = """
            INSERT INTO interview_answer (INTV_ANS_ID, INTV_Q_ASSIGN_ID, ANS_TXT, RGS_DTM) 
            VALUES (%s, %s, %s, NOW())
            ON DUPLICATE KEY UPDATE RGS_DTM = NOW()
            """
            await cursor.execute(insert_sql, (intv_ans_id, assign_id, f"면접태도 분석 - user:{user_id}, question:{question_num}"))

            logger.info(f"interview_answer 레코드 생성 완료: INTV_ANS_ID={intv_ans_id}")
            return True

        except Exception as e:
            logger.error(f"interview_answer 레코드 생성 중 오류: {e}")
            return False

    async def _ensure_question_assignment_exists(self, cursor, user_id: str, question_num: int):
        """interview_question_assignment 테이블에 레코드가 존재하는지 확인하고 없으면 생성"""
        try:
            # interview_question_assignment 테이블이 존재하는지 확인
            if not await self._table_exists(cursor, "interview_question_assignment"):
                logger.warning("interview_question_assignment 테이블이 존재하지 않습니다.")
                return None

            # 임시 assign_id 생성 (user_id + question_num 조합)
            assign_id = self._generate_safe_id(user_id, question_num, "1")
            
            # 레코드 존재 확인
            await cursor.execute("SELECT INTV_Q_ASSIGN_ID FROM interview_question_assignment WHERE INTV_Q_ASSIGN_ID = %s", (assign_id,))
            result = await cursor.fetchone()

            if result:
                logger.info(f"interview_question_assignment 레코드가 이미 존재합니다: INTV_Q_ASSIGN_ID={assign_id}")
                return assign_id

            # 레코드가 없으면 생성 (최소한의 필수 필드만 사용)
            logger.info(f"interview_question_assignment 레코드 생성 중: INTV_Q_ASSIGN_ID={assign_id}")
            
            # 테이블 구조를 확인하여 필수 필드 파악
            await cursor.execute("DESCRIBE interview_question_assignment")
            columns = await cursor.fetchall()
            
            # 기본 INSERT 시도 (테이블 구조에 따라 조정 필요)
            try:
                insert_sql = """
                INSERT INTO interview_question_assignment (INTV_Q_ASSIGN_ID) 
                VALUES (%s)
                ON DUPLICATE KEY UPDATE INTV_Q_ASSIGN_ID = VALUES(INTV_Q_ASSIGN_ID)
                """
                await cursor.execute(insert_sql, (assign_id,))
                logger.info(f"interview_question_assignment 레코드 생성 완료: INTV_Q_ASSIGN_ID={assign_id}")
                return assign_id
            
            except Exception as e:
                logger.error(f"interview_question_assignment 레코드 생성 실패: {e}")
                logger.info("테이블 구조:")
                for col in columns:
                    logger.info(f"  {col}")
                return None

        except Exception as e:
            logger.error(f"interview_question_assignment 처리 중 오류: {e}")
            return None

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