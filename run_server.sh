#!/bin/bash
# ----
# 작성목적 : FastAPI 서버 실행 스크립트
# 작성일 : 2025-06-14

# 변경사항 내역 (날짜 | 변경목적 | 변경내용 | 작성자 순으로 기입)
# 2025-06-14 | 최초 구현 | FastAPI 베스트 프랙티스에 따른 구조로 재구성 | 이재인
# 2025-01-17 | 기능 최적화 | 자동 분석 전용으로 수정 | 이재인
# ----
echo "FastAPI 자동 분석 서버를 시작합니다..."

# 현재 디렉토리 확인
echo "현재 디렉토리: $(pwd)"

# Python 경로에 현재 디렉토리 추가
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# 환경 변수 확인
echo "Python 경로: $PYTHONPATH"

# FastAPI 서버 실행 (자동 분석 전용)
echo "자동 분석 서버 실행 중..."
echo "서버 시작 시 S3 영상 자동 분석이 시작됩니다."
python -m uvicorn src.main:app --reload --host 0.0.0.0 --port 8001

echo "서버가 종료되었습니다." 