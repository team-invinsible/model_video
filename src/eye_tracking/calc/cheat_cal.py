# ----------------------------------------------------------------------------------------------------
# 작성목적 : 부정행위 감지 및 분석 모듈
# 작성일 : 2024-06-01

# 변경사항 내역 (날짜 | 변경목적 | 변경내용 | 작성자 순으로 기입)
# 2024-06-01 | 최초 구현 | 고개 방향 기반 부정행위 감지 구현 | 이소미
# 2024-06-14 | 기능 추가 | 얼굴 수 변화 기반 부정행위 감지 추가, 부정행위 판단 기준 및 로직 개선 | 이소미
# 2024-06-15 | 기능 추가 | 로그 파일 자동 탐지 기능 추가 | 이소미
# 2025-06-24 | 기능 개선 | S3 경로 기반 동적 user_id, question_num 설정 | 이재인
# 2025-06-24 | 기능 추가 | 부정행위 횟수 기반 판단 및 얼굴 2개 이상 감지 추가 | 이재인
# ----------------------------------------------------------------------------------------------------

import json
import sys
import os
import re
from pathlib import Path

def extract_s3_path_info(video_path_or_s3_path):
    """
    S3 경로에서 userId와 question_num을 추출합니다.
    경로 형식: */interview_audio/{userId}/{question_num}/* 또는 기타 형식
    
    Args:
        video_path_or_s3_path (str): S3 경로 또는 비디오 파일 경로
        
    Returns:
        tuple: (user_id, question_num) 또는 (None, None)
    """
    try:
        print(f"🔍 cheat_cal.py: 경로 파싱 시작 - {video_path_or_s3_path}")
        
        # S3 경로 패턴 매칭 (더 포괄적)
        patterns = [
            r'interview_audio/([^/]+)/([^/]+)',  # */interview_audio/{user_id}/{question_num}/*
            r'skala25a/team12/interview_audio/([^/]+)/([^/]+)',  # 기존 패턴
            r'/([^/]+)/([^/]+)/[^/]*\.(mp4|webm|mov)',  # /{user_id}/{question_num}/filename.ext
        ]
        
        for pattern in patterns:
            match = re.search(pattern, video_path_or_s3_path)
            if match:
                user_id = match.group(1)
                question_num = match.group(2)
                print(f"🔍 cheat_cal.py: 패턴 매칭 성공 - user_id={user_id}, question_num={question_num}")
                return user_id, question_num
        
        # 로컬 파일 경로에서 정보 추출 시도
        path_parts = video_path_or_s3_path.split('/')
        print(f"🔍 cheat_cal.py: 경로 분할 - {path_parts}")
        
        if len(path_parts) >= 2:
            # 파일명에서 user_id와 question 정보 추출 시도
            filename = path_parts[-1]
            if '_' in filename:
                parts = filename.split('_')
                if len(parts) >= 2:
                    user_id = parts[0]
                    question_num = parts[1].replace('.mp4', '').replace('.webm', '')
                    print(f"🔍 cheat_cal.py: 파일명 파싱 성공 - user_id={user_id}, question_num={question_num}")
                    return user_id, question_num
        
        print(f"🔍 cheat_cal.py: 경로 파싱 실패")
        return None, None
        
    except Exception as e:
        print(f"경로 파싱 오류: {e}")
        return None, None

def detect_cheating(head_log_path, anomaly_log_path, user_id, question_num=None, video_path=None):
    """
    부정행위 감지 결과를 생성합니다.
    
    Args:
        head_log_path (str): 머리 방향 로그 파일 경로
        anomaly_log_path (str): 이상 상황 로그 파일 경로  
        user_id (str): 사용자 ID
        question_num (str): 질문 번호 (예: Q1, Q2 등)
        video_path (str): 비디오 파일 경로 (S3 경로 파싱용)
        
    Returns:
        dict: 부정행위 감지 결과
    """
    results = []
    idx = 1
    
    # 부정행위 감지 카운터
    total_violations = 0
    face_multiple_detected = False
    
    print(f"🔍 detect_cheating 시작: user_id={user_id}, question_num={question_num}, video_path={video_path}")
    
    # S3 경로에서 정보 추출 시도 (video_path가 있는 경우)
    if video_path:
        extracted_user_id, extracted_question_num = extract_s3_path_info(video_path)
        print(f"🔍 S3 경로 추출 결과: user_id={extracted_user_id}, question_num={extracted_question_num}")
        if extracted_user_id:
            user_id = extracted_user_id
        if extracted_question_num:
            question_num = extracted_question_num

    # question_num이 없는 경우 기본값 설정
    if not question_num:
        question_num = "Q001"
    
    # question_num을 그대로 사용 (변환하지 않음)
    question_key = str(question_num)
    
    print(f"🔍 최종 설정: user_id={user_id}, question_key={question_key}")

    # 1. anomalies 로그에서 face_count가 0이거나 2개 이상인 경우
    if os.path.exists(anomaly_log_path):
        with open(anomaly_log_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    if "face_count" in data:
                        face_count = data["face_count"]
                        
                        # 얼굴 0개 감지
                        if face_count == 0:
                            total_violations += 1
                            reason = "얼굴이 {}개 감지됨".format(face_count)
                            results.append({
                                "category": "부정행위",
                                "index": idx,
                                "comments": reason
                            })
                            idx += 1
                            
                        # 얼굴 2개 이상 감지 (새로운 조건)
                        elif face_count >= 2:
                            total_violations += 1
                            face_multiple_detected = True
                            reason = "얼굴이 {}개 감지됨 (다른 사람 존재 의심)".format(face_count)
                            print(f"🔍 다중얼굴 감지: {face_count}개 얼굴, face_multiple_detected={face_multiple_detected}")
                            results.append({
                                "category": "부정행위",
                                "index": idx,
                                "comments": reason
                            })
                            idx += 1
                except Exception:
                    continue

    # 2. head 로그에서 direction이 center가 아닌 경우
    if os.path.exists(head_log_path):
        with open(head_log_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    if "direction" in data and data["direction"] != "center":
                        total_violations += 1
                        reason = "머리 방향: {}".format(data["direction"])
                        results.append({
                            "category": "부정행위",
                            "index": idx,
                            "comments": reason
                        })
                        idx += 1
                except Exception:
                    continue

    # 3. 부정행위 의심 종합 판단 및 요약 추가
    summary_parts = []
    
    # 5번 이상 부정행위 감지 확인
    if total_violations >= 5:
        summary_parts.append(f"⚠️ 총 {total_violations}회 부정행위 감지 (5회 이상)")
    
    # 얼굴 2개 이상 감지 확인
    if face_multiple_detected:
        summary_parts.append("⚠️ 다른 사람 존재 의심 감지됨")
    
    # 기본 결과 처리
    if not results:
        results.append({
            "category": "부정행위",
            "index": 1,
            "comments": "부정행위 의심 상황 없음"
        })
    
    # 의심 상황이 있으면 요약 정보 추가
    if summary_parts:
        summary_comment = " | ".join(summary_parts)
        results.insert(0, {
            "category": "부정행위 요약",
            "index": 0,
            "comments": summary_comment
        })

    print(f"🔍 부정행위 감지 최종 결과: total_violations={total_violations}, face_multiple_detected={face_multiple_detected}")
    print(f"🔍 반환 데이터: user_id={user_id}, question_key={question_key}, results_count={len(results)}")
    
    return {
        "user_id": user_id,
        question_key: results
    }

def find_latest_logs():
    """logs 디렉토리에서 가장 최근의 로그 파일들 찾기"""
    log_dir = Path("logs")
    if not log_dir.exists():
        print("Error: logs 디렉토리를 찾을 수 없습니다.")
        return None, None, None

    # 모든 로그 파일 찾기
    log_files = list(log_dir.glob("*_Q*_*.jsonl"))
    if not log_files:
        print("Error: 로그 파일을 찾을 수 없습니다.")
        return None, None, None

    # 파일명에서 user_id와 question_id 추출
    latest_file = max(log_files, key=lambda x: x.stat().st_mtime)
    parts = latest_file.stem.split("_")
    if len(parts) < 2:
        print("Error: 잘못된 로그 파일 이름 형식입니다.")
        return None, None, None

    user_id = parts[0]
    question_id = parts[1]

    # 필요한 로그 파일 경로 생성
    head_log = log_dir / f"{user_id}_{question_id}_head.jsonl"
    anomaly_log = log_dir / f"{user_id}_{question_id}_anomalies.jsonl"

    if not head_log.exists() or not anomaly_log.exists():
        print("Error: 필요한 로그 파일이 없습니다.")
        return None, None, None

    return str(head_log), str(anomaly_log), user_id

if __name__ == "__main__":
    # 자동으로 로그 파일 찾기
    head_log_path, anomaly_log_path, user_id = find_latest_logs()
    if not head_log_path or not anomaly_log_path or not user_id:
        sys.exit(1)

    print(f"처리할 로그 파일:")
    print(f"- 머리 방향 로그: {head_log_path}")
    print(f"- 이상 상황 로그: {anomaly_log_path}")
    print(f"- 사용자 ID: {user_id}")
    print("분석을 시작합니다...")

    # 질문 번호 추출 (로그 파일명에서)
    question_num = None
    if "_Q" in head_log_path:
        parts = head_log_path.split("_Q")
        if len(parts) > 1:
            question_num = "Q" + parts[1].split("_")[0]

    res = detect_cheating(head_log_path, anomaly_log_path, user_id, question_num)
    
    # 결과 저장 (jsonl 형식)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(base_dir, "cheating_detected.jsonl")
    with open(save_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(res, ensure_ascii=False, indent=2) + "\n\n")
    print("\n분석 결과가 저장되었습니다:", save_path)
    print(json.dumps(res, ensure_ascii=False, indent=2))