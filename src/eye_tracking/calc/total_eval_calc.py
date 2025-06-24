# ----------------------------------------------------------------------------------------------------
# 작성목적 : 의사소통능력 및 면접태도 평가 계산
# 작성일 : 2024-06-01

# 변경사항 내역 (날짜 | 변경목적 | 변경내용 | 작성자 순으로 기입)
# 2024-06-01 | 최초 구현 | 깜빡임 기반 의사소통능력 평가 구현 | 이소미
# 2024-06-05 | 기능 추가 | 아이컨택 기반 면접태도 평가 추가 | 이소미
# 2024-06-07 | 기능 개선 | 평가 기준 및 점수 체계 개선 | 이소미
# 2024-06-09 | 기능 추가 | 로그 파일 자동 탐지 기능 추가 | 이소미
# 2025-06-24 | 기능 개선 | S3 경로 기반 동적 user_id, question_num 설정 | 이재인
# ----------------------------------------------------------------------------------------------------

import json
import sys
import os
import re
from pathlib import Path

def calc_blink_score(log_path, user_id):
    """깜빡임 점수 계산"""
    # 로그 파일에서 깜빡임 타임스탬프 읽기
    with open(log_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
        if not lines:
            return {
                "category": "의사소통능력",
                "score": 0,
                "comments": "로그 데이터 없음"
            }
        
        # 각 줄에서 time 값 추출 (JSON 형식 지원)
        timestamps = []
        for line in lines:
            try:
                data = json.loads(line)
                if "time" in data:
                    timestamps.append(float(data["time"]))
            except:
                continue
    
    if not timestamps:
        return {
            "category": "의사소통능력",
            "score": 0,
            "comments": "유효한 시간 데이터 없음"
        }

    # 마지막 시각으로 분당 깜빡임 계산
    last_time = max(timestamps)
    duration_min = last_time / 60 if last_time > 0 else 1
    blink_per_min = len(timestamps) / duration_min

    # 점수 기준
    if blink_per_min <= 20:
        score = 10
        comment = "친근하고 자연스러운 인상으로 평가"
    elif blink_per_min <= 30:
        score = 4
        comment = "다소 긴장, 불안, 산만함 등으로 인식될 가능성 높음"
    else:
        score = 2
        comment = "매우 긴장, 불안, 산만함 등으로 인식될 가능성 높음"

    return {
        "category": "의사소통능력",
        "score": score,
        "comments": comment
    }

def calc_eye_contact_score(gaze_log_path, user_id):
    """아이컨택 점수 계산"""
    # 기준표
    score_table = [
        {
            "score": 40,
            "min": 0.6,
            "max": 1.0,
            "comment": "전체 면접 시간 대비 60% 이상 시선이 화면 중앙에 위치하여 답변동안 아주 적절할 정도의 아이컨택이 이루어짐"
        },
        {
            "score": 20,
            "min": 0.4,
            "max": 0.6,
            "comment": "전체 면접 시간 대비 40% 이상 60% 미만 시선이 화면 중앙에 위치하여 답변동안 살짝 부족한 아이컨택 시간"
        },
        {
            "score": 0,
            "min": 0.0,
            "max": 0.2,
            "comment": "전체 면접 시간 대비 20% 이하 시선이 화면 중앙에 위치하여 답변동안 아이컨택이 제대로 이루어지지 않음"
        }
    ]

    # gaze 로그 읽기
    with open(gaze_log_path, "r", encoding="utf-8") as f:
        logs = [json.loads(line) for line in f if line.strip()]

    # 전체 시간 계산 (마지막 end_time)
    if not logs:
        return {
            "category": "면접태도",
            "score": 0,
            "comments": "로그 데이터 없음"
        }

    total_time = max(log["end_time"] for log in logs)
    center_time = 0.0

    # 정면(center) 응시 시간 합산
    for log in logs:
        if log["direction"].strip() == "center":
            # start_time == end_time인 경우가 많으니 0.2초로 가정(프레임 단위)
            duration = max(0.2, log["end_time"] - log["start_time"])
            center_time += duration

    # 정면 비율
    center_ratio = center_time / total_time if total_time > 0 else 0

    # 점수 및 코멘트 결정
    for row in score_table:
        if row["min"] <= center_ratio < row["max"]:
            score = row["score"]
            comment = row["comment"]
            break
    else:
        # 20~40% 구간 등 예외 처리
        score = 0
        comment = "전체 면접 시간 대비 20~40% 구간(기준표 외) 시선이 화면 중앙에 위치함"

    return {
        "category": "면접태도",
        "score": score,
        "comments": comment
    }

def find_latest_logs():
    """logs 디렉토리에서 가장 최근의 로그 파일들 찾기"""
    log_dir = Path("logs")
    if not log_dir.exists():
        print("Error: logs 디렉토리를 찾을 수 없습니다.")
        return None, None, None

    # 모든 로그 파일 찾기
    log_files = list(log_dir.glob("*_Q*.jsonl"))
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
    blink_log = log_dir / f"{user_id}_{question_id}.jsonl"
    gaze_log = log_dir / f"{user_id}_{question_id}_gaze.jsonl"

    if not blink_log.exists() or not gaze_log.exists():
        print("Error: 필요한 로그 파일이 없습니다.")
        return None, None, None

    return str(blink_log), str(gaze_log), user_id

def extract_s3_path_info(video_path_or_s3_path):
    """
    S3 경로에서 userId와 question_num을 추출합니다.
    경로 형식: skala25a/team12/interview_audio/{userId}/{question_num}
    
    Args:
        video_path_or_s3_path (str): S3 경로 또는 비디오 파일 경로
        
    Returns:
        tuple: (user_id, question_num) 또는 (None, None)
    """
    try:
        # S3 경로 패턴 매칭
        pattern = r'skala25a/team12/interview_audio/([^/]+)/([^/]+)'
        match = re.search(pattern, video_path_or_s3_path)
        
        if match:
            user_id = match.group(1)
            question_num = match.group(2)
            return user_id, question_num
        
        # 로컬 파일 경로에서 정보 추출 시도
        path_parts = video_path_or_s3_path.split('/')
        if len(path_parts) >= 2:
            # 파일명에서 user_id와 question 정보 추출 시도
            filename = path_parts[-1]
            if '_' in filename:
                parts = filename.split('_')
                if len(parts) >= 2:
                    return parts[0], parts[1].replace('.mp4', '').replace('.webm', '')
        
        return None, None
        
    except Exception as e:
        print(f"경로 파싱 오류: {e}")
        return None, None

def save_total_eval(user_id, blink_result, eye_contact_result, question_num=None, video_path=None):
    """
    통합 평가 결과를 jsonl로 저장
    
    Args:
        user_id (str): 사용자 ID
        blink_result (dict): 깜빡임 평가 결과
        eye_contact_result (dict): 아이컨택 평가 결과
        question_num (str): 질문 번호 (예: Q1, Q2 등)
        video_path (str): 비디오 파일 경로 (S3 경로 파싱용)
    """
    # S3 경로에서 정보 추출 시도 (video_path가 있는 경우)
    if video_path:
        extracted_user_id, extracted_question_num = extract_s3_path_info(video_path)
        if extracted_user_id:
            user_id = extracted_user_id
        if extracted_question_num:
            question_num = extracted_question_num

    # question_num이 없는 경우 기본값 설정
    if not question_num:
        question_num = "001"
    
  

    q_num = str(question_num)
    
    question_key = q_num
    
    result = {
        "user_id": user_id,
        question_key: [blink_result, eye_contact_result]
    }
    
    # 저장 경로 설정
    base_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(base_dir, "total_eval.jsonl")
    
    # 결과 저장 (append 모드) - 보기 좋게 정렬
    with open(save_path, "a", encoding="utf-8") as f:
        formatted_result = {
            "user_id": user_id,
            question_key: {
                "의사소통능력": {
                    "score": blink_result["score"],
                    "comments": blink_result["comments"]
                },
                "면접태도": {
                    "score": eye_contact_result["score"],
                    "comments": eye_contact_result["comments"]
                }
            }
        }
        f.write(json.dumps(formatted_result, ensure_ascii=False, indent=2) + "\n\n")
    
    return formatted_result

if __name__ == "__main__":
    # 자동으로 로그 파일 찾기
    blink_log_path, gaze_log_path, user_id = find_latest_logs()
    if not blink_log_path or not gaze_log_path or not user_id:
        sys.exit(1)

    print(f"처리할 로그 파일:")
    print(f"- 깜빡임 로그: {blink_log_path}")
    print(f"- 시선 방향 로그: {gaze_log_path}")
    print(f"- 사용자 ID: {user_id}")
    print("분석을 시작합니다...")
    
    # 질문 번호 추출 (로그 파일명에서)
    question_num = None
    if "_Q" in blink_log_path:
        parts = blink_log_path.split("_Q")
        if len(parts) > 1:
            question_num = "Q" + parts[1].split("_")[0]
    
    # 두 평가 수행
    blink_result = calc_blink_score(blink_log_path, user_id)
    eye_contact_result = calc_eye_contact_score(gaze_log_path, user_id)
    
    # 통합 결과 저장
    res = save_total_eval(user_id, blink_result, eye_contact_result, question_num)
    print("\n평가 결과가 저장되었습니다.")
    print(json.dumps(res, ensure_ascii=False, indent=2))
    
    # 디버그용 정보 출력
    print("\n디버그 정보:")
    print(f"깜빡임 점수: {blink_result['score']}")
    print(f"아이컨택 점수: {eye_contact_result['score']}") 