# ----------------------------------------------------------------------------------------------------
# 작성목적 : 이상 상황(얼굴 수 변화) 감지 및 로깅
# 작성일 : 2024-06-01

# 변경사항 내역 (날짜 | 변경목적 | 변경내용 | 작성자 순으로 기입)
# 2024-06-01 | 최초 구현 | 다중 얼굴 감지 로깅 기능 구현 | 이소미
# 2024-06-014 | 기능 추가 | 얼굴 미감지 상황 로깅 추가,이상 상황 판단 로직 개선 | 이소미
# 2024-06-15 | 기능 개선 | 로깅 포맷 및 저장 방식 개선 | 이소미
# ----------------------------------------------------------------------------------------------------

import json

class AnomalyLogger:
    def __init__(self, filepath):
        self.filepath = filepath
        self.active = False
        self.current_start_time = None
        self.current_reason = None
        self.current_face_count = None
        self.anomaly_indices = {
            "multiple_faces_detected": 0,
            "no_face": 0
        }

        # 파일 초기화
        with open(self.filepath, "w") as f:
            f.write("")

    def update_state(self, timestamp, face_count):
        """
        얼굴 수를 바탕으로 상태를 업데이트합니다.
        1명: 정상 → active 상태면 resolve
        2명 이상: 얼굴 수가 다르면 resolve + begin
        """
        if face_count == 1:
            if self.active:
                self.resolve_anomaly(timestamp)
        else:
            reason = "multiple_faces_detected" if face_count > 1 else "no_face"
            if not self.active:
                self.begin_anomaly(timestamp, reason, face_count)
            elif self.current_reason == reason and self.current_face_count != face_count:
                # 얼굴 수가 바뀌었으면 이전 종료하고 새로 시작
                self.resolve_anomaly(timestamp)
                self.begin_anomaly(timestamp, reason, face_count)

    def begin_anomaly(self, timestamp, reason, face_count):
        self.active = True
        self.current_start_time = timestamp
        self.current_reason = reason
        self.current_face_count = face_count
        print(f"[Anomaly Start] {reason} (faces={face_count}) at {timestamp:.2f}s")

    def resolve_anomaly(self, timestamp):
        if not self.active:
            return

        idx = self.anomaly_indices.get(self.current_reason, 0)
        log_entry = {
            "start_time": round(self.current_start_time, 2),
            "end_time": round(timestamp, 2),
            "reason": self.current_reason,
            "face_count": self.current_face_count,
            "index": idx
        }
        with open(self.filepath, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
        print(f"[Anomaly End] {log_entry}")

        self.anomaly_indices[self.current_reason] = idx + 1
        self.active = False
        self.current_reason = None
        self.current_face_count = None

    def force_resolve(self, timestamp):
        """종료 시 강제 저장용"""
        if self.active:
            self.resolve_anomaly(timestamp)
