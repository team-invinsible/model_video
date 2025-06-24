# ----------------------------------------------------------------------------------------------------
# 작성목적 : 시선, 깜빡임, 고개 방향 로깅 클래스 구현
# 작성일 : 2024-06-01

# 변경사항 내역 (날짜 | 변경목적 | 변경내용 | 작성자 순으로 기입)
# 2024-06-01 | 최초 구현 | 깜빡임 로깅 기능 구현 | 이소미
# 2024-06-14 | 기능 추가 | 시선 방향,고개 방향 로깅 기능 추가 | 이소미
# 2024-06-15 | 기능 개선 | 로깅 포맷 및 저장 방식 개선 | 이소미
# ----------------------------------------------------------------------------------------------------

import json

class BlinkLogger:
    def __init__(self, filepath):
        self.filepath = filepath
        with open(self.filepath, 'w') as f:  # 빈 파일로 초기화
            pass
        self.blink_index = 0

    def log_blink(self, timestamp):
        self.blink_index += 1
        log_entry = {
            "time": round(timestamp, 2),
            "event": "blink",
            "eye": "both",
            "blink_index": self.blink_index
        }
        with open(self.filepath, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
        print(f"[LOG] {log_entry}")

    def force_resolve(self, timestamp):
        """프로그램 종료 시 호출되는 메서드"""
        # BlinkLogger는 각 깜빡임을 즉시 기록하므로 
        # 종료 시 특별히 처리할 내용이 없음
        pass


class MultiFaceAnomalyLogger:
    def __init__(self, filepath):
        self.filepath = filepath
        self.active = False
        self.current_start_time = None
        self.anomaly_index = 0

    def begin_anomaly(self, timestamp):
        if not self.active:
            self.active = True
            self.current_start_time = timestamp
            print(f"[Anomaly Start] Multiple faces detected at {timestamp:.2f}s")

    def resolve_anomaly(self, timestamp):
        if self.active:
            log_entry = {
                "start_time": round(self.current_start_time, 2),
                "end_time": round(timestamp, 2),
                "reason": "multiple_faces_detected",
                "index": self.anomaly_index
            }
            with open(self.filepath, "a") as f:
                f.write(json.dumps(log_entry) + "\n")
            print(f"[Anomaly End] {log_entry}")
            self.anomaly_index += 1
            self.active = False


class GazeLogger:
    def __init__(self, filepath):
        self.filepath = filepath
        with open(self.filepath, 'w') as f:  # 빈 파일로 초기화
            pass
        self.active = False
        self.current_start_time = None
        self.current_direction = None
        self.gaze_index = 0

    def update_gaze(self, timestamp, direction):
        """
        시선 방향이 변경될 때마다 호출
        direction: 현재 시선 방향 ("center", "left", "right", "up", "down" 등)
        """
        if not self.active:
            # 처음 방향이 바뀔 때
            self.active = True
            self.current_start_time = timestamp
            self.current_direction = direction
        else:
            # 이미 방향을 보고 있는 상태에서
            if direction != self.current_direction:
                # 이전 방향의 로그 기록
                self._log_gaze(timestamp)
                # 새로운 방향으로 시작
                self.current_start_time = timestamp
                self.current_direction = direction

    def _log_gaze(self, timestamp):
        """현재 시선 로그 기록"""
        if self.active:
            log_entry = {
                "start_time": round(self.current_start_time, 2),
                "end_time": round(timestamp, 2),
                "direction": self.current_direction,
                "index": self.gaze_index
            }
            with open(self.filepath, "a") as f:
                f.write(json.dumps(log_entry) + "\n")
            print(f"[Gaze Log] {log_entry}")
            self.gaze_index += 1
            self.active = False
            self.current_direction = None

    def force_resolve(self, timestamp):
        """프로그램 종료 시 강제 저장"""
        if self.active:
            self._log_gaze(timestamp)


class HeadLogger:
    def __init__(self, filepath):
        self.filepath = filepath
        with open(self.filepath, 'w') as f:  # 빈 파일로 초기화
            pass
        self.active = False
        self.current_start_time = None
        self.current_direction = None
        self.head_index = 0

    def update_head(self, timestamp, direction):
        """
        고개 방향이 변경될 때마다 호출
        direction: 현재 고개 방향 ("center", "left", "right", "up", "down" 등)
        """
        if not self.active:
            # 처음 방향이 바뀔 때
            self.active = True
            self.current_start_time = timestamp
            self.current_direction = direction
        else:
            # 이미 방향을 보고 있는 상태에서
            if direction != self.current_direction:
                # 이전 방향의 로그 기록
                self._log_head(timestamp)
                # 새로운 방향으로 시작
                self.current_start_time = timestamp
                self.current_direction = direction

    def _log_head(self, timestamp):
        """현재 고개 방향 로그 기록"""
        if self.active:
            log_entry = {
                "start_time": round(self.current_start_time, 2),
                "end_time": round(timestamp, 2),
                "direction": self.current_direction,
                "index": self.head_index
            }
            with open(self.filepath, "a") as f:
                f.write(json.dumps(log_entry) + "\n")
            print(f"[Head Log] {log_entry}")
            self.head_index += 1
            self.active = False
            self.current_direction = None

    def force_resolve(self, timestamp):
        """프로그램 종료 시 강제 저장"""
        if self.active:
            self._log_head(timestamp)

