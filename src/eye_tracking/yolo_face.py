# ----------------------------------------------------------------------------------------------------
# 작성목적 : YOLO 기반 얼굴 감지 클래스 구현
# 작성일 : 2024-06-09

# 변경사항 내역
# 2024-06-01 | 최초 구현 | YOLOv8 기반 얼굴 감지 기능 구현 | 이소미
# ----------------------------------------------------------------------------------------------------

from ultralytics import YOLO
import cv2

class YOLOFaceDetector:
    def __init__(self, model_path="yolov8n-face-lindevs.pt"):
        self.model = YOLO(model_path)

    def detect_faces(self, frame):
        results = self.model.predict(source=frame, imgsz=640, conf=0.5, iou=0.5, verbose=False)[0]
        boxes = []
        for box in results.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box[:4])
            boxes.append((x1, y1, x2, y2))
        return boxes

    def draw_faces(self, frame, boxes):
        for (x1, y1, x2, y2) in boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
