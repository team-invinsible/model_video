# 통합 영상 분석 API - 프로젝트 구조 및 개발 문서

## 프로젝트 개요

### 기본 정보
- **프로젝트명**: 통합 영상 분석 API (model_video)
- **위치**: `/Users/9dongb/Documents/Git/model_video`
- **주요 언어**: Python 3.9.23
- **프레임워크**: FastAPI 2.1.0
- **환경 관리**: Conda (environment.yml)

### 목적
S3에 저장된 webm 면접 영상을 분석하여 **감정 분석**, **시선 추적**, **GPT-4 기반 종합 피드백**을 제공하는 AI 기반 면접 평가 시스템입니다.

## 시스템 아키텍처

### 주요 기술 스택

**백엔드 프레임워크**:
- FastAPI 0.104.1 (웹 API 프레임워크)
- Uvicorn 0.24.0.post1 (ASGI 서버)
- Pydantic 2.11.7 (데이터 검증 및 직렬화)

**AI/ML 라이브러리**:
- PyTorch 2.7.1, TorchVision 0.22.1 (딥러닝 프레임워크)
- OpenCV 4.8.1.78 (컴퓨터 비전 처리)
- MediaPipe 0.10.21 (얼굴 랜드마크 및 시선 감지)
- Ultralytics 8.3.154 (YOLO v8 객체 감지)
- EfficientNet (감정 분류 모델)

**데이터베이스**:
- MongoDB (PyMongo 4.13.1, Motor 3.7.1) - 원시 분석 데이터 저장
- MariaDB (PyMySQL 1.1.1) - LLM 결과 및 면접태도 평가 저장

**클라우드 서비스**:
- AWS S3 (Boto3 1.38.36) - 영상 파일 저장소
- OpenAI GPT-4 API (1.86.0) - 종합 피드백 생성

### 데이터 플로우
```
영상 요청 → S3 다운로드 → [감정분석 + 시선추적] → GPT 분석 → [MongoDB + MariaDB 저장] → 결과 반환
```

## 프로젝트 구조

```
model_video/                           # 프로젝트 루트
├── main.py                           # FastAPI 애플리케이션 진입점 (루트)
├── environment.yml                   # Conda 환경 설정 파일 (권장)
├── requirements-dev.txt              # 개발용 패키지 목록
├── run_server.sh                     # 서버 실행 스크립트
├── README.md                         # 프로젝트 문서 (현재 파일)
├── .env                             # 환경변수 설정 파일
├── logs/                            # 분석 로그 저장소
│   ├── api_user_*_Q*.jsonl         # 사용자별 분석 로그
│   ├── *_gaze.jsonl                # 시선 추적 로그
│   ├── *_head.jsonl                # 머리 움직임 로그
│   ├── *_anomalies.jsonl           # 이상 행동 감지 로그
│   └── recalib_log.jsonl           # 재보정 로그
└── src/                             # 소스코드 메인 디렉토리
    ├── main.py                      # FastAPI 애플리케이션 (src 내부)
    ├── emotion/                     # 감정 분석 모듈
    │   ├── analyzer.py              # 감정 분석 메인 클래스
    │   ├── model_eff.pth            # EfficientNet 사전 훈련된 모델
    │   ├── face_classifier.xml      # Haar Cascade 얼굴 검출 모델
    │   ├── utils.py                 # 감정 분석 유틸리티
    │   └── models/                  # 딥러닝 모델 구현
    │       ├── efficientnet.py     # EfficientNet 모델 구현
    │       ├── cnn.py               # 기본 CNN 모델
    │       ├── resnet.py            # ResNet 모델
    │       ├── vgg.py               # VGG 모델
    │       └── utils.py             # 모델 공통 유틸리티
    ├── eye_tracking/                # 시선 추적 모듈
    │   ├── analyzer.py              # 시선 추적 메인 클래스
    │   ├── face.py                  # 얼굴 감지 (MediaPipe)
    │   ├── eye.py                   # 눈 감지 및 추적
    │   ├── gaze_analyzer.py         # 시선 방향 분석 로직
    │   ├── yolo_face.py             # YOLO 기반 얼굴 검출
    │   ├── yolov8n-face-lindevs.pt  # YOLO 얼굴 검출 모델
    │   ├── logger.py                # 분석 결과 로깅
    │   ├── anomaly_logger.py        # 이상 행동 감지 로깅
    │   ├── utils.py                 # 시선 추적 유틸리티
    │   └── calc/                    # 점수 계산 모듈
    │       ├── cheat_cal.py         # 부정행위 감지 계산
    │       ├── total_eval_calc.py   # 전체 평가 점수 계산
    │       ├── cheating_detected.jsonl # 부정행위 로그
    │       └── total_eval.jsonl     # 전체 평가 로그
    ├── llm/                         # LLM 분석 모듈
    │   ├── gpt_analyzer.py          # OpenAI GPT-4 피드백 생성
    │   ├── keyword_analyzer.py      # 키워드 추출 및 분석
    │   └── interview_prompts.yaml   # GPT 프롬프트 템플릿
    ├── db/                          # 데이터베이스 관리
    │   ├── database.py              # MongoDB 연결 및 설정
    │   ├── models.py                # Pydantic 데이터 모델 정의
    │   ├── crud.py                  # MongoDB CRUD 연산
    │   └── mariadb_handler.py       # MariaDB 연결 및 LLM 결과 저장
    └── utils/                       # 공통 유틸리티
        ├── s3_handler.py            # AWS S3 파일 관리
        └── file_utils.py            # FFmpeg 비디오 처리
```

## 주요 기능 모듈

### 1. 감정 분석 모듈 (src/emotion/)

**목적**: EfficientNet 기반 실시간 감정 인식 및 면접 점수 산출

**핵심 기능**:
- 7가지 감정 분류: 기쁨, 당황, 분노, 불안, 상처, 슬픔, 중립
- 60점 만점 면접 감정 점수 계산
- Haar Cascade 기반 얼굴 검출
- 1초 간격 프레임 분석으로 성능 최적화
- 감정 안정성 및 일관성 평가

**주요 파일**:
- `analyzer.py`: 감정 분석 메인 클래스, 비디오 처리 및 점수 산출
- `models/efficientnet.py`: EfficientNet-B5 모델 구현
- `model_eff.pth`: 사전 훈련된 EfficientNet 가중치
- `face_classifier.xml`: 얼굴 검출용 Haar Cascade 모델

### 2. 시선 추적 모듈 (src/eye_tracking/)

**목적**: MediaPipe와 YOLO 기반 시선 방향 및 집중도 분석

**핵심 기능**:
- 시선 방향 감지 (center, left, right, up, down)
- 깜빡임 빈도 분석 (분당 15-20회 정상 기준)
- 집중도 점수 (15점), 안정성 점수 (15점), 깜빡임 점수 (10점)
- 총 40점 만점 시선 추적 점수 산출
- 부정행위 감지 (커닝, 대리시험 등)
- 머리 움직임 및 이상 행동 로깅

**주요 파일**:
- `analyzer.py`: 시선 추적 메인 클래스, 비디오 분석 워크플로우
- `gaze_analyzer.py`: 시선 방향 계산 및 분석 로직
- `yolo_face.py`: YOLO v8 기반 고정밀 얼굴 검출
- `calc/total_eval_calc.py`: 종합 점수 계산 알고리즘
- `logger.py`: 시선 추적 결과 구조화된 로깅

### 3. LLM 분석 모듈 (src/llm/)

**목적**: OpenAI GPT-4 기반 종합적인 면접 피드백 생성

**핵심 기능**:
- 감정 분석 + 시선 추적 결과 종합 해석
- YAML 기반 구조화된 프롬프트 관리
- 키워드 추출 및 카테고리별 평가
- API 크레딧 소진 시 Fallback 모드 지원
- 재시도 로직 및 Rate Limit 관리
- 배치 처리를 통한 효율적인 GPT 호출

**주요 파일**:
- `gpt_analyzer.py`: GPT-4 API 호출, 응답 파싱, 오류 처리
- `interview_prompts.yaml`: 프롬프트 템플릿 및 평가 기준
- `keyword_analyzer.py`: 키워드 기반 분석 및 동적 피드백 생성

### 4. 데이터베이스 모듈 (src/db/)

**목적**: 이중 데이터베이스 구조로 안전하고 효율적인 데이터 저장

**핵심 기능**:
- **MongoDB**: 원시 분석 데이터, 로그, 메타데이터 저장
- **MariaDB**: LLM 결과, 면접태도 요약, 통계 데이터 저장
- 연결 실패 시 자동 스킵 및 복구 메커니즘
- 안전한 CRUD 연산 및 트랜잭션 관리
- 데이터 무결성 보장

**주요 파일**:
- `database.py`: MongoDB 연결 관리 (연결 실패 시 스킵 기능)
- `mariadb_handler.py`: MariaDB 연결 풀 관리 및 면접태도 저장
- `crud.py`: 안전한 MongoDB CRUD 연산
- `models.py`: Pydantic 데이터 모델 및 검증 스키마

### 5. 유틸리티 모듈 (src/utils/)

**목적**: 공통 기능 및 외부 서비스 연동

**핵심 기능**:
- AWS S3 파일 업로드/다운로드 관리
- FFmpeg 기반 비디오 전처리 및 포맷 변환
- 임시 파일 관리 및 자동 정리
- 오류 처리 및 로깅

**주요 파일**:
- `s3_handler.py`: S3 파일 관리, 사용자별 영상 검색
- `file_utils.py`: 비디오 파일 처리 및 품질 최적화

## API 엔드포인트

### 메인 분석 API
- `POST /analysis/attitude`: 메인 영상 분석 API
  - S3 Object Key를 받아 감정+시선+GPT 종합 분석 수행
  - 폴링 방식으로 분석 완료까지 대기 후 결과 반환
  - 15분 타임아웃 설정

### 분석 결과 조회 API
- `GET /analysis/{analysis_id}`: 완전한 분석 결과 조회
- `GET /analysis/{analysis_id}/llm-comment`: GPT 피드백만 조회
- `GET /analysis/{analysis_id}/status`: 분석 진행 상태 조회
- `GET /analysis/recent`: 최근 분석 결과 목록 (기본 10개)

### 면접태도 평가 API
- `GET /interview-attitude/{user_id}`: 사용자별 모든 면접태도 조회
- `GET /interview-attitude/{user_id}/{question_num}`: 질문별 면접태도 조회

### S3 관리 API
- `GET /s3/available-users-questions`: S3에 저장된 영상 목록 조회
- `GET /s3/find-video/{user_id}/{question_num}`: 특정 사용자/질문 영상 검색

### 시스템 관리 API
- `GET /health`: 시스템 상태 확인 (MongoDB, MariaDB, S3 연결 상태)
- `POST /gpt-batch/trigger`: GPT 배치 처리 수동 실행
- `GET /gpt-batch/status`: GPT 배치 처리 상태 조회

### 개발 및 테스트 API
- `POST /test/yaml-keywords`: YAML 키워드 분석 기능 테스트
- `POST /test/mariadb-save`: MariaDB 저장 기능 테스트
- `GET /test/yaml-all-features`: 전체 YAML 기반 기능 테스트
- `POST /analysis/{analysis_id}/cancel`: 진행 중인 분석 취소

## 개발 환경 설정

### 시스템 요구사항
- Python 3.9.23
- FFmpeg (비디오 처리)
- MongoDB (선택사항, 연결 실패 시 스킵)
- MariaDB (면접태도 평가 저장)
- 최소 8GB RAM (AI 모델 로딩)
- CUDA 지원 GPU (선택사항, 성능 향상)

### Conda 환경 설정 (권장)
```bash
# 환경 생성 및 활성화
conda env create -f environment.yml
conda activate new_pipeline

# 서버 실행
cd src && uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 환경변수 설정 (.env)
```env
# MongoDB 설정 (선택사항)
MONGODB_CONNECTION_STRING=mongodb://localhost:27017/
MONGODB_DATABASE=video_analysis

# MariaDB 설정 (필수)
MARIADB_HOST=localhost
MARIADB_PORT=3306
MARIADB_USER=root
MARIADB_PASSWORD=your_password
MARIADB_DATABASE=interview_analysis

# AWS S3 설정 (필수)
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
S3_BUCKET_NAME=skala25a

# OpenAI 설정 (선택사항)
OPENAI_API_KEY=your_openai_api_key
OPENAI_ENABLED=true
```

## 분석 워크플로우

### 1단계: 영상 수신 및 전처리
- S3 Object Key 파싱 (user_id, question_num 추출)
- S3에서 영상 다운로드 및 임시 저장
- FFmpeg를 통한 비디오 전처리

### 2단계: 병렬 AI 분석
- **감정 분석**: EfficientNet 기반 7종 감정 분류 (60점 만점)
- **시선 추적**: MediaPipe + YOLO 기반 시선 방향 및 집중도 분석 (40점 만점)

### 3단계: 데이터 저장
- MongoDB: 원시 분석 데이터, 로그, 메타데이터
- MariaDB: 구조화된 면접태도 평가 점수

### 4단계: GPT 종합 분석
- 감정 + 시선 분석 결과 종합
- YAML 템플릿 기반 구조화된 피드백 생성
- 키워드 추출 및 개선 제안

### 5단계: 결과 반환
- 완전한 분석 결과 JSON 응답
- 폴링 방식으로 실시간 상태 업데이트

## 점수 체계

### 감정 분석 (60점 만점)
- **긍정 감정** (기쁨, 중립): 높은 점수 부여
- **부정 감정** (분노, 불안, 슬픔 등): 점수 차감
- **감정 안정성**: 일관된 감정 표현 평가
- **표현력**: 적절한 감정 변화 평가

### 시선 추적 (40점 만점)
- **집중도** (15점): 중앙 시선 유지 비율
- **안정성** (15점): 시선 변화 빈도 및 패턴
- **깜빡임** (10점): 분당 15-20회 정상 범위 기준

### 부정행위 감지
- **커닝 의심**: 시선 이탈 빈도 5회 이상
- **대리시험 의심**: 다중 얼굴 감지
- **이상 행동**: 급격한 머리 움직임, 화면 이탈

### GPT 종합 평가
- 정량적 점수를 바탕으로 한 정성적 피드백
- 강점 및 약점 키워드 추출
- 개인화된 면접 개선 제안
- 카테고리별 세부 분석

## 안정성 및 성능 최적화

### 오류 처리 메커니즘
- MongoDB 연결 실패 시 자동 스킵 및 계속 진행
- GPT API 크레딧 소진 시 Fallback 모드 전환
- 영상 처리 오류 시 안전한 예외 처리 및 로깅
- 자동 재시도 로직 (지수 백오프)

### 성능 최적화 기법
- 1초 간격 프레임 분석으로 처리 속도 3배 향상
- 비동기 처리 및 백그라운드 태스크 활용
- GPT 배치 처리로 API 효율성 증대
- 메모리 사용량 최적화 및 가비지 컬렉션

### 로깅 및 모니터링
- 사용자별/질문별 상세 분석 로그 파일 생성
- 분석 단계별 진행상황 실시간 추적
- 오류 발생 시 상세 스택 트레이스 기록
- 성능 메트릭 및 처리 시간 측정

## 데이터베이스 스키마

### MongoDB Collections
- `analysis_results`: 완전한 분석 결과 및 메타데이터
- `llm_comments`: GPT 피드백 및 키워드 분석 결과

### MariaDB Tables
- `answer_score`: 면접태도 점수 (emotion_score, eye_score)
- `interview_answer`: 면접 답변 메타데이터
- `answer_category_result`: GPT 키워드 분석 결과

## 확장성 및 유지보수

### 모듈화 설계
- 각 분석 모듈이 독립적으로 작동
- 새로운 AI 모델 추가 용이
- 데이터베이스 백엔드 교체 가능

### 설정 관리
- 환경변수를 통한 동적 설정
- YAML 파일 기반 프롬프트 관리
- 버전별 호환성 유지

### 코딩 표준
- 한글 주석 및 커밋 메시지 템플릿
- Pydantic 기반 타입 안전성
- 비동기 프로그래밍 패턴 준수

이 시스템은 면접자의 태도를 다각도로 분석하여 객관적이고 구체적인 피드백을 제공하는 통합 AI 면접 평가 솔루션으로, 높은 정확도와 안정성을 보장합니다.



