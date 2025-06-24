# 🎥 통합 영상 분석 API

S3에 저장된 webm 영상을 분석하여 **감정 분석**과 **시선 추적**을 수행하고, **GPT를 통한 종합적인 면접 피드백**을 제공하는 FastAPI 기반 서비스입니다.

## 🌟 주요 기능

- **🎭 감정 분석**: EfficientNet 기반 실시간 감정 인식 및 면접 점수 산출
- **👁️ 시선 추적**: MediaPipe와 YOLO를 활용한 시선 방향 및 집중도 분석  
- **🤖 GPT 피드백**: OpenAI GPT-4를 통한 종합적인 면접 평가 및 개선 제안
- **⚡ 비동기 처리**: FastAPI 백그라운드 태스크를 통한 효율적인 영상 처리
- **🗄️ 이중 데이터베이스**: MongoDB(분석 데이터) + MariaDB(LLM 결과) 구조

## 📁 프로젝트 구조

### 🏗️ 전체 아키텍처
이 프로젝트는 **모듈형 구조**로 설계되어 각 기능이 독립적으로 작동합니다:

```
model_video/                           # 프로젝트 루트 디렉토리
├── 📄 README.md                       # 프로젝트 문서 (현재 파일)
├── 📄 requirements.txt                # Python 의존성 패키지 목록 (레거시)
├── 📄 environment.yml                 # Conda 환경 설정 파일 (권장)
├── 🚀 run_server.sh                   # 서버 실행 스크립트
├── 📊 logs/                          # 분석 로그 저장소
│   └── recalib_log.jsonl             # 재보정 로그
└── 📦 src/                           # 소스코드 메인 디렉토리
    ├── 🎯 main.py                    # FastAPI 애플리케이션 진입점
    ├── 📁 temp_uploads/              # 임시 파일 저장 공간
    ├── 🛠️ utils/                     # 공통 유틸리티 모듈
    │   ├── s3_handler.py             # AWS S3 파일 다운로드/업로드
    │   ├── file_utils.py             # FFmpeg 비디오 처리
    │   └── __init__.py
    ├── 🗃️ db/                        # 데이터베이스 관리 모듈
    │   ├── database.py               # MongoDB 연결 및 설정
    │   ├── models.py                 # Pydantic 데이터 모델 정의
    │   ├── crud.py                   # MongoDB CRUD 연산
    │   ├── mariadb_handler.py        # MariaDB 연결 및 LLM 결과 저장
    │   └── __init__.py
    ├── 😊 emotion/                   # 감정 분석 모듈
    │   ├── analyzer.py               # 감정 분석 메인 클래스
    │   ├── face_classifier.xml       # Haar Cascade 얼굴 검출 모델
    │   ├── utils.py                  # 감정 분석 유틸리티
    │   └── models/                   # 딥러닝 모델 컬렉션
    │       ├── cnn.py                # 기본 CNN 모델
    │       ├── efficientnet.py       # EfficientNet 구현
    │       ├── resnet.py             # ResNet 모델
    │       ├── vgg.py                # VGG 모델
    │       └── utils.py              # 모델 공통 유틸리티
    ├── 👀 eye_tracking/              # 시선 추적 모듈
    │   ├── analyzer.py               # 시선 추적 메인 클래스
    │   ├── eye.py                    # 눈 감지 및 추적
    │   ├── face.py                   # 얼굴 감지
    │   ├── gaze_analyzer.py          # 시선 방향 분석
    │   ├── logger.py                 # 분석 결과 로깅
    │   ├── utils.py                  # 시선 추적 유틸리티
    │   └── calc/                     # 점수 계산 모듈
    │       ├── cheat_cal.py          # 부정행위 감지 계산
    │       ├── total_eval_calc.py    # 전체 평가 점수 계산
    │       └── *.jsonl               # 계산 결과 저장
    └── 🧠 llm/                       # LLM 분석 모듈
        ├── gpt_analyzer.py           # OpenAI GPT 피드백 생성
        ├── keyword_analyzer.py       # 키워드 추출 및 분석
        └── interview_prompts.yaml    # GPT 프롬프트 템플릿
```

### 🔍 각 모듈의 역할

#### 1. **main.py** - FastAPI 애플리케이션
- HTTP API 엔드포인트 정의
- 백그라운드 태스크 관리
- 요청/응답 처리

#### 2. **emotion/** - 감정 분석 시스템
- EfficientNet 딥러닝 모델로 7가지 감정 분류
- 실시간 얼굴 검출 및 감정 추론
- 60점 만점 면접 점수 산출

#### 3. **eye_tracking/** - 시선 추적 시스템
- MediaPipe와 YOLO 기반 얼굴/눈 검출
- 시선 방향, 깜박임 빈도 분석
- 집중도 및 안정성 점수 계산

#### 4. **db/** - 데이터베이스 관리
- **MongoDB**: 원시 분석 데이터 저장
- **MariaDB**: GPT 피드백 및 요약 데이터 저장

#### 5. **llm/** - 인공지능 피드백
- GPT-4 API를 통한 종합 분석
- 키워드 기반 분석 및 카테고리별 점수
- 개인화된 면접 개선 제안

## 🛠️ 설치 및 설정

### 1. 사전 요구사항

```bash
# 시스템 패키지 설치 (macOS)
brew install ffmpeg
brew install mongodb-community
brew install mariadb

# 시스템 패키지 설치 (Ubuntu/Debian)
sudo apt update
sudo apt install ffmpeg mongodb mariadb-server python3-dev
```

### 2. Conda 환경 설정 (권장)

```bash
# 프로젝트 디렉토리로 이동
cd model_video

# environment.yml을 사용한 자동 환경 구성
conda env create -f environment.yml

# 환경 활성화
conda activate new_pipeline

# 설치 확인
conda list | grep opencv
conda list | grep fastapi
```

### 3. 대안: 수동 환경 설정

```bash
# Conda 환경 생성 (Python 3.9 권장)
conda create -n new_pipeline python=3.9 -y

# 환경 활성화
conda activate new_pipeline

# 기본 패키지 설치
conda install -c conda-forge pip setuptools wheel -y

# pip를 통한 전체 패키지 설치
pip install -r requirements.txt
```

### 4. 환경변수 설정

프로젝트 루트에 `.env` 파일을 생성하세요:

```env
# MongoDB 설정
MONGODB_URL=mongodb://localhost:27017
MONGODB_DATABASE=interview_analysis

# MariaDB 설정  
MARIADB_HOST=localhost
MARIADB_PORT=3306
MARIADB_USER=root
MARIADB_PASSWORD=your_password
MARIADB_DATABASE=interview_analysis

# AWS S3 설정
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_DEFAULT_REGION=ap-northeast-2

# OpenAI GPT 설정
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-4

# 로그 레벨 (선택사항)
LOG_LEVEL=INFO
```

### 5. 데이터베이스 초기화

**MongoDB 시작:**
```bash
# macOS
brew services start mongodb-community

# Ubuntu/Debian  
sudo systemctl start mongodb
sudo systemctl enable mongodb
```

**MariaDB 시작 및 설정:**
```bash
# macOS
brew services start mariadb

# Ubuntu/Debian
sudo systemctl start mariadb
sudo systemctl enable mariadb

# 초기 설정 (비밀번호 설정)
sudo mysql_secure_installation

# 데이터베이스 생성
mysql -u root -p
CREATE DATABASE interview_analysis;
```

## 🚀 실행 방법

### 개발 환경에서 실행

```bash
# 1. Conda 환경 활성화
conda activate new_pipeline

# 2. 소스코드 디렉토리로 이동
cd src

# 3. FastAPI 서버 실행 (개발 모드)
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 프로덕션 환경에서 실행

```bash
# 자동 실행 스크립트 사용
chmod +x run_server.sh
./run_server.sh

# 또는 직접 실행
cd src
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

### 🌐 서버 접속 확인

브라우저에서 다음 URL에 접속하여 서버 상태를 확인하세요:

- **API 문서**: http://localhost:8000/docs
- **헬스체크**: http://localhost:8000/health
- **대안 API 문서**: http://localhost:8000/redoc

## 📚 API 사용 가이드

### 1. 영상 분석 요청 🎬

**요청 예시:**
```bash
curl -X POST "http://localhost:8000/analyze" \
     -H "Content-Type: application/json" \
     -d '{
       "s3_bucket": "your-bucket-name",
       "s3_key": "interviews/2025/01/27/interview_001.webm",
       "user_id": "user123",
       "session_id": "session456"
     }'
```

**응답:**
```json
{
  "analysis_id": "analysis_20250127_143022_user123",
  "status": "processing",
  "message": "영상 분석이 시작되었습니다. 약 2-3분 후 결과를 확인하세요.",
  "estimated_completion": "2025-01-27T14:33:22Z"
}
```

### 2. 분석 결과 조회 📊

```bash
curl "http://localhost:8000/analysis/analysis_20250127_143022_user123"
```

**응답 예시:**
```json
{
  "analysis_id": "analysis_20250127_143022_user123",
  "status": "completed",
  "emotion_analysis": {
    "interview_score": 45.2,
    "emotions": {
      "happy": 0.15,
      "neutral": 0.45,
      "sad": 0.10,
      "angry": 0.05,
      "fear": 0.15,
      "disgust": 0.05,
      "surprise": 0.05
    }
  },
  "eye_tracking": {
    "attention_score": 78.5,
    "gaze_stability": 0.82,
    "blink_frequency": 15.3
  }
}
```

### 3. GPT 피드백 조회 🤖

```bash
curl "http://localhost:8000/analysis/analysis_20250127_143022_user123/llm-comment"
```

### 4. 최근 분석 목록 📝

```bash
curl "http://localhost:8000/analysis/recent?limit=10"
```

## 🔧 분석 과정 상세

### 1단계: 데이터 수집 📥
- S3에서 webm 영상 파일 다운로드
- FFmpeg를 통한 비디오 전처리

### 2단계: 병렬 분석 ⚡
- **감정 분석**: 프레임별 얼굴 검출 → EfficientNet 감정 분류
- **시선 추적**: MediaPipe 얼굴 랜드마크 → 시선 벡터 계산


### 3단계: 데이터 저장 💾
- MongoDB: 원시 분석 데이터 및 메타데이터
- MariaDB: 집계된 점수 및 통계 (새로운 테이블 구조)

### 4단계: AI 피드백 🧠
- GPT-4 API 호출로 종합 평가
- YAML 기반 구조화된 출력
- 키워드별 카테고리 점수 분석
- 개인화된 개선 제안 생성

## 📊 결과 해석 가이드

### 감정 분석 점수 (60점 만점)
- **50-60점**: 우수한 감정 표현
- **40-49점**: 양호한 수준
- **30-39점**: 보통 수준  
- **30점 미만**: 개선 필요

### 시선 추적 점수 (100점 만점)
- **80-100점**: 매우 안정적인 시선
- **60-79점**: 양호한 집중도
- **40-59점**: 보통 수준
- **40점 미만**: 집중도 개선 필요

### 음성 분석 점수

- **말하기 속도**: 적절한 템포 유지 여부

## 🚨 문제 해결

### 자주 발생하는 오류들

1. **Conda 환경 문제**
```bash
# 환경 재생성
conda env remove -n new_pipeline
conda env create -f environment.yml
conda activate new_pipeline
```

2. **패키지 버전 충돌**
```bash
# 환경 업데이트
conda env update -f environment.yml --prune
```

3. **모델 파일 없음 오류**
```bash
# EfficientNet 모델 파일 확인
ls -la src/emotion/model_eff.pth
# 파일이 없다면 별도로 다운로드 필요
```

4. **MongoDB 연결 오류**
```bash
# MongoDB 서비스 상태 확인
brew services list | grep mongodb  # macOS
sudo systemctl status mongodb      # Linux
```

5. **메모리 부족 오류**
```bash
# 시스템 메모리 확인 (최소 8GB 권장)
free -h  # Linux
vm_stat  # macOS
```

6. **OpenAI API 키 오류**
```bash
# .env 파일의 API 키 확인
cat .env | grep OPENAI_API_KEY
```



## 🤝 개발에 참여하기

```bash
# 1. 저장소 포크 및 클론
git clone https://github.com/your-username/model_video.git

# 2. 새 기능 브랜치 생성
git checkout -b feature/amazing-feature

# 3. 변경사항 커밋 (한글 템플릿 사용)
git commit -m 'feat(emotion): 새로운 감정 분류 모델 추가'

# 4. 브랜치 푸시
git push origin feature/amazing-feature
```

## 📝 커밋 메시지 템플릿

```
<type>(<scope>): <subject>

[상세 설명 - 필요할 때만 작성]

[Footer - 선택, 예: Resolves: #123]
```

**type 예시:**
- `feat`: 새로운 기능 추가
- `fix`: 버그 수정
- `docs`: 문서 수정
- `style`: 코드 포맷팅
- `refactor`: 코드 리팩토링
- `test`: 테스트 코드 추가/수정
- `chore`: 빌드, 패키지 매니저 설정 등



