# ----
# 작성목적 : S3 파일 업로드/다운로드 핸들러
# 작성일 : 2025-06-14

# 변경사항 내역 (날짜 | 변경목적 | 변경내용 | 작성자 순으로 기입)
# 2025-06-14 | 최초 구현 | FastAPI 베스트 프랙티스에 따른 구조로 재구성 | 이재인
# 2025-06-16 | 기능 확장 | 업로드 기능 및 메타데이터 관리 추가 | 이재인
# ----

import boto3
import os
import asyncio
from typing import Optional, Dict, Any, List
from botocore.exceptions import ClientError, NoCredentialsError
import aiofiles
import tempfile
from datetime import datetime
import mimetypes

class S3Handler:
    """S3에서 webm 파일을 다운로드하는 핸들러"""
    
    def __init__(self, aws_access_key_id: Optional[str] = None, 
                 aws_secret_access_key: Optional[str] = None,
                 region_name: str = 'ap-northeast-2'):
        """
        S3 클라이언트 초기화
        
        Args:
            aws_access_key_id: AWS 액세스 키 (환경변수에서 자동 로드 가능)
            aws_secret_access_key: AWS 시크릿 키 (환경변수에서 자동 로드 가능)
            region_name: AWS 리전
        """
        try:
            if aws_access_key_id and aws_secret_access_key:
                self.s3_client = boto3.client(
                    's3',
                    aws_access_key_id=aws_access_key_id,
                    aws_secret_access_key=aws_secret_access_key,
                    region_name=region_name
                )
            else:
                # 환경변수나 IAM 역할에서 자동으로 인증 정보 로드
                self.s3_client = boto3.client('s3', region_name=region_name)
                
        except NoCredentialsError:
            raise Exception("AWS 인증 정보를 찾을 수 없습니다. 환경변수나 IAM 역할을 설정해주세요.")
    
    async def download_file(self, bucket_name: str, s3_key: str, 
                          local_dir: str) -> str:
        """
        S3에서 파일을 비동기적으로 다운로드합니다.
        
        Args:
            bucket_name: S3 버킷 이름
            s3_key: S3 객체 키
            local_dir: 로컬 저장 디렉토리
            
        Returns:
            str: 다운로드된 파일의 로컬 경로
        """
        try:
            # 파일명 추출
            filename = os.path.basename(s3_key)
            if not filename:
                filename = "downloaded_video.webm"
            
            # 로컬 파일 경로
            local_path = os.path.join(local_dir, filename)
            
            # 비동기적으로 파일 다운로드
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None, 
                self._download_file_sync, 
                bucket_name, s3_key, local_path
            )
            
            return local_path
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'NoSuchBucket':
                raise Exception(f"S3 버킷을 찾을 수 없습니다: {bucket_name}")
            elif error_code == 'NoSuchKey':
                raise Exception(f"S3 객체를 찾을 수 없습니다: {s3_key}")
            else:
                raise Exception(f"S3 다운로드 오류: {str(e)}")
        except Exception as e:
            raise Exception(f"파일 다운로드 중 오류 발생: {str(e)}")
    
    def _download_file_sync(self, bucket_name: str, s3_key: str, local_path: str):
        """동기적으로 파일을 다운로드하는 내부 메서드"""
        self.s3_client.download_file(bucket_name, s3_key, local_path)
    
    async def check_file_exists(self, bucket_name: str, s3_key: str) -> bool:
        """
        S3에서 파일 존재 여부를 확인합니다.
        
        Args:
            bucket_name: S3 버킷 이름
            s3_key: S3 객체 키
            
        Returns:
            bool: 파일 존재 여부
        """
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: self.s3_client.head_object(Bucket=bucket_name, Key=s3_key)
            )
            return True
        except ClientError:
            return False
    
    async def get_file_info(self, bucket_name: str, s3_key: str) -> dict:
        """
        S3 파일의 메타데이터를 가져옵니다.
        
        Args:
            bucket_name: S3 버킷 이름
            s3_key: S3 객체 키
            
        Returns:
            dict: 파일 메타데이터
        """
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.s3_client.head_object(Bucket=bucket_name, Key=s3_key)
            )
            
            return {
                'size': response.get('ContentLength', 0),
                'last_modified': response.get('LastModified'),
                'content_type': response.get('ContentType'),
                'etag': response.get('ETag')
            }
        except ClientError as e:
            raise Exception(f"파일 정보 조회 오류: {str(e)}")
    
    async def upload_file(self, local_path: str, bucket_name: str, s3_key: str, 
                         metadata: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        로컬 파일을 S3에 업로드합니다.
        
        Args:
            local_path: 업로드할 로컬 파일 경로
            bucket_name: S3 버킷 이름
            s3_key: S3 객체 키
            metadata: 파일 메타데이터
            
        Returns:
            Dict[str, Any]: 업로드 결과 정보
        """
        try:
            # 파일 존재 확인
            if not os.path.exists(local_path):
                raise Exception(f"로컬 파일을 찾을 수 없습니다: {local_path}")
            
            # 파일 정보 수집
            file_size = os.path.getsize(local_path)
            content_type = mimetypes.guess_type(local_path)[0] or 'application/octet-stream'
            
            # 메타데이터 설정
            upload_metadata = {
                'uploaded_at': datetime.now().isoformat(),
                'file_size': str(file_size),
                'content_type': content_type
            }
            if metadata:
                upload_metadata.update(metadata)
            
            # 비동기적으로 파일 업로드
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                self._upload_file_sync,
                local_path, bucket_name, s3_key, content_type, upload_metadata
            )
            
            return {
                'bucket': bucket_name,
                's3_key': s3_key,
                'file_size': file_size,
                'content_type': content_type,
                'uploaded_at': upload_metadata['uploaded_at']
            }
            
        except ClientError as e:
            raise Exception(f"S3 업로드 오류: {str(e)}")
        except Exception as e:
            raise Exception(f"파일 업로드 중 오류 발생: {str(e)}")
    
    def _upload_file_sync(self, local_path: str, bucket_name: str, s3_key: str,
                         content_type: str, metadata: Dict[str, str]):
        """동기적으로 파일을 업로드하는 내부 메서드"""
        self.s3_client.upload_file(
            local_path, bucket_name, s3_key,
            ExtraArgs={
                'ContentType': content_type,
                'Metadata': metadata
            }
        )
    
    async def delete_file(self, bucket_name: str, s3_key: str) -> bool:
        """
        S3에서 파일을 삭제합니다.
        
        Args:
            bucket_name: S3 버킷 이름
            s3_key: S3 객체 키
            
        Returns:
            bool: 삭제 성공 여부
        """
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: self.s3_client.delete_object(Bucket=bucket_name, Key=s3_key)
            )
            return True
        except ClientError as e:
            print(f"S3 파일 삭제 오류: {str(e)}")
            return False
    
    async def list_available_users_and_questions(self, bucket_name: str, 
                                               base_prefix: str = "team12/interview_video/") -> Dict[str, List[str]]:
        """
        S3 버킷에서 사용 가능한 사용자 ID와 질문 번호 목록을 스캔합니다.
        
        Args:
            bucket_name: S3 버킷 이름
            base_prefix: 기본 경로 (예: "team12/interview_video/")
            
        Returns:
            Dict[str, List[str]]: {user_id: [question_nums]} 형태의 딕셔너리
        """
        try:
            loop = asyncio.get_event_loop()
            user_questions = await loop.run_in_executor(
                None, 
                self._scan_user_questions_sync, 
                bucket_name, base_prefix
            )
            
            return user_questions
            
        except ClientError as e:
            raise Exception(f"S3 디렉토리 스캔 오류: {str(e)}")
    
    def _scan_user_questions_sync(self, bucket_name: str, base_prefix: str) -> Dict[str, List[str]]:
        """동기적으로 사용자와 질문 목록을 스캔하는 내부 메서드"""
        user_questions = {}
        
        try:
            # S3에서 모든 객체 나열
            paginator = self.s3_client.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=bucket_name, Prefix=base_prefix)
            
            for page in pages:
                if 'Contents' not in page:
                    continue
                    
                for obj in page['Contents']:
                    key = obj['Key']
                    
                    # base_prefix 제거하고 경로 분석
                    relative_path = key[len(base_prefix):]
                    
                    # 경로가 userId/questionNum/파일명 구조인지 확인
                    path_parts = relative_path.split('/')
                    if len(path_parts) >= 3:  # userId/questionNum/filename 최소 3개 부분
                        user_id = path_parts[0]
                        question_num = path_parts[1]
                        filename = path_parts[2]
                        
                        # 영상 파일인지 확인 (확장자 체크)
                        if filename.lower().endswith(('.mp4', '.webm', '.avi', '.mov')):
                            if user_id not in user_questions:
                                user_questions[user_id] = []
                            
                            if question_num not in user_questions[user_id]:
                                user_questions[user_id].append(question_num)
            
            # 질문 번호 정렬
            for user_id in user_questions:
                user_questions[user_id].sort()
                
            return user_questions
            
        except Exception as e:
            print(f"S3 스캔 중 오류: {e}")
            return {}

    async def find_video_file(self, bucket_name: str, user_id: str, question_num: str,
                            base_prefix: str = "team12/interview_video/") -> Optional[str]:
        """
        특정 사용자와 질문에 해당하는 영상 파일을 찾습니다.
        
        Args:
            bucket_name: S3 버킷 이름
            user_id: 사용자 ID
            question_num: 질문 번호
            base_prefix: 기본 경로
            
        Returns:
            Optional[str]: 찾은 영상 파일의 S3 키, 없으면 None
        """
        try:
            loop = asyncio.get_event_loop()
            video_key = await loop.run_in_executor(
                None,
                self._find_video_file_sync,
                bucket_name, user_id, question_num, base_prefix
            )
            
            return video_key
            
        except Exception as e:
            print(f"영상 파일 검색 오류: {e}")
            return None
    
    def _find_video_file_sync(self, bucket_name: str, user_id: str, question_num: str, base_prefix: str) -> Optional[str]:
        """동기적으로 영상 파일을 찾는 내부 메서드"""
        try:
            search_prefix = f"{base_prefix}{user_id}/{question_num}/"
            
            response = self.s3_client.list_objects_v2(
                Bucket=bucket_name,
                Prefix=search_prefix
            )
            
            if 'Contents' not in response:
                return None
            
            # 영상 파일 확장자 우선순위
            video_extensions = ['.mp4', '.webm', '.avi', '.mov']
            
            for obj in response['Contents']:
                key = obj['Key']
                filename = key.split('/')[-1]
                
                for ext in video_extensions:
                    if filename.lower().endswith(ext):
                        return key
            
            return None
            
        except Exception as e:
            print(f"영상 파일 검색 중 오류: {e}")
            return None

    async def generate_presigned_url(self, bucket_name: str, s3_key: str, 
                                   expiration: int = 3600) -> str:
        """
        S3 파일에 대한 사전 서명된 URL을 생성합니다.
        
        Args:
            bucket_name: S3 버킷 이름
            s3_key: S3 객체 키
            expiration: URL 만료 시간 (초)
            
        Returns:
            str: 사전 서명된 URL
        """
        try:
            loop = asyncio.get_event_loop()
            url = await loop.run_in_executor(
                None,
                lambda: self.s3_client.generate_presigned_url(
                    'get_object',
                    Params={'Bucket': bucket_name, 'Key': s3_key},
                    ExpiresIn=expiration
                )
            )
            return url
        except ClientError as e:
            raise Exception(f"사전 서명된 URL 생성 오류: {str(e)}")

# 환경변수에서 AWS 설정을 로드하는 헬퍼 함수
def create_s3_handler_from_env() -> S3Handler:
    """환경변수에서 AWS 설정을 로드하여 S3Handler를 생성합니다."""
    aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
    aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
    region_name = os.getenv('AWS_DEFAULT_REGION', 'ap-northeast-2')
    
    return S3Handler(
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=region_name
    ) 