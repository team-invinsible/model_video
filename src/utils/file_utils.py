import os
import asyncio
import subprocess
from typing import Optional, Tuple
import tempfile
from pathlib import Path

class FileProcessor:
    """ffmpeg를 사용한 영상/음성 파일 처리 클래스"""
    
    def __init__(self):
        """FileProcessor 초기화"""
        self._check_ffmpeg()
    
    def _check_ffmpeg(self):
        """ffmpeg 설치 여부 확인"""
        try:
            subprocess.run(['ffmpeg', '-version'], 
                         capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise Exception("ffmpeg가 설치되어 있지 않습니다. ffmpeg를 설치해주세요.")
    
    async def process_video(self, video_path: str, 
                          extract_audio: bool = False) -> str:
        """
        비디오 파일을 처리합니다.
        
        Args:
            video_path: 입력 비디오 파일 경로
            extract_audio: 오디오 추출 여부
            
        Returns:
            str: 처리된 비디오 파일 경로
        """
        try:
            # 파일 존재 확인
            if not os.path.exists(video_path):
                raise Exception(f"비디오 파일을 찾을 수 없습니다: {video_path}")
            
            # webm 파일을 mp4로 변환 (분석 호환성을 위해)
            if video_path.lower().endswith('.webm'):
                converted_path = await self._convert_webm_to_mp4(video_path)
                return converted_path
            
            return video_path
            
        except Exception as e:
            raise Exception(f"비디오 처리 중 오류 발생: {str(e)}")
    
    async def _convert_webm_to_mp4(self, webm_path: str) -> str:
        """
        webm 파일을 mp4로 변환합니다.
        
        Args:
            webm_path: webm 파일 경로
            
        Returns:
            str: 변환된 mp4 파일 경로
        """
        try:
            # 출력 파일 경로 생성
            base_name = os.path.splitext(webm_path)[0]
            mp4_path = f"{base_name}.mp4"
            
            # ffmpeg 명령어 구성
            cmd = [
                'ffmpeg',
                '-i', webm_path,
                '-c:v', 'libx264',  # 비디오 코덱
                '-c:a', 'aac',      # 오디오 코덱
                '-preset', 'fast',   # 인코딩 속도 우선
                '-y',               # 덮어쓰기 허용
                mp4_path
            ]
            
            # 비동기적으로 변환 실행
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._run_ffmpeg_command, cmd)
            
            return mp4_path
            
        except Exception as e:
            raise Exception(f"webm to mp4 변환 오류: {str(e)}")
    
    async def extract_audio(self, video_path: str, 
                          output_format: str = 'wav') -> str:
        """
        비디오에서 오디오를 추출합니다.
        
        Args:
            video_path: 비디오 파일 경로
            output_format: 출력 오디오 형식 (wav, mp3 등)
            
        Returns:
            str: 추출된 오디오 파일 경로
        """
        try:
            # 출력 파일 경로 생성
            base_name = os.path.splitext(video_path)[0]
            audio_path = f"{base_name}.{output_format}"
            
            # ffmpeg 명령어 구성
            cmd = [
                'ffmpeg',
                '-i', video_path,
                '-vn',              # 비디오 스트림 제외
                '-acodec', 'pcm_s16le' if output_format == 'wav' else 'mp3',
                '-ar', '44100',     # 샘플링 레이트
                '-ac', '2',         # 스테레오
                '-y',               # 덮어쓰기 허용
                audio_path
            ]
            
            # 비동기적으로 추출 실행
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._run_ffmpeg_command, cmd)
            
            return audio_path
            
        except Exception as e:
            raise Exception(f"오디오 추출 오류: {str(e)}")
    
    async def get_video_info(self, video_path: str) -> dict:
        """
        비디오 파일의 정보를 가져옵니다.
        
        Args:
            video_path: 비디오 파일 경로
            
        Returns:
            dict: 비디오 정보 (duration, fps, resolution 등)
        """
        try:
            cmd = [
                'ffprobe',
                '-v', 'quiet',
                '-print_format', 'json',
                '-show_format',
                '-show_streams',
                video_path
            ]
            
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, self._run_ffprobe_command, cmd
            )
            
            import json
            info = json.loads(result)
            
            # 비디오 스트림 찾기
            video_stream = None
            for stream in info.get('streams', []):
                if stream.get('codec_type') == 'video':
                    video_stream = stream
                    break
            
            if not video_stream:
                raise Exception("비디오 스트림을 찾을 수 없습니다.")
            
            return {
                'duration': float(info.get('format', {}).get('duration', 0)),
                'fps': eval(video_stream.get('r_frame_rate', '0/1')),
                'width': video_stream.get('width', 0),
                'height': video_stream.get('height', 0),
                'codec': video_stream.get('codec_name', ''),
                'bitrate': int(info.get('format', {}).get('bit_rate', 0))
            }
            
        except Exception as e:
            raise Exception(f"비디오 정보 조회 오류: {str(e)}")
    
    def _run_ffmpeg_command(self, cmd: list):
        """ffmpeg 명령어를 동기적으로 실행합니다."""
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout
        except subprocess.CalledProcessError as e:
            raise Exception(f"ffmpeg 실행 오류: {e.stderr}")
    
    def _run_ffprobe_command(self, cmd: list) -> str:
        """ffprobe 명령어를 동기적으로 실행합니다."""
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout
        except subprocess.CalledProcessError as e:
            raise Exception(f"ffprobe 실행 오류: {e.stderr}")
    
    async def resize_video(self, video_path: str, 
                          width: int, height: int) -> str:
        """
        비디오 크기를 조정합니다.
        
        Args:
            video_path: 입력 비디오 파일 경로
            width: 출력 너비
            height: 출력 높이
            
        Returns:
            str: 크기 조정된 비디오 파일 경로
        """
        try:
            # 출력 파일 경로 생성
            base_name = os.path.splitext(video_path)[0]
            resized_path = f"{base_name}_resized_{width}x{height}.mp4"
            
            # ffmpeg 명령어 구성
            cmd = [
                'ffmpeg',
                '-i', video_path,
                '-vf', f'scale={width}:{height}',
                '-c:a', 'copy',     # 오디오는 복사
                '-y',               # 덮어쓰기 허용
                resized_path
            ]
            
            # 비동기적으로 크기 조정 실행
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._run_ffmpeg_command, cmd)
            
            return resized_path
            
        except Exception as e:
            raise Exception(f"비디오 크기 조정 오류: {str(e)}")
    
    def cleanup_temp_files(self, *file_paths: str):
        """임시 파일들을 정리합니다."""
        for file_path in file_paths:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except Exception:
                pass  # 정리 실패는 무시 