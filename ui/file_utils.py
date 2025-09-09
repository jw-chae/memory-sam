import os
import glob
import tempfile
import shutil
import subprocess
import sys
from typing import List, Dict, Tuple, Any, Optional, Union
from pathlib import Path

class FileManager:
    """파일 및 폴더 처리를 위한 유틸리티 클래스"""
    
    @staticmethod
    def browse_directory() -> str:
        """
        시스템의 디렉토리 브라우저를 열고 선택된 경로를 반환
        
        Returns:
            선택된 디렉토리 경로 (또는 취소 시 빈 문자열)
        """
        # 1) tkinter를 별도 서브프로세스로 실행해 메인 스레드 제약을 회피
        try:
            py = sys.executable or "python3"
            code = (
                "import tkinter as tk;"
                "from tkinter import filedialog;"
                "root=tk.Tk();root.withdraw();"
                "p=filedialog.askdirectory(title='폴더 선택');"
                "print(p,end='')"
            )
            result = subprocess.run([py, "-c", code], capture_output=True, text=True)
            if result.returncode == 0:
                path = result.stdout.strip()
                if path:
                    return path
        except Exception as e:
            print(f"tkinter(subprocess) 폴더 브라우저 실패: {e}")

        # 2) 리눅스 환경에서 zenity가 있으면 사용
        try:
            if shutil.which("zenity"):
                result = subprocess.run(['zenity', '--file-selection', '--directory'],
                                        capture_output=True, text=True)
                if result.returncode == 0:
                    return result.stdout.strip()
        except Exception as e:
            print(f"zenity 기반 폴더 브라우저 실패: {e}")

        # 3) KDE 환경에서 kdialog가 있으면 사용
        try:
            if shutil.which("kdialog"):
                result = subprocess.run(['kdialog', '--getexistingdirectory'],
                                        capture_output=True, text=True)
                if result.returncode == 0:
                    return result.stdout.strip()
        except Exception as e:
            print(f"kdialog 기반 폴더 브라우저 실패: {e}")

        # 4) 환경에 GUI가 없을 수 있음: 안전한 폴백으로 현재 작업 디렉토리 반환
        #    사용자가 텍스트 박스에서 경로를 수정할 수 있도록 기본값을 채워줌
        cwd = os.getcwd()
        print("폴더 브라우저 열기 실패: GUI 미지원 환경으로 판단. 현재 작업 디렉토리를 기본값으로 반환합니다:", cwd)
        return cwd
    
    @staticmethod
    def collect_image_files(folder_path: str) -> List[str]:
        """
        폴더에서 이미지 파일 목록 수집
        
        Args:
            folder_path: 이미지를 검색할 폴더 경로
            
        Returns:
            이미지 파일 경로 목록
        """
        if not folder_path or not os.path.isdir(folder_path):
            return []
        
        image_extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.gif")
        image_files = []
        
        for ext in image_extensions:
            pattern = os.path.join(folder_path, ext)
            image_files.extend(glob.glob(pattern))
            
            # 하위 폴더 검색
            pattern = os.path.join(folder_path, "**", ext)
            image_files.extend(glob.glob(pattern, recursive=True))
        
        return sorted(image_files)
    
    @staticmethod
    def create_temp_directory() -> Tuple[tempfile.TemporaryDirectory, str]:
        """
        임시 디렉토리 생성
        
        Returns:
            (임시 디렉토리 객체, 임시 디렉토리 경로)
        """
        temp_dir = tempfile.TemporaryDirectory()
        return temp_dir, temp_dir.name
    
    @staticmethod
    def copy_files_to_temp(file_paths: List[str], temp_dir: str) -> List[str]:
        """
        파일을 임시 디렉토리에 복사
        
        Args:
            file_paths: 복사할 파일 경로 목록
            temp_dir: 임시 디렉토리 경로
            
        Returns:
            임시 디렉토리의 파일 경로 목록
        """
        temp_paths = []
        
        for file_path in file_paths:
            file_name = os.path.basename(file_path)
            temp_path = os.path.join(temp_dir, file_name)
            shutil.copy2(file_path, temp_path)
            temp_paths.append(temp_path)
        
        return temp_paths