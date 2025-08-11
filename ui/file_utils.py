import os
import glob
import tempfile
import shutil
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
        try:
            import tkinter as tk
            from tkinter import filedialog
            
            # tkinter 초기화 및 숨기기
            root = tk.Tk()
            root.withdraw()
            
            # 폴더 선택 대화상자 열기
            folder_path = filedialog.askdirectory(title="폴더 선택")
            
            # 취소 시 빈 문자열 반환
            return folder_path
        except Exception as e:
            print(f"폴더 브라우저 열기 실패: {e}")
            return ""
    
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