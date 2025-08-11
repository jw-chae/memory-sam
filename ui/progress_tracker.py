import time
from typing import List, Dict, Tuple, Any, Optional, Union, Callable
from tqdm import tqdm

class ProgressTracker:
    """처리 진행 상황을 추적하는 클래스"""
    
    def __init__(self, total_steps: int = 100):
        """
        프로그레스 트래커 초기화
        
        Args:
            total_steps: 전체 단계 수
        """
        self.total_steps = total_steps
        self.current_step = 0
        self.start_time = None
        self.tqdm_instance = None
    
    def start(self, desc: str = "처리 중"):
        """
        진행 상황 추적 시작
        
        Args:
            desc: 진행 상황 설명
        """
        self.start_time = time.time()
        self.current_step = 0
        self.tqdm_instance = tqdm(total=self.total_steps, desc=desc)
    
    def update(self, steps: int = 1):
        """
        진행 상황 업데이트
        
        Args:
            steps: 진행된 단계 수
        """
        if self.tqdm_instance:
            self.current_step += steps
            self.tqdm_instance.update(steps)
    
    def finish(self):
        """진행 상황 추적 완료"""
        if self.tqdm_instance:
            self.tqdm_instance.close()
        
        total_time = time.time() - self.start_time if self.start_time else 0
        return total_time
    
    def get_progress(self) -> float:
        """
        현재 진행률 반환
        
        Returns:
            0.0 ~ 1.0 사이의 진행률
        """
        if self.total_steps <= 0:
            return 1.0
        return min(1.0, self.current_step / self.total_steps)