#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
향상된 Memory SAM UI를 실행하는 스크립트

이 스크립트는 다음 기능이 추가된 UI를 실행합니다:
- 이미지 리사이징 옵션 (25%, 50%, 75%, 100%)
- 폴더와 이미지 선택을 위한 버튼 인터페이스
- 클러스터링 하이퍼파라미터 실시간 조절 (슬라이더)
- 폴더 처리 진행 상황을 보여주는 프로그레스 바
"""

import os
import sys

# 현재 스크립트 디렉토리를 경로에 추가
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)

# 메인 모듈 임포트
from sam2.main import main

if __name__ == "__main__":
    # 'enhanced' 인터페이스를 기본값으로 하는 sys.argv를 설정
    if len(sys.argv) == 1:
        # 아무 인자도 제공되지 않은 경우, enhanced 인터페이스 사용 인자 추가
        sys.argv.append("--interface")
        sys.argv.append("enhanced")
    else:
        # 인터페이스 타입이 명시적으로 지정되지 않은 경우, enhanced 설정
        interface_specified = False
        for i, arg in enumerate(sys.argv):
            if arg == "--interface" and i + 1 < len(sys.argv):
                interface_specified = True
                break
        
        if not interface_specified:
            sys.argv.append("--interface")
            sys.argv.append("enhanced")
    
    # 메인 함수 실행
    main()