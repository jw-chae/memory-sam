# 향상된 Memory SAM UI

이 UI는 SAM2와 DINOv2 기반의 메모리 시스템을 활용한 지능형 이미지 세그멘테이션 인터페이스의 향상된 버전입니다.

## 주요 기능

1. **이미지 리사이징 옵션**: 25%, 50%, 75%, 100% 비율로 이미지 크기 조절 가능
2. **폴더와 이미지 버튼 인터페이스**: 시각적으로 폴더와 이미지를 선택할 수 있는 버튼 제공
3. **클러스터링 하이퍼파라미터 실시간 조절**: 슬라이더를 통해 유사도 임계값 등의 매개변수 조절 가능
4. **폴더 처리 진행 상황 모니터링**: 프로그레스 바를 통해 처리 진행 상황 실시간 확인 기능

## 사용 방법

향상된 UI 실행:
```bash
python run_enhanced_ui.py
```

실행 시 추가 옵션:
```bash
python run_enhanced_ui.py --model_type hiera_l --device cuda --share
```

기본 옵션:
- `--model_type`: SAM2 모델 유형 (기본값: hiera_l)
- `--device`: 사용할 디바이스 (기본값: cuda)
- `--share`: Gradio 인터페이스를 공개 URL로 공유 (선택 사항)

## 문제 해결

이 UI는 기존 Memory SAM 시스템을 기반으로 합니다. 아래는 발생할 수 있는 일반적인 문제와 해결 방법입니다:

1. **이미지 처리 속도가 느린 경우**: 
   - 리사이징 옵션을 낮추어 처리 속도 향상
   - 더 작은 이미지 사용

2. **메모리 부족 오류**:
   - 리사이징 옵션을 낮추어 메모리 사용량 감소
   - 작은 배치 크기로 이미지 처리

3. **매칭 품질이 좋지 않은 경우**:
   - 유사도 임계값 조정
   - 배경 가중치 조정
   - 클러스터링 옵션 활성화/비활성화 실험

## 파일 구조

- `ui/memory_sam_ui.py`: 메인 UI 클래스
- `ui/components.py`: UI 구성 요소
- `ui/image_utils.py`: 이미지 처리 유틸리티
- `ui/file_utils.py`: 파일 및 폴더 처리 유틸리티
- `ui/progress_tracker.py`: 진행 상황 추적기
- `ui/segmentation_enhanced.py`: 향상된 세그멘테이션 모듈
- `ui/main.py`: UI 진입점
- `run_enhanced_ui.py`: 실행 스크립트