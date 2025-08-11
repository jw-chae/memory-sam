# DINOv2 Sparse Matching for SAM2

이 문서는 SAM2에 DINOv2 스파스 매칭 기능을 추가하는 방법에 대해 설명합니다.

## 개요

기존 Memory SAM 시스템은 DINOv2의 CLS 토큰을 사용한 글로벌 특징 매칭을 사용했습니다. 이 업데이트는 DINOv2의 패치 수준 특징을 사용하여 더 세밀한 스파스 매칭 기능을 추가합니다.

## 구현 내용

1. `dinov2_matcher.py`: DINOv2 스파스 매칭을 위한 새로운 클래스 구현
2. `memory_system.py`: 패치 특징을 저장하고 스파스 매칭을 통해 유사 이미지를 찾는 기능 추가
3. `memory_sam_predictor.py`: 스파스 매칭 사용을 위한 클래스 수정 및 시각화 기능 추가

## 사용 방법

### 기본 사용법

```python
from scripts.memory_sam_predictor import MemorySAMPredictor

# 스파스 매칭으로 초기화
predictor = MemorySAMPredictor(
    model_type="hiera_l",
    device="cuda",
    use_sparse_matching=True  # 스파스 매칭 활성화
)

# 이미지 처리
result = predictor.process_image(
    image_path="path/to/image.jpg",
    use_sparse_matching=True  # 이 호출에서만 스파스 매칭 사용 (옵션)
)
```

### 비교 실행

제공된 예제 스크립트를 사용하여 글로벌 매칭과 스파스 매칭을 비교할 수 있습니다:

```bash
python scripts/example_sparse_matching.py --image path/to/image.jpg --method both
```

이 스크립트는 두 방법으로 이미지를 처리하고 결과를 비교합니다.

## 구현 세부 사항

### DINOv2 Matcher

`Dinov2Matcher` 클래스는 이미지에서 패치 수준 특징을 추출하고 이미지 간 특징 매칭을 수행합니다.

주요 메서드:
- `prepare_image`: 이미지를 DINOv2 처리를 위해 준비
- `extract_features`: 패치 수준 특징 추출
- `prepare_mask`: 마스크를 특징 그리드 크기로 조정
- `find_matches`: 두 특징 집합 간 매칭 찾기

### Memory System

메모리 시스템이 확장되어 패치 특징도 저장하고 사용할 수 있습니다:

- `add_memory`: 이미지, 마스크, 전역 특징과 함께 패치 특징도 저장
- `get_most_similar_sparse`: 스파스 매칭을 사용하여 유사한 항목 찾기

### MemorySAMPredictor

주요 변경 사항:
- `extract_patch_features`: 이미지에서 패치 특징 추출
- `process_image`: 스파스 매칭 사용 옵션 추가
- `visualize_sparse_matches`: 두 이미지 간 특징 매칭 시각화

## 예제 결과

글로벌 매칭과 스파스 매칭을 모두 사용하면 `comparison_results` 디렉토리에 비교 결과가 저장됩니다:
- `input.png`: 입력 이미지
- `global_mask.png`: 글로벌 매칭으로 생성된 마스크
- `sparse_mask.png`: 스파스 매칭으로 생성된 마스크
- `global_visualization.png`: 글로벌 매칭 결과 시각화
- `sparse_visualization.png`: 스파스 매칭 결과 시각화
- `comparison.png`: 두 방법 비교
- `feature_matches.png`: 스파스 특징 매칭 시각화 (가능한 경우)