# Memory SAM 알고리즘 분석

## 1. 개요

Memory SAM은 SAM2(Segment Anything Model 2)와 DINOv2(Vision Transformer 기반 자기지도학습 모델)를 결합하여 메모리 기반 세그멘테이션 시스템을 구현한 프레임워크입니다. 이 시스템은 이전에 처리한 이미지와 마스크를 메모리에 저장하고, 새로운 이미지가 입력되면 유사한 이미지를 메모리에서 검색하여 세그멘테이션 성능을 향상시킵니다.

## 2. 핵심 컴포넌트

### 2.1 SAM2 (Segment Anything Model 2)

SAM2는 Meta AI에서 개발한 최신 세그멘테이션 모델로, 다양한 프롬프트(포인트, 박스 등)를 입력으로 받아 이미지의 객체를 세그멘테이션합니다.

- **모델 구조**: Hiera 아키텍처 기반 (tiny, small, large, base+ 등 다양한 크기)
- **입력 프롬프트 유형**:
  - 포인트 프롬프트: 전경(객체)과 배경을 나타내는 점들의 집합
  - 박스 프롬프트: 객체를 둘러싸는 경계 상자

### 2.2 DINOv2 (Vision Transformer)

DINOv2는 Facebook Research에서 개발한 자기지도학습 기반 비전 트랜스포머 모델로, 이미지의 의미적 특징을 추출하는 데 사용됩니다.

- **특징 추출 방식**:
  - **글로벌 특징**: 이미지 전체의 의미적 특징을 CLS 토큰을 통해 추출
  - **패치 특징**: 이미지를 패치로 분할하여 각 패치별 특징 추출 (스파스 매칭에 사용)

### 2.3 메모리 시스템

메모리 시스템은 처리된 이미지, 마스크, 특징 벡터를 저장하고 검색하는 역할을 합니다.

- **저장 정보**:
  - 원본 이미지
  - 세그멘테이션 마스크
  - DINOv2 글로벌 특징
  - DINOv2 패치 특징 (스파스 매칭용)
  - 메타데이터 (타임스탬프, 원본 경로 등)

## 3. 알고리즘 작동 방식

### 3.1 초기화 과정

```python
def __init__(self, 
            model_type: str = "hiera_l", 
            checkpoint_path: str = None,
            dinov2_model: str = "facebook/dinov2-base",
            dinov2_matching_repo: str = "facebookresearch/dinov2",
            dinov2_matching_model: str = "dinov2_vitb14",
            memory_dir: str = "memory",
            results_dir: str = "results",
            device: str = "cuda",
            use_sparse_matching: bool = True):
```

1. **SAM2 모델 로드**:
   - 지정된 모델 타입(hiera_b+, hiera_l, hiera_s, hiera_t)에 따라 체크포인트 로드
   - Hydra 설정 파일을 통해 모델 구성 로드

2. **DINOv2 모델 초기화**:
   - 글로벌 특징 추출용 DINOv2 모델 로드
   - 스파스 매칭용 DINOv2 매처 초기화 (선택적)

3. **메모리 시스템 초기화**:
   - 지정된 디렉토리에서 기존 메모리 인덱스 로드 또는 새 인덱스 생성

### 3.2 이미지 처리 파이프라인

```python
def process_image(self, 
                 image_path: str, 
                 reference_path: str = None,
                 prompt_type: str = "points",
                 use_sparse_matching: bool = None,
                 match_background: bool = True) -> Dict:
```

#### 3.2.1 특징 추출

1. **이미지 로드**:
   - 단일 이미지 또는 폴더 내 이미지들 로드

2. **글로벌 특징 추출**:
   ```python
   def extract_features(self, image: np.ndarray) -> np.ndarray:
       # DINOv2를 사용하여 이미지에서 글로벌 특징 추출
       inputs = self.image_processor(images=image_pil, return_tensors="pt").to(self.device)
       with torch.no_grad():
           outputs = self.dinov2_model(**inputs)
       features = outputs.last_hidden_state[:, 0].cpu().numpy()
       return features[0]  # CLS 토큰 특징 반환
   ```

3. **패치 특징 추출** (스파스 매칭 활성화 시):
   ```python
   def extract_patch_features(self, image: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int], float]:
       # 이미지를 DINOv2 형식으로 준비
       image_tensor, grid_size, resize_scale = self.dinov2_matcher.prepare_image(image)
       # 패치 특징 추출
       patch_features = self.dinov2_matcher.extract_features(image_tensor)
       return patch_features, grid_size, resize_scale
   ```

#### 3.2.2 메모리 검색

1. **참조 이미지 사용** (제공된 경우):
   - 참조 이미지의 마스크를 사용하여 프롬프트 생성

2. **스파스 매칭을 통한 유사 항목 검색** (활성화된 경우):
   ```python
   def get_most_similar_sparse(self, patch_features: np.ndarray, mask: Optional[np.ndarray] = None, 
                              grid_size: Optional[Tuple[int, int]] = None, top_k: int = 1, 
                              match_threshold: float = 0.8, match_background: bool = True) -> List[Dict]:
   ```
   - **수학적 원리**:
     - 패치 특징 간 최근접 이웃(Nearest Neighbors) 매칭
     - 전경과 배경 영역을 분리하여 매칭 (마스크 기반)
     - 매칭 거리 임계값(match_threshold)을 기준으로 매칭 비율 계산
     - 유사도 = 매칭 비율 × (1.0 - 평균 거리)
     - 전경과 배경 유사도 가중 결합: `0.7 × 전경_유사도 + 0.3 × 배경_유사도`

3. **글로벌 특징 기반 유사 항목 검색** (스파스 매칭 실패 또는 비활성화 시):
   ```python
   def get_most_similar(self, features: np.ndarray, top_k: int = 1, method: str = "global") -> List[Dict]:
   ```
   - **수학적 원리**:
     - 코사인 유사도 계산: `cos(θ) = (a·b) / (||a|| × ||b||)`
     - 유사도가 높은 순으로 정렬하여 상위 k개 항목 반환

#### 3.2.3 프롬프트 생성

1. **메모리 항목의 마스크에서 프롬프트 생성**:
   ```python
   def generate_prompt(self, mask: np.ndarray, prompt_type: str = "points") -> Dict:
   ```

   - **포인트 프롬프트**:
     - 마스크 영역(전경)에서 최대 5개 포인트 무작위 샘플링
     - 비마스크 영역(배경)에서 최대 5개 포인트 무작위 샘플링
     - 각 포인트에 레이블 할당 (1: 전경, 0: 배경)

   - **박스 프롬프트**:
     - 마스크의 경계 상자(bounding box) 계산
     - 약간의 패딩 추가 (5픽셀)

2. **기본 프롬프트 생성** (메모리 항목 없을 때):
   - 이미지 중앙에 포인트 또는 박스 생성

#### 3.2.4 세그멘테이션 수행

```python
def segment_with_sam(self, 
                    image: np.ndarray, 
                    prompt: Dict,
                    multimask_output: bool = True) -> Tuple[np.ndarray, float]:
```

1. **SAM2 예측기에 이미지 설정**:
   ```python
   self.predictor.set_image(image)
   ```

2. **프롬프트 유형에 따른 마스크 생성**:
   - **포인트 프롬프트**:
     ```python
     masks, scores, _ = self.predictor.predict(
         point_coords=prompt["points"],
         point_labels=prompt["labels"],
         multimask_output=multimask_output
     )
     ```
   - **박스 프롬프트**:
     ```python
     masks, scores, _ = self.predictor.predict(
         box=prompt["box"][None, :],
         multimask_output=multimask_output
     )
     ```

3. **최적 마스크 선택**:
   - 점수가 가장 높은 마스크 선택
   ```python
   best_idx = np.argmax(scores)
   best_mask = masks[best_idx]
   best_score = scores[best_idx]
   ```

#### 3.2.5 결과 저장 및 메모리 업데이트

1. **결과 저장**:
   - 원본 이미지, 마스크, 시각화 이미지 저장
   - 폴더 처리 시 모든 이미지에 대한 결과 저장

2. **메모리에 추가**:
   ```python
   memory_id = self.memory.add_memory(
       image=image,
       mask=mask,
       features=features,
       patch_features=patch_features,
       grid_size=grid_size,
       resize_scale=resize_scale,
       metadata={...}
   )
   ```

### 3.3 스파스 매칭 알고리즘

```python
def visualize_sparse_matches(self, image1: np.ndarray, image2: np.ndarray, 
                           mask1: Optional[np.ndarray] = None, 
                           mask2: Optional[np.ndarray] = None) -> np.ndarray:
```

1. **특징 추출**:
   - 두 이미지에서 패치 특징 추출
   - 마스크가 있는 경우 마스크 영역만 필터링

2. **최근접 이웃 매칭**:
   ```python
   knn = NearestNeighbors(n_neighbors=1)
   knn.fit(features1_masked)
   distances, matches = knn.kneighbors(features2_masked)
   ```

3. **매칭 시각화**:
   - 두 이미지를 나란히 배치
   - 매칭된 패치 간 연결선 그리기
   - 각 매칭에 무작위 색상 할당

## 4. 메모리 시스템 구현

### 4.1 메모리 항목 저장

```python
def add_memory(self, 
              image: np.ndarray, 
              mask: np.ndarray, 
              features: np.ndarray,
              patch_features: Optional[np.ndarray] = None,
              grid_size: Optional[Tuple[int, int]] = None,
              resize_scale: Optional[float] = None,
              metadata: Dict = None) -> int:
```

1. **고유 ID 할당**:
   - 각 메모리 항목에 순차적 ID 할당

2. **데이터 저장**:
   - 이미지, 마스크, 특징 벡터를 디스크에 저장
   - 패치 특징, 그리드 크기, 크기 조정 비율 저장 (있는 경우)
   - 메타데이터 저장

3. **인덱스 업데이트**:
   - 메모리 인덱스에 새 항목 추가
   - 인덱스를 JSON 형식으로 디스크에 저장

### 4.2 유사 항목 검색

#### 4.2.1 글로벌 특징 기반 검색

```python
def get_most_similar(self, features: np.ndarray, top_k: int = 1, method: str = "global") -> List[Dict]:
```

1. **코사인 유사도 계산**:
   ```python
   def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
       return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
   ```

2. **유사도 기반 정렬 및 반환**:
   - 모든 메모리 항목과의 유사도 계산
   - 유사도가 높은 순으로 정렬
   - 상위 k개 항목 반환

#### 4.2.2 스파스 매칭 기반 검색

```python
def get_most_similar_sparse(self, patch_features: np.ndarray, mask: Optional[np.ndarray] = None, 
                           grid_size: Optional[Tuple[int, int]] = None, top_k: int = 1, 
                           match_threshold: float = 0.8, match_background: bool = True) -> List[Dict]:
```

1. **전경 매칭**:
   - 메모리 항목의 마스크 영역 특징 추출
   - 최근접 이웃 매칭 수행
   - 매칭 통계 계산 (평균 거리, 매칭 비율, 유사도)

2. **배경 매칭** (선택적):
   - 메모리 항목의 비마스크 영역 특징 추출
   - 최근접 이웃 매칭 수행
   - 매칭 통계 계산

3. **결합 유사도 계산**:
   - 전경과 배경 유사도 가중 결합 (0.7:0.3)
   - 유사도 기반 정렬 및 반환

## 5. DINOv2 매처 구현

```python
class Dinov2Matcher:
    def __init__(self, 
                repo_name="facebookresearch/dinov2", 
                model_name="dinov2_vitb14", 
                smaller_edge_size=448, 
                half_precision=False, 
                device="cuda"):
```

### 5.1 이미지 준비

```python
def prepare_image(self, rgb_image_numpy: np.ndarray) -> Tuple[torch.Tensor, Tuple[int, int], float]:
```

1. **이미지 전처리**:
   - 크기 조정 (작은 쪽 크기를 smaller_edge_size로 조정)
   - 정규화 (ImageNet 평균 및 표준편차 사용)
   - 패치 크기의 배수가 되도록 크기 조정

2. **그리드 크기 계산**:
   - 이미지 크기를 패치 크기로 나누어 그리드 크기 계산
   - 크기 조정 비율 계산 (원본 크기 / 조정된 크기)

### 5.2 특징 추출

```python
def extract_features(self, image_tensor: torch.Tensor) -> np.ndarray:
```

1. **DINOv2 모델 추론**:
   - 이미지 텐서를 모델에 입력
   - 중간 레이어의 토큰 추출
   - CPU로 이동 및 NumPy 배열로 변환

### 5.3 매칭 및 시각화

```python
def find_matches(self, features1: np.ndarray, features2: np.ndarray, k: int = 1) -> Tuple[np.ndarray, np.ndarray]:
```

1. **최근접 이웃 매칭**:
   - scikit-learn의 NearestNeighbors 사용
   - 첫 번째 특징 집합으로 모델 학습
   - 두 번째 특징 집합으로 쿼리 수행
   - 거리와 매칭 인덱스 반환

## 6. 사용자 인터페이스 구현

### 6.1 Gradio 인터페이스

```python
class MemoryGradioInterface:
    def __init__(self, 
                memory_sam_predictor=None, 
                model_type="hiera_l", 
                checkpoint_path=None, 
                dinov2_model="facebook/dinov2-base",
                dinov2_matching_repo="facebookresearch/dinov2",
                dinov2_matching_model="dinov2_vitb14",
                memory_dir="memory", 
                results_dir="results",
                use_sparse_matching=True):
```

1. **모듈 구성**:
   - 세그멘테이션 모듈: 메모리 기반 세그멘테이션 처리
   - 마스크 생성기 모듈: 수동 마스크 생성 기능
   - 메모리 관리자 모듈: 메모리 항목 관리 기능

2. **인터페이스 탭**:
   - 메모리 기반 세그멘테이션 탭
   - 마스크 생성기 탭
   - 메모리 관리자 탭

## 7. 수학적 알고리즘 분석

### 7.1 코사인 유사도

두 벡터 a와 b 사이의 코사인 유사도는 다음과 같이 계산됩니다:

$$\text{similarity} = \cos(\theta) = \frac{\mathbf{a} \cdot \mathbf{b}}{||\mathbf{a}|| \cdot ||\mathbf{b}||}$$

여기서:
- $\mathbf{a} \cdot \mathbf{b}$는 두 벡터의 내적
- $||\mathbf{a}||$와 $||\mathbf{b}||$는 각 벡터의 L2 노름(크기)

### 7.2 스파스 매칭 유사도

스파스 매칭에서 유사도는 다음과 같이 계산됩니다:

1. **매칭 비율 계산**:
   $$\text{match\_ratio} = \frac{1}{n} \sum_{i=1}^{n} \mathbf{1}(d_i < \text{threshold})$$
   
   여기서:
   - $d_i$는 i번째 특징 벡터의 최근접 이웃까지의 거리
   - $\mathbf{1}$은 지시 함수(indicator function)
   - threshold는 매칭 거리 임계값

2. **평균 거리 계산**:
   $$\text{mean\_distance} = \frac{1}{n} \sum_{i=1}^{n} d_i$$

3. **유사도 계산**:
   $$\text{similarity} = \text{match\_ratio} \times (1.0 - \text{mean\_distance})$$

4. **전경과 배경 유사도 결합**:
   $$\text{combined\_similarity} = 0.7 \times \text{fg\_similarity} + 0.3 \times \text{bg\_similarity}$$

### 7.3 최근접 이웃 알고리즘

스파스 매칭에서는 scikit-learn의 NearestNeighbors 알고리즘을 사용합니다:

1. **k-d 트리 구축**:
   - 특징 공간을 효율적으로 분할하는 트리 구조 생성
   - 시간 복잡도: O(n log n), 여기서 n은 특징 벡터의 수

2. **최근접 이웃 검색**:
   - 쿼리 포인트에서 가장 가까운 이웃 찾기
   - 시간 복잡도: O(log n)

## 8. 결론

Memory SAM은 SAM2의 세그멘테이션 능력과 DINOv2의 특징 추출 능력을 결합하여 메모리 기반 세그멘테이션 시스템을 구현합니다. 이 시스템은 이전에 처리한 이미지와 마스크를 메모리에 저장하고, 새로운 이미지가 입력되면 유사한 이미지를 메모리에서 검색하여 세그멘테이션 성능을 향상시킵니다.

핵심 알고리즘은 다음과 같습니다:
1. DINOv2를 사용한 글로벌 및 패치 특징 추출
2. 코사인 유사도 및 스파스 매칭을 통한 유사 이미지 검색
3. 메모리 항목의 마스크를 기반으로 한 프롬프트 생성
4. SAM2를 사용한 세그멘테이션 수행
5. 결과의 메모리 저장 및 재활용

이러한 접근 방식은 세그멘테이션의 정확도와 일관성을 향상시키며, 특히 유사한 객체나 장면이 반복되는 경우에 효과적입니다. 