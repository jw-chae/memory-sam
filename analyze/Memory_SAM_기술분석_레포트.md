# Memory-SAM 기술 분석 레포트

## 1. 기술 개요

Memory-SAM은 SAM2(Segment Anything Model 2), DINOv2(Vision Transformer 기반 자가 감독 학습 모델), 그리고 효율적인 메모리 시스템을 통합한 혁신적인 이미지 세그멘테이션 프레임워크입니다. 이 프레임워크는 TCM(전통 중국 의학) 망진 진단에서 혀 세그멘테이션을 위해 설계되었으며, 과거 처리된 이미지와 마스크를 저장하고 검색하여 새로운 이미지의 세그멘테이션을 향상시키는 메모리 메커니즘을 특징으로 합니다.

Memory-SAM의 핵심적인 혁신은 유사성 기반 검색을 통해 과거 사례를 활용하여 세그멘테이션을 안내하는 데 있으며, 이는 특히 소수 샷 학습(few-shot learning)과 도메인 이동(domain shift) 문제가 빈번한 의료 영상 분야에서 큰 가치를 지닙니다.

## 2. 핵심 기술 구성요소

### 2.1 SAM2 (Segment Anything Model 2)

SAM2는 Meta AI에서 개발한 최신 세그멘테이션 모델로, Memory-SAM의 세그멘테이션 엔진 역할을 담당합니다.

**주요 특징**:
- **Hiera 아키텍처**: 계층적 특징 추출을 통해 다양한 스케일의 객체를 세그멘테이션
- **다양한 모델 크기**: tiny, small, large, base+ 등 다양한 크기로 제공
- **프롬프트 유형**: 점(point), 박스(box) 등 다양한 프롬프트 지원
- **다중 마스크 예측**: 여러 후보 마스크를 생성하고 신뢰도 점수를 통해 최적 마스크 선택

코드 구현에서 SAM2는 Hydra 설정 파일을 통해 구성되며, 다음과 같이 초기화됩니다:
```python
self.sam_model = build_sam2(config_file, checkpoint_path, device=self.device)
self.predictor = SAM2ImagePredictor(self.sam_model)
```

### 2.2 DINOv2 (Vision Transformer)

DINOv2는 Facebook Research에서 개발한 자가 감독 학습 기반 Vision Transformer 모델로, 이미지의 의미적 특징을 추출하는 역할을 담당합니다.

**주요 특징**:
- **자가 감독 학습**: 레이블 없이 이미지의 의미적 표현 학습
- **전역 특징 추출**: CLS 토큰을 통해 이미지 전체의 의미적 특징 추출
- **패치 특징 추출**: 이미지를 패치로 분할하여 각 패치별 특징 추출
- **계층적 특징 표현**: 다양한 수준의 의미적 정보 포착

Memory-SAM에서는 두 가지 방식으로 DINOv2를 활용합니다:
1. **전역 특징 추출** (Transformers 라이브러리 사용):
   ```python
   inputs = self.image_processor(images=image_pil, return_tensors="pt").to(self.device)
   outputs = self.dinov2_model(**inputs)
   features = outputs.last_hidden_state[:, 0].cpu().numpy()  # CLS 토큰
   ```

2. **패치 특징 추출** (스파스 매칭용):
   ```python
   image_tensor, grid_size, resize_scale = self.dinov2_matcher.prepare_image(image)
   patch_features = self.dinov2_matcher.extract_features(image_tensor)
   ```

### 2.3 메모리 시스템

메모리 시스템은 Memory-SAM의 핵심으로, 과거 처리된 이미지, 마스크, 특징 벡터를 저장하고 검색하는 역할을 담당합니다.

**주요 특징**:
- **구조화된 저장**: 각 항목은 이미지, 마스크, 특징 벡터, 메타데이터로 구성
- **FAISS 인덱싱**: 고차원 특징 벡터의 효율적인 유사성 검색을 위한 FAISS 활용
- **검색 메커니즘**: 전역 특징 기반 검색과 패치 특징 기반 희소 매칭 지원
- **메타데이터 관리**: 타임스탬프, 소스 경로 등 추가 정보 저장

메모리 시스템의 핵심 구성 요소:
```python
class MemorySystem:
    def __init__(self, memory_dir: str = "memory"):
        # 메모리 디렉토리, 인덱스, FAISS 인덱스 초기화
        
    def add_memory(self, image, mask, features, patch_features, grid_size, resize_scale, metadata):
        # 새 항목 저장 및 인덱싱
        
    def get_most_similar(self, features, top_k, method):
        # 전역 특징 기반 유사 항목 검색
        
    def get_most_similar_sparse(self, patch_features, mask, grid_size, top_k, match_threshold, match_background):
        # 패치 특징 기반 희소 매칭
```

## 3. 알고리즘 작동 원리

### 3.1 초기화 과정

Memory-SAM의 초기화는 다음 단계로 진행됩니다:

1. **환경 설정**: CUDA 가용성 확인 및 TF32 최적화 설정
2. **SAM2 모델 로드**: 지정된 모델 타입에 따라 체크포인트 및 설정 로드
3. **DINOv2 모델 초기화**: 전역 특징 추출용과 스파스 매칭용 모델 설정
4. **메모리 시스템 초기화**: 기존 메모리 인덱스 로드 또는 새 인덱스 생성
5. **결과 디렉토리 설정**: 세그멘테이션 결과 저장 위치 정의

### 3.2 이미지 처리 파이프라인

Memory-SAM의 이미지 처리 과정은 다음과 같은 핵심 단계로 구성됩니다:

#### 1) 이미지 로드 및 특징 추출
- 입력 이미지 로드 및 전처리
- DINOv2를 통한 전역 특징 추출
- 스파스 매칭 활성화 시 패치 특징 추출

#### 2) 메모리 검색
메모리 검색은 두 가지 방식으로 수행됩니다:

**a. 희소 매칭 기반 검색** (기본 방식):
```python
similar_items = self.memory.get_most_similar_sparse(
    patch_features=patch_features,
    mask=mask if mask is not None and mask_valid else None,
    grid_size=grid_size,
    top_k=1,
    match_threshold=self.similarity_threshold,
    match_background=match_background
)
```

희소 매칭의 수학적 원리:
- 각 패치 특징에 대해 최근접 이웃 검색
- 마스크를 기반으로 전경과 배경 특징 분리
- 매칭 비율(match ratio) 계산: `매칭 비율 = (1/n) × Σ 1(d_j < τ)`
- 평균 거리(mean distance) 계산: `평균 거리 = (1/n) × Σ d_j`
- 유사성 점수: `유사성 = 매칭 비율 × (1.0 - 평균 거리)`
- 전경과 배경 유사성 가중 결합: `최종 유사성 = 0.7 × 전경_유사성 + 0.3 × 배경_유사성`

**b. 전역 특징 기반 검색** (희소 매칭 실패 또는 비활성화 시):
```python
similar_items = self.memory.get_most_similar(
    features=features,
    top_k=1
)
```

전역 특징 검색의 수학적 원리:
- 코사인 유사도 계산: `cos(θ) = (f_g^q · f_g^i) / (‖f_g^q‖ × ‖f_g^i‖)`
- FAISS 인덱스를 통한 효율적인 최근접 이웃 검색
- 유사도 순으로 정렬하여 상위 k개 항목 반환

#### 3) 프롬프트 생성
메모리에서 검색된 항목의 마스크를 기반으로 SAM2에 입력할 프롬프트를 생성합니다:

**a. 점 프롬프트 생성**:
```python
# 전경 점 샘플링 (마스크=1인 영역)
fg_points = fg_points_pool[np.random.choice(len(fg_points_pool), min(len(fg_points_pool), num_fg_points), replace=False)]

# 배경 점 샘플링 (마스크=0인 영역)
bg_points = bg_points_pool[np.random.choice(len(bg_points_pool), min(len(bg_points_pool), num_bg_points), replace=False)]

# 좌표와 레이블 결합
points = np.vstack([fg_points, bg_points])
labels = np.array([1] * len(fg_points) + [0] * len(bg_points))
```

**b. 박스 프롬프트 생성**:
```python
# 마스크에서 비-0 픽셀 좌표 찾기
y_indices, x_indices = np.where(mask > 0)

# 경계 상자 계산 (최소/최대 좌표)
x_min, y_min = np.min(x_indices), np.min(y_indices)
x_max, y_max = np.max(x_indices), np.max(y_indices)

# 패딩 추가
x_min = max(0, x_min - 5)
y_min = max(0, y_min - 5)
x_max = min(mask.shape[1] - 1, x_max + 5)
y_max = min(mask.shape[0] - 1, y_max + 5)

box = np.array([x_min, y_min, x_max, y_max])
```

#### 4) SAM2를 통한 세그멘테이션
생성된 프롬프트를 SAM2에 입력하여 세그멘테이션을 수행합니다:

```python
# 이미지 설정
self.predictor.set_image(image)

# 프롬프트 유형에 따른 예측
if "points" in prompt and prompt["points"] is not None:
    masks, scores, _ = self.predictor.predict(
        point_coords=prompt["points"],
        point_labels=prompt["labels"],
        multimask_output=multimask_output
    )
elif "box" in prompt and prompt["box"] is not None:
    masks, scores, _ = self.predictor.predict(
        box=prompt["box"][None, :],
        multimask_output=multimask_output
    )

# 최적 마스크 선택
best_idx = np.argmax(scores)
best_mask = masks[best_idx]
best_score = scores[best_idx]
```

#### 5) 결과 저장 및 메모리 업데이트
세그멘테이션 결과 저장 및 필요시 메모리에 추가:

```python
# 결과 저장
result_path = os.path.join(self.results_dir, f"{os.path.splitext(os.path.basename(image_path))[0]}_mask.png")
cv2.imwrite(result_path, (mask * 255).astype(np.uint8))

# 메모리에 추가 (필요시)
if auto_add_to_memory:
    memory_id = self.memory.add_memory(
        image=image,
        mask=mask,
        features=features,
        patch_features=patch_features,
        grid_size=grid_size,
        resize_scale=resize_scale,
        metadata={"source": image_path, "timestamp": datetime.now().isoformat()}
    )
```

## 4. 기술적 특징 및 최적화

### 4.1 효율적인 유사성 검색

Memory-SAM은 고차원 특징 벡터의 효율적인 유사성 검색을 위해 FAISS 라이브러리를 활용합니다:

```python
# FAISS 인덱스 초기화
self.faiss_index = faiss.IndexFlatL2(self.feature_dim)

# 특징 정규화 및 추가
features = features.reshape(1, -1).astype(np.float32)
faiss.normalize_L2(features)
self.faiss_index.add(features)

# 유사성 검색
D, I = self.faiss_index.search(query_features, top_k)
```

이는 다음과 같은 이점을 제공합니다:
- **시간 복잡도 개선**: 선형 검색 $O(n)$에서 $O(\log n)$으로 최적화
- **GPU 가속**: CUDA 지원으로 대규모 인덱스 처리 가속화
- **메모리 효율성**: 효율적인 인덱싱으로 대규모 특징 벡터 처리 가능

### 4.2 스파스 매칭 및 클러스터링

패치 특징 기반 희소 매칭은 다음과 같은 과정으로 최적화됩니다:

1. **전경/배경 분리**: 마스크를 기준으로 특징 분리하여 관련 영역에 집중
2. **FAISS 기반 매칭**: 최적화된 최근접 이웃 알고리즘 사용
3. **클러스터링 최적화**: 매칭된 특징 포인트를 클러스터링하여 노이즈 제거
   ```python
   def _cluster_feature_points(self, coords1, coords2, similarities, n_clusters=2):
       # 특징 포인트 결합 및 클러스터링
       combined_data = np.hstack([coords1, coords2])
       kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(combined_data)
       
       # 클러스터별 평균 유사도 계산
       cluster_similarities = {}
       for i in range(n_clusters):
           cluster_indices = np.where(kmeans.labels_ == i)[0]
           if len(cluster_indices) > 0:
               cluster_similarities[i] = np.mean(similarities[cluster_indices])
       
       # 유사도가 높은 클러스터 선택
       best_cluster = max(cluster_similarities, key=cluster_similarities.get)
       best_indices = np.where(kmeans.labels_ == best_cluster)[0]
       
       return coords1[best_indices], coords2[best_indices], similarities[best_indices]
   ```

### 4.3 하이브리드 매칭 전략

Memory-SAM은 다양한 시나리오에 적응하기 위해 하이브리드 매칭 전략을 도입합니다:

1. **스파스 매칭 우선**: 세부적인 유사성 탐지에 유리
2. **전역 매칭 폴백**: 스파스 매칭 실패 시 대안으로 사용
3. **하이브리드 클러스터링**: 전경과 배경을 함께 고려한 클러스터링
   ```python
   if self.hybrid_clustering:
       # 하이브리드 모드: 전경과 배경 함께 클러스터링
       all_fg_features = np.vstack([fg_features1, fg_features2])
       all_bg_features = np.vstack([bg_features1, bg_features2])
       combined_similarity = self._compute_hybrid_similarity(all_fg_features, all_bg_features)
   ```

## 5. TCM 망진 진단에의 적용

Memory-SAM은 TCM 망진 진단에서 혀 세그멘테이션을 위한 효과적인 도구를 제공합니다:

### 5.1 혀 영역 세그멘테이션

TCM 진단에서는 혀의 다양한 특징(색상, 코팅, 질감 등)을 분석하기 위해 정확한 혀 영역 세그멘테이션이 필수적입니다:

- **색상 분석**: 붉은 혀, 창백한 혀 등의 색상 패턴 식별
- **코팅 분석**: 노란 코팅, 흰 코팅의 두께와 분포 평가
- **질감 분석**: 혀 표면의 균열, 패턴 등 분석

Memory-SAM은 유사한 혀 이미지에서 학습한 패턴을 활용하여 정확한 세그멘테이션을 제공합니다.

### 5.2 소수 샷 학습 및 도메인 이동 문제 해결

TCM 망진 진단에서 흔히 발생하는 문제들을 Memory-SAM이 효과적으로 해결합니다:

1. **소수 샷 학습**: 제한된 수의 주석 데이터로도 정확한 세그멘테이션 가능
   - 메모리 시스템이 기존 사례를 재활용하여 데이터 효율성 증대
   - 유사 사례 기반 프롬프트 생성으로 주석 의존성 감소

2. **도메인 이동**: 다양한 촬영 조건과 환자 간 변동성 처리
   - 특징 기반 매칭으로 조명, 각도 등의 변화에 강인함
   - 스파스 매칭이 세부적인 유사성을 포착하여 환자별 차이 극복

## 6. 결론 및 확장 가능성

Memory-SAM은 최신 컴퓨터 비전 기술과 메모리 기반 접근법을 결합하여 TCM 망진 진단의 혀 세그멘테이션을 효과적으로 자동화합니다. 이 프레임워크는 다음과 같은 장점을 제공합니다:

1. **데이터 효율성**: 과거 사례를 활용하여 제한된 데이터 환경에서도 우수한 성능
2. **적응성**: 다양한 촬영 조건과 환자 간 변동성에 적응
3. **투명성**: 유사 사례 참조를 통한 결정 근거 제공
4. **확장성**: 혀 세그멘테이션 외에도 다양한 의료 영상 분야로 확장 가능

Memory-SAM은 단순한 세그멘테이션 도구를 넘어, 의료 영상 분석의 새로운 패러다임을 제시하는 혁신적인 프레임워크입니다. 특히 TCM의 고대 지혜와 현대 AI 기술을 연결하여, 전통 의학의 객관화와 표준화에 기여할 수 있는 잠재력을 보여줍니다. 