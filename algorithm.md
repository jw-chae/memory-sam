# SAM2 메모리 시스템 알고리즘 및 기능 설명

본 문서는 SAM2 메모리 시스템에서 구현된 주요 알고리즘과 기능들에 대한 수학적 설명을 제공합니다.

## 목차

1. [특징 추출 알고리즘](#1-특징-추출-알고리즘)
   - [글로벌 특징 추출](#11-글로벌-특징-추출)
   - [패치 특징 추출](#12-패치-특징-추출)
2. [유사도 계산 알고리즘](#2-유사도-계산-알고리즘)
   - [코사인 유사도](#21-코사인-유사도)
   - [L2 거리 기반 유사도](#22-l2-거리-기반-유사도)
3. [스파스 매칭 알고리즘](#3-스파스-매칭-알고리즘)
   - [특징점 매칭](#31-특징점-매칭)
   - [클러스터링 기반 매칭](#32-클러스터링-기반-매칭)
4. [프롬프트 생성 알고리즘](#4-프롬프트-생성-알고리즘)
   - [포인트 프롬프트 생성](#41-포인트-프롬프트-생성)
   - [박스 프롬프트 생성](#42-박스-프롬프트-생성)
5. [마스크 처리 알고리즘](#5-마스크-처리-알고리즘)
   - [다중 채널 마스크 처리](#51-다중-채널-마스크-처리)
   - [마스크 시각화](#52-마스크-시각화)
6. [메모리 시스템](#6-메모리-시스템)
   - [FAISS 인덱싱](#61-faiss-인덱싱)
   - [메모리 검색](#62-메모리-검색)
7. [클러스터링 기반 세그멘테이션 안정화](#7-클러스터링-기반-세그멘테이션-안정화)
   - [영역 분할 클러스터링](#71-영역-분할-클러스터링)
   - [모호성 검출 및 제거](#72-모호성-검출-및-제거)
   - [신뢰도 맵 생성](#73-신뢰도-맵-생성)

---

## 1. 특징 추출 알고리즘

### 1.1 글로벌 특징 추출

DINOv2 모델을 사용하여 이미지에서 글로벌 특징 벡터를 추출합니다. 이 과정은 다음과 같은 수학적 단계를 따릅니다:

1. 이미지를 DINOv2 모델의 입력 형식으로 변환합니다.
2. 모델을 통해 특징 벡터를 추출합니다.
3. CLS 토큰과 패치 특징의 평균을 가중 결합합니다:

$$\text{features} = \alpha \cdot \text{CLS} + (1 - \alpha) \cdot \text{mean}(\text{patches})$$

여기서 $\alpha = 0.7$은 CLS 토큰의 가중치입니다.

4. L2 정규화를 적용하여 단위 벡터로 변환합니다:

$$\text{normalized\_features} = \frac{\text{features}}{||\text{features}||_2}$$

이렇게 추출된 특징 벡터는 이미지의 전체적인 의미적 정보를 담고 있으며, 메모리 시스템에서 유사한 이미지를 검색하는 데 사용됩니다.

### 1.2 패치 특징 추출

DINOv2 Matcher를 사용하여 이미지의 각 패치에 대한 특징 벡터를 추출합니다:

1. 이미지를 패치 크기의 배수가 되도록 크기를 조정합니다.
2. 각 패치에 대한 특징 벡터를 추출합니다.
3. 각 패치 특징 벡터를 L2 정규화합니다:

$$\text{normalized\_patch\_features}_i = \frac{\text{patch\_features}_i}{||\text{patch\_features}_i||_2}$$

패치 특징은 이미지의 지역적 특성을 포착하며, 스파스 매칭에 사용됩니다.

## 2. 유사도 계산 알고리즘

### 2.1 코사인 유사도

두 특징 벡터 간의 유사도를 계산하기 위해 코사인 유사도를 사용합니다:

$$\text{similarity}(A, B) = \frac{A \cdot B}{||A||_2 \cdot ||B||_2}$$

특징 벡터가 이미 정규화되어 있다면, 단순히 내적으로 계산할 수 있습니다:

$$\text{similarity}(A, B) = A \cdot B$$

코사인 유사도는 -1에서 1 사이의 값을 가지며, 1에 가까울수록 두 벡터가 유사함을 의미합니다.

### 2.2 L2 거리 기반 유사도

FAISS 인덱싱에서는 L2 거리를 기반으로 유사도를 계산합니다:

$$\text{distance}(A, B) = ||A - B||_2$$

이 거리를 유사도로 변환하기 위해 다음 공식을 사용합니다:

$$\text{similarity} = \frac{1}{1 + \text{distance}}$$

## 3. 스파스 매칭 알고리즘

### 3.1 특징점 매칭

두 이미지 간의 특징점 매칭은 다음 단계로 수행됩니다:

1. 두 이미지에서 패치 특징을 추출합니다.
2. 마스크를 사용하여 전경과 배경 특징점을 분리합니다.
3. 각 특징점에 대해 코사인 유사도 행렬을 계산합니다:

$$\text{similarities} = \text{features1\_norm} \cdot \text{features2\_norm}^T$$

4. 각 특징점에 대해 가장 유사한 매칭을 찾습니다:

$$\text{best\_match}_i = \arg\max_j \text{similarities}_{i,j}$$

5. 유사도 임계값보다 높은 매칭만 선택합니다:

$$\text{matches} = \{(i, j, \text{sim}) | \text{sim} \geq \text{threshold}\}$$

6. 유사도에 따라 매칭을 정렬하고 상위 `max_matches`개만 선택합니다.

### 3.2 클러스터링 기반 매칭

매칭된 특징점을 클러스터링하여 대표 매칭을 선택합니다:

1. K-means 클러스터링을 사용하여 매칭된 좌표를 `n_clusters`개의 그룹으로 나눕니다:

$$\text{clusters} = \text{KMeans}(\text{coords}, \text{n\_clusters})$$

2. 각 클러스터에서 유사도가 가장 높은 매칭을 선택합니다:

$$\text{best\_match\_in\_cluster}_i = \arg\max_{j \in \text{cluster}_i} \text{similarities}_j$$

이 방법은 매칭 결과를 더 간결하게 만들고, 중복된 매칭을 제거하는 데 도움이 됩니다.

## 4. 프롬프트 생성 알고리즘

### 4.1 포인트 프롬프트 생성

마스크에서 SAM2 모델을 위한 포인트 프롬프트를 생성합니다:

1. 전경 포인트 생성:
   - 마스크에서 전경 픽셀(값이 1인 픽셀)의 위치를 찾습니다.
   - 무작위로 최대 5개의 전경 포인트를 샘플링합니다.

2. 배경 포인트 생성:
   - 마스크에서 배경 픽셀(값이 0인 픽셀)의 위치를 찾습니다.
   - K-means 클러스터링을 사용하여 배경 픽셀을 클러스터링합니다:
     
     $$\text{clusters} = \text{KMeans}(\text{bg\_coords}, \text{n\_clusters}=3)$$
     
   - 각 클러스터의 중심에 가장 가까운 실제 배경 픽셀을 선택합니다:
     
     $$\text{closest\_point}_i = \arg\min_{p \in \text{cluster}_i} ||\text{center}_i - p||_2$$

3. 전경 및 배경 포인트를 결합하여 최종 프롬프트를 생성합니다.

### 4.2 박스 프롬프트 생성

마스크에서 SAM2 모델을 위한 박스 프롬프트를 생성합니다:

1. 마스크에서 전경 픽셀의 위치를 찾습니다.
2. 전경 픽셀의 경계 상자를 계산합니다:

$$\begin{align}
x_{\min} &= \min(x) - \text{padding} \\
y_{\min} &= \min(y) - \text{padding} \\
x_{\max} &= \max(x) + \text{padding} \\
y_{\max} &= \max(y) + \text{padding}
\end{align}$$

3. 경계 상자가 이미지 범위를 벗어나지 않도록 조정합니다.

## 5. 마스크 처리 알고리즘

### 5.1 다중 채널 마스크 처리

다중 채널 마스크(예: RGB 마스크)를 단일 채널 이진 마스크로 변환합니다:

1. 마스크의 차원을 확인합니다.
2. 다중 채널 마스크인 경우:
   - 모든 채널이 동일한지 확인합니다:
     
     $$\text{is\_identical} = \forall i,j: \text{mask}[:,:,i] = \text{mask}[:,:,j]$$
     
   - 동일하면 첫 번째 채널만 사용합니다.
   - 그렇지 않으면 그레이스케일로 변환합니다:
     
     $$\text{mask\_gray} = 0.299 \cdot R + 0.587 \cdot G + 0.114 \cdot B$$

3. 이진 마스크로 변환합니다:

$$\text{binary\_mask} = \text{mask} > 0$$

### 5.2 마스크 시각화

마스크를 시각적으로 표현하기 위한 알고리즘:

1. 마스크를 단일 채널 이진 마스크로 변환합니다.
2. 색상 오버레이를 생성합니다:
   
   $$\text{color\_mask}[mask > 0] = [30, 144, 255]$$

3. 원본 이미지와 색상 마스크를 블렌딩합니다:
   
   $$\text{vis} = (1 - \alpha) \cdot \text{image} + \alpha \cdot \text{color\_mask}$$

4. 마스크의 윤곽선을 추출하고 그립니다:
   
   $$\text{contours} = \text{findContours}(\text{mask})$$

## 6. 메모리 시스템

### 6.1 FAISS 인덱싱

FAISS(Facebook AI Similarity Search)를 사용하여 특징 벡터를 효율적으로 인덱싱합니다:

1. 특징 벡터의 차원 `d`를 결정합니다.
2. L2 거리를 사용하는 FAISS 인덱스를 생성합니다:
   
   $$\text{index} = \text{IndexFlatL2}(d)$$

3. 특징 벡터를 정규화하고 인덱스에 추가합니다:
   
   $$\text{normalized\_features} = \frac{\text{features}}{||\text{features}||_2}$$
   $$\text{index.add}(\text{normalized\_features})$$

### 6.2 메모리 검색

메모리에서 유사한 항목을 검색하는 알고리즘:

1. 글로벌 특징 기반 검색:
   - 쿼리 이미지의 글로벌 특징을 추출합니다.
   - FAISS 인덱스를 사용하여 가장 유사한 `top_k`개의 항목을 검색합니다:
     
     $$\text{distances}, \text{indices} = \text{index.search}(\text{query\_features}, \text{top\_k})$$
     
   - 거리를 유사도로 변환합니다:
     
     $$\text{similarity} = \frac{1}{1 + \text{distance}}$$

2. 스파스 매칭 기반 검색:
   - 쿼리 이미지의 패치 특징을 추출합니다.
   - 각 메모리 항목에 대해 스파스 매칭을 수행합니다.
   - 전경과 배경 영역을 분리하여 매칭합니다.
   - 매칭 비율과 평균 거리를 계산합니다:
     
     $$\text{match\_ratio} = \frac{\text{num\_matches}}{\min(\text{num\_features1}, \text{num\_features2})}$$
     $$\text{mean\_distance} = \frac{1}{|\text{matches}|} \sum_{(i,j) \in \text{matches}} ||\text{features1}_i - \text{features2}_j||_2$$
     
   - 전경과 배경의 유사도를 결합하여 최종 유사도를 계산합니다:
     
     $$\text{similarity} = w_{\text{fg}} \cdot \text{similarity\_fg} + w_{\text{bg}} \cdot \text{similarity\_bg}$$

## 7. 클러스터링 기반 세그멘테이션 안정화

### 7.1 영역 분할 클러스터링

세그멘테이션의 안정성을 높이기 위해 특징 공간에서 전경, 배경 및 전체 이미지에 대한 클러스터링을 수행합니다:

1. 전경 및 배경 특징 분리:

$$\begin{align}
\mathcal{F}_{fg} &= \{f_i \mid i \in \text{전경 픽셀 인덱스}\} \\
\mathcal{F}_{bg} &= \{f_i \mid i \in \text{배경 픽셀 인덱스}\}
\end{align}$$

2. 영역별 클러스터링:

**전경 클러스터링:**
$$\mathcal{C}_{fg} = \{C_{fg,1}, C_{fg,2}, \ldots, C_{fg,k_{fg}}\}$$

여기서 각 클러스터 $C_{fg,j}$는 다음과 같이 정의됩니다:
$$C_{fg,j} = \{f_i \in \mathcal{F}_{fg} \mid \arg\min_l ||f_i - \mu_{fg,l}||_2^2 = j\}$$

$\mu_{fg,j}$는 전경 클러스터 $j$의 중심입니다.

**배경 클러스터링:**
$$\mathcal{C}_{bg} = \{C_{bg,1}, C_{bg,2}, \ldots, C_{bg,k_{bg}}\}$$

3. 전체 이미지 클러스터링:
$$\mathcal{C}_{all} = \{C_{all,1}, C_{all,2}, \ldots, C_{all,k_{all}}\}$$

### 7.2 모호성 검출 및 제거

세그멘테이션의 불확실한 영역을 식별하고 제거합니다:

1. 클러스터 순도 계산:

$$\begin{align}
p_{fg}(C_{all,j}) &= \frac{|\{f_i \in C_{all,j} \mid i \in \text{전경 픽셀 인덱스}\}|}{|C_{all,j}|} \\
p_{bg}(C_{all,j}) &= \frac{|\{f_i \in C_{all,j} \mid i \in \text{배경 픽셀 인덱스}\}|}{|C_{all,j}|}
\end{align}$$

2. 모호한 클러스터 식별:

$$\text{ambiguity}(C_{all,j}) = 1 - |p_{fg}(C_{all,j}) - p_{bg}(C_{all,j})|$$

모호성이 임계값보다 높은 클러스터를 모호한 클러스터로 식별합니다:

$$\mathcal{C}_{ambiguous} = \{C_{all,j} \mid \text{ambiguity}(C_{all,j}) > \tau_{amb}\}$$

여기서 $\tau_{amb}$는 모호성 임계값입니다(예: 0.3).

3. 모호한 영역 제거:

$$\text{mask}_{refined}(i) = \begin{cases}
\text{mask}_{original}(i), & \text{if } f_i \notin \bigcup_{C \in \mathcal{C}_{ambiguous}} C \\
\text{undefined}, & \text{otherwise}
\end{cases}$$

### 7.3 신뢰도 맵 생성

각 픽셀의 세그멘테이션 신뢰도를 계산하여 후속 처리에 활용합니다:

$$\text{confidence}(i) = \begin{cases}
|p_{fg}(C_{all,j}) - p_{bg}(C_{all,j})|, & \text{if } f_i \in C_{all,j} \\
0, & \text{if } f_i \in \bigcup_{C \in \mathcal{C}_{ambiguous}} C
\end{cases}$$

이 신뢰도 맵은 다음과 같은 용도로 활용할 수 있습니다:

1. 시간적 일관성 유지: 연속된 프레임에서 신뢰도가 낮은 영역은 이전 프레임의 세그멘테이션을 참조하여 보정합니다.
2. 사용자 피드백 요청: 신뢰도가 낮은 영역에 대해 사용자의 추가 입력을 요청합니다.
3. 세그멘테이션 가중치 조정: 신뢰도 맵을 기반으로 최종 세그멘테이션 결과의 가중치를 조정합니다.

이러한 클러스터링 기반 세그멘테이션 안정화 알고리즘은 특히 객체의 경계 부분이나 텍스처가 복잡한 영역에서 세그멘테이션의 정확도와 일관성을 크게 향상시킬 수 있습니다.

