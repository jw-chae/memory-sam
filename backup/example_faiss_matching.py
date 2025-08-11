#!/usr/bin/env python3
"""
FAISS를 사용한 이미지 유사도 검색 예제

이 스크립트는 DINOv2 모델과 FAISS를 사용하여 이미지 유사도 검색을 수행하는 방법을 보여줍니다.
"""

import os
import sys
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import faiss
import argparse
import matplotlib.pyplot as plt
from pathlib import Path
import time

def parse_args():
    parser = argparse.ArgumentParser(description='FAISS를 사용한 이미지 유사도 검색 예제')
    parser.add_argument('--dataset', type=str, default='./dataset', help='이미지 데이터셋 경로')
    parser.add_argument('--query', type=str, required=True, help='쿼리 이미지 경로')
    parser.add_argument('--output', type=str, default='./faiss_results', help='결과 저장 경로')
    parser.add_argument('--top_k', type=int, default=5, help='반환할 유사 이미지 수')
    parser.add_argument('--model', type=str, default='facebook/dinov2-small', help='사용할 모델')
    parser.add_argument('--device', type=str, default='cuda', help='사용할 장치 (cuda 또는 cpu)')
    parser.add_argument('--index_path', type=str, default='vector.index', help='FAISS 인덱스 저장 경로')
    parser.add_argument('--build_index', action='store_true', help='인덱스를 새로 구축할지 여부')
    return parser.parse_args()

def load_model(model_name, device):
    """모델 및 프로세서 로드"""
    print(f"모델 로드 중: {model_name}")
    device = torch.device(device if torch.cuda.is_available() and device == 'cuda' else "cpu")
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    return processor, model, device

def get_image_paths(dataset_path):
    """데이터셋 폴더에서 모든 이미지 경로 가져오기"""
    image_paths = []
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.lower().endswith(('jpg', 'jpeg', 'png', 'bmp')):
                image_paths.append(os.path.join(root, file))
    return image_paths

def extract_features(image_path, processor, model, device):
    """이미지에서 특징 추출"""
    img = Image.open(image_path).convert('RGB')
    with torch.no_grad():
        inputs = processor(images=img, return_tensors="pt").to(device)
        outputs = model(**inputs)
    # 전역 특징 사용 (CLS 토큰 또는 평균 풀링)
    features = outputs.last_hidden_state.mean(dim=1)
    return features.cpu().numpy()

def build_faiss_index(image_paths, processor, model, device, index_path):
    """FAISS 인덱스 구축"""
    # 첫 번째 이미지로 특징 차원 결정
    first_features = extract_features(image_paths[0], processor, model, device)
    d = first_features.shape[1]  # 특징 차원
    
    # FAISS 인덱스 생성
    index = faiss.IndexFlatL2(d)
    
    # 모든 이미지 처리
    print(f"인덱스 구축 중: {len(image_paths)} 이미지 처리...")
    t0 = time.time()
    
    for i, image_path in enumerate(image_paths):
        if i % 10 == 0:
            print(f"처리 중: {i}/{len(image_paths)}")
        
        try:
            # 특징 추출
            features = extract_features(image_path, processor, model, device)
            
            # 특징 정규화 및 추가
            features = features.astype(np.float32)
            faiss.normalize_L2(features)
            index.add(features)
        except Exception as e:
            print(f"이미지 처리 중 오류 발생: {image_path}, {e}")
    
    print(f"인덱스 구축 완료: {time.time() - t0:.2f}초 소요, {index.ntotal} 항목")
    
    # 인덱스 저장
    faiss.write_index(index, index_path)
    print(f"인덱스 저장됨: {index_path}")
    
    return index

def search_similar_images(query_image_path, image_paths, processor, model, device, index, top_k=5):
    """쿼리 이미지와 유사한 이미지 검색"""
    # 쿼리 이미지 특징 추출
    query_features = extract_features(query_image_path, processor, model, device)
    query_features = query_features.astype(np.float32)
    faiss.normalize_L2(query_features)
    
    # 검색 수행
    k = min(top_k, index.ntotal)
    distances, indices = index.search(query_features, k)
    
    # 결과 변환
    results = []
    for i in range(len(indices[0])):
        idx = indices[0][i]
        distance = distances[0][i]
        if idx < len(image_paths):
            results.append({
                "image_path": image_paths[idx],
                "distance": distance,
                "similarity": 1.0 / (1.0 + distance)
            })
    
    return results

def visualize_results(query_image_path, results, output_path):
    """검색 결과 시각화"""
    # 결과 디렉토리 생성
    os.makedirs(output_path, exist_ok=True)
    
    # 쿼리 이미지 로드
    query_img = Image.open(query_image_path).convert('RGB')
    
    # 결과 이미지 로드
    result_imgs = []
    for result in results:
        img = Image.open(result["image_path"]).convert('RGB')
        result_imgs.append(img)
    
    # 시각화
    n = len(result_imgs)
    fig, axs = plt.subplots(1, n+1, figsize=(4*(n+1), 4))
    
    # 쿼리 이미지
    axs[0].imshow(query_img)
    axs[0].set_title("쿼리 이미지")
    axs[0].axis('off')
    
    # 결과 이미지
    for i, (img, result) in enumerate(zip(result_imgs, results)):
        axs[i+1].imshow(img)
        axs[i+1].set_title(f"유사도: {result['similarity']:.3f}")
        axs[i+1].axis('off')
    
    plt.tight_layout()
    result_path = os.path.join(output_path, "results.png")
    plt.savefig(result_path, dpi=300)
    print(f"결과 저장됨: {result_path}")
    
    # 결과 텍스트 파일 저장
    with open(os.path.join(output_path, "results.txt"), "w") as f:
        f.write(f"쿼리 이미지: {query_image_path}\n\n")
        for i, result in enumerate(results):
            f.write(f"{i+1}. 이미지: {result['image_path']}\n")
            f.write(f"   유사도: {result['similarity']:.6f}\n")
            f.write(f"   거리: {result['distance']:.6f}\n\n")

def main():
    # 인자 파싱
    args = parse_args()
    
    # 출력 디렉토리 생성
    os.makedirs(args.output, exist_ok=True)
    
    # 모델 로드
    processor, model, device = load_model(args.model, args.device)
    
    # 이미지 경로 가져오기
    image_paths = get_image_paths(args.dataset)
    if not image_paths:
        print(f"오류: 데이터셋 경로 '{args.dataset}'에서 이미지를 찾을 수 없습니다.")
        return
    
    print(f"데이터셋에서 {len(image_paths)} 이미지를 찾았습니다.")
    
    # FAISS 인덱스 구축 또는 로드
    if args.build_index or not os.path.exists(args.index_path):
        print("새 FAISS 인덱스 구축 중...")
        index = build_faiss_index(image_paths, processor, model, device, args.index_path)
    else:
        print(f"기존 FAISS 인덱스 로드 중: {args.index_path}")
        index = faiss.read_index(args.index_path)
        print(f"인덱스 로드됨: {index.ntotal} 항목")
    
    # 유사 이미지 검색
    print(f"쿼리 이미지 처리 중: {args.query}")
    results = search_similar_images(args.query, image_paths, processor, model, device, index, args.top_k)
    
    # 결과 시각화
    visualize_results(args.query, results, args.output)
    
    print("처리 완료!")

if __name__ == "__main__":
    main() 