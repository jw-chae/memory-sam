import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import io
import gradio as gr

# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# device 설정
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

if device.type == "cuda":
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print(
        "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS. "
        "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    )

# SAM2 모델 및 프리딕터 초기화
from sam2.build_sam import build_sam2 
from sam2.sam2_image_predictor import SAM2ImagePredictor

# 환경 변수에서 체크포인트 경로 가져오기 (없으면 기본값 사용)
sam2_checkpoint = os.environ.get("SAM2_CHECKPOINT", "/home/joongwon00/sam2/checkpoints/sam2.1_hiera_large.pt")
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
print(f"사용 체크포인트: {sam2_checkpoint}")
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
predictor = SAM2ImagePredictor(sam2_model)

# --- 시각화 함수들 ---
def show_mask(mask, ax, random_color=False, borders=True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        import cv2
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2) 
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    if coords is None or len(coords) == 0:
        return
    pos_points = coords[labels==1] if np.any(labels==1) else np.empty((0,2))
    neg_points = coords[labels==0] if np.any(labels==0) else np.empty((0,2))
    if pos_points.size > 0:
        ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    if neg_points.size > 0:
        ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def show_box(box, ax):
    if box is None:
        return
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

def visualize_masks(image, masks, scores, point_coords=None, point_labels=None, box_coords=None):
    outputs = []
    for i, (mask, score) in enumerate(zip(masks, scores)):
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(image)
        show_mask(mask, ax, borders=True)
        if point_coords is not None and point_labels is not None:
            show_points(point_coords, point_labels, ax)
        if box_coords is not None:
            show_box(box_coords, ax)
        ax.set_title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        ax.axis('off')
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        out_img = Image.open(buf)
        outputs.append(out_img)
        plt.close(fig)
    return outputs

# --- 입력 문자열 파싱 함수 ---
def parse_points(point_str):
    """
    점 좌표 입력: "x1,y1; x2,y2; ..." 형태의 문자열을 numpy array로 변환
    """
    if not point_str.strip():
        return None
    try:
        points = []
        for pt in point_str.split(';'):
            pt = pt.strip()
            if not pt:
                continue
            x, y = pt.split(',')
            points.append([float(x.strip()), float(y.strip())])
        if len(points) == 0:
            return None
        return np.array(points)
    except Exception as e:
        print("Error parsing points:", e)
        return None

def parse_box(box_str):
    """
    박스 좌표 입력: "x0,y0,x1,y1" 형태의 문자열을 numpy array로 변환
    """
    if not box_str.strip():
        return None
    try:
        vals = [float(x.strip()) for x in box_str.split(',') if x.strip() != '']
        if len(vals) != 4:
            return None
        return np.array(vals)
    except Exception as e:
        print("Error parsing box:", e)
        return None

# --- SAM2 분할 수행 함수 (인터랙티브 입력용) ---
def segment_image(image, points_str, box_str):
    if image is None:
        return []
    predictor.set_image(image)
    
    # points_str를 바탕으로 좌표를 파싱하고, 모든 점은 양성(1)으로 처리
    if points_str and points_str.strip():
        point_coords = parse_points(points_str)
        point_labels = np.ones((point_coords.shape[0],), dtype=int) if point_coords is not None else None
    else:
        point_coords, point_labels = None, None

    box_coords = parse_box(box_str) if box_str and box_str.strip() else None

    try:
        masks, scores, _ = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box_coords,
            multimask_output=True
        )
    except Exception as e:
        print("Prediction error:", e)
        return []
    
    outputs = visualize_masks(image, masks, scores, point_coords, point_labels, box_coords)
    return outputs

# --- Gradio 인터페이스 구성 ---
with gr.Blocks() as demo:
    gr.Markdown("# SAM2 Segmentation GUI (Interactive)")
    
    # 현재 선택 모드 상태 (positive 또는 negative)
    point_mode = gr.State("positive")
    
    with gr.Row():
        with gr.Column():
            # 업로드된 이미지를 인터랙티브하게 보여줌 (클릭 이벤트로 좌표를 얻음)
            image_input = gr.Image(label="Input Image", type="pil", interactive=True)
            
            with gr.Row():
                pos_btn = gr.Button("Positive Points (Green)", variant="primary")
                neg_btn = gr.Button("Negative Points (Red)", variant="secondary")
            
            points_textbox = gr.Textbox(label="Positive Points", placeholder="이미지 클릭 시 좌표가 추가됩니다.", lines=2, visible=True)
            neg_points_textbox = gr.Textbox(label="Negative Points", placeholder="이미지 클릭 시 좌표가 추가됩니다.", lines=2, visible=True)
            box_input = gr.Textbox(label="Box Coordinates", placeholder="예: 50,50,300,300")
            
            with gr.Row():
                clear_btn = gr.Button("Clear Points")
                reset_btn = gr.Button("Reset All", variant="stop")
                
        with gr.Column():
            output_gallery = gr.Gallery(label="Segmentation Masks", columns=1, height="auto")

    # 모드 변경 함수
    def set_positive_mode():
        return "positive"
    
    def set_negative_mode():
        return "negative"
    
    # 점을 추가하는 함수를 수정하여 현재 모드에 따라 positive 또는 negative 포인트 추가
    def add_point_and_segment(current_mode, pos_points, neg_points, box_str, image, evt: gr.SelectData):
        # 점 추가
        x, y = evt.index
        new_point = f"{x},{y}"
        
        # 현재 모드에 따라 적절한 포인트 목록 업데이트
        if current_mode == "positive":
            if pos_points is None or pos_points.strip() == "":
                updated_pos_points = new_point
            else:
                updated_pos_points = pos_points + "; " + new_point
            updated_neg_points = neg_points
        else:  # negative 모드
            if neg_points is None or neg_points.strip() == "":
                updated_neg_points = new_point
            else:
                updated_neg_points = neg_points + "; " + new_point
            updated_pos_points = pos_points
        
        # 세그멘테이션 실행 (positive와 negative 포인트 모두 전달)
        masks = segment_with_pos_neg(image, updated_pos_points, updated_neg_points, box_str)
        
        return updated_pos_points, updated_neg_points, masks

    # positive와 negative 포인트를 모두 사용하는 세그멘테이션 함수
    def segment_with_pos_neg(image, pos_points_str, neg_points_str, box_str):
        if image is None:
            return []
        predictor.set_image(image)
        
        # positive 포인트 파싱
        pos_coords = parse_points(pos_points_str) if pos_points_str and pos_points_str.strip() else None
        
        # negative 포인트 파싱
        neg_coords = parse_points(neg_points_str) if neg_points_str and neg_points_str.strip() else None
        
        # 모든 포인트와 라벨 준비
        if pos_coords is not None and neg_coords is not None:
            point_coords = np.vstack([pos_coords, neg_coords])
            point_labels = np.concatenate([np.ones(len(pos_coords)), np.zeros(len(neg_coords))])
        elif pos_coords is not None:
            point_coords = pos_coords
            point_labels = np.ones(len(pos_coords))
        elif neg_coords is not None:
            point_coords = neg_coords
            point_labels = np.zeros(len(neg_coords))
        else:
            point_coords, point_labels = None, None

        box_coords = parse_box(box_str) if box_str and box_str.strip() else None

        try:
            masks, scores, _ = predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=box_coords,
                multimask_output=True
            )
        except Exception as e:
            print("Prediction error:", e)
            return []
        
        outputs = visualize_masks(image, masks, scores, point_coords, point_labels, box_coords)
        return outputs

    # 점 초기화 함수
    def clear_points():
        return "", ""
    
    # 모든 것을 초기화하는 함수
    def reset_all():
        return None, "", "", "", []

    # 이벤트 연결
    pos_btn.click(fn=set_positive_mode, inputs=[], outputs=point_mode)
    neg_btn.click(fn=set_negative_mode, inputs=[], outputs=point_mode)
    
    image_input.select(
        fn=add_point_and_segment, 
        inputs=[point_mode, points_textbox, neg_points_textbox, box_input, image_input], 
        outputs=[points_textbox, neg_points_textbox, output_gallery]
    )
    
    clear_btn.click(
        fn=clear_points, 
        inputs=[], 
        outputs=[points_textbox, neg_points_textbox]
    )
    
    reset_btn.click(
        fn=reset_all, 
        inputs=[], 
        outputs=[image_input, points_textbox, neg_points_textbox, box_input, output_gallery]
    )
    
    # 박스 입력이 변경될 때도 자동으로 세그멘테이션 실행
    box_input.change(
        fn=segment_with_pos_neg,
        inputs=[image_input, points_textbox, neg_points_textbox, box_input],
        outputs=output_gallery
    )

# 직접 실행될 때만 launch 호출
if __name__ == "__main__":
    demo.launch(share=True)
