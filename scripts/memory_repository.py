import json
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from PIL import Image


class MemoryRepository:
    """파일시스템 기반 메모리 항목 저장/로드 담당.

    I/O 책임을 분리하여 상위 로직은 순수 계산에 집중할 수 있게 합니다.
    """

    def __init__(self, memory_dir: str):
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(exist_ok=True, parents=True)
        self.index_path = self.memory_dir / "index.json"

    # ---- Index ----
    def load_index(self) -> Dict:
        if self.index_path.exists():
            with open(self.index_path, "r") as f:
                return json.load(f)
        return {"items": [], "next_id": 0}

    def save_index(self, index: Dict) -> None:
        with open(self.index_path, "w") as f:
            json.dump(index, f, indent=2)

    # ---- Item I/O ----
    def create_item_dir(self, item_id: int) -> Path:
        d = self.memory_dir / f"item_{item_id}"
        d.mkdir(exist_ok=True)
        return d

    def save_image(self, item_dir: Path, image: np.ndarray) -> str:
        path = item_dir / "image.png"
        Image.fromarray(image).save(str(path))
        return str(path.relative_to(self.memory_dir))

    def save_mask(self, item_dir: Path, mask: np.ndarray) -> str:
        path = item_dir / "mask.png"
        if mask.dtype != np.uint8:
            mask = (mask * 255).astype(np.uint8)
        Image.fromarray(mask).save(str(path))
        return str(path.relative_to(self.memory_dir))

    def save_features(self, item_dir: Path, features: np.ndarray) -> str:
        path = item_dir / "features.npy"
        np.save(str(path), features)
        return str(path.relative_to(self.memory_dir))

    def save_patch_features(
        self, item_dir: Path, patch_features: Optional[np.ndarray], grid_size: Optional[Tuple[int, int]], resize_scale: Optional[float]
    ) -> Optional[str]:
        if patch_features is None:
            return None
        path = item_dir / "patch_features.npy"
        np.save(str(path), patch_features)
        # patch info
        if grid_size is not None and resize_scale is not None:
            with open(item_dir / "patch_info.json", "w") as f:
                json.dump({"grid_size": grid_size, "resize_scale": resize_scale}, f)
        return str(path.relative_to(self.memory_dir))

    def load_item_files(self, item: Dict) -> Dict:
        image = np.array(Image.open(str(self.memory_dir / item["image_path"])))
        mask = np.array(Image.open(str(self.memory_dir / item["mask_path"])))
        features = np.load(str(self.memory_dir / item["features_path"]))
        result = {"image": image, "mask": mask, "features": features}
        if "patch_features_path" in item:
            pf = self.memory_dir / item["patch_features_path"]
            if os.path.exists(pf):
                result["patch_features"] = np.load(str(pf))
                info = self.memory_dir / Path(item["patch_features_path"]).parent / "patch_info.json"
                if info.exists():
                    with open(info, "r") as f:
                        meta = json.load(f)
                    result["grid_size"] = tuple(meta["grid_size"])  # type: ignore
                    result["resize_scale"] = meta["resize_scale"]
        return result


