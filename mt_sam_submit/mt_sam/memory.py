from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import faiss
import numpy as np
from PIL import Image


class MemoryBank:
    def __init__(self, memory_dir: str):
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.memory_dir / "index.json"
        self.index = self._load_index()
        self.faiss_index = None
        self.position_to_id: Dict[int, int] = {}
        self._build_faiss_index()

    def _load_index(self) -> Dict:
        if self.index_path.exists():
            return json.loads(self.index_path.read_text(encoding="utf-8"))
        return {"items": [], "next_id": 0}

    def _save_index(self) -> None:
        self.index_path.write_text(json.dumps(self.index, indent=2, ensure_ascii=False), encoding="utf-8")

    def _item_by_id(self, item_id: int) -> Dict:
        for item in self.index["items"]:
            if int(item["id"]) == int(item_id):
                return item
        raise KeyError(f"memory item not found: {item_id}")

    @staticmethod
    def _normalize(vec: np.ndarray) -> np.ndarray:
        arr = np.asarray(vec, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(arr))
        return arr / norm if norm > 0 else arr

    def _build_faiss_index(self) -> None:
        self.faiss_index = None
        self.position_to_id = {}
        for item in self.index.get("items", []):
            path = self.memory_dir / item["features_path"]
            desc = self._normalize(np.load(str(path))).reshape(1, -1).astype(np.float32)
            faiss.normalize_L2(desc)
            if self.faiss_index is None:
                self.faiss_index = faiss.IndexFlatIP(desc.shape[1])
            pos = int(self.faiss_index.ntotal)
            self.faiss_index.add(desc)
            self.position_to_id[pos] = int(item["id"])

    def add(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        global_features: np.ndarray,
        patch_features: np.ndarray,
        grid_size: Tuple[int, int],
    ) -> int:
        item_id = int(self.index["next_id"])
        self.index["next_id"] = item_id + 1
        item_dir = self.memory_dir / f"item_{item_id}"
        item_dir.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(item_dir / "image.png"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(item_dir / "mask.png"), (np.asarray(mask) > 0).astype(np.uint8) * 255)
        np.save(str(item_dir / "features.npy"), np.asarray(global_features, dtype=np.float32))
        np.save(str(item_dir / "patch_features.npy"), np.asarray(patch_features, dtype=np.float32))
        (item_dir / "patch_info.json").write_text(
            json.dumps({"grid_size": [int(grid_size[0]), int(grid_size[1])]}, indent=2),
            encoding="utf-8",
        )

        item = {
            "id": item_id,
            "image_path": f"item_{item_id}/image.png",
            "mask_path": f"item_{item_id}/mask.png",
            "features_path": f"item_{item_id}/features.npy",
            "patch_features_path": f"item_{item_id}/patch_features.npy",
        }
        self.index["items"].append(item)
        self._save_index()
        self._build_faiss_index()
        return item_id

    def search(self, global_features: np.ndarray, top_k: int = 5) -> List[Dict]:
        if self.faiss_index is None or self.faiss_index.ntotal == 0:
            return []
        query = self._normalize(global_features).reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(query)
        scores, positions = self.faiss_index.search(query, min(int(top_k), int(self.faiss_index.ntotal)))
        out = []
        for score, pos in zip(scores[0], positions[0]):
            item_id = self.position_to_id[int(pos)]
            out.append({"item": self._item_by_id(item_id), "similarity": float(score)})
        return out

    def count(self) -> int:
        return int(len(self.index.get("items", [])))

    def list_items(self) -> List[Dict]:
        return [dict(item) for item in self.index.get("items", [])]

    def get_item(self, item_id: int) -> Dict:
        return dict(self._item_by_id(item_id))

    def load_image_mask(self, item_id: int) -> Tuple[np.ndarray, np.ndarray]:
        item = self._item_by_id(item_id)
        image = np.array(Image.open(str(self.memory_dir / item["image_path"])).convert("RGB"))
        mask = np.array(Image.open(str(self.memory_dir / item["mask_path"])).convert("L")) > 0
        return image, mask

    def delete(self, item_id: int) -> None:
        item = self._item_by_id(item_id)
        item_dir = self.memory_dir / Path(item["image_path"]).parent
        self.index["items"] = [row for row in self.index["items"] if int(row["id"]) != int(item_id)]
        if item_dir.exists():
            shutil.rmtree(item_dir)
        self._save_index()
        self._build_faiss_index()

    def clear(self) -> None:
        for item in list(self.index.get("items", [])):
            item_dir = self.memory_dir / Path(item["image_path"]).parent
            if item_dir.exists():
                shutil.rmtree(item_dir)
        self.index = {"items": [], "next_id": 0}
        self._save_index()
        self._build_faiss_index()

    def load_sparse(self, item_id: int) -> Dict:
        item = self._item_by_id(item_id)
        info = json.loads((self.memory_dir / Path(item["patch_features_path"]).parent / "patch_info.json").read_text())
        return {
            "item": item,
            "mask": np.array(Image.open(str(self.memory_dir / item["mask_path"]))),
            "patch_features": np.load(str(self.memory_dir / item["patch_features_path"])),
            "grid_size": tuple(info["grid_size"]),
        }
