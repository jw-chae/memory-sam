import numpy as np
from typing import List, Tuple

class SparseMatcher:
    """Handles sparse feature matching and clustering."""

    def __init__(self, kmeans_fg_clusters: int = 10):
        self.kmeans_fg_clusters = kmeans_fg_clusters

    def _match_features_with_coords(self, features1, features2, coords1, coords2, 
                                    grid_size1, grid_size2, image1_shape, image2_shape, 
                                    similarity_threshold=0.7, max_matches=100000):
        """Helper function to perform matching between two feature sets using coordinates."""
        if len(features1) == 0 or len(features2) == 0:
            return [], [], []
        
        features1_norm = features1 / np.linalg.norm(features1, axis=1, keepdims=True)
        features2_norm = features2 / np.linalg.norm(features2, axis=1, keepdims=True)
        similarities = np.matmul(features1_norm, features2_norm.T)
        
        best_matches = []
        for i in range(len(features1_norm)):
            best_idx = np.argmax(similarities[i])
            best_sim = similarities[i][best_idx]
            if best_sim >= similarity_threshold:
                best_matches.append((i, best_idx, best_sim))
        
        best_matches.sort(key=lambda x: x[2], reverse=True)
        best_matches = best_matches[:max_matches]
        
        match_coords1, match_coords2, match_similarities = [], [], []
        for i, j, sim in best_matches:
            y1, x1 = coords1[0][i], coords1[1][i]
            y2, x2 = coords2[0][j], coords2[1][j]
            img1_x = int(x1 * (image1_shape[1] / grid_size1[1]))
            img1_y = int(y1 * (image1_shape[0] / grid_size1[0]))
            img2_x = int(x2 * (image2_shape[1] / grid_size2[1]))
            img2_y = int(y2 * (image2_shape[0] / grid_size2[0]))
            match_coords1.append((img1_x, img1_y))
            match_coords2.append((img2_x, img2_y))
            match_similarities.append(sim)
            
        return match_coords1, match_coords2, match_similarities

    def _cluster_feature_points(self, coords1, coords2, similarities, n_clusters=2, is_foreground=True):
        """Helper function to cluster feature points."""
        if len(coords1) <= n_clusters or n_clusters <= 0:
            return coords1, coords2, similarities
        
        try:
            from sklearn.cluster import KMeans
            coords1_array = np.array(coords1)
            coords2_array = np.array(coords2)
            similarities_array = np.array(similarities)
            adjusted_n_clusters = min(n_clusters, len(coords1))
            
            # Use coords1, coords2, and similarities for clustering features
            features = np.column_stack((
                coords1_array, 
                coords2_array, 
                similarities_array.reshape(-1, 1) * 100
            ))
            
            kmeans = KMeans(n_clusters=adjusted_n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(features)
            
            clustered_coords1, clustered_coords2, clustered_similarities = [], [], []
            for i in range(adjusted_n_clusters):
                cluster_indices = np.where(cluster_labels == i)[0]
                if len(cluster_indices) > 0:
                    best_idx_in_cluster = cluster_indices[np.argmax(similarities_array[cluster_indices])]
                    clustered_coords1.append(coords1[best_idx_in_cluster])
                    clustered_coords2.append(coords2[best_idx_in_cluster])
                    clustered_similarities.append(similarities[best_idx_in_cluster])
            
            if clustered_coords1:
                sorted_indices = np.argsort(clustered_similarities)[::-1]
                clustered_coords1 = [clustered_coords1[i] for i in sorted_indices]
                clustered_coords2 = [clustered_coords2[i] for i in sorted_indices]
                clustered_similarities = [clustered_similarities[i] for i in sorted_indices]

            return clustered_coords1, clustered_coords2, clustered_similarities
        except Exception as e:
            print(f"Error during keypoint clustering: {e}")
            return coords1, coords2, similarities
