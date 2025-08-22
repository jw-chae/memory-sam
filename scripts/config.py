from dataclasses import dataclass


@dataclass
class ResizeConfig:
    enabled: bool = False
    scale: object = 1.0  # "512x512" or float


@dataclass
class MatchingConfig:
    similarity_threshold: float = 0.8
    background_weight: float = 0.3
    skip_clustering: bool = False
    hybrid_clustering: bool = False
    max_positive_points: int = 10
    max_negative_points: int = 5
    use_positive_kmeans: bool = False
    positive_kmeans_clusters: int = 5


@dataclass
class AppConfig:
    resize: ResizeConfig = ResizeConfig()
    matching: MatchingConfig = MatchingConfig()


